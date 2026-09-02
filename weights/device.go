// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package weights

import (
	"encoding/binary"
	"errors"
	"fmt"

	"golang.design/x/accel"
)

// poolChunk is how much a pool reserves when a tensor does not need more. Pools
// never grow and a device caps how many live allocations a process may hold, so
// a pool per tensor is not an option: a Qwen3-0.6B load is 311 buffers.
const poolChunk = 64 << 20

// poolGranularity is accel's suballocation granularity, from
// specs/001-device-resources.md §3.1. It is not queryable, so a pool sized to
// the exact byte total of its buffers would run out on the last one; the
// arena rounds every request to it before deciding whether the current pool has
// room.
const poolGranularity = 256

// arena hands out weight buffers, creating pools as it fills them.
//
// It also decides how bytes reach the device. On unified memory a MemoryShared
// pool's mapping *is* device memory, so the conversion writes straight into the
// destination through Buffer.Access and the converted host plane never exists
// (001-D8). Access refuses a pool with no host mapping — reported rather than
// discovered — so on a discrete device the arena allocates MemoryDevice and
// stages through Queue.WriteBuffer, which is the honest shape there (001-D9).
type arena struct {
	dev   *accel.Device
	kind  accel.MemoryKind
	max   int
	pools []*accel.Pool
	cur   *accel.Pool
	free  int
}

// forceStaging makes newArena take the Queue.WriteBuffer path even where the
// device has unified memory. The fallback 001-D9 specifies is the shape a
// discrete GPU always gets and the shape this project's CI machines never do,
// so without this the branch would ship untested. Nothing outside this
// package's own tests sets it.
var forceStaging bool

func newArena(dev *accel.Device) *arena {
	kind := accel.MemoryDevice
	if dev.Capabilities().SharedMemoryKind && !forceStaging {
		kind = accel.MemoryShared
	}
	max := dev.Limits().MaxPoolBytes
	if max <= 0 {
		max = poolChunk
	}
	return &arena{dev: dev, kind: kind, max: max}
}

// mapped reports whether the arena writes into device memory directly. It is
// what the load report prints, because "the converted copy does not exist" and
// "the converted copy is staged" are different peak-memory promises and a
// caller sizing a machine needs to know which one it got.
func (a *arena) mapped() bool { return a.kind == accel.MemoryShared }

// poolHeadroom is the fraction of a pool's size kept spare, as a divisor.
//
// A pool sized to exactly the bytes it will hold can refuse its own contents:
// accel's suballocator rounds a request up to a size class before searching for
// a block, so a request for the whole pool lands in the class above the one the
// pool's single free block sits in. A sixteenth plus one granularity is past
// the widest that round-up can be, and alloc retries in a fresh pool anyway, so
// the arithmetic is a first choice rather than a requirement.
const poolHeadroom = 16

// alloc returns a buffer of count elements of dt.
func (a *arena) alloc(dt accel.DType, count int, label string) (*accel.Buffer, error) {
	need := roundUp(dt.Size()*count, poolGranularity)
	desc := accel.BufferDescriptor{
		DType: dt, Count: count, Label: label,
		Usage: accel.BufferStorage | accel.BufferCopyDst | accel.BufferCopySrc,
	}
	if a.cur != nil && a.free >= need {
		if b, err := a.cur.AllocBuffer(desc); err == nil {
			a.free -= need
			return b, nil
		}
		// The pool reports the space and cannot hand it out contiguously. Open
		// another rather than failing: fragmentation is the allocator's business
		// and the loader's answer to it is a fresh arena.
	}
	if err := a.grow(need, label); err != nil {
		return nil, err
	}
	b, err := a.cur.AllocBuffer(desc)
	if err != nil {
		return nil, fmt.Errorf("weights: %s: %w", label, err)
	}
	a.free -= need
	return b, nil
}

// grow opens a pool that can hold at least need bytes.
func (a *arena) grow(need int, label string) error {
	want := need + need/poolHeadroom + poolGranularity
	size := min(max(want, poolChunk), a.max)
	if size < want {
		// The pool has to be larger than the buffer it holds: accel's
		// suballocator rounds a request up to a size class before searching, so
		// a pool sized to exactly its contents refuses them. Naming `need`
		// alone here read as "needs N bytes, more than MaxPoolBytes of N" when
		// a tensor landed exactly on the cap, which is a self-contradiction.
		return fmt.Errorf("weights: %s is %d bytes and needs a pool of %d, more than "+
			"this device's MaxPoolBytes of %d", label, need, want, a.max)
	}
	p, err := a.dev.NewPool(accel.PoolDescriptor{
		Kind: a.kind, Bytes: size, Policy: accel.PoolGeneral, Label: "tgo.weights",
	})
	if err != nil {
		return fmt.Errorf("weights: %s: %w", label, err)
	}
	a.pools = append(a.pools, p)
	a.cur, a.free = p, size
	return nil
}

// fill runs write over the buffer's bytes.
//
// On a host-visible pool that slice is the device's own memory and write
// produces the converted plane in place. Otherwise the bytes are built once on
// the host and staged, which costs the copy §7.1 removes on unified memory.
func (a *arena) fill(b *accel.Buffer, write func(dst []byte) error) error {
	if a.mapped() {
		return b.Access(write)
	}
	host := make([]byte, b.Bytes())
	if err := write(host); err != nil {
		return err
	}
	return a.stage(b, host)
}

// stage copies host bytes through the queue. Queue.WriteBuffer takes a typed
// slice rather than bytes, so the plane is re-read at the buffer's dtype here.
// The extra pass exists only on the fallback path; making the conversion itself
// dtype-generic would cost it on the path that matters.
//
// One case per width a weight is stored in, and the set is closed: f16, int8's
// codes, and int4's, which are packed eight to a u32 word. A width with no case
// is refused rather than guessed, because guessing means writing a buffer full
// of plausible numbers.
//
// U32 was missing until 2026-08-27 and int4 could not load on a discrete
// device. It survived because the branch runs only where memory is not
// unified, which is every GPU this project targets and no machine it is tested
// on — so [forceStaging] exists, and every width now has a test that takes it.
func (a *arena) stage(b *accel.Buffer, host []byte) error {
	switch b.DType() {
	case accel.F16:
		u := make([]uint16, len(host)/2)
		for i := range u {
			u[i] = binary.LittleEndian.Uint16(host[2*i:])
		}
		return a.dev.Queue().WriteBuffer(b, 0, u)
	case accel.I8:
		q := make([]int8, len(host))
		for i, v := range host {
			q[i] = int8(v)
		}
		return a.dev.Queue().WriteBuffer(b, 0, q)
	case accel.U32:
		w := make([]uint32, len(host)/4)
		for i := range w {
			w[i] = binary.LittleEndian.Uint32(host[4*i:])
		}
		return a.dev.Queue().WriteBuffer(b, 0, w)
	default:
		return fmt.Errorf("weights: no staging path for %v", b.DType())
	}
}

// flush waits for every staged write. It is a no-op on the mapped path, where
// the bytes were already the device's.
func (a *arena) flush() error {
	if a.mapped() {
		return nil
	}
	return a.dev.Queue().Flush().Wait()
}

// close releases the pools. Buffers are closed by their owner first; a pool
// reports its live children rather than freeing memory out from under them.
func (a *arena) close() error {
	var errs []error
	for _, p := range a.pools {
		if err := p.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	a.pools, a.cur, a.free = nil, nil, 0
	return errors.Join(errs...)
}

func roundUp(n, to int) int { return (n + to - 1) / to * to }
