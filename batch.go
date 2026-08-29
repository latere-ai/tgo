// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"errors"
	"fmt"
	"maps"
	"sync"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/internal/prefix"
	"github.com/latere-ai/tgo/model"
)

// Batch runs several sequences in one forward pass.
//
// # What it is, and what it is not
//
// It is specs/008-scheduler.md §2's slots and the dispatch over them: the
// device buffers a batched step binds, the plan it submits, and the arithmetic
// that turns "these slots contributed these tokens" into one set of ports. It
// is the mechanism.
//
// It is **not** the scheduler. Which requests are admitted, when a slot is
// preempted and which victim is chosen are §3 and §4, and they are policy over
// this. Building both at once would make a batching bug and an admission bug
// indistinguishable, which is 008-D8's argument applied a second time.
//
// # Why a decode step is memory-bound, and why that is the whole point
//
// A decode reads every weight to produce one token. Two sequences decoding
// together read those weights **once** and produce two tokens, so at batch B
// one step costs (W + B·A)/β for weight bytes W, per-sequence traffic A and
// bandwidth β. Throughput is near-linear in B until B·A approaches W
// (008 §1).
//
// # It requires a shared block pool
//
// Sequences that step together have different lengths, so a contiguous cache
// would pad every one of them to the longest — which is the allocation paging
// exists to avoid. Open the model with WithPrefixCache(CacheProcess, ...).
type Batch struct {
	m     *Model
	slots []*batchSlot

	// rows is the largest number of query rows one step may carry, and
	// buckets rounds a step's total up to a plan shape.
	rows    int
	buckets tensor.Buckets

	// The step's ports, allocated once at the largest shape they can take.
	ids, posq, posk, slots1 *accel.Buffer
	lengths, extents, last  *accel.Buffer
	pageBuf, logits         *accel.Buffer

	host    model.BatchStep
	hLogits []float32

	// binds caches one Bindings per plan shape, because a step that rebuilt
	// three hundred weight entries would put a map allocation in every
	// submission (007 §5.1).
	binds map[int]tensor.Bindings

	mu     sync.Mutex
	closed bool
}

// batchSlot is one slot's state: its blocks, and what it holds.
//
// It is not a [Session]. A Session owns the buffers one step binds, and in a
// batch there is one set of those for the whole dispatch — which is the same
// move Wave 8 made for the key and value states, one layer up.
type batchSlot struct {
	lease   *prefix.Lease
	pages   []int
	length  int
	history []int

	// out is this slot's logits from its last step, copied out of the shared
	// readback. One buffer per slot, allocated once: the copy is host to host
	// against a readback of the same bytes off the device, so it is noise
	// beside what it makes safe.
	out []float32
}

// Work is what one slot contributes to a step.
type Work struct {
	// Slot is the index in [0, Batch.Slots()).
	Slot int

	// Tokens are the ids this slot contributes. A slot that contributes
	// nothing is simply absent from the step's work.
	Tokens []int
}

// ErrNoBlockPool is what [Model.NewBatch] refuses a model with no shared block
// pool with.
var ErrNoBlockPool = errors.New("tgo: a batched step needs a shared block pool")

// NewBatch reserves n slots over the model's shared block pool.
//
// n is fixed for the life of the batch, and 008-D1 is why: batch is a leading
// dimension on every port, so a step that changed it would be a different
// compiled graph. A slot with nothing to do is absent from the step's work and
// costs nothing rather than a second plan.
//
// The buffers are allocated now rather than on the first step that needs them,
// for [Model.NewPool]'s reason: what this process will hold for its life is
// something a device that cannot hold it should refuse at startup rather than
// under load.
func (m *Model) NewBatch(n int) (*Batch, error) {
	if n < 2 {
		return nil, fmt.Errorf("tgo: a batch of %d slot(s) is a session; a batch is "+
			"two or more sequences stepping together", n)
	}
	if m.blocks == nil {
		return nil, fmt.Errorf("%w: sequences that step together have different "+
			"lengths, so a contiguous cache would pad every one of them to the "+
			"longest. Open the model with WithPrefixCache(CacheProcess, positions) "+
			"(specs/008-scheduler.md §2)", ErrNoBlockPool)
	}
	rows, err := m.batchRows(n)
	if err != nil {
		return nil, err
	}
	buckets, err := batchBuckets(n, rows)
	if err != nil {
		return nil, fmt.Errorf("tgo: %w", err)
	}

	c := m.cfg
	b := &Batch{
		m: m, rows: rows, buckets: buckets,
		slots: make([]*batchSlot, n),
		binds: map[int]tensor.Bindings{},
	}
	for i := range b.slots {
		b.slots[i] = &batchSlot{}
	}
	alloc := func(dst **accel.Buffer, dt accel.DType, count int, label string) error {
		buf, err := m.dev.NewBuffer(accel.BufferDescriptor{
			DType: dt, Count: count, Label: label,
			Usage: accel.BufferStorage | accel.BufferCopyDst | accel.BufferCopySrc,
		})
		if err != nil {
			return fmt.Errorf("tgo: allocating %s for a batch of %d: %w", label, n, err)
		}
		*dst = buf
		return nil
	}
	pages := m.blocks.maxPages()
	for _, a := range []struct {
		dst   **accel.Buffer
		dt    accel.DType
		n     int
		label string
	}{
		{&b.ids, accel.U32, rows, model.PortIDs},
		{&b.posq, accel.U32, rows * c.NumHeads, model.PortPosQ},
		{&b.posk, accel.U32, rows * c.NumKVHeads, model.PortPosK},
		{&b.slots1, accel.U32, rows, model.PortSlots},
		{&b.lengths, accel.U32, n, model.PortLengths},
		{&b.extents, accel.U32, n, model.PortExtents},
		{&b.last, accel.U32, n, model.PortLast},
		{&b.pageBuf, accel.U32, n * pages, model.PortPages},
		{&b.logits, accel.F32, n * c.VocabSize, model.PortLogits},
	} {
		if err := alloc(a.dst, a.dt, a.n, a.label); err != nil {
			_ = b.Close()
			return nil, err
		}
	}
	b.hLogits = make([]float32, n*c.VocabSize)
	b.host = model.BatchStep{
		IDs: make([]uint32, rows), PosQ: make([]uint32, rows*c.NumHeads),
		PosK: make([]uint32, rows*c.NumKVHeads), Slots: make([]uint32, rows),
		Lengths: make([]uint32, n), Extents: make([]uint32, n), Last: make([]uint32, n),
	}
	return b, nil
}

// batchRows is the largest query-row count one step may carry.
//
// The pool's positions, which is an upper bound rather than a tight one: a
// step's rows are what the slots contribute, and the pool holds every slot's
// whole cache rather than one step's queries. The looseness costs nothing,
// because [batchBuckets] compiles a plan per rung actually used and not per
// rung named, and the alternative would be a bound derived from what a
// scheduler intends to contribute -- a number this type does not have, and one
// 008 section 5's chunk size is free to change.
func (m *Model) batchRows(n int) (int, error) {
	rows := m.blocks.positions
	if rows < n {
		return 0, fmt.Errorf("tgo: a shared pool of %d positions cannot carry a batch "+
			"of %d slots, each of which needs at least one", rows, n)
	}
	return rows, nil
}

// Slots is how many sequences this batch holds.
func (b *Batch) Slots() int { return len(b.slots) }

// Admit gives a slot the blocks for a prompt **and for the answer it will
// generate**, and reports how many leading positions the pool already holds.
//
// # Why the reserve is not optional
//
// specs/008-scheduler.md §3: a request admitted on a free slot alone is how a
// server deadlocks. Every slot occupied, the pool empty, no sequence able to
// grow -- so nothing finishes and nothing can be evicted into progress. The
// blocks for
//
//	ceil((T_prompt + R) / B_block)
//
// are taken together or not at all, so a slot that was admitted has been shown
// to fit. reserve of zero says the sequence will not grow, which is right for a
// scoring pass and wrong for generation.
//
// R is policy and not a derived number. Setting it to the request's max_tokens
// is safe and admits too little; setting it to zero maximises occupancy and
// deadlocks. This takes it from the caller and refuses loudly, because a server
// that quietly admits fewer requests than it could is indistinguishable from a
// slow one.
//
// # The reuse is not only this slot's
//
// The pool is keyed on chained block hashes and shared across every slot, so a
// slot that has never seen these tokens can still find most of them
// (specs/016-prefix-cache.md §4).
func (b *Batch) Admit(slot int, ids []int, salt string, reserve int) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.usable(slot); err != nil {
		return 0, err
	}
	if len(ids) == 0 {
		return 0, errors.New("tgo: the prompt is empty; there is nothing to condition on")
	}
	s := b.slots[slot]
	s.release()
	l, err := b.m.blocks.pool.Acquire(prefix.Request{
		IDs: ids, Session: salt, Salt: salt, Reserve: reserve,
	})
	if err != nil {
		return 0, fmt.Errorf("tgo: admitting a %d-token prompt with a reserve of %d "+
			"to slot %d: %w", len(ids), reserve, slot, err)
	}
	s.lease, s.pages = l, l.Blocks()
	s.length = l.Reused()
	s.history = append(s.history[:0], ids[:l.Reused()]...)
	return l.Reused(), nil
}

// Evict returns a slot's blocks and empties it.
//
// 008-D2: a preempted sequence drops its blocks and re-prefills on readmission,
// rather than swapping its cache to the host. Prefill is compute-bound and
// parallel over its tokens; a swap is two serial transfers over a bus and needs
// a host mirror of every swapped sequence.
func (b *Batch) Evict(slot int) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.usable(slot); err != nil {
		return err
	}
	s := b.slots[slot]
	s.release()
	s.length, s.history = 0, s.history[:0]
	return nil
}

// Length is how many positions a slot's cache holds.
func (b *Batch) Length(slot int) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	if slot < 0 || slot >= len(b.slots) {
		return 0
	}
	return b.slots[slot].length
}

// Step runs one forward pass over every slot in work and returns each one's
// logits, in the order work names them.
//
// # How long a slice is yours
//
// Until **that slot** steps again. Another slot stepping does not disturb it,
// and no allocation happens after the first step: each slot owns one buffer and
// a step copies into the buffers of the slots that worked.
//
// The first draft returned slices of one shared readback, valid until the next
// step whichever slot it was for, and the author's own test broke that rule
// within minutes — it held a prefill's logits across a decode and compared
// stale numbers, which read as a batching bug and was not. A lifetime nobody
// can keep is not a lifetime, and the copy that fixes it is host-to-host
// against a readback of the same bytes off the device.
func (b *Batch) Step(work []Work) ([][]float32, error) {
	out, _, err := b.step(work)
	return out, err
}

// step is [Batch.Step] with the three device terms measured.
//
// The split is [Session.run]'s and means the same things: submit is building
// the bindings and handing the plan to the queue, device is the fence wait, and
// readback is the logits coming back. specs/017-benchmarks.md §1 treats the
// four terms as exhaustive, so a batched loop that recorded a wall clock under
// one of their names would report a device cost as host time.
func (b *Batch) step(work []Work) ([][]float32, timings, error) {
	var t timings
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil, t, errors.New("tgo: the batch is closed")
	}
	if len(work) == 0 {
		return nil, t, errors.New("tgo: a step with no work computes nothing")
	}

	members := make([]model.Member, len(b.slots))
	seen := make([]bool, len(b.slots))
	total := 0
	for _, w := range work {
		if err := b.usable(w.Slot); err != nil {
			return nil, t, err
		}
		if seen[w.Slot] {
			return nil, t, fmt.Errorf("tgo: slot %d appears twice in one step; a slot is "+
				"one sequence and its tokens are consecutive", w.Slot)
		}
		seen[w.Slot] = true
		if len(w.Tokens) == 0 {
			return nil, t, fmt.Errorf("tgo: slot %d contributes no tokens; a slot with "+
				"nothing to do is absent from the work rather than empty in it", w.Slot)
		}
		total += len(w.Tokens)
	}
	if total > b.rows {
		return nil, t, fmt.Errorf("tgo: %d tokens do not fit a batch sized for %d",
			total, b.rows)
	}

	// Every slot is a member, because batch is a leading dimension of the plan
	// and a slot that contributes nothing is still one row of it (008-D1).
	for i, s := range b.slots {
		members[i] = model.Member{First: s.length, Pages: s.pages}
	}
	for _, w := range work {
		s := b.slots[w.Slot]
		if err := s.reserve(w.Tokens); err != nil {
			return nil, t, fmt.Errorf("tgo: slot %d: %w", w.Slot, err)
		}
		members[w.Slot].Tokens = w.Tokens
		members[w.Slot].Pages = s.pages
	}

	rows, err := b.buckets.For(total)
	if err != nil {
		return nil, t, fmt.Errorf("tgo: %w", err)
	}
	step, err := model.NewBatchStep(b.m.cfg, rows, members, CacheBlock, b.m.blocks.positions)
	if err != nil {
		return nil, t, err
	}
	b.host = step

	plan, err := b.m.plan(rows, b.m.blocks.positions, CacheBlock, len(b.slots), b.m.blocks.dtype)
	if err != nil {
		return nil, t, err
	}
	bind, err := b.bindings(rows)
	if err != nil {
		return nil, t, err
	}

	b.m.mu.Lock()
	defer b.m.mu.Unlock()
	q := b.m.dev.Queue()
	mark := time.Now()
	for _, wr := range []struct {
		buf  *accel.Buffer
		data any
	}{
		{b.ids, step.IDs}, {b.posq, step.PosQ}, {b.posk, step.PosK},
		{b.slots1, step.Slots}, {b.lengths, step.Lengths},
		{b.extents, step.Extents}, {b.last, step.Last},
	} {
		if err := q.WriteBuffer(wr.buf, 0, wr.data); err != nil {
			return nil, t, fmt.Errorf("tgo: binding a batched step's inputs: %w", err)
		}
	}
	table := make([]uint32, len(b.slots)*b.m.blocks.maxPages())
	for i, s := range b.slots {
		for j, id := range s.pages {
			table[i*b.m.blocks.maxPages()+j] = uint32(id)
		}
	}
	if err := q.WriteBuffer(b.pageBuf, 0, table); err != nil {
		return nil, t, fmt.Errorf("tgo: binding the batch's page tables: %w", err)
	}
	fence := plan.Submit(q, bind)
	t.submit = time.Since(mark)
	mark = time.Now()
	if err := fence.Wait(); err != nil {
		return nil, t, fmt.Errorf("tgo: submitting a %d-slot step of %d tokens: %w",
			len(work), total, err)
	}
	t.device = time.Since(mark)
	mark = time.Now()
	// Only the rows this step produced, as one read.
	//
	// A batched readback is V floats per slot and V is 151936, so a step is
	// 593.5 KiB per slot against 4 bytes of useful output. Reading the whole
	// buffer charged every step for every *idle* slot as well: an eight-slot
	// scheduler with one live decoder moved 4.86 MB to learn one token.
	//
	// The span rather than a read per slot, because the measurement that would
	// justify N calls does not exist yet -- specs/017-benchmarks.md §4.1 puts
	// the readback at 807 MB/s off a mapped buffer, which is far enough below
	// memcpy bandwidth that most of it may be fixed per-call cost. One call
	// over [lo, hi] is never more bytes than needed and never more calls than
	// before, which is the choice that does not depend on the number.
	lo, hi := len(b.slots), 0
	for _, w := range work {
		lo, hi = min(lo, w.Slot), max(hi, w.Slot)
	}
	v := b.m.cfg.VocabSize
	span := b.hLogits[lo*v : (hi+1)*v]
	if err := q.ReadBuffer(b.logits, lo*v, span); err != nil {
		return nil, t, fmt.Errorf("tgo: reading a batched step's logits back: %w", err)
	}
	t.readback = time.Since(mark)

	// The step landed, so the slots advance and their complete blocks are
	// offered to the pool. After the step and never before: a published block
	// is immutable and another sequence may attend to it before the call
	// returns (016 §5).
	out := make([][]float32, len(work))
	for i, w := range work {
		s := b.slots[w.Slot]
		if err := s.commit(w.Tokens); err != nil {
			return nil, t, fmt.Errorf("tgo: slot %d: %w", w.Slot, err)
		}
		s.history = append(s.history, w.Tokens...)
		s.length += len(w.Tokens)
		if s.out == nil {
			s.out = make([]float32, v)
		}
		copy(s.out, b.hLogits[w.Slot*v:(w.Slot+1)*v])
		out[i] = s.out
	}
	return out, t, nil
}

// usable reports whether a slot index names a slot of a live batch.
func (b *Batch) usable(slot int) error {
	if b.closed {
		return errors.New("tgo: the batch is closed")
	}
	if slot < 0 || slot >= len(b.slots) {
		return fmt.Errorf("tgo: slot %d is outside a batch of %d", slot, len(b.slots))
	}
	return nil
}

// bindings returns the batch's bindings for a plan of rows query rows.
func (b *Batch) bindings(rows int) (tensor.Bindings, error) {
	if bind, ok := b.binds[rows]; ok {
		return bind, nil
	}
	c, n := b.m.cfg, len(b.slots)
	bufs := make(map[string]accel.BufferView, len(b.m.weightBind)+10)
	maps.Copy(bufs, b.m.weightBind)
	for _, e := range []struct {
		name  string
		buf   *accel.Buffer
		count int
	}{
		{model.PortIDs, b.ids, rows},
		{model.PortPosQ, b.posq, rows * c.NumHeads},
		{model.PortPosK, b.posk, rows * c.NumKVHeads},
		{model.PortSlots, b.slots1, rows},
		{model.PortLengths, b.lengths, n},
		{model.PortExtents, b.extents, n},
		{model.PortLast, b.last, n},
		{model.PortPages, b.pageBuf, n * b.m.blocks.maxPages()},
		{model.PortKeys, b.m.blocks.keys,
			c.NumLayers * b.m.blocks.positions * c.NumKVHeads * c.HeadDim},
		{model.PortValues, b.m.blocks.values,
			c.NumLayers * b.m.blocks.positions * c.NumKVHeads * c.HeadDim},
		{model.PortLogits, b.logits, n * c.VocabSize},
	} {
		view, err := e.buf.View(0, e.count)
		if err != nil {
			return tensor.Bindings{}, fmt.Errorf("tgo: binding %q: %w", e.name, err)
		}
		bufs[e.name] = view
	}
	// No ScalarBase: a ragged step derives each token's position from its own
	// sequence's length and count, so there is no single first position for a
	// base to name and accel refuses one.
	bind := tensor.Bindings{Buffers: bufs, Scalars: map[string]tensor.ScalarValue{
		model.ScalarRoPEBase: tensor.F32(c.RoPETheta),
		model.ScalarScale:    tensor.F32(rsqrt(c.HeadDim)),
	}}
	b.binds[rows] = bind
	return bind, nil
}

// Close releases the batch's buffers and every slot's blocks.
func (b *Batch) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true
	for _, s := range b.slots {
		if s != nil {
			s.release()
		}
	}
	var errs []error
	if b.m != nil && b.m.dev != nil {
		errs = append(errs, b.m.dev.Queue().Flush().Wait())
	}
	for _, buf := range []*accel.Buffer{
		b.ids, b.posq, b.posk, b.slots1, b.lengths, b.extents, b.last,
		b.pageBuf, b.logits,
	} {
		if buf != nil {
			errs = append(errs, buf.Close())
		}
	}
	return errors.Join(errs...)
}

// reserve makes sure the slot's blocks cover the positions this step will
// write, and takes the page table they give.
//
// It records no tokens. What a step's tokens *are* is not settled until the
// step lands: a submission that fails is one whose tokens the caller may
// replace, and a hash chained over a token nobody computed names a block
// holding something else. [batchSlot.commit] is the other half and runs after
// the step.
//
// Only the positions the lease does not already cover. [Batch.Admit] leases the
// whole prompt, so a prefill needs nothing more, and growing for it a second
// time would give the lease twice the sequence.
func (s *batchSlot) reserve(toks []int) error {
	if s.lease == nil {
		return errors.New("holds no blocks; admit it before stepping it")
	}
	if need := s.length + len(toks); need > s.lease.Len() {
		if err := s.lease.Grow(need - s.lease.Len()); err != nil {
			return err
		}
	}
	s.pages = s.lease.Blocks()
	return nil
}

// commit records the tokens a step computed and offers every block that step
// completed, and it runs only after the step landed.
//
// The published extent is the slot's own length and not the lease's. A lease
// covers the positions the slot *may* write -- Admit takes the whole prompt so
// admission is a promise -- so publishing on its extent offers another sequence
// a block holding nothing, which it reads as context. A chunked prefill 32
// tokens into a 192-token prompt published all six blocks, and the next request
// with the same prefix reused 192 positions of which 160 were never written.
func (s *batchSlot) commit(toks []int) error {
	if s.lease == nil {
		return nil
	}
	if need := s.length + len(toks); need > s.lease.Len() {
		if err := s.lease.Commit(toks[len(toks)-(need-s.lease.Len()):]...); err != nil {
			return err
		}
	}
	s.pages = s.lease.Publish(s.length + len(toks))
	return nil
}

// release gives the slot's blocks back. Idempotent.
func (s *batchSlot) release() {
	if s.lease != nil {
		s.lease.Release()
		s.lease, s.pages = nil, nil
	}
}
