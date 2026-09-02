// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package nn

import "fmt"

// The flat convolution window, and the host arithmetic that addresses it.
//
// specs/023-cache-kinds.md §3. [ConvState]'s comment used to say the window was
// [slots, K-1+T, C] and the code was one slot, and putting the slot axis in as
// a real tensor axis does not work: tensor.ScatterRows computes one row's width
// as elements/shape[0], so on a [slots, K-1+T, C] state a "row" is a whole slot
// and writing one token's C values is inexpressible. tensor.Slice takes
// compile-time bounds, so the per-tap read cannot select a slot at runtime
// either.
//
// [023-D2] makes the window flat — [R, C] — and the slot axis arithmetic in the
// u32 index ports, exactly as paging is arithmetic in slots
// (specs/005-kv-cache.md §2.2). The per-tap read becomes GatherRows instead of
// Slice + Contiguous, which copies the same bytes through an index.
//
// The layout, for a step of T flat token rows over B slots:
//
//	R = B(K-1) + T
//
// Rows 0 .. B(K-1)-1 are the carry: slot j owns j(K-1) .. j(K-1)+K-2, holding
// the K-1 inputs before this step. Rows B(K-1) .. R-1 are this step's tokens,
// in the same flat order the queries are in.

// ConvIndices are the index ports a step binds into a [ConvState].
//
// Every one of them is host arithmetic over what each slot contributes, which
// is the same extents a batched step already builds for its query rows.
type ConvIndices struct {
	// Write is where this step's rows go: a [T] u32 port, one entry per row of
	// x. A pad row's entry is R, and tensor.ScatterRows drops a write at or
	// above capacity — the rule specs/007-engine.md 007-D3 already uses for a
	// bucketed prefill.
	Write []uint32

	// Taps is one [T] u32 port per tap: Taps[i][r] is the row tap i reads for
	// output row r. A pad row's entry is R, and tensor.GatherRows writes zeros
	// for an index at or above the table's rows — so a pad row reads zeros,
	// writes nothing, and needs no mask.
	Taps [][]uint32

	// Carry is which rows become the next step's leading K-1 per slot, and
	// CarryWrite is where they go: both [B(K-1)] u32 ports.
	Carry, CarryWrite []uint32

	// Rows is R, the window's row count, and is the index every out-of-range
	// entry above holds. It is the state's **declared** row count and not the
	// minimum this step needs: a plan is compiled per bucket over one shared
	// buffer, so a smaller bucket's window is a prefix of a larger one's, and a
	// sentinel of the minimum would address a real row of the larger.
	Rows int
}

// ConvIndex computes a step's window addressing.
//
// counts[j] is how many token rows slot j contributes and rows is the step's
// row count, which is at least their sum and larger when the step is padded to
// a bucket. k is the tap count. capacity is the window state's declared row
// count, and zero takes the minimum this step needs, B(K-1) + rows.
//
// # Why the pad rows are not a special case
//
// A bucketed step's rows past the last slot's tokens read zeros and write
// nothing because their indices are R, which is out of range for both
// operators. That is the same shape [C23](../specs/010-conformance.md) gave the
// ragged step, and it is what keeps a padded step's window identical to an
// exact step's.
func ConvIndex(counts []int, rows, k, capacity int) (ConvIndices, error) {
	if k <= 1 {
		return ConvIndices{}, fmt.Errorf("nn: the kernel is %d taps; a causal "+
			"convolution has at least two, and one tap is an elementwise scale", k)
	}
	if len(counts) == 0 {
		return ConvIndices{}, fmt.Errorf("nn: a step over no slots convolves nothing")
	}
	total := 0
	for j, n := range counts {
		if n < 0 {
			return ConvIndices{}, fmt.Errorf("nn: slot %d contributes %d rows", j, n)
		}
		total += n
	}
	if rows < total {
		return ConvIndices{}, fmt.Errorf("nn: a step of %d rows cannot carry %d tokens",
			rows, total)
	}
	b := len(counts)
	carryRows := b * (k - 1)
	r := carryRows + rows
	if capacity != 0 {
		if capacity < r {
			return ConvIndices{}, fmt.Errorf("nn: the window holds %d rows and this "+
				"step needs %d: %d carry rows over %d slots and %d token rows",
				capacity, r, carryRows, b, rows)
		}
		r = capacity
	}

	// start[j] is where slot j's tokens begin inside the token region, and
	// slot[r]/local[r] invert that for a token row.
	starts := make([]int, b)
	slotOf := make([]int, rows)
	localOf := make([]int, rows)
	for i := range slotOf {
		// Past the last slot's tokens: a pad row, addressed out of range.
		slotOf[i] = -1
	}
	at := 0
	for j, n := range counts {
		starts[j] = at
		for i := range n {
			slotOf[at+i], localOf[at+i] = j, i
		}
		at += n
	}

	// rowOf maps a slot-local input position to the window row holding it.
	// Negative is before this step, which is the slot's own carry band.
	rowOf := func(j, local int) uint32 {
		if local < 0 {
			return uint32(j*(k-1) + (k - 1) + local)
		}
		return uint32(carryRows + starts[j] + local)
	}

	out := ConvIndices{
		Write: make([]uint32, rows),
		Taps:  make([][]uint32, k),
		Rows:  r,
	}
	for i := range out.Taps {
		out.Taps[i] = make([]uint32, rows)
	}
	for row := range rows {
		j := slotOf[row]
		if j < 0 {
			out.Write[row] = uint32(r)
			for i := range out.Taps {
				out.Taps[i][row] = uint32(r)
			}
			continue
		}
		out.Write[row] = uint32(carryRows + row)
		p := localOf[row]
		for i := range out.Taps {
			// Tap i lands on position p-K+1+i, so tap K-1 is the current row
			// and tap 0 is the one K-1 back:
			//
			//	y[p] = sum_i taps[i] * x[p-K+1+i]
			//
			// which is the convention the weights are trained under.
			out.Taps[i][row] = rowOf(j, p-k+1+i)
		}
	}

	out.Carry = make([]uint32, carryRows)
	out.CarryWrite = make([]uint32, carryRows)
	for j, n := range counts {
		for m := range k - 1 {
			// The last K-1 inputs of slot j, in the order the next step's
			// negative locals read them back: local -1 is the last input and
			// lands at m = K-2.
			out.Carry[j*(k-1)+m] = rowOf(j, n-(k-1)+m)
			out.CarryWrite[j*(k-1)+m] = uint32(j*(k-1) + m)
		}
	}
	return out, nil
}
