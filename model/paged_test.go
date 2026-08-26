// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// pagedStep binds the ports of a step over a cache addressed through pages.
//
// The table is the sequence's own row -- [1, MaxPages] -- and the slots come
// from it, so the two bindings cannot drift apart: a test that wrote the KV
// contiguously and read it through a table would be measuring nothing, which is
// the shape of accel issue 10.
func (r *rig) pagedStep(c *Config, ids []uint32, first int, pages []int,
	block, maxPages int) Step {

	r.t.Helper()
	s, err := NewPagedStep(c, first, len(ids), pages, block)
	if err != nil {
		r.t.Fatalf("NewPagedStep: %v", err)
	}
	r.u32(PortIDs, ids)
	r.u32(PortPosQ, s.PosQ)
	r.u32(PortPosK, s.PosK)
	r.u32(PortSlots, s.Slots)
	r.u32(PortLengths, s.Lengths)
	table := make([]uint32, maxPages)
	for i := range table {
		if i < len(pages) {
			table[i] = uint32(pages[i])
		}
	}
	r.u32(PortPages, table)
	r.scalars[ScalarRoPEBase] = tensor.F32(c.RoPETheta)
	r.scalars[ScalarScale] = tensor.F32(float32(1 / math.Sqrt(float64(c.HeadDim))))
	if len(ids) > 1 {
		r.scalars[ScalarBase] = tensor.U32(s.Base)
	}
	return s
}

// TestAPermutedPageTableComputesWhatAContiguousCacheDoes is the whole claim of
// the page-table port in one assertion.
//
// Paging moves where a token's key and value are *stored*. It does not move
// where the token is in the sequence, what it attends to, or what any of it
// multiplies. So a prefill over a cache whose blocks are scattered has to
// produce the logits a contiguous one produces, exactly -- not within a
// tolerance, because the two runs perform the same reductions over the same
// values in the same order and differ only in the addresses they read them
// from.
//
// The table is a permutation and deliberately not the identity. An identity
// table would pass whether the kernels honoured it or read the cache
// contiguously, which is how accel issue 10 stayed invisible for a day: a paged
// prefill that dropped its table returned a fluent wrong answer and a probe
// that only checked the graph compiled recorded it as working.
func TestAPermutedPageTableComputesWhatAContiguousCacheDoes(t *testing.T) {
	m := synthetic(t)
	c := m.Config()
	w := synthWeights(c)
	ids := []uint32{3, 17, 42, 8, 21}

	const capacity, block = 16, 4
	const maxPages = capacity / block

	contiguous := newRig(t)
	contiguous.record(m, GraphSpec{Tokens: len(ids), Capacity: capacity, Cache: accel.F32})
	contiguous.weights(w)
	contiguous.cache(c, capacity)
	contiguous.step(c, ids, 0)
	want := contiguous.submit(c)

	// Every block somewhere other than where it would be contiguously, and the
	// order reversed rather than rotated: a rotation by one still puts block 0
	// before block 1.
	pages := []int{3, 1, 2, 0}

	paged := newRig(t)
	paged.record(m, GraphSpec{
		Tokens: len(ids), Capacity: capacity, Cache: accel.F32, Block: block,
	})
	paged.weights(w)
	paged.cache(c, capacity)
	paged.pagedStep(c, ids, 0, pages, block, maxPages)
	got := paged.submit(c)

	if len(got) != len(want) {
		t.Fatalf("%d logits paged and %d contiguous", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit %d is %v through a permuted page table and %v over a "+
				"contiguous cache; paging changes where a key lives and nothing "+
				"about what is computed from it", i, got[i], want[i])
		}
	}

	// The plan has to say it paged. Two runs agreeing is what a page table
	// being ignored also looks like, if the slots were contiguous too -- so
	// the selection is the half of the evidence the values cannot carry.
	var kernels []string
	for _, s := range paged.plan.Selections() {
		if s.Op == "Attention" {
			kernels = append(kernels, s.Kernel)
		}
	}
	if len(kernels) != c.NumLayers {
		t.Fatalf("%d attention selections over %d layers", len(kernels), c.NumLayers)
	}
	for _, k := range kernels {
		if k == "" {
			t.Fatal("an attention node selected no kernel")
		}
	}
	t.Logf("paged prefill over blocks %v: %s", pages, kernels[0])
}

// TestThePageTableIsRead is the negative control the test above needs.
//
// It writes the KV where a *contiguous* cache would put it and reads through a
// permuted table. If the table were ignored the two would agree, and the test
// above would be green whether or not paging worked.
func TestThePageTableIsRead(t *testing.T) {
	m := synthetic(t)
	c := m.Config()
	w := synthWeights(c)
	ids := []uint32{3, 17, 42, 8, 21}

	const capacity, block = 16, 4
	const maxPages = capacity / block
	pages := []int{3, 1, 2, 0}

	agreed := newRig(t)
	agreed.record(m, GraphSpec{
		Tokens: len(ids), Capacity: capacity, Cache: accel.F32, Block: block,
	})
	agreed.weights(w)
	agreed.cache(c, capacity)
	agreed.pagedStep(c, ids, 0, pages, block, maxPages)
	want := agreed.submit(c)

	// The same graph and the same table, with the writes put where a
	// contiguous cache would put them.
	crossed := newRig(t)
	crossed.record(m, GraphSpec{
		Tokens: len(ids), Capacity: capacity, Cache: accel.F32, Block: block,
	})
	crossed.weights(w)
	crossed.cache(c, capacity)
	crossed.pagedStep(c, ids, 0, pages, block, maxPages)
	contiguous, err := NewStep(c, 0, len(ids))
	if err != nil {
		t.Fatalf("NewStep: %v", err)
	}
	crossed.u32(PortSlots, contiguous.Slots)
	got := crossed.submit(c)

	for i := range want {
		if got[i] != want[i] {
			return
		}
	}
	t.Fatal("writing the KV contiguously and reading it through a permuted page " +
		"table produced the same logits as writing it through the table; the table " +
		"reaches nothing, and the test that compares paged against contiguous is " +
		"green for the wrong reason")
}

// TestTheGraphRefusesHalfAPagedBinding: a page table and a block size are one
// binding in two values.
func TestTheGraphRefusesHalfAPagedBinding(t *testing.T) {
	m := synthetic(t)

	for _, tc := range []struct {
		name string
		spec GraphSpec
		want string
	}{
		{"a negative block", GraphSpec{Tokens: 2, Capacity: 16, Block: -1},
			"contiguous cache"},
		{"a capacity that is not whole blocks", GraphSpec{Tokens: 2, Capacity: 18, Block: 4},
			"whole blocks"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig(t)
			_, _, err := Record(r.rt.NewBuilder("qwen3"), m, tc.spec)
			if err == nil {
				t.Fatal("recorded")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("the refusal does not say %q: %v", tc.want, err)
			}
		})
	}
}

// TestNewPagedStepRefusesATableThatDoesNotReach: a step past the blocks the
// table names would write into the block after this sequence's last one, which
// belongs to another sequence.
func TestNewPagedStepRefusesATableThatDoesNotReach(t *testing.T) {
	c := synthetic(t).Config()
	if _, err := NewPagedStep(c, 0, 9, []int{0, 1}, 4); err == nil {
		t.Fatal("nine positions over two blocks of four recorded")
	} else if !strings.Contains(err.Error(), "page table entries") {
		t.Fatalf("the refusal does not name the table: %v", err)
	}
	if _, err := NewPagedStep(c, 0, 4, []int{0}, 0); err == nil {
		t.Fatal("a block size of zero recorded")
	}
	if _, err := NewPagedStep(c, 0, 4, []int{-1}, 4); err == nil {
		t.Fatal("a negative block index recorded")
	}
	// And the positions are unchanged by paging: a token's rotary position is
	// where it sits in the sequence.
	plain, err := NewStep(c, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	paged, err := NewPagedStep(c, 2, 3, []int{7, 4}, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i := range plain.PosQ {
		if plain.PosQ[i] != paged.PosQ[i] {
			t.Fatalf("query position %d is %d paged and %d contiguous; paging moves "+
				"storage and not position", i, paged.PosQ[i], plain.PosQ[i])
		}
	}
	// Position 2 is in logical block 0, which the table puts at physical 7.
	if want := uint32(7*4 + 2); paged.Slots[0] != want {
		t.Fatalf("position 2 through table [7, 4] is row %d, want %d",
			paged.Slots[0], want)
	}
	// Position 4 crosses into logical block 1, which the table puts at 4.
	if want := uint32(4 * 4); paged.Slots[2] != want {
		t.Fatalf("position 4 through table [7, 4] is row %d, want %d",
			paged.Slots[2], want)
	}
}
