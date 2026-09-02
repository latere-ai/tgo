// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// batchStep binds a batched step's ports.
func (r *rig) batchStep(c *Config, rows int, members []Member, block, capacity, maxPages int) {
	r.t.Helper()
	s, err := NewBatchStep(c, rows, members, block, capacity)
	if err != nil {
		r.t.Fatalf("NewBatchStep: %v", err)
	}
	r.u32(PortIDs, s.IDs)
	r.u32(PortPosQ, s.PosQ)
	r.u32(PortPosK, s.PosK)
	r.u32(PortSlots, s.Slots)
	r.u32(PortLengths, s.Lengths)
	r.u32(PortExtents, s.Extents)
	r.u32(PortLast, s.Last)
	table := make([]uint32, len(members)*maxPages)
	for i, m := range members {
		for j := range maxPages {
			if j < len(m.Pages) {
				table[i*maxPages+j] = uint32(m.Pages[j])
			}
		}
	}
	r.u32(PortPages, table)
	r.scalars[ScalarRoPEBase] = tensor.F32(c.RoPETheta)
	r.scalars[ScalarScale] = tensor.F32(float32(1 / math.Sqrt(float64(c.HeadDim))))
}

// submitBatch runs a batched plan and reads back [batch, VocabSize].
func (r *rig) submitBatch(c *Config, batch int) []float32 {
	r.t.Helper()
	if _, ok := r.buffers[PortLogits]; !ok {
		r.f32(PortLogits, make([]float32, batch*c.VocabSize))
	}
	fence := r.plan.Submit(r.dev.Queue(), tensor.Bindings{Buffers: r.bufs, Scalars: r.scalars})
	if err := fence.Wait(); err != nil {
		r.t.Fatalf("submit: %v", err)
	}
	out := make([]float32, batch*c.VocabSize)
	if err := r.dev.Queue().ReadBuffer(r.buffers[PortLogits], 0, out); err != nil {
		r.t.Fatalf("readback: %v", err)
	}
	return out
}

// TestABatchProducesWhatTheSameStepsProduceApart is the whole claim of a
// batched step, and the one assertion that has to hold before any scheduler is
// worth writing.
//
// Three sequences of different lengths, at different positions, on scattered
// blocks, stepping together: a multi-token chunk, a single decode token, and a
// sequence contributing nothing. Each must produce the logits it produces when
// it runs on its own.
//
// It is asserted bit for bit. The two runs perform the same reductions over the
// same values in the same order -- batching changes which dispatch a row is in,
// not what the row computes -- so any difference is a sequence seeing tokens
// that are not its own, which is the whole failure mode of a shared dispatch.
func TestABatchProducesWhatTheSameStepsProduceApart(t *testing.T) {
	m := synthetic(t)
	c := m.Config()
	w := synthWeights(c)

	const block, capacity = 4, 64
	const maxPages = capacity / block

	// Deliberately not the identity and not in order: a table that happened to
	// be [0,1,2] would pass whether the kernel read it or read the cache
	// contiguously.
	members := []Member{
		{Tokens: []int{3, 17, 42}, First: 0, Pages: []int{9, 2, 5}},
		{Tokens: []int{8}, First: 5, Pages: []int{7, 1, 12}},
		{Tokens: nil, First: 4, Pages: []int{3, 11, 0}},
	}
	rows := 0
	for _, mem := range members {
		rows += len(mem.Tokens)
	}

	batched := newRig(t)
	batched.record(m, GraphSpec{
		Tokens: rows, Capacity: capacity, Cache: accel.F32,
		Block: block, Batch: len(members),
	})
	batched.weights(w)
	batched.cache(c, capacity)
	batched.batchStep(c, rows, members, block, capacity, maxPages)
	together := batched.submitBatch(c, len(members))

	// The same sequences, one plan each, against a cache prepared the same way.
	for i, mem := range members {
		if len(mem.Tokens) == 0 {
			continue
		}
		alone := newRig(t)
		alone.record(m, GraphSpec{
			Tokens: len(mem.Tokens), Capacity: capacity, Cache: accel.F32, Block: block,
		})
		alone.weights(w)
		alone.cache(c, capacity)
		ids := make([]uint32, len(mem.Tokens))
		for j, id := range mem.Tokens {
			ids[j] = uint32(id)
		}
		alone.pagedStep(c, ids, mem.First, mem.Pages, block, maxPages)
		want := alone.submit(c)

		// Within a relative budget rather than bit for bit, since accel
		// 2026-09-02: a step alone is M=1 and takes the matrix-vector kernel,
		// whose lanes fold K in sixteen phases and sum the partials, while the
		// batch is M>1 and takes the tile, which folds K in order. The same
		// products in a different order, through 28 layers, move a logit by a
		// few parts in a million; 1e-5 relative is well above what was seen
		// (13 ULP at 1.96) and well below anything a sampler could tell
		// apart. What says a batched step is the steps it batches is the
		// argmax, which is checked exactly.
		const relative = 1e-5
		got := together[i*c.VocabSize : (i+1)*c.VocabSize]
		argGot, argWant := 0, 0
		for j := range want {
			if math.Abs(float64(got[j]-want[j])) > relative*math.Max(1, math.Abs(float64(want[j]))) {
				t.Fatalf("sequence %d logit %d is %v in a batch of %d and %v on its "+
					"own; a batched step is the steps it batches",
					i, j, got[j], len(members), want[j])
			}
			if got[j] > got[argGot] {
				argGot = j
			}
			if want[j] > want[argWant] {
				argWant = j
			}
		}
		if argGot != argWant {
			t.Fatalf("sequence %d picks token %d in a batch of %d and %d on its own",
				i, argGot, len(members), argWant)
		}
	}

	var kernels []string
	for _, s := range batched.plan.Selections() {
		if s.Op == "Attention" {
			kernels = append(kernels, s.Kernel)
		}
	}
	if len(kernels) != c.NumLayers {
		t.Fatalf("%d attention selections over %d layers", len(kernels), c.NumLayers)
	}
	if !strings.Contains(kernels[0], "Ragged") {
		t.Fatalf("the batched step selected %q; a step whose sequences contribute "+
			"different counts runs the ragged kernel, and the rectangular one gives "+
			"every sequence the same count", kernels[0])
	}
	t.Logf("a %d-token chunk, a decode and an empty slot in one %s dispatch",
		len(members[0].Tokens), kernels[0])

	// And again with the plan bucketed above the tokens, which is what a
	// scheduler will actually submit. The padding is charged to a real
	// sequence's extent (C23), so the inflated extent reaches the kernel and
	// the real tokens have to keep their positions through it -- the one thing
	// a test with rows == total cannot show.
	t.Run("bucketed above the member tokens", func(t *testing.T) {
		padded := newRig(t)
		padded.record(m, GraphSpec{
			Tokens: rows + 5, Capacity: capacity, Cache: accel.F32,
			Block: block, Batch: len(members),
		})
		padded.weights(w)
		padded.cache(c, capacity)
		padded.batchStep(c, rows+5, members, block, capacity, maxPages)
		got := padded.submitBatch(c, len(members))

		for i := range members {
			if len(members[i].Tokens) == 0 {
				continue
			}
			a := together[i*c.VocabSize : (i+1)*c.VocabSize]
			b := got[i*c.VocabSize : (i+1)*c.VocabSize]
			for j := range a {
				if a[j] != b[j] {
					t.Fatalf("sequence %d logit %d is %v in a %d-row plan and %v in a "+
						"%d-row one; padding a batch must not move a real token",
						i, j, b[j], rows+5, a[j], rows)
				}
			}
		}
	})
}

// TestABatchPadsWithRowsNoSequenceClaims is C23 after accel closed it.
//
// A bucketed batch has more plan rows than member tokens. Those rows belong to
// no sequence and the ragged kernel reads them as padding, so no real
// sequence's extent or length moves -- which is what the earlier version of
// this had to do, and which was one more thing to get right.
func TestABatchPadsWithRowsNoSequenceClaims(t *testing.T) {
	c := synthetic(t).Config()
	members := []Member{
		{Tokens: []int{3, 17}, First: 0, Pages: []int{0, 1}},
		{Tokens: []int{8}, First: 5, Pages: []int{2, 3}},
	}
	const rows = 8
	s, err := NewBatchStep(c, rows, members, 4, 64)
	if err != nil {
		t.Fatalf("NewBatchStep: %v", err)
	}

	sum := uint32(0)
	for _, e := range s.Extents {
		sum += e
	}
	if int(sum) != 3 {
		t.Fatalf("the extents sum to %d and the members contribute 3; a pad row is "+
			"claimed by nobody and reaches nothing", sum)
	}

	// No real sequence moved.
	if s.Extents[1] != 1 {
		t.Fatalf("the second member's extent is %d, want its own 1: the padding is "+
			"not charged to it", s.Extents[1])
	}
	if s.Lengths[1] != 6 {
		t.Fatalf("the second member's length is %d, want 6: five cached positions "+
			"and the one it contributes", s.Lengths[1])
	}
	// Its logits still come from the real last token, not from a pad row.
	if s.Last[1] != 2 {
		t.Fatalf("the second member reads its logits from flat row %d; the pad rows "+
			"carry a repeat of its last token and are not its answer", s.Last[1])
	}
	// And every pad row's write is dropped.
	for i := 3; i < rows; i++ {
		if s.Slots[i] != 64 {
			t.Fatalf("pad row %d writes to cache row %d; a pad row's key and value "+
				"are computed from a repeat and must reach no cache", i, s.Slots[i])
		}
	}
}

// TestNewBatchStepRefusals: each names the field.
func TestNewBatchStepRefusals(t *testing.T) {
	c := synthetic(t).Config()
	sound := []Member{
		{Tokens: []int{1, 2}, First: 0, Pages: []int{0, 1}},
		{Tokens: []int{3}, First: 0, Pages: []int{2, 3}},
	}
	for _, tc := range []struct {
		name    string
		rows    int
		members []Member
		block   int
		want    string
	}{
		{"one member", 2, sound[:1], 4, "a batch is two or more"},
		{"a contiguous cache", 4, sound, 0, "page table"},
		{"nobody contributing", 4, []Member{
			{Pages: []int{0}}, {Pages: []int{1}}}, 4, "computes nothing"},
		{"more tokens than rows", 2, sound, 4, "do not fit"},
		{"a negative position", 4, []Member{
			{Tokens: []int{1}, First: -1, Pages: []int{0}},
			{Tokens: []int{2}, First: 0, Pages: []int{1}}}, 4, "not negative"},
		{"a token outside the vocabulary", 4, []Member{
			{Tokens: []int{c.VocabSize}, Pages: []int{0}},
			{Tokens: []int{1}, Pages: []int{1}}}, 4, "vocabulary"},
		{"a table that does not reach", 4, []Member{
			{Tokens: []int{1}, First: 9, Pages: []int{0}},
			{Tokens: []int{2}, First: 0, Pages: []int{1}}}, 4, "page table entries"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewBatchStep(c, tc.rows, tc.members, tc.block, 64)
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("the refusal does not say %q: %v", tc.want, err)
			}
		})
	}
}

// TestTheGraphRefusesABatchItCannotRecord.
func TestTheGraphRefusesABatchItCannotRecord(t *testing.T) {
	m := synthetic(t)
	for _, tc := range []struct {
		name string
		spec GraphSpec
		want string
	}{
		{"a negative batch", GraphSpec{Tokens: 4, Capacity: 16, Batch: -1},
			"zero or one is a single sequence"},
		{"a batch on a contiguous cache", GraphSpec{Tokens: 4, Capacity: 16, Batch: 2},
			"paging exists to avoid"},
		{"fewer rows than sequences", GraphSpec{Tokens: 2, Capacity: 16, Block: 4, Batch: 3},
			"total across the batch"},
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
