// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tgo

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/model"
)

// defaultBuckets is specs/007-engine.md §3's prefill bucket set.
//
// Powers of two, because rounding T up to the next power of two wastes at most
// T rows and, for T uniform on a bucket's span, averages
//
//	E[(B(T) - T)/T] = ∫₁² (2-u)/u du = 2·ln2 - 1 ≈ 0.386
//
// so about 39% of the bucketed dimension is padding, against a plan count
// logarithmic in the context. Halving the waste means doubling the bucket
// count, and each new bucket is a compile. The set is defensible, not optimal
// (007-D2).
//
// §3 also calls the set "configurable" and §1 exposes nothing to configure it
// with, so it is unexported here rather than published as a mutable package
// variable: what would justify a different set is the compile-time-per-bucket
// measurement 010 §3 asks for, which has not been taken.
var defaultBuckets = []int{32, 64, 128, 256, 512, 1024, 2048, 4096}

// bucketsFor is the bucket set a cache of this capacity can actually run.
//
// specs/007-engine.md §3 names a fixed set and §1 makes capacity a caller's
// number, and the two do not meet: a graph writes every token of a step into
// the cache, so model.GraphSpec refuses Tokens > Capacity, and a 90-token
// prompt in a 100-position cache would round up to a 128-token plan that cannot
// be recorded. Every default below the capacity is kept and the capacity itself
// is the last bucket, so a bucket exists for every admissible T and none of
// them exceeds C.
func bucketsFor(capacity int) (tensor.Buckets, error) {
	sizes := make([]int, 0, len(defaultBuckets)+1)
	for _, b := range defaultBuckets {
		if b < capacity {
			sizes = append(sizes, b)
		}
	}
	sizes = append(sizes, capacity)
	return tensor.NewBuckets(sizes...)
}

// batchBuckets is the bucket set a batch of n slots over a pool of rows
// positions can run.
//
// [defaultBuckets] starts at 32 because a single sequence's prefill is the
// thing it rounds, and a decode is one token and gets its own plan below the
// ladder. A batch inverts that: its steady state is **every slot decoding**,
// which is exactly n rows, and rounding that up to 32 would run a 16x wider
// plan than the step needs at the batch sizes a single device holds.
//
// So n is a bucket of its own, and the default ladder above it covers the
// prefill chunks a scheduler mixes in (008 §5). The set is small on purpose:
// each entry is a compile, and 007-D2's argument for a logarithmic ladder is
// unchanged by there being one more rung at the bottom.
func batchBuckets(n, rows int) (tensor.Buckets, error) {
	sizes := []int{n}
	for _, b := range defaultBuckets {
		if b > n && b < rows {
			sizes = append(sizes, b)
		}
	}
	if rows > n {
		sizes = append(sizes, rows)
	}
	return tensor.NewBuckets(sizes...)
}

// plan returns the compiled plan for one step shape, compiling it at most once.
//
// Every step goes through the cache rather than through a memoized field.
// Recording is a few slices and a digest against a model step, and it is what
// makes "N decode steps compile exactly one plan" a measurement rather than a
// tautology: a plan held in a variable cannot miss, whatever §6's Weight and
// Input ports are declared as.
func (m *Model) plan(tokens, capacity, block, batch int, cache accel.DType) (*tensor.Plan, error) {
	spec := model.GraphSpec{
		Tokens:   tokens,
		Capacity: capacity,
		Block:    block,
		Batch:    batch,
		Cache:    cache,
		Stored:   m.stored,
	}
	label := "prefill"
	switch {
	case batch > 1:
		label = "batch"
	case tokens == 1:
		label = "decode"
	}
	var recErr error
	p, err := m.cache.Compile(func(b *tensor.Builder) {
		_, _, recErr = model.Record(b, m.builder, spec)
	}, tensor.CompileOptions{Label: label})
	// The recording error first: model.Record refuses a step the graph cannot
	// hold before it records anything, and an empty graph compiles.
	if recErr != nil {
		return nil, fmt.Errorf("tgo: recording the %s graph at T=%d, C=%d, B=%d: %w",
			label, tokens, capacity, max(batch, 1), recErr)
	}
	if err != nil {
		return nil, fmt.Errorf("tgo: compiling the %s graph at T=%d, C=%d, B=%d: %w",
			label, tokens, capacity, max(batch, 1), err)
	}
	return p, nil
}

// stepData is one submission's input ports, in host memory.
//
// The slices are the session's and are reused: a decode step that allocated
// five vectors per token would put the allocator in the four-term breakdown
// specs/017-benchmarks.md §1 attributes to the host.
type stepData struct {
	ids     []uint32
	posq    []uint32
	posk    []uint32
	slots   []uint32
	lengths []uint32
	pages   []uint32
	base    uint32
}

// cacheLayout says where a session's positions live inside the key and value
// states it binds.
//
// The two numbers were one number until a pool existed. A session's own cache
// is exactly as long as its context, so "does this step fit" and "which index
// does ScatterRows drop" had the same answer. Sharing splits them: the state
// holds every session's blocks, and a session may occupy a small part of it.
// Keeping one number would make a pad row's sentinel land on a real row inside
// another conversation's block -- a write nothing reports, read back later as a
// fluent answer to somebody else's prompt.
type cacheLayout struct {
	// rows is how many positions the bound states hold, whoever owns them. It
	// is the sentinel a dropped write uses (007-D3).
	rows int

	// limit is how many positions this session may occupy.
	limit int

	// pages is this sequence's page table and block is how many positions one
	// entry holds. A nil table is a contiguous cache, which is the same thing
	// with an identity table and a block of one -- and which does not pay the
	// indirection.
	pages []int
	block int
}

// row is the physical state row logical position p lives at.
func (l cacheLayout) row(p int) int {
	if l.pages == nil {
		return p
	}
	return l.pages[p/l.block]*l.block + p%l.block
}

// reaches reports whether the layout can address position p at all.
func (l cacheLayout) reaches(p int) bool {
	if p < 0 || p >= l.limit {
		return false
	}
	if l.pages == nil {
		return p < l.rows
	}
	return p/l.block < len(l.pages)
}

// fill writes one step of tokens at positions first..first+len(tokens)-1 into
// the port slices, padding out to the plan's row count.
//
// # The padding, which is §4's whole point
//
// A bucketed prefill computes rows nobody reads. Their logits are discarded,
// which is free. What must not happen is that they write KV: a pad row's key
// and value are computed from a pad token and corrupt the cache for every
// later step, as a quality loss much later and never as an error.
//
// tensor.ScatterRows documents that an index at or above capacity writes
// nothing, because a GPU cannot report one. So a pad row's slot is the capacity
// itself and the write is dropped by the operator's own contract (007-D3). No
// mask, no scratch row, no extra allocation.
//
// Lengths stays the real length. Every *real* row's causal window is then
// unchanged, because the kernel masks with `pos <= base+s && pos < lengths[0]`.
//
// # A pad row is the last real token, not a pad token
//
// specs/007-engine.md §4 says a pad row's logits "are discarded, which is
// free", and specs/004-model-graph.md §3.2 says the graph computes logits for
// the *last row only*. With trailing padding those are the same row: a bucketed
// prefill that filled its pad rows with a pad token would return the pad token's
// logits and sample the first token of every completion from nothing.
//
// So a pad row carries the last real token at the last real position. Its
// causal limit is base+s, which is past the last real position and clamped by
// lengths, so it sees exactly what the last real row sees; its query is the same
// token at the same rotary angle; and every later row is elementwise in the one
// before. Row B-1 therefore computes what row T-1 computes, and the slice reads
// the answer it was written to read. The KV write is still dropped by the slot,
// so nothing reaches the cache twice.
func (d *stepData) fill(c *model.Config, rows int, tokens []int, first int,
	lay cacheLayout) error {

	t := len(tokens)
	if t == 0 || t > rows {
		return fmt.Errorf("tgo: a %d-row plan cannot score %d tokens", rows, t)
	}
	if first < 0 || first+t > lay.limit {
		return fmt.Errorf("tgo: %d tokens at position %d do not fit a %d-position cache",
			t, first, lay.limit)
	}
	if t > 0 && !lay.reaches(first+t-1) {
		return fmt.Errorf("tgo: %d tokens at position %d reach position %d, which this "+
			"sequence holds no block for; the write would land in a block another "+
			"sequence owns", t, first, first+t-1)
	}
	d.ids = d.ids[:rows]
	d.posq = d.posq[:rows*c.NumHeads]
	d.posk = d.posk[:rows*c.NumKVHeads]
	d.slots = d.slots[:rows]
	d.lengths = d.lengths[:1]

	for i := range t {
		id := tokens[i]
		if id < 0 || id >= c.VocabSize {
			return fmt.Errorf("tgo: token id %d is outside the model's vocabulary of %d",
				id, c.VocabSize)
		}
		p := uint32(first + i)
		d.ids[i] = uint32(id)
		d.slots[i] = uint32(lay.row(first + i))
		for h := 0; h < c.NumHeads; h++ {
			d.posq[i*c.NumHeads+h] = p
		}
		for h := 0; h < c.NumKVHeads; h++ {
			d.posk[i*c.NumKVHeads+h] = p
		}
	}
	// The pad rows: the last real token at the last real position, and a slot
	// at the capacity so the write is dropped. See the note above for why this
	// is not a pad token.
	last := uint32(tokens[t-1])
	lastPos := uint32(first + t - 1)
	for i := t; i < rows; i++ {
		d.ids[i] = last
		d.slots[i] = uint32(lay.rows)
		for h := 0; h < c.NumHeads; h++ {
			d.posq[i*c.NumHeads+h] = lastPos
		}
		for h := 0; h < c.NumKVHeads; h++ {
			d.posk[i*c.NumKVHeads+h] = lastPos
		}
	}
	d.lengths[0] = uint32(first + t)
	d.base = uint32(first)
	return nil
}
