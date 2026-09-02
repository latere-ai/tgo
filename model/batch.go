// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
)

// Member is one sequence's contribution to a batched step.
type Member struct {
	// Tokens are the ids this sequence contributes, in order. Empty is legal:
	// a slot admitted with nothing to say yet is an ordinary member of the
	// batch and costs one row of arithmetic (specs/008-scheduler.md §2).
	Tokens []int

	// First is the position of Tokens[0] within this sequence, so the tokens
	// occupy positions First..First+len(Tokens)-1.
	First int

	// Pages is this sequence's page-table row: Pages[i] is the physical block
	// holding its i-th logical block.
	Pages []int
}

// BatchStep is the runtime data one *batched* forward pass binds.
//
// It is [Step] with the per-sequence ports widened and two more: the extents
// that say which flat query row belongs to whom, and the row each sequence's
// logits are read from.
type BatchStep struct {
	// IDs, PosQ, PosK and Slots are flat: every sequence's rows end to end.
	IDs, PosQ, PosK, Slots []uint32

	// Lengths, Extents and Last are one entry per sequence.
	Lengths, Extents, Last []uint32
}

// NewBatchStep builds the port data for a step over several sequences.
//
// # The padding, and the upstream fix that simplified it
//
// A bucketed step has more plan rows than member tokens. The rows past the
// extents belong to no sequence, and accel's ragged kernel treats them as
// padding: they contribute nothing and read nothing
// ([C23](../specs/010-conformance.md)).
//
// It did not, and the difference is worth keeping. The kernel found a token's
// sequence by counting the rows that end at or before it, so a row past the
// last segment counted every one of them and indexed one past the offsets
// array -- another sequence's cache on a GPU, read back as a fluent answer.
// The extents are device data, so [tensor.Attention] could not check the sum
// at record time either. tgo reported it as accel#24 and argued for "a row
// past the last extent contributes nothing" over clamping it into the last
// sequence, because clamping makes the read in-range and the answer wrong.
// accel took that shape.
//
// So the padding is what a single-sequence step does: the pad rows carry the
// last real token at its position with a slot at the capacity, which
// [tensor.ScatterRows] drops. Before the fix they had to be charged to a real
// sequence's extent, which moved that sequence's length and was one more thing
// to get right.
//
// rows is the plan's row count, which is at or above the total the members
// contribute.
func NewBatchStep(c *Config, rows int, members []Member, block, capacity int) (BatchStep, error) {
	if c == nil {
		return BatchStep{}, fmt.Errorf("model: NewBatchStep: the config is nil")
	}
	if len(members) < 2 {
		return BatchStep{}, fmt.Errorf("model: NewBatchStep: %d member(s); a batch is "+
			"two or more sequences, and one is NewStep", len(members))
	}
	if block <= 0 {
		return BatchStep{}, fmt.Errorf("model: NewBatchStep: block is %d; sequences "+
			"that step together are addressed through a page table", block)
	}
	total := 0
	for _, m := range members {
		total += len(m.Tokens)
	}
	if total == 0 {
		return BatchStep{}, fmt.Errorf("model: NewBatchStep: every member contributes " +
			"nothing; a step with no rows computes nothing and has no logits to read")
	}
	if total > rows {
		return BatchStep{}, fmt.Errorf("model: NewBatchStep: %d member tokens do not "+
			"fit a %d-row plan", total, rows)
	}

	s := BatchStep{
		IDs:     make([]uint32, rows),
		PosQ:    make([]uint32, rows*c.NumHeads),
		PosK:    make([]uint32, rows*c.NumKVHeads),
		Slots:   make([]uint32, rows),
		Lengths: make([]uint32, len(members)),
		Extents: make([]uint32, len(members)),
		Last:    make([]uint32, len(members)),
	}

	at := 0
	pad := -1
	for i, m := range members {
		s.Lengths[i] = uint32(m.First + len(m.Tokens))
		s.Extents[i] = uint32(len(m.Tokens))
		if m.First < 0 {
			return BatchStep{}, fmt.Errorf("model: NewBatchStep: member %d starts at "+
				"position %d; a position is not negative", i, m.First)
		}
		if len(m.Tokens) > 0 {
			pad = i
			// The row this sequence's logits come from: its last token.
			s.Last[i] = uint32(at + len(m.Tokens) - 1)
		}
		for j, id := range m.Tokens {
			if id < 0 || id >= c.VocabSize {
				return BatchStep{}, fmt.Errorf("model: NewBatchStep: member %d token "+
					"id %d is outside the model's vocabulary of %d", i, id, c.VocabSize)
			}
			p := m.First + j
			if need := p/block + 1; need > len(m.Pages) {
				return BatchStep{}, fmt.Errorf("model: NewBatchStep: member %d reaches "+
					"position %d, which needs %d page table entries and it has %d; the "+
					"missing block is one another sequence owns", i, p, need, len(m.Pages))
			}
			s.IDs[at] = uint32(id)
			s.Slots[at] = uint32(m.Pages[p/block]*block + p%block)
			fillRow(s.PosQ, at, c.NumHeads, uint32(p))
			fillRow(s.PosK, at, c.NumKVHeads, uint32(p))
			at++
		}
	}

	// The padding: rows no extent claims, which the kernel reads as padding
	// and which write nothing because their slot is the cache capacity.
	//
	// No sequence's extent or length moves. The pad rows carry the last real
	// token at its position rather than a pad token, so nothing about them is
	// unusual if a later kernel ever does look at them.
	if at < rows && pad >= 0 {
		m := members[pad]
		last := uint32(m.Tokens[len(m.Tokens)-1])
		lastPos := uint32(m.First + len(m.Tokens) - 1)
		for ; at < rows; at++ {
			s.IDs[at] = last
			s.Slots[at] = uint32(capacity)
			fillRow(s.PosQ, at, c.NumHeads, lastPos)
			fillRow(s.PosK, at, c.NumKVHeads, lastPos)
		}
	}
	return s, nil
}

// fillRow writes one position into every head's row of a positions tensor.
func fillRow(dst []uint32, token, heads int, p uint32) {
	for h := range heads {
		dst[token*heads+h] = p
	}
}
