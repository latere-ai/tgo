// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"fmt"
	"sort"
)

// DefaultChunk is how many prompt tokens one step prefills for one sequence.
//
// specs/008-scheduler.md §5. The chunk trades prefill efficiency against how
// much of a step it takes: larger is a better GEMM, and a chunk far larger than
// the number of decoding sequences makes the step a prefill with passengers,
// which is fine. Far smaller wastes the weight read, because the weights are
// read once per step whatever the step carries.
//
// The useful range is 512 to 2048 and this is the low end, because a chunk that
// does not fit the plan's rows is refused and a small default fits every pool.
const DefaultChunk = 512

// schedState is what the scheduler knows about one slot, and it is deliberately
// separable from the device.
//
// specs/008-scheduler.md §2 to §5 are policy over [Batch]'s mechanism, and
// policy that can only be tested by running a forward pass is policy nobody
// tests exhaustively. Everything here is decided from integers.
type schedState struct {
	// live says the slot holds a sequence.
	live bool

	// prompt is the sequence's whole prompt and prefilled is how much of it a
	// step has scored. A slot with prefilled < len(prompt) is mid-prefill.
	prompt    []int
	prefilled int

	// feed is the token a decoding slot contributes next: the one its last
	// step sampled.
	feed int

	// arrived orders admission for eviction. Last arrived is first evicted
	// (008-D5), which bounds the worst-case latency of the sequences already
	// in flight rather than spreading the damage across all of them.
	arrived uint64
}

// schedPlan is one step's decisions.
type schedPlan struct {
	// Work is what to submit, in slot order.
	Work []Work

	// Prefill and Decode count what the step carries, so a caller can report
	// the mix rather than infer it.
	Prefill, Decode int
}

// nextStep decides what the next step runs.
//
// # The mix is the point
//
// A prefill chunk and a decode are one dispatch, because a ragged step's
// sequences contribute different token counts ([C16](specs/010-conformance.md)).
// So a long prompt does not stall the sequences decoding beside it and does not
// pay for its own weight read: the weights are read once for the chunk and the
// decodes together, where two dispatches read them twice.
//
// # Prefills first, then decodes, then what is left
//
// A slot mid-prefill is a request with no first token yet, so its chunk is
// placed before any decode: time to first token is what a waiting caller
// measures, and a decode's own latency is bounded by the step either way.
// Decodes then take one row each. If the rows run out, the sequences that did
// not fit simply wait a step -- they are not evicted, because a step they were
// left out of is a step, and eviction is for a pool that cannot grow (§4).
func nextStep(slots []schedState, chunk, rows int) (schedPlan, error) {
	if chunk <= 0 {
		return schedPlan{}, fmt.Errorf("tgo: the prefill chunk is %d; a chunk is at "+
			"least one token", chunk)
	}
	if rows <= 0 {
		return schedPlan{}, fmt.Errorf("tgo: a step of %d rows carries nothing", rows)
	}
	var p schedPlan
	left := rows
	for i := range slots {
		s := &slots[i]
		if !s.live || s.prefilled >= len(s.prompt) {
			continue
		}
		n := min(chunk, len(s.prompt)-s.prefilled, left)
		if n <= 0 {
			continue
		}
		p.Work = append(p.Work, Work{
			Slot: i, Tokens: s.prompt[s.prefilled : s.prefilled+n],
		})
		p.Prefill += n
		left -= n
	}
	for i := range slots {
		s := &slots[i]
		if !s.live || s.prefilled < len(s.prompt) || left <= 0 {
			continue
		}
		p.Work = append(p.Work, Work{Slot: i, Tokens: []int{s.feed}})
		p.Decode++
		left--
	}
	// Slot order, so a step's rows are laid out the way the ports are and a
	// reader comparing the two is comparing the same order.
	sort.Slice(p.Work, func(a, b int) bool { return p.Work[a].Slot < p.Work[b].Slot })
	return p, nil
}

// victim is the slot 008-D5 evicts: the one that arrived last.
//
// Last-arrived-first-evicted bounds the worst-case latency of the sequences
// already in flight. Round-robin or longest-running spreads the damage instead,
// which turns one slow request into several.
//
// A slot mid-prefill is preferred over one already decoding where both arrived
// after the survivors, because the mid-prefill one has produced nothing a
// caller has seen: 008-D2 makes eviction a recompute, and recomputing a prefill
// nobody has read is cheaper in the only currency that matters.
//
// It returns -1 when nothing is evictable.
func victim(slots []schedState) int {
	best, bestKey := -1, uint64(0)
	for i := range slots {
		s := &slots[i]
		if !s.live {
			continue
		}
		// Mid-prefill sorts above decoding at equal arrival, by adding a bit
		// above every arrival counter this process will reach.
		key := s.arrived
		if s.prefilled < len(s.prompt) {
			key |= 1 << 63
		}
		if best < 0 || key > bestKey {
			best, bestKey = i, key
		}
	}
	return best
}
