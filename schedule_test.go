// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tgo

import (
	"strings"
	"testing"
)

// The scheduling policy is tested without a device, on purpose.
//
// specs/008-scheduler.md §2 to §5 are decisions over integers, and a policy
// that can only be exercised by running a forward pass is a policy nobody tests
// exhaustively -- the fixture cost alone would stop it. Every case below is
// microseconds, so the cases can be the ones that matter rather than the ones
// that are affordable.

func seq(prompt, prefilled, feed int, arrived uint64) schedState {
	ids := make([]int, prompt)
	for i := range ids {
		ids[i] = i + 1
	}
	return schedState{live: true, prompt: ids, prefilled: prefilled,
		feed: feed, arrived: arrived}
}

func tokenCount(p schedPlan) int {
	n := 0
	for _, w := range p.Work {
		n += len(w.Tokens)
	}
	return n
}

// TestAChunkRidesWithTheDecodesBesideIt is 008 §5, which is the thing the
// ragged step made possible and the reason this scheduler is worth having.
//
// One sequence mid-prefill and three decoding produce **one** step of
// chunk+3 rows, not a prefill step and then a decode step. The weights are read
// once for all of it.
func TestAChunkRidesWithTheDecodesBesideIt(t *testing.T) {
	slots := []schedState{
		seq(2000, 0, 0, 1), // mid-prefill
		seq(10, 10, 42, 2), // decoding
		seq(10, 10, 43, 3), // decoding
		seq(10, 10, 44, 4), // decoding
	}
	p, err := nextStep(slots, 512, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if p.Prefill != 512 || p.Decode != 3 {
		t.Fatalf("the step carries %d prefill tokens and %d decodes, want 512 and 3",
			p.Prefill, p.Decode)
	}
	if got := len(p.Work); got != 4 {
		t.Fatalf("%d slots contribute to the step, want 4; a chunk that ran alone "+
			"would stall the three decoding beside it and read the weights twice", got)
	}
	if got := tokenCount(p); got != 515 {
		t.Fatalf("the step is %d rows, want 515", got)
	}
	// Slot order, so the work lines up with the ports.
	for i, w := range p.Work {
		if w.Slot != i {
			t.Fatalf("work %d is slot %d; the step is laid out in slot order", i, w.Slot)
		}
	}
}

// TestTheChunkIsCappedByTheChunkSizeAndByTheRows: two different bounds, and the
// smaller one wins.
func TestTheChunkIsCappedByTheChunkSizeAndByTheRows(t *testing.T) {
	for _, c := range []struct {
		name          string
		prompt, chunk int
		rows          int
		want          int
	}{
		{"by the chunk", 2000, 512, 4096, 512},
		{"by the rows", 2000, 512, 100, 100},
		{"by what is left of the prompt", 40, 512, 4096, 40},
	} {
		t.Run(c.name, func(t *testing.T) {
			p, err := nextStep([]schedState{seq(c.prompt, 0, 0, 1)}, c.chunk, c.rows)
			if err != nil {
				t.Fatal(err)
			}
			if p.Prefill != c.want {
				t.Fatalf("the chunk is %d, want %d", p.Prefill, c.want)
			}
		})
	}
}

// TestPrefillsComeBeforeDecodesWhenRowsAreScarce.
//
// A slot mid-prefill is a request with no first token yet, and time to first
// token is what a waiting caller measures. A decode that waits a step is a
// caller already receiving text.
func TestPrefillsComeBeforeDecodesWhenRowsAreScarce(t *testing.T) {
	slots := []schedState{
		seq(10, 10, 42, 1), // decoding
		seq(100, 0, 0, 2),  // mid-prefill
		seq(10, 10, 43, 3), // decoding
	}
	// Four rows: the chunk takes them, and the decodes wait.
	p, err := nextStep(slots, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if p.Prefill != 4 || p.Decode != 0 {
		t.Fatalf("with four rows the step carries %d prefill and %d decode; the "+
			"prefill is the one with no first token yet", p.Prefill, p.Decode)
	}
	// Six rows: the chunk and both decodes fit.
	p, err = nextStep(slots, 4, 6)
	if err != nil {
		t.Fatal(err)
	}
	if p.Prefill != 4 || p.Decode != 2 {
		t.Fatalf("with six rows the step carries %d prefill and %d decode, want 4 and 2",
			p.Prefill, p.Decode)
	}
}

// TestASlotLeftOutOfAStepIsNotEvicted: waiting a step is not preemption.
func TestASlotLeftOutOfAStepIsNotEvicted(t *testing.T) {
	slots := []schedState{seq(10, 10, 1, 1), seq(10, 10, 2, 2)}
	p, err := nextStep(slots, 512, 1)
	if err != nil {
		t.Fatal(err)
	}
	if p.Decode != 1 {
		t.Fatalf("a one-row step carries %d decodes, want 1", p.Decode)
	}
	for i := range slots {
		if !slots[i].live {
			t.Fatalf("slot %d stopped being live because it did not fit a step; "+
				"eviction is for a pool that cannot grow, not for a busy step", i)
		}
	}
}

// TestAnEmptyOrFinishedSlotContributesNothing.
func TestAnEmptyOrFinishedSlotContributesNothing(t *testing.T) {
	slots := []schedState{{}, seq(10, 10, 5, 1), {}}
	p, err := nextStep(slots, 512, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Work) != 1 || p.Work[0].Slot != 1 {
		t.Fatalf("the step is %+v; a slot holding nothing is absent from the work "+
			"rather than empty in it", p.Work)
	}
}

// TestNextStepRefusals.
func TestNextStepRefusals(t *testing.T) {
	for _, c := range []struct {
		name        string
		chunk, rows int
		want        string
	}{
		{"a chunk of zero", 0, 32, "at least one token"},
		{"a negative chunk", -1, 32, "at least one token"},
		{"no rows", 512, 0, "carries nothing"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := nextStep([]schedState{seq(4, 0, 0, 1)}, c.chunk, c.rows)
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("the refusal does not say %q: %v", c.want, err)
			}
		})
	}
}

// TestTheVictimIsTheLastToArrive is 008-D5.
//
// Last-arrived-first-evicted bounds the worst-case latency of the sequences
// already in flight. Round-robin or longest-running spreads the damage, which
// turns one slow request into several.
func TestTheVictimIsTheLastToArrive(t *testing.T) {
	slots := []schedState{
		seq(10, 10, 1, 3),
		seq(10, 10, 2, 7),
		seq(10, 10, 3, 5),
	}
	if got := victim(slots); got != 1 {
		t.Fatalf("victim = %d, want 1: the slot that arrived last", got)
	}
}

// TestAMidPrefillVictimIsPreferredAtEqualArrival.
//
// 008-D2 makes eviction a recompute. A sequence that has not finished
// prefilling has produced nothing a caller has read, so recomputing it costs
// the same device work and none of the latency a caller has already been
// promised.
func TestAMidPrefillVictimIsPreferredAtEqualArrival(t *testing.T) {
	slots := []schedState{
		seq(100, 100, 1, 9), // decoding, arrived latest among the decoders
		seq(100, 20, 0, 4),  // mid-prefill, arrived earlier
	}
	if got := victim(slots); got != 1 {
		t.Fatalf("victim = %d, want 1: a sequence with no first token yet is the "+
			"cheaper recompute", got)
	}
	// But a decoding slot is still evicted when it is the only live one.
	if got := victim(slots[:1]); got != 0 {
		t.Fatalf("victim = %d over one decoding slot, want 0", got)
	}
}

// TestThereIsNoVictimInAnEmptyBatch.
func TestThereIsNoVictimInAnEmptyBatch(t *testing.T) {
	if got := victim(nil); got != -1 {
		t.Fatalf("victim(nil) = %d, want -1", got)
	}
	if got := victim([]schedState{{}, {}}); got != -1 {
		t.Fatalf("victim over dead slots = %d, want -1", got)
	}
}
