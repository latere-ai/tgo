// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"errors"
	"strings"
	"testing"
)

// batchModel is the fixture with a shared block pool large enough for a few
// slots, which a batch requires.
func batchModel(t *testing.T) *Model {
	t.Helper()
	return openSynthetic(t, WithPrefixCache(CacheProcess, blockPoolCap))
}

// newBatch opens a batch of n slots and closes it with the test.
func newBatch(t *testing.T, m *Model, n int) *Batch {
	t.Helper()
	b, err := m.NewBatch(n)
	if err != nil {
		t.Fatalf("NewBatch(%d): %v", n, err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Errorf("Batch.Close: %v", err)
		}
	})
	return b
}

// greedyOf is the token a set of logits picks under a greedy policy, which is
// what a comparison between a batched step and a single one has to agree on:
// two runs that agree on every logit agree on this, and asserting the token as
// well says what a caller would actually have observed.
func greedyOf(logits []float32) int {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return best
}

// TestABatchStepEqualsTheSessionsItBatches is the engine-level half of the
// claim the model layer already makes about shapes.
//
// Two slots prefilling different prompts in one dispatch, then two slots
// decoding in one dispatch, against two ordinary sessions doing the same thing
// one at a time. Every logit has to agree: the batch changes which submission a
// sequence's rows are in, not what they compute.
//
// **Two slots and not eight**, deliberately. A batched step runs B forward
// passes on the CPU backend and CONTRIBUTING's race budget is 3500s of CPU
// against a suite already at 2547s, so the fixture is the smallest one that can
// fail — one slot is not a batch, and three would assert nothing two does not.
func TestABatchStepEqualsTheSessionsItBatches(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	prompts := [][]int{promptIDs(1, 20), promptIDs(2, 12)}

	b := newBatch(t, m, 2)
	work := make([]Work, len(prompts))
	for i, p := range prompts {
		reused, err := b.Admit(i, p, "")
		if err != nil {
			t.Fatalf("Admit(%d): %v", i, err)
		}
		if reused != 0 {
			t.Fatalf("slot %d reused %d positions of an empty pool", i, reused)
		}
		work[i] = Work{Slot: i, Tokens: p}
	}
	stepped, err := b.Step(work)
	if err != nil {
		t.Fatalf("the prefill step: %v", err)
	}
	// Copied, because a slot's logits are its own only until that slot steps
	// again and this test steps both slots below. A generation loop samples
	// before the next step and needs no copy; a test that compares across one
	// does.
	prefill := make([][]float32, len(stepped))
	for i, v := range stepped {
		prefill[i] = append([]float32(nil), v...)
	}

	// One decode each, fed the token the prefill chose, which is what a
	// generation loop would do.
	next := make([]Work, len(prompts))
	for i := range prompts {
		next[i] = Work{Slot: i, Tokens: []int{greedyOf(prefill[i])}}
	}
	steppedAgain, err := b.Step(next)
	if err != nil {
		t.Fatalf("the decode step: %v", err)
	}
	decode := make([][]float32, len(steppedAgain))
	for i, v := range steppedAgain {
		decode[i] = append([]float32(nil), v...)
	}

	// The same two sequences, one session each, one step at a time.
	for i, p := range prompts {
		s := session(t, m, WithSessionContext(sharedCap))
		rows, err := s.buckets.For(len(p))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.acquire(p, ""); err != nil {
			t.Fatalf("acquire: %v", err)
		}
		wantPrefill, _, err := s.run(rows, p, 0)
		if err != nil {
			t.Fatalf("session %d prefill: %v", i, err)
		}
		compareLogits(t, "prefill", i, prefill[i], wantPrefill)

		fed := greedyOf(wantPrefill)
		s.length, s.history = len(p), append(s.history, p...)
		if err := s.reserve(fed); err != nil {
			t.Fatalf("reserve: %v", err)
		}
		wantDecode, _, err := s.run(1, []int{fed}, len(p))
		if err != nil {
			t.Fatalf("session %d decode: %v", i, err)
		}
		compareLogits(t, "decode", i, decode[i], wantDecode)
	}
	t.Logf("two prompts of %d and %d tokens prefilled in one dispatch and decoded "+
		"in one more, matching two sessions exactly", len(prompts[0]), len(prompts[1]))
}

func compareLogits(t *testing.T, phase string, slot int, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s slot %d: %d logits against %d", phase, slot, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s slot %d logit %d is %v in a batch and %v in a session; a "+
				"batched step is the steps it batches", phase, slot, i, got[i], want[i])
		}
	}
}

// TestABatchDoesNotLetOneSlotSeeAnother is the failure a shared dispatch exists
// to risk, asserted directly.
//
// Two slots whose prompts share no leading run, one long and one short, in one
// step. If a slot's causal window reached past its own extent it would attend to
// the other's rows, and the answer would still be fluent. The single-session run
// is the only thing that can tell.
func TestABatchDoesNotLetOneSlotSeeAnother(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	long, short := promptIDs(3, 30), promptIDs(4, 3)

	b := newBatch(t, m, 2)
	for i, p := range [][]int{long, short} {
		if _, err := b.Admit(i, p, ""); err != nil {
			t.Fatalf("Admit(%d): %v", i, err)
		}
	}
	got, err := b.Step([]Work{{Slot: 0, Tokens: long}, {Slot: 1, Tokens: short}})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}

	// The short one alone. If the batch let it see the long sequence's rows,
	// this is where the two part.
	s := session(t, m, WithSessionContext(sharedCap))
	if _, err := s.acquire(short, ""); err != nil {
		t.Fatal(err)
	}
	rows, err := s.buckets.For(len(short))
	if err != nil {
		t.Fatal(err)
	}
	want, _, err := s.run(rows, short, 0)
	if err != nil {
		t.Fatal(err)
	}
	compareLogits(t, "a 3-token slot beside a 30-token one", 1, got[1], want)
}

// TestABatchSharesBlocksBetweenItsSlots: the pool is the same pool, so two
// slots with the same system prompt prefill it once between them.
func TestABatchSharesBlocksBetweenItsSlots(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	system := promptIDs(1, 2*CacheBlock)

	b := newBatch(t, m, 2)
	first := extend(system, 7, 6, 0)
	if _, err := b.Admit(0, first, ""); err != nil {
		t.Fatalf("Admit(0): %v", err)
	}
	if _, err := b.Step([]Work{{Slot: 0, Tokens: first}}); err != nil {
		t.Fatalf("Step: %v", err)
	}

	// A different conversation, into a slot that has never held these tokens.
	second := extend(system, 11, 6, 0)
	reused, err := b.Admit(1, second, "")
	if err != nil {
		t.Fatalf("Admit(1): %v", err)
	}
	if want := len(system); reused != want {
		t.Fatalf("slot 1 reused %d of a %d-token opening slot 0 had already paid "+
			"for; a batch draws from the same pool every session does", reused, want)
	}
}

// TestBatchRefusals: each names the field.
func TestBatchRefusals(t *testing.T) {
	t.Parallel()
	t.Run("a model with no shared pool", func(t *testing.T) {
		t.Parallel()
		m := openSynthetic(t)
		_, err := m.NewBatch(2)
		if !errors.Is(err, ErrNoBlockPool) {
			t.Fatalf("NewBatch on a contiguous model = %v, want ErrNoBlockPool", err)
		}
		if !strings.Contains(err.Error(), "CacheProcess") {
			t.Errorf("the refusal does not say how to fix it: %v", err)
		}
	})
	t.Run("a batch of one", func(t *testing.T) {
		t.Parallel()
		m := batchModel(t)
		if _, err := m.NewBatch(1); err == nil {
			t.Fatal("a batch of one slot was accepted")
		} else if !strings.Contains(err.Error(), "two or more") {
			t.Errorf("the refusal does not say what a batch is: %v", err)
		}
	})

	m := batchModel(t)
	b := newBatch(t, m, 2)
	ids := promptIDs(1, 8)
	if _, err := b.Admit(0, ids, ""); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		work []Work
		want string
	}{
		{"no work", nil, "computes nothing"},
		{"a slot outside the batch", []Work{{Slot: 5, Tokens: ids}}, "outside a batch"},
		{"a slot twice", []Work{{Slot: 0, Tokens: ids}, {Slot: 0, Tokens: ids}},
			"appears twice"},
		{"a slot with no tokens", []Work{{Slot: 0, Tokens: nil}}, "absent from the work"},
		{"a slot never admitted", []Work{{Slot: 1, Tokens: ids}}, "admit it before"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := b.Step(c.work); err == nil {
				t.Fatal("accepted")
			} else if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("the refusal does not say %q: %v", c.want, err)
			}
		})
	}
	if _, err := b.Admit(9, ids, ""); err == nil {
		t.Error("Admit on a slot outside the batch was accepted")
	}
	if err := b.Evict(9); err == nil {
		t.Error("Evict on a slot outside the batch was accepted")
	}
}

// TestEvictDropsASlotsBlocksAndItsHistory is 008-D2: a preempted sequence drops
// its blocks and re-prefills on readmission rather than swapping to the host.
func TestEvictDropsASlotsBlocksAndItsHistory(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	b := newBatch(t, m, 2)
	ids := promptIDs(1, 2*CacheBlock)
	if _, err := b.Admit(0, ids, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Step([]Work{{Slot: 0, Tokens: ids}}); err != nil {
		t.Fatal(err)
	}
	if got := b.Length(0); got != len(ids) {
		t.Fatalf("slot 0 holds %d positions after prefilling %d", got, len(ids))
	}
	held := m.blocks.pool.Stats().InUse
	if held == 0 {
		t.Fatal("a slot that just prefilled holds no blocks")
	}

	if err := b.Evict(0); err != nil {
		t.Fatalf("Evict: %v", err)
	}
	if got := b.Length(0); got != 0 {
		t.Fatalf("an evicted slot holds %d positions", got)
	}
	if got := m.blocks.pool.Stats().InUse; got != 0 {
		t.Fatalf("%d blocks are still referenced after an eviction; the point of "+
			"evicting is that another sequence can have them", got)
	}
	// Not freed outright: the published blocks are the cache, so readmitting
	// the same prompt recomputes nothing.
	reused, err := b.Admit(0, ids, "")
	if err != nil {
		t.Fatal(err)
	}
	if reused == 0 {
		t.Fatal("readmitting an evicted sequence reused nothing; eviction drops the " +
			"reference and the blocks stay in the cache (016-D5)")
	}
}

// TestASlotsLogitsSurviveAnotherSlotsStep pins the lifetime [Batch.Step]
// documents, because the first draft of that lifetime was one nobody could keep
// and it produced a false failure that read as a batching bug.
//
// The rule is: until *that* slot steps again. So another slot stepping must
// leave it alone, and the slot's own next step is allowed to replace it.
func TestASlotsLogitsSurviveAnotherSlotsStep(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	b := newBatch(t, m, 2)
	for i, p := range [][]int{promptIDs(1, 10), promptIDs(2, 6)} {
		if _, err := b.Admit(i, p, ""); err != nil {
			t.Fatalf("Admit(%d): %v", i, err)
		}
	}
	first, err := b.Step([]Work{{Slot: 0, Tokens: promptIDs(1, 10)}})
	if err != nil {
		t.Fatal(err)
	}
	held := first[0]
	before := append([]float32(nil), held...)

	// Slot 1 steps. Slot 0 said nothing and must be untouched.
	if _, err := b.Step([]Work{{Slot: 1, Tokens: promptIDs(2, 6)}}); err != nil {
		t.Fatal(err)
	}
	for i := range before {
		if held[i] != before[i] {
			t.Fatalf("slot 0's logit %d changed from %v to %v while slot 1 stepped; a "+
				"slot's logits are its own until it steps again", i, before[i], held[i])
		}
	}
}

// TestAdmitAndStepDoNotLeaseTheSamePromptTwice is the bug that had no symptom
// a value test could see.
//
// Admit leases the whole prompt; Step then reserved the tokens it was handed,
// which for a prefill is that same prompt. The lease grew to twice the
// sequence, took twice the blocks, and -- the part that matters -- chained its
// block hashes over `prompt+prompt`. Those hashes name blocks holding only
// `prompt`, so a later request whose prompt genuinely was the doubled one would
// match them and attend to a cache holding something else. The logits of the
// step that caused it are correct, so nothing downstream reports it.
//
// The lease's own length is the thing to assert, because it is the quantity
// that was wrong.
func TestAdmitAndStepDoNotLeaseTheSamePromptTwice(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	b := newBatch(t, m, 2)
	prompt := promptIDs(1, 2*CacheBlock+5)
	if _, err := b.Admit(0, prompt, ""); err != nil {
		t.Fatal(err)
	}
	if got := b.slots[0].lease.Len(); got != len(prompt) {
		t.Fatalf("admitting a %d-token prompt leased %d positions", len(prompt), got)
	}
	if _, err := b.Step([]Work{{Slot: 0, Tokens: prompt}}); err != nil {
		t.Fatal(err)
	}
	if got := b.slots[0].lease.Len(); got != len(prompt) {
		t.Fatalf("after prefilling the prompt it was admitted with, the lease covers "+
			"%d positions and the sequence is %d long; the prefill was leased twice",
			got, len(prompt))
	}

	// A generated token does extend it, which is the other half: a reserve
	// that appended nothing would leave the token with no row to be written to.
	if _, err := b.Step([]Work{{Slot: 0, Tokens: []int{7}}}); err != nil {
		t.Fatal(err)
	}
	if got, want := b.slots[0].lease.Len(), len(prompt)+1; got != want {
		t.Fatalf("after one generated token the lease covers %d positions, want %d",
			got, want)
	}
	if got := b.Length(0); got != len(prompt)+1 {
		t.Fatalf("the slot holds %d positions and the lease covers %d", got,
			b.slots[0].lease.Len())
	}
}
