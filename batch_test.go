// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"errors"

	"strings"
	"testing"

	"github.com/latere-ai/tgo/internal/prefix"
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
		reused, err := b.Admit(i, p, "", 0)
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
		if _, err := b.Admit(i, p, "", 0); err != nil {
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
	if _, err := b.Admit(0, first, "", 0); err != nil {
		t.Fatalf("Admit(0): %v", err)
	}
	if _, err := b.Step([]Work{{Slot: 0, Tokens: first}}); err != nil {
		t.Fatalf("Step: %v", err)
	}

	// A different conversation, into a slot that has never held these tokens.
	second := extend(system, 11, 6, 0)
	reused, err := b.Admit(1, second, "", 0)
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
	if _, err := b.Admit(0, ids, "", 0); err != nil {
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
	if _, err := b.Admit(9, ids, "", 0); err == nil {
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
	if _, err := b.Admit(0, ids, "", 0); err != nil {
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
	reused, err := b.Admit(0, ids, "", 0)
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
		if _, err := b.Admit(i, p, "", 0); err != nil {
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
	if _, err := b.Admit(0, prompt, "", 0); err != nil {
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

// TestAdmissionRefusesWhatThePoolCannotHoldWithItsAnswer is 008 §3, and the
// difference between it and admitting on a free slot alone.
//
// Two slots are free. The pool can hold the second prompt. It cannot hold the
// second prompt *and* the answer that prompt is going to generate, and that is
// the case a scheduler must refuse: admitting it fills both slots with
// sequences that cannot grow, so nothing finishes and there is nothing to
// evict into progress.
func TestAdmissionRefusesWhatThePoolCannotHoldWithItsAnswer(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	b := newBatch(t, m, 2)
	blocks := m.blocks.positions / CacheBlock

	// The first slot takes most of the pool, prompt and reserve together.
	first := promptIDs(1, (blocks-2)*CacheBlock)
	if _, err := b.Admit(0, first, "", CacheBlock); err != nil {
		t.Fatalf("Admit(0): %v", err)
	}

	// The second prompt fits on its own: one block is left over.
	second := promptIDs(2, CacheBlock)
	if _, err := b.Admit(1, second, "", 0); err != nil {
		t.Fatalf("a prompt that fits the pool on its own was refused: %v", err)
	}
	if err := b.Evict(1); err != nil {
		t.Fatal(err)
	}

	// With an answer to write, it does not, and the refusal says so rather
	// than admitting a sequence that cannot grow.
	_, err := b.Admit(1, second, "", 4*CacheBlock)
	if err == nil {
		t.Fatal("a prompt whose answer the pool cannot hold was admitted; a slot " +
			"filled with a sequence that cannot grow is 008 §3's deadlock")
	}
	if !errors.Is(err, prefix.ErrExhausted) {
		t.Fatalf("the refusal is not ErrExhausted: %v", err)
	}
	for _, want := range []string{"reserve of", "exhausted"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not say %q: %v", want, err)
		}
	}
}

// TestAReservedSlotGrowsWithoutAskingAgain: the blocks were taken at
// admission, so generating into them cannot fail for want of a pool.
func TestAReservedSlotGrowsWithoutAskingAgain(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	b := newBatch(t, m, 2)

	prompt := promptIDs(1, CacheBlock)
	if _, err := b.Admit(0, prompt, "", 2*CacheBlock); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Step([]Work{{Slot: 0, Tokens: prompt}}); err != nil {
		t.Fatal(err)
	}

	// Fill the pool with the other slot, so nothing is left to allocate.
	rest := promptIDs(2, m.blocks.positions-3*CacheBlock)
	if _, err := b.Admit(1, rest, "", 0); err != nil {
		t.Fatalf("filling the pool: %v", err)
	}
	if free := m.blocks.pool.Stats().Free; free != 0 {
		t.Fatalf("%d blocks are still free; this test needs an exhausted pool", free)
	}
	held := m.blocks.pool.Stats().InUse

	// Slot 0 grows into its reserve, across a block boundary, on a pool with
	// nothing left to allocate. Two steps rather than sixty-four one-token
	// ones: what is under test is the reserve, not the decode, and a batched
	// step over a real forward pass is what CONTRIBUTING's CPU budget is spent
	// on.
	grow := make([]int, CacheBlock+2)
	for i := range grow {
		grow[i] = 7
	}
	if _, err := b.Step([]Work{{Slot: 0, Tokens: grow}}); err != nil {
		t.Fatalf("growing into a reserve on an exhausted pool: %v", err)
	}
	if _, err := b.Step([]Work{{Slot: 0, Tokens: []int{7}}}); err != nil {
		t.Fatalf("one more token into the reserve: %v", err)
	}
	if got := m.blocks.pool.Stats().InUse; got != held {
		t.Fatalf("growing into the reserve took the pool from %d blocks to %d; the "+
			"reserve is what makes admission a promise", held, got)
	}
	// The assertion above cannot fail on an exhausted pool -- there is nothing
	// left to take, so InUse cannot rise -- and a review caught that. What
	// makes it mean something is that the growth *succeeded*: a slot without
	// its reserve would have been refused by the steps above, so this checks
	// the pool is genuinely full and that the slot reached its whole reserve.
	if free := m.blocks.pool.Stats().Free; free != 0 {
		t.Fatalf("%d blocks are free, so nothing above was constrained", free)
	}
	if got, want := b.Length(0), CacheBlock+CacheBlock+3; got != want {
		t.Fatalf("the slot is %d positions in, want %d: the prompt and the whole "+
			"reserve it was admitted with", got, want)
	}
}

// TestStepReturnsResultsInTheOrderTheWorkNamed pins what Batch.Step documents:
// "each one's logits, in the order work names them".
//
// Every other test passes work already in slot order, so a Step that returned
// results in *slot* order would have satisfied all of them. The two orders are
// the same until a caller has a reason to submit otherwise -- a scheduler that
// placed prefills first, for instance -- and then a caller reading out[i] for
// work[i] gets another slot's distribution and samples from it.
func TestStepReturnsResultsInTheOrderTheWorkNamed(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	b := newBatch(t, m, 2)

	prompts := [][]int{promptIDs(1, 12), promptIDs(2, 5)}
	for i, p := range prompts {
		if _, err := b.Admit(i, p, "", 0); err != nil {
			t.Fatalf("Admit(%d): %v", i, err)
		}
	}

	// Slot 1 first, which is what makes this test different from the others.
	out, err := b.Step([]Work{
		{Slot: 1, Tokens: prompts[1]},
		{Slot: 0, Tokens: prompts[0]},
	})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}

	// Each sequence's own logits, taken from a run where it is alone, so the
	// comparison is against a value and not against the other entry.
	for i, p := range prompts {
		alone := newBatch(t, m, 2)
		if _, err := alone.Admit(i, p, "solo", 0); err != nil {
			t.Fatal(err)
		}
		want, err := alone.Step([]Work{{Slot: i, Tokens: p}})
		if err != nil {
			t.Fatal(err)
		}
		// out[0] is slot 1's and out[1] is slot 0's, because that is the order
		// the work named.
		got := out[1]
		if i == 1 {
			got = out[0]
		}
		compareLogits(t, "work order", i, got, want[0])
	}
}

// TestAStepReadsBackOnlyTheRowsItProduced.
//
// A readback is VocabSize floats per slot, which for Qwen3 is 593.5 KiB
// against four bytes of useful output. Reading the whole buffer charged every
// step for every idle slot too, so an eight-slot scheduler with one live
// decoder moved 4.86 MB to learn one token.
//
// The assertion is on the values rather than on the byte count, because the
// byte count is not observable from here and the thing that would break is:
// a slot outside the span read must keep what it had, and a slot inside it
// must get what the device computed.
func TestAStepReadsBackOnlyTheRowsItProduced(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	b := newBatch(t, m, 3)

	prompts := [][]int{promptIDs(1, 8), promptIDs(2, 9), promptIDs(3, 10)}
	for i, p := range prompts {
		if _, err := b.Admit(i, p, "", 0); err != nil {
			t.Fatalf("Admit(%d): %v", i, err)
		}
	}
	// Slot 1 alone: the span is one row in the middle, so a read that started
	// at zero or ran to the end would still produce it and a read of the wrong
	// row would not.
	one, err := b.Step([]Work{{Slot: 1, Tokens: prompts[1]}})
	if err != nil {
		t.Fatal(err)
	}
	mid := append([]float32(nil), one[0]...)

	// The same sequence in a batch of its own, to say what row 1 should hold.
	solo := newBatch(t, m, 3)
	if _, err := solo.Admit(1, prompts[1], "solo", 0); err != nil {
		t.Fatal(err)
	}
	want, err := solo.Step([]Work{{Slot: 1, Tokens: prompts[1]}})
	if err != nil {
		t.Fatal(err)
	}
	compareLogits(t, "a one-slot span in the middle", 1, mid, want[0])

	// Now slots 0 and 2, which spans all three rows, and slot 1's buffer must
	// still hold what its own step produced.
	if _, err := b.Step([]Work{
		{Slot: 0, Tokens: prompts[0]}, {Slot: 2, Tokens: prompts[2]},
	}); err != nil {
		t.Fatal(err)
	}
	for i := range mid {
		if b.slots[1].out[i] != mid[i] {
			t.Fatalf("slot 1's logit %d changed from %v to %v while slots 0 and 2 "+
				"stepped over it", i, mid[i], b.slots[1].out[i])
		}
	}
}
