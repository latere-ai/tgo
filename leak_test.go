// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"math"
	"testing"
)

// Three defects a multi-agent review of Wave 9 found and this file keeps out.
//
// None of them had a symptom the value tests could see: in every one the logits
// of the step that caused the damage are correct, and the wrong answer is
// handed to a *later* request whose own caller did nothing unusual. They share
// one root cause -- a lease covering positions no step has computed -- and one
// fix, which is that a lease grows before a step and records after it, and
// publishes only what its slot has written.

// divergence is the largest absolute difference between two logit rows.
func divergence(a, b []float32) float64 {
	worst := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > worst {
			worst = d
		}
	}
	return worst
}

// drain steps until the named slot's prompt is scored and returns its logits.
func drain(t *testing.T, s *Scheduler, slot int) []float32 {
	t.Helper()
	for range 64 {
		r, err := s.Step()
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		for _, p := range r.Produced {
			if p.Slot == slot && p.Sampleable() {
				return append([]float32(nil), p.Logits...)
			}
		}
	}
	t.Fatalf("slot %d never finished its prompt", slot)
	return nil
}

// TestAChunkedPrefillPublishesOnlyWhatItWrote is the worst of the three.
//
// `Admit` leases the whole prompt so admission is a promise, and the lease
// chains a hash over every complete block of it at that moment. Publishing on
// the lease's extent rather than on what the slot has written therefore named
// blocks holding nothing: a slot 32 tokens into a 192-token prompt published
// all six, and the next request with the same prefix reused 192 positions of
// which 160 were never computed. It attends to that and answers fluently.
//
// Reached by any prompt longer than the scheduler's chunk, which is the
// ordinary serving case.
func TestAChunkedPrefillPublishesOnlyWhatItWrote(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t, WithPrefixCache(CacheProcess, 32*CacheBlock))
	b, err := m.NewBatch(2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Errorf("Batch.Close: %v", err)
		}
	})

	prompt := promptIDs(11, 6*CacheBlock)
	if _, err := b.Admit(0, prompt, "tenant", CacheBlock); err != nil {
		t.Fatal(err)
	}
	// One chunk of six.
	if _, err := b.Step([]Work{{Slot: 0, Tokens: prompt[:CacheBlock]}}); err != nil {
		t.Fatal(err)
	}

	// A second conversation on the same prefix may reuse at most what the
	// first has actually written.
	reused, err := b.Admit(1, prompt, "tenant", CacheBlock)
	if err != nil {
		t.Fatal(err)
	}
	if reused > CacheBlock {
		t.Fatalf("a second request reused %d positions after the first had written "+
			"%d; the blocks past that hold nothing, and it attends to them",
			reused, CacheBlock)
	}

	// And the whole prompt, once written, is publishable.
	for i := CacheBlock; i < len(prompt); i += CacheBlock {
		if _, err := b.Step([]Work{{Slot: 0, Tokens: prompt[i : i+CacheBlock]}}); err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Evict(1); err != nil {
		t.Fatal(err)
	}
	full, err := b.Admit(1, prompt, "tenant", CacheBlock)
	if err != nil {
		t.Fatal(err)
	}
	if want := len(prompt) - CacheBlock; full != want {
		t.Fatalf("after the whole prompt was written a second request reused %d "+
			"positions, want %d; publishing what *is* written is the other half "+
			"of the rule", full, want)
	}
}

// TestAFailedStepRecordsNoToken is the same root cause reached the other way.
//
// A step reserved its blocks and recorded its tokens before submitting. When
// the submission failed the slot's length did not advance but the lease's did,
// so a retry carrying a *different* token wrote that token's key and value into
// a block whose hash names the abandoned one. A later request whose prompt
// genuinely ends in the abandoned token matches that hash and attends to
// something else.
//
// Blocks before the step and tokens after it is the fix, and this is the
// assertion that keeps it: after a failed step, nothing is reachable by a hash
// over a token nobody computed.
func TestAFailedStepRecordsNoToken(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t, WithPrefixCache(CacheProcess, 8*CacheBlock))
	b, err := m.NewBatch(2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Errorf("Batch.Close: %v", err)
		}
	})

	// One token short of a block, so the next token completes one and makes it
	// publishable -- which is what turns the divergence into a cache entry.
	p0 := promptIDs(1, CacheBlock-1)
	if _, err := b.Admit(0, p0, "s", CacheBlock); err != nil {
		t.Fatal(err)
	}
	p1 := promptIDs(2, 6*CacheBlock)
	if _, err := b.Admit(1, p1, "s", 0); err != nil {
		t.Fatal(err)
	}
	for _, w := range [][]Work{{{Slot: 0, Tokens: p0}}, {{Slot: 1, Tokens: p1}}} {
		if _, err := b.Step(w); err != nil {
			t.Fatal(err)
		}
	}

	// A step whose second slot cannot grow. Slot 0 is reserved first, so this
	// is the interleaving that used to record its token.
	const abandoned, scored = 5, 9
	if _, err := b.Step([]Work{
		{Slot: 0, Tokens: []int{abandoned}}, {Slot: 1, Tokens: []int{7}},
	}); err == nil {
		t.Fatal("a step over an exhausted pool ran")
	}
	if got := b.Length(0); got != CacheBlock-1 {
		t.Fatalf("slot 0 advanced to %d on a step that did not run", got)
	}

	// The retry carries a different token, and lands.
	if _, err := b.Step([]Work{{Slot: 0, Tokens: []int{scored}}}); err != nil {
		t.Fatal(err)
	}

	// Nothing is now reachable under the abandoned token's prefix.
	if err := b.Evict(1); err != nil {
		t.Fatal(err)
	}
	// Longer than one block, or 016-D10's cap at T-1 would keep the lookup
	// from reaching the block at all and the test would be green for a reason
	// that has nothing to do with the defect.
	ghost := append(append([]int(nil), p0...), abandoned)
	ghost = append(ghost, promptIDs(3, CacheBlock)...)
	reused, err := b.Admit(1, ghost, "s", CacheBlock)
	if err != nil {
		t.Fatal(err)
	}
	if reused != 0 {
		t.Fatalf("a prompt whose first block ends in the abandoned token reused %d "+
			"positions; the block it matched holds the token that was scored "+
			"instead, and it attends to that", reused)
	}

	// The control: the prompt that *was* scored does match, so the assertion
	// above is about which token the hash names and not about the pool having
	// nothing in it.
	if err := b.Evict(1); err != nil {
		t.Fatal(err)
	}
	real := append(append([]int(nil), p0...), scored)
	real = append(real, promptIDs(3, CacheBlock)...)
	if hit, err := b.Admit(1, real, "s", CacheBlock); err != nil {
		t.Fatal(err)
	} else if hit != CacheBlock {
		t.Fatalf("the prompt that was actually scored reused %d positions, want %d; "+
			"if this is zero the refusal above proves nothing", hit, CacheBlock)
	}
}

// TestAdmitDoesNotWriteIntoTheCallersPrompt.
//
// `Feed` appends the generated token to the slot's prompt. Keeping the caller's
// slice meant that append writing into the caller's array whenever it had spare
// capacity -- and a caller carving prompts out of one buffer, `all[:20]` then
// `all[20:40]`, had the first request rewrite the first token of the second.
// Both answers stay fluent.
func TestAdmitDoesNotWriteIntoTheCallersPrompt(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t, WithPrefixCache(CacheProcess, 64*CacheBlock))
	s, err := m.NewScheduler(2, SchedulerOptions{Chunk: CacheBlock, Reserve: CacheBlock})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Scheduler.Close: %v", err)
		}
	})

	// One backing array, two prompts carved from it: cap(all[:20]) is 40.
	all := promptIDs(7, 40)
	first, err := s.Admit(all[:20], "")
	if err != nil {
		t.Fatal(err)
	}
	drain(t, s, first)

	before := all[20]
	second := append([]int(nil), all[20:40]...)
	if _, err := s.Admit(all[20:40], ""); err != nil {
		t.Fatal(err)
	}
	if err := s.Feed(first, 1); err != nil {
		t.Fatal(err)
	}
	if all[20] != before {
		t.Fatalf("feeding slot %d rewrote the caller's all[20] from %d to %d, which "+
			"is the first token of the prompt admitted after it",
			first, before, all[20])
	}
	if _, err := s.Admit(second, "reference"); err == nil {
		t.Fatal("a third admission into a batch of two was accepted")
	}
}
