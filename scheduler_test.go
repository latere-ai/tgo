// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"errors"
	"strings"
	"testing"
)

func newScheduler(t *testing.T, m *Model, n int, o SchedulerOptions) *Scheduler {
	t.Helper()
	s, err := m.NewScheduler(n, o)
	if err != nil {
		t.Fatalf("NewScheduler(%d, %+v): %v", n, o, err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Scheduler.Close: %v", err)
		}
	})
	return s
}

// TestAChunkAndTheDecodesRunInOneDispatch is 008 §5 on the device, and it is
// the claim the ragged step exists for.
//
// One long prompt being chunked and two sequences decoding beside it. The
// scheduler must put them in **one** step, so the weights are read once for all
// three -- and the tokens the two decoding sequences produce must be the ones
// they produce on their own, because a chunk riding along must not change what
// its passengers see.
//
// The chunk is 8 and the prompts are tens of tokens, not the 512 and thousands
// a deployment uses: every step here is a real forward pass on the CPU backend,
// and CONTRIBUTING's race budget is what a fixture sized for realism would
// spend. What is under test is the mix, and the mix is the same at any size.
func TestAChunkAndTheDecodesRunInOneDispatch(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	s := newScheduler(t, m, 3, SchedulerOptions{Chunk: 8, Reserve: CacheBlock})

	long := promptIDs(1, 20)
	short := [][]int{promptIDs(2, 4), promptIDs(3, 4)}

	// The two short ones first, so they are decoding by the time the long one
	// is still being chunked.
	for i, p := range short {
		slot, err := s.Admit(p, "")
		if err != nil {
			t.Fatalf("Admit(short %d): %v", i, err)
		}
		if slot != i {
			t.Fatalf("the %dth admission took slot %d", i, slot)
		}
	}
	first, err := s.Step()
	if err != nil {
		t.Fatalf("the first step: %v", err)
	}
	if first.Decodes != 0 || first.PrefillTokens != 8 {
		t.Fatalf("the first step carried %d prefill tokens and %d decodes, want 8 and 0",
			first.PrefillTokens, first.Decodes)
	}
	for _, p := range first.Produced {
		if !p.Sampleable() {
			t.Fatalf("slot %d's 4-token prompt is not scored after a step that "+
				"carried 8 prefill tokens", p.Slot)
		}
		if err := s.Feed(p.Slot, greedyOf(p.Logits)); err != nil {
			t.Fatal(err)
		}
	}

	// Now the long one arrives and is chunked beside them.
	slot, err := s.Admit(long, "")
	if err != nil {
		t.Fatalf("Admit(long): %v", err)
	}
	mixed, err := s.Step()
	if err != nil {
		t.Fatalf("the mixed step: %v", err)
	}
	if mixed.PrefillTokens != 8 || mixed.Decodes != 2 {
		t.Fatalf("the mixed step carried %d prefill tokens and %d decodes, want 8 and "+
			"2; a chunk that ran alone would stall the two beside it and read the "+
			"weights twice", mixed.PrefillTokens, mixed.Decodes)
	}
	if got := len(mixed.Produced); got != 3 {
		t.Fatalf("%d slots contributed to the mixed step, want 3", got)
	}
	for _, p := range mixed.Produced {
		if p.Slot == slot && p.Sampleable() {
			t.Fatal("a slot 8 tokens into a 20-token prompt reports a sampleable " +
				"distribution; those logits are over a token the caller already has")
		}
	}
	t.Logf("one dispatch: %d prompt tokens of a chunked prefill and %d decodes",
		mixed.PrefillTokens, mixed.Decodes)
}

// TestASchedulersDecodesMatchTheSessionsTheyBatch is the value half, and the
// only thing that can tell a working mix from a plausible one.
func TestASchedulersDecodesMatchTheSessionsTheyBatch(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	s := newScheduler(t, m, 2, SchedulerOptions{Chunk: 8, Reserve: CacheBlock})

	prompts := [][]int{promptIDs(4, 6), promptIDs(5, 10)}
	for _, p := range prompts {
		if _, err := s.Admit(p, ""); err != nil {
			t.Fatalf("Admit: %v", err)
		}
	}
	// Step until both prompts are scored, feeding what each samples, then take
	// one more token from each.
	var got [][]int
	for range 6 {
		r, err := s.Step()
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		if len(r.Produced) == 0 {
			break
		}
		row := make([]int, s.Slots())
		for i := range row {
			row[i] = -1
		}
		for _, p := range r.Produced {
			if !p.Sampleable() {
				continue
			}
			tok := greedyOf(p.Logits)
			row[p.Slot] = tok
			if err := s.Feed(p.Slot, tok); err != nil {
				t.Fatal(err)
			}
		}
		got = append(got, row)
	}

	// The same prompts, one session each, greedy, in sequence.
	for i, p := range prompts {
		sess := session(t, m, WithSessionContext(sharedCap))
		st, err := sess.start(t.Context(), p, greedy(3))
		if err != nil {
			t.Fatalf("session %d: %v", i, err)
		}
		var want []int
		for st.Next() {
		}
		if err := st.Err(); err != nil {
			t.Fatal(err)
		}
		want = append(want, sess.history[len(p):]...)

		var batched []int
		for _, row := range got {
			if row[i] >= 0 {
				batched = append(batched, row[i])
			}
		}
		n := min(len(want), len(batched))
		if n == 0 {
			t.Fatalf("sequence %d produced nothing through the scheduler", i)
		}
		for j := range n {
			if batched[j] != want[j] {
				t.Fatalf("sequence %d generated token %d as %d through the scheduler "+
					"and %d through a session; a batched step is the steps it batches",
					i, j, batched[j], want[j])
			}
		}
	}
}

// TestAdmitRefusesWithoutASlotAndSaysWhich: a caller has to be able to tell
// "wait" from "this will never fit".
func TestAdmitRefusesWithoutASlotAndSaysWhich(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	s := newScheduler(t, m, 2, SchedulerOptions{Chunk: 8, Reserve: CacheBlock})
	for range 2 {
		if _, err := s.Admit(promptIDs(1, 4), ""); err != nil {
			t.Fatalf("Admit: %v", err)
		}
	}
	if _, err := s.Admit(promptIDs(2, 4), ""); !errors.Is(err, ErrNoSlot) {
		t.Fatalf("a third admission into a batch of two = %v, want ErrNoSlot", err)
	}
	// A finished slot is available again.
	if err := s.Finish(0); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if slot, err := s.Admit(promptIDs(2, 4), ""); err != nil {
		t.Fatalf("Admit after Finish: %v", err)
	} else if slot != 0 {
		t.Fatalf("the readmission took slot %d, want the freed 0", slot)
	}
}

// TestEvictChoosesTheLastToArriveOnTheDevice wires 008-D5 to the pool: the
// victim's blocks go back.
func TestEvictChoosesTheLastToArriveOnTheDevice(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	s := newScheduler(t, m, 2, SchedulerOptions{Chunk: 8, Reserve: CacheBlock})
	for range 2 {
		if _, err := s.Admit(promptIDs(1, 4), ""); err != nil {
			t.Fatal(err)
		}
	}
	held := m.blocks.pool.Stats().InUse
	v, err := s.Evict()
	if err != nil {
		t.Fatal(err)
	}
	if v != 1 {
		t.Fatalf("the victim is slot %d, want 1: the one that arrived last", v)
	}
	if got := m.blocks.pool.Stats().InUse; got >= held {
		t.Fatalf("the pool holds %d blocks after evicting and %d before; evicting is "+
			"how a full pool makes room", got, held)
	}
	// And the slot is free.
	if slot, err := s.Admit(promptIDs(3, 4), ""); err != nil {
		t.Fatalf("Admit after Evict: %v", err)
	} else if slot != 1 {
		t.Fatalf("the readmission took slot %d, want the evicted 1", slot)
	}
	if v, err := s.Evict(); err != nil || v < 0 {
		t.Fatalf("Evict over live slots = %d, %v", v, err)
	}
}

// TestSchedulerRefusals.
func TestSchedulerRefusals(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	if _, err := m.NewScheduler(2, SchedulerOptions{Chunk: 8}); err == nil {
		t.Fatal("a scheduler with no reserve was accepted")
	} else if !strings.Contains(err.Error(), "cannot grow") {
		t.Errorf("the refusal does not say what a reserve is for: %v", err)
	}
	if _, err := m.NewScheduler(2, SchedulerOptions{Chunk: -1, Reserve: 8}); err == nil {
		t.Fatal("a negative chunk was accepted")
	}

	s := newScheduler(t, m, 2, SchedulerOptions{Reserve: CacheBlock})
	if _, err := s.Step(); err != nil {
		t.Fatalf("a step with nothing to do = %v, want no work and no error", err)
	}
	if err := s.Feed(0, 1); err == nil {
		t.Error("feeding a slot holding no sequence was accepted")
	}
	if err := s.Feed(9, 1); err == nil {
		t.Error("feeding a slot outside the batch was accepted")
	}

	long := promptIDs(1, 40)
	slot, err := s.Admit(long, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Feed(slot, 1); err == nil {
		t.Fatal("feeding a slot whose prompt is not scored was accepted; the token " +
			"would have been generated from a distribution over the middle of it")
	} else if !strings.Contains(err.Error(), "middle of it") {
		t.Errorf("the refusal does not say why: %v", err)
	}
	if err := s.Finish(9); err == nil {
		t.Error("finishing a slot outside the batch was accepted")
	}
}
