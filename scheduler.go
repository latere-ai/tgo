// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"errors"
	"fmt"
	"sync"
)

// Scheduler drives a [Batch]: it admits requests into slots, decides what each
// step carries, and reports what the step produced.
//
// specs/008-scheduler.md §2 to §5. It is the policy; [Batch] is the mechanism,
// and the split is 008-D8 -- building both at once makes a batching bug and an
// admission bug indistinguishable.
//
// # What it does not do
//
// It does not sample. A step returns each slot's logits and the caller says
// what the next token is, which keeps [specs/006-sampling.md]'s policy where it
// already lives and keeps this file about §3's admission and §5's mix. A
// scheduler that also sampled would be two designs in one type, and the second
// one is the reference for accel's device-side path (006-D1).
//
// # Concurrency
//
// Safe for concurrent use. The step itself is serial -- there is one device
// queue and one plan -- so callers admitting and finishing around it are what
// the lock is for.
type Scheduler struct {
	b       *Batch
	chunk   int
	reserve int

	mu    sync.Mutex
	slots []schedState
	next  uint64
}

// SchedulerOptions are the two policy numbers §3 and §5 leave to a deployment.
type SchedulerOptions struct {
	// Chunk is how many prompt tokens one step prefills for one sequence.
	// Zero takes [DefaultChunk].
	Chunk int

	// Reserve is how many positions beyond its prompt an admitted sequence
	// holds blocks for: §3's R, and the difference between admitting a request
	// and promising it can finish. Zero is refused, because a scheduler that
	// admits on the prompt alone is the deadlock §3 is about.
	Reserve int
}

// NewScheduler builds a scheduler over a batch of n slots.
func (m *Model) NewScheduler(n int, o SchedulerOptions) (*Scheduler, error) {
	if o.Reserve <= 0 {
		return nil, fmt.Errorf("tgo: the scheduler's reserve is %d; it is how many "+
			"positions an admitted sequence may grow by, and admitting without one "+
			"fills every slot with sequences that cannot grow "+
			"(specs/008-scheduler.md §3)", o.Reserve)
	}
	if o.Chunk == 0 {
		o.Chunk = DefaultChunk
	}
	if o.Chunk < 0 {
		return nil, fmt.Errorf("tgo: the prefill chunk is %d; a chunk is at least one "+
			"token", o.Chunk)
	}
	b, err := m.NewBatch(n)
	if err != nil {
		return nil, err
	}
	return &Scheduler{b: b, chunk: o.Chunk, reserve: o.Reserve,
		slots: make([]schedState, n)}, nil
}

// Slots is how many sequences the scheduler holds.
func (s *Scheduler) Slots() int { return s.b.Slots() }

// ErrNoSlot is what [Scheduler.Admit] refuses with when every slot is live.
var ErrNoSlot = errors.New("tgo: every slot is occupied")

// Admit places a prompt in a free slot and returns which one.
//
// It refuses rather than queueing, and refuses for two different reasons that a
// caller must be able to tell apart: [ErrNoSlot] means wait, and
// [prefix.ErrExhausted] wrapped means this request is too large for the pool as
// it currently stands and evicting something would be the way to make room.
// A server that reported one number for both would be indistinguishable from a
// slow one, which §3 says is the thing to avoid.
func (s *Scheduler) Admit(prompt []int, salt string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	free := -1
	for i := range s.slots {
		if !s.slots[i].live {
			free = i
			break
		}
	}
	if free < 0 {
		return -1, ErrNoSlot
	}
	if _, err := s.b.Admit(free, prompt, salt, s.reserve); err != nil {
		return -1, err
	}
	// The reuse is positions the pool already holds, so they are prefilled
	// already and this sequence starts partway through its own prompt.
	reused := s.b.Length(free)
	s.next++
	s.slots[free] = schedState{
		// Copied, and this is not defensive style.
		//
		// [Scheduler.Feed] appends the generated token to this slice, and an
		// append onto a caller's slice with spare capacity writes into the
		// caller's array -- which for `all[:20]` out of a 40-element prompt
		// buffer is `all[20]`, the first token of the request the caller is
		// about to submit next. That is one request silently rewriting
		// another's prompt, and both answers stay fluent.
		live: true, prompt: append([]int(nil), prompt...),
		prefilled: reused, arrived: s.next,
	}
	return free, nil
}

// Finish frees a slot and gives its blocks back.
func (s *Scheduler) Finish(slot int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if slot < 0 || slot >= len(s.slots) {
		return fmt.Errorf("tgo: slot %d is outside a batch of %d", slot, len(s.slots))
	}
	if err := s.b.Evict(slot); err != nil {
		return err
	}
	s.slots[slot] = schedState{}
	return nil
}

// Evict preempts the slot 008-D5 chooses and returns which one, or -1 when
// nothing is live.
//
// The sequence drops its blocks and re-prefills on readmission (008-D2), so
// what a caller does with the answer is resubmit the prompt: recompute wins
// over swapping the cache to the host because a prefill is compute-bound and
// parallel over its tokens, while a swap is two serial transfers over a bus and
// needs a host mirror of every swapped sequence.
func (s *Scheduler) Evict() (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v := victim(s.slots)
	if v < 0 {
		return -1, nil
	}
	if err := s.b.Evict(v); err != nil {
		return -1, err
	}
	s.slots[v] = schedState{}
	return v, nil
}

// Produced is what one slot got out of a step.
type Produced struct {
	// Slot is which slot this is.
	Slot int

	// Prefill says the slot contributed prompt tokens rather than one
	// generated one. A slot still mid-prefill after this step has no token to
	// sample yet, which [Produced.Sampleable] reports.
	Prefill bool

	// Done says the slot's prompt is now fully scored, so Logits is the
	// distribution over its next token.
	Done bool

	// Logits is the slot's output. It is the slot's own buffer and is valid
	// until that slot steps again.
	Logits []float32
}

// Sampleable reports whether Logits is a distribution worth sampling: the
// prompt is fully scored, so the row this came from is the last one of it.
//
// A chunk that is not the last chunk produces logits for a token in the middle
// of the prompt, which is a real distribution over a token the caller already
// has. Reading it is the mistake this method exists to prevent.
func (p Produced) Sampleable() bool { return p.Done }

// StepResult is one step's outcome.
type StepResult struct {
	// Produced is one entry per slot that contributed, in slot order.
	Produced []Produced

	// PrefillTokens and Decodes are what the step carried: the mix §5 is
	// about, reported rather than inferred.
	PrefillTokens, Decodes int
}

// Step runs one batched forward pass over whatever the slots have to do.
//
// A prefill chunk and the decodes beside it are one dispatch, so the weights
// are read once for both -- which is what makes chunking recover throughput
// rather than only bound latency (§5).
func (s *Scheduler) Step() (StepResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, err := nextStep(s.slots, s.chunk, s.b.rows)
	if err != nil {
		return StepResult{}, err
	}
	if len(p.Work) == 0 {
		return StepResult{}, nil
	}
	out, err := s.b.Step(p.Work)
	if err != nil {
		return StepResult{}, err
	}
	res := StepResult{PrefillTokens: p.Prefill, Decodes: p.Decode}
	for i, w := range p.Work {
		st := &s.slots[w.Slot]
		prefill := st.prefilled < len(st.prompt)
		if prefill {
			st.prefilled += len(w.Tokens)
		}
		res.Produced = append(res.Produced, Produced{
			Slot: w.Slot, Prefill: prefill,
			Done:   st.prefilled >= len(st.prompt),
			Logits: out[i],
		})
	}
	return res, nil
}

// Feed sets the token a slot contributes next.
//
// It is separate from [Scheduler.Step] because sampling is the caller's: the
// scheduler decides what runs and the caller decides what the token is.
func (s *Scheduler) Feed(slot, token int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if slot < 0 || slot >= len(s.slots) {
		return fmt.Errorf("tgo: slot %d is outside a batch of %d", slot, len(s.slots))
	}
	st := &s.slots[slot]
	if !st.live {
		return fmt.Errorf("tgo: slot %d holds no sequence", slot)
	}
	if st.prefilled < len(st.prompt) {
		return fmt.Errorf("tgo: slot %d is %d tokens into a %d-token prompt; a token "+
			"fed before the prompt is scored would be generated from a distribution "+
			"over the middle of it", slot, st.prefilled, len(st.prompt))
	}
	if token < 0 || token >= s.b.m.cfg.VocabSize {
		return fmt.Errorf("tgo: token id %d is outside the model's vocabulary of %d",
			token, s.b.m.cfg.VocabSize)
	}
	st.feed = token
	// A fed token is one more position of the sequence, and the prompt is what
	// nextStep measures a prefill against -- so the token joins it and the slot
	// stays fully prefilled rather than becoming mid-prefill again.
	st.prompt = append(st.prompt, token)
	st.prefilled++
	return nil
}

// Close releases the scheduler's batch.
func (s *Scheduler) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Close()
}
