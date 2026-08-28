// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"errors"
	"fmt"
	"sync"

	"github.com/latere-ai/tgo/internal/prefix"
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

	// capacity carries one signal per event that could make a refused
	// admission succeed: a slot freed, blocks returned, or a step taken. It
	// has capacity 1 and is sent to without blocking, so a [Queue] waits on an
	// event rather than on a timer and a burst of releases collapses into one
	// re-evaluation.
	capacity chan struct{}

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
	//
	// It is the **default**, used for a request that names no reserve of its
	// own. specs/022-batched-serving.md 022-D7: one R for the whole deployment
	// is either larger than most requests need, which admits fewer than the
	// device holds, or smaller than some request needs, which is the admission
	// the promise exists to prevent — and the request already carries the
	// number, in Policy.MaxTokens.
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
		capacity: make(chan struct{}, 1), slots: make([]schedState, n)}, nil
}

// Slots is how many sequences the scheduler holds.
func (s *Scheduler) Slots() int { return s.b.Slots() }

// Capacity fires when something happened that could make a refused admission
// succeed. It is the signal a [Queue] waits on.
//
// specs/021-admission-queue.md §2. The channel holds one signal, so a reader
// that wakes once after ten releases has lost nothing: the answer to "can this
// prompt be admitted now" is recomputed from the slots and the pool rather than
// counted from the signals. A send that would block is dropped for the same
// reason, which is what keeps [Scheduler.Finish] and [Scheduler.Step] from
// waiting on whoever is listening.
func (s *Scheduler) Capacity() <-chan struct{} { return s.capacity }

// signal offers one capacity event and drops it if one is already pending.
func (s *Scheduler) signal() {
	select {
	case s.capacity <- struct{}{}:
	default:
	}
}

// Feasible reports whether a prompt of this length could ever be admitted:
// whether the pool is large enough for it and its reserve, not whether the
// blocks are free right now.
//
// specs/021-admission-queue.md §3 and 021-D3. [prefix.ErrExhausted] means two
// things -- "the pool is too small for this prompt", which never resolves, and
// "live sequences hold the blocks", which resolves when one finishes -- and
// [Scheduler.Admit] returns whichever it got. A queue that classified by
// inspecting the error would enqueue a request that can never be admitted, and
// its head-of-line bound would stop being finite. So the first case is
// arithmetic at the door instead, over the pool's own block size and count, so
// that this check and the pool's cannot drift apart.
func (s *Scheduler) Feasible(prompt, reserve int) error {
	if prompt <= 0 {
		return errors.New("tgo: the prompt is empty; there is nothing to condition on")
	}
	reserve, err := s.reserveFor(reserve)
	if err != nil {
		return err
	}
	p := s.b.m.blocks.pool
	block, blocks := p.Block(), p.Blocks()
	need := (prompt + reserve + block - 1) / block
	if need > blocks {
		return fmt.Errorf("tgo: a %d-token prompt with a reserve of %d needs %d "+
			"blocks of %d positions and the pool holds %d: %w", prompt, reserve,
			need, block, blocks, prefix.ErrExhausted)
	}
	return nil
}

// reserveFor resolves a request's reserve against the deployment's default.
//
// Zero takes the default, which is what a caller with no budget of its own
// passes; a negative one is refused rather than clamped, because a reserve is
// positions and a caller that computed a negative number computed something
// else (022-D7).
func (s *Scheduler) reserveFor(reserve int) (int, error) {
	if reserve == 0 {
		return s.reserve, nil
	}
	if reserve < 0 {
		return 0, fmt.Errorf("tgo: the reserve is %d; it is how many positions "+
			"beyond its prompt an admitted sequence may grow by", reserve)
	}
	return reserve, nil
}

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
// reserve is how many positions beyond its prompt this sequence may grow by,
// which is the request's own budget. Zero takes [SchedulerOptions.Reserve].
func (s *Scheduler) Admit(prompt []int, salt string, reserve int) (int, error) {
	reserve, err := s.reserveFor(reserve)
	if err != nil {
		return -1, err
	}
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
	if _, err := s.b.Admit(free, prompt, salt, reserve); err != nil {
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
		prefilled: reused, reused: reused, arrived: s.next,
	}
	return free, nil
}

// Reused is how many leading positions of a slot's prompt the shared pool
// already held when it was admitted, which is the prefill this request did not
// pay for. It is [Usage.CachedPromptTokens] for a caller holding a slot.
func (s *Scheduler) Reused(slot int) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if slot < 0 || slot >= len(s.slots) || !s.slots[slot].live {
		return 0
	}
	return s.slots[slot].reused
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
	s.signal()
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
	s.signal()
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
	res, _, err := s.step()
	return res, err
}

// step is [Scheduler.Step] with the three device terms measured, for a caller
// that instruments the batched loop. specs/017-benchmarks.md §1 treats the four
// terms as exhaustive, and a wall clock recorded under one of their names would
// report a device cost as host time.
func (s *Scheduler) step() (StepResult, timings, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var t timings
	p, err := nextStep(s.slots, s.chunk, s.b.rows)
	if err != nil {
		return StepResult{}, t, err
	}
	if len(p.Work) == 0 {
		return StepResult{}, t, nil
	}
	out, t, err := s.b.step(p.Work)
	if err != nil {
		return StepResult{}, t, err
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
	// A step frees nothing by itself. It signals anyway, because a step is the
	// only event that happens while the batch is busy, and a [Queue] whose
	// pass raced with a release would otherwise sleep until the next one. The
	// cost of a signal nobody needed is one re-evaluation over the waiter list.
	s.signal()
	return res, t, nil
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
