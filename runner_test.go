// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/latere-ai/tgo/bench"
)

// newRunner opens a runner over the shared block pool and closes it with the
// test.
func newRunner(t *testing.T, m *Model, o RunnerOptions) *Runner {
	t.Helper()
	r, err := m.NewRunner(o)
	if err != nil {
		t.Fatalf("NewRunner(%+v): %v", o, err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Errorf("Runner.Close: %v", err)
		}
	})
	return r
}

// drain reads a slot stream to its end and returns the text and the events.
func drainSlot(t *testing.T, st *SlotStream) (string, []Event) {
	t.Helper()
	var text string
	var evs []Event
	for st.Next() {
		e := st.Event()
		evs = append(evs, e)
		text += e.Text
	}
	if err := st.Err(); err != nil {
		t.Fatalf("SlotStream.Err: %v", err)
	}
	return text, evs
}

// TestTwoRequestsShareOneForwardPass is 022's gate, and it is read from the
// recorder rather than from a clock (010-D7): two requests decoding together
// produce their tokens from **one** step per pair, so the step count is half
// the token count and the batch width the step reports is 2.
//
// A clock would measure the machine. The batch width is the claim.
func TestTwoRequestsShareOneForwardPass(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	rec := bench.NewRecorder(256)
	r := newRunner(t, m, RunnerOptions{Slots: 2, Chunk: 8, Reserve: CacheBlock,
		Recorder: rec})

	// Sixteen and not four: what is under test is that two requests overlap,
	// and on a device where a completion is milliseconds a short one can finish
	// before the other is admitted. The overlap has to be the likely outcome of
	// the fixture rather than of the machine.
	const tokens = 16
	var wg sync.WaitGroup
	for i := range 2 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			st, err := r.Complete(context.Background(), RunRequest{},
				strings.Repeat("a ", 4+i), greedy(tokens))
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				return
			}
			for st.Next() {
			}
			if err := st.Err(); err != nil {
				t.Errorf("request %d: %v", i, err)
			}
			if got := st.Usage().CompletionTokens; got != tokens {
				t.Errorf("request %d produced %d tokens, want %d", i, got, tokens)
			}
		}(i)
	}
	wg.Wait()

	rep := rec.Report()
	var decodes, shared int
	for _, s := range rec.Steps() {
		if s.Phase != bench.Decode {
			continue
		}
		decodes++
		if s.Batch >= 2 {
			shared++
		}
	}
	if rep.Dropped != 0 {
		t.Fatalf("the recorder dropped %d steps; the fixture must fit", rep.Dropped)
	}
	if shared == 0 {
		t.Fatalf("no decode step carried two sequences; %d decode steps, and a "+
			"batch of one is what a session per request already gives", decodes)
	}
	// Two requests of `tokens` each is 2*tokens tokens. If every decode step
	// carried both, that is `tokens` steps; admission and the prefills stagger
	// them, so the bound is what matters: strictly fewer steps than tokens.
	if decodes >= 2*tokens {
		t.Errorf("%d decode steps for %d tokens: the sequences did not share a pass",
			decodes, 2*tokens)
	}
}

// TestRunnerAgreesWithASingleSession is the ordering the extraction had to get
// right: a session appends the drawn token to its history on the *next* step,
// and a runner appends it through Scheduler.Feed after sampling. The two orders
// must give the same tokens, and a disagreement here reads as a sampling bug
// rather than as the ordering one it would be.
func TestRunnerAgreesWithASingleSession(t *testing.T) {
	t.Parallel()
	const prompt = "the quick brown fox"
	p := Policy{MaxTokens: 5, Temperature: 0.8, TopK: 20, Seed: 7,
		RepetitionPenalty: 1.1}

	s := session(t, openSynthetic(t))
	sst, err := s.Complete(context.Background(), prompt, p)
	if err != nil {
		t.Fatalf("Session.Complete: %v", err)
	}
	want, _ := collect(t, sst)

	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})
	rst, err := r.Complete(context.Background(), RunRequest{}, prompt, p)
	if err != nil {
		t.Fatalf("Runner.Complete: %v", err)
	}
	got, _ := drainSlot(t, rst)

	if got != want {
		t.Errorf("the batched path produced %q and one session produced %q", got, want)
	}
	if a, b := rst.Usage(), sst.Usage(); a.CompletionTokens != b.CompletionTokens ||
		a.PromptTokens != b.PromptTokens {
		t.Errorf("usage = %+v, want %+v", a, b)
	}
	if a, b := rst.StopReason(), sst.StopReason(); a != b {
		t.Errorf("stop reason = %v, want %v", a, b)
	}
}

// TestEachRequestKeepsItsOwnSeed is 022-D4: reproducibility is a stream
// property, so a request's draws must not depend on which other requests were
// batched with it.
func TestEachRequestKeepsItsOwnSeed(t *testing.T) {
	t.Parallel()
	const prompt = "a shared opening"
	hot := Policy{MaxTokens: 4, Temperature: 0.9, Seed: 1}
	other := hot
	other.Seed = 2

	// Alone, on its own runner.
	alone := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})
	st, err := alone.Complete(context.Background(), RunRequest{}, prompt, hot)
	if err != nil {
		t.Fatalf("alone: %v", err)
	}
	want, _ := drainSlot(t, st)

	// Beside a request with a different seed.
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})
	var wg sync.WaitGroup
	var got, neighbour string
	for i, pol := range []Policy{hot, other} {
		wg.Add(1)
		go func(i int, pol Policy) {
			defer wg.Done()
			st, err := r.Complete(context.Background(), RunRequest{}, prompt, pol)
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				return
			}
			text, _ := drainSlot(t, st)
			if i == 0 {
				got = text
			} else {
				neighbour = text
			}
		}(i, pol)
	}
	wg.Wait()

	if got != want {
		t.Errorf("seeded 1 in a batch produced %q and alone produced %q", got, want)
	}
	if neighbour == want {
		t.Errorf("two seeds produced the same completion %q, so the fixture cannot "+
			"tell a shared generator from a per-request one", want)
	}
}

// TestASlowConsumerDoesNotStallTheBatch is §4's bound. Two things must hold and
// one of them alone would let the other fail silently: a request that stops
// reading is dropped rather than allowed to accumulate an unbounded queue on
// the driver, and the sequences beside it keep producing while it does.
func TestASlowConsumerDoesNotStallTheBatch(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	r := newRunner(t, m, RunnerOptions{Slots: 2, Chunk: 8, Reserve: CacheBlock,
		Backlog: 4})

	// A request nobody reads, with a budget many times the backlog.
	slow, err := r.Complete(context.Background(), RunRequest{}, "the slow one",
		greedy(64))
	if err != nil {
		t.Fatalf("the slow request: %v", err)
	}
	// One beside it that is read to the end.
	done := make(chan string, 1)
	go func() {
		st, err := r.Complete(context.Background(), RunRequest{}, "the fast one",
			greedy(4))
		if err != nil {
			t.Errorf("the fast request: %v", err)
			done <- ""
			return
		}
		text, _ := drainSlot(t, st)
		done <- text
	}()
	select {
	case text := <-done:
		if text == "" {
			t.Fatal("the fast request produced nothing")
		}
	case <-time.After(30 * time.Second):
		t.Fatal("the fast request did not finish while a slow one held a slot")
	}

	// And the slow one is told why it ended, rather than being truncated in
	// silence: a client that fell behind must be able to tell that from a
	// completion the model chose to end.
	// Polled on the run rather than on the stream, because reading the stream
	// is what stops it being a slow consumer: a test that drained while it
	// waited would be waiting for a condition it was busy preventing.
	waitFor(t, "the slow consumer to be dropped", func() bool {
		return slow.run.gone.Load()
	})
	for slow.Next() {
	}
	if err := slow.Err(); !errors.Is(err, ErrSlowConsumer) {
		t.Errorf("the slow request ended with %v, want ErrSlowConsumer", err)
	}
	if got := slow.Usage().CompletionTokens; got >= 64 {
		t.Errorf("the dropped request produced %d of its 64 tokens, so it was not "+
			"dropped at all", got)
	}
}

// TestRunnerReleasesASlotWhenTheCallerLeaves is 022-D5 at the step boundary: a
// caller that stops reading gives its slot and its blocks back, so the next
// request is admitted rather than waiting for a budget to elapse.
//
// The waiting half is asserted as a **positive observation** -- the third
// request is seen in the queue -- and not as a deadline that elapsed. A
// deadline says only that nothing happened in the time allowed, which on a fast
// machine is also what a batch that emptied under the test looks like: the
// first draft of this test used one and passed on the CPU backend and failed on
// Metal, where the two holders finished before the third request arrived.
func TestRunnerReleasesASlotWhenTheCallerLeaves(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	r := newRunner(t, m, RunnerOptions{Slots: 2, Chunk: 8, Reserve: CacheBlock})

	// Both slots, with completions long enough that neither can end while the
	// third request is being admitted -- 96 steps against the microseconds a
	// goroutine takes to start -- and which nobody reads to the end. The test
	// finishes long before they do, and closing the runner ends them.
	held := make([]*SlotStream, 2)
	for i := range held {
		st, err := r.Complete(context.Background(), RunRequest{},
			strings.Repeat("x ", 3+i), greedy(96))
		if err != nil {
			t.Fatalf("holding request %d: %v", i, err)
		}
		if !st.Next() {
			t.Fatalf("holding request %d produced nothing: %v", i, st.Err())
		}
		held[i] = st
	}

	third := admit2(r, "the third")
	waitFor(t, "the third request to be queued behind a full batch", func() bool {
		return r.Queue().Depth() == 1
	})
	if !third.pending() {
		t.Fatal("the third request was admitted while both slots were live")
	}

	if err := held[0].Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if text := third.text(t); text == "" {
		t.Error("the request admitted into the freed slot produced nothing")
	}
}

// admit2 runs one completion in the background, so a test can watch it wait.
type backgroundRun struct {
	out chan string
	err chan error
}

func admit2(r *Runner, prompt string) *backgroundRun {
	b := &backgroundRun{out: make(chan string, 1), err: make(chan error, 1)}
	go func() {
		st, err := r.Complete(context.Background(), RunRequest{}, prompt, greedy(2))
		if err != nil {
			b.out <- ""
			b.err <- err
			return
		}
		var text string
		for st.Next() {
			text += st.Event().Text
		}
		b.out <- text
		b.err <- st.Err()
	}()
	return b
}

func (b *backgroundRun) pending() bool {
	select {
	case err := <-b.err:
		b.err <- err
		return false
	default:
		return true
	}
}

func (b *backgroundRun) text(t *testing.T) string {
	t.Helper()
	select {
	case err := <-b.err:
		if err != nil {
			t.Fatalf("the background request: %v", err)
		}
		return <-b.out
	case <-time.After(30 * time.Second):
		t.Fatal("the background request did not finish")
		return ""
	}
}

// TestRunnerRefusesAModelWithNoBlockPool is 022-D1: the batched path and the
// process scope are one configuration, so there is nothing to build under the
// other two.
func TestRunnerRefusesAModelWithNoBlockPool(t *testing.T) {
	t.Parallel()
	_, err := openSynthetic(t).NewRunner(RunnerOptions{Slots: 2})
	if !errors.Is(err, ErrNoBlockPool) {
		t.Fatalf("NewRunner without a shared pool = %v, want ErrNoBlockPool", err)
	}
}

// TestRunnerRefusesWhatWillNotRun is the set of refusals that must happen
// before a slot is taken, so a request that cannot run leaves the batch as it
// found it.
func TestRunnerRefusesWhatWillNotRun(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	r := newRunner(t, m, RunnerOptions{Slots: 2, Chunk: 8, Reserve: CacheBlock})
	ctx := context.Background()

	for _, c := range []struct {
		name string
		run  func() error
	}{
		{"an empty prompt", func() error {
			_, err := r.Complete(ctx, RunRequest{}, "", greedy(2))
			return err
		}},
		{"a nil context", func() error {
			_, err := r.Complete(nil, RunRequest{}, "hi", greedy(2)) //nolint:staticcheck
			return err
		}},
		{"a policy the model refuses", func() error {
			_, err := r.Complete(ctx, RunRequest{}, "hi", Policy{Temperature: -1})
			return err
		}},
		{"a schema the compiler refuses", func() error {
			_, err := r.Complete(ctx, RunRequest{}, "hi",
				Policy{MaxTokens: 2, Schema: []byte(`{"type":"nonsense"}`)})
			return err
		}},
		{"a budget past the context", func() error {
			_, err := r.Complete(ctx, RunRequest{}, "hi",
				Policy{MaxTokens: m.Info().Context + 1})
			return err
		}},
	} {
		if err := c.run(); err == nil {
			t.Errorf("%s was accepted", c.name)
		}
	}
	if d := r.Queue().Depth(); d != 0 {
		t.Errorf("Queue.Depth = %d after five refusals, want 0: none of them may "+
			"take a slot or a place in the queue", d)
	}
}

// TestRunnerRecordsTheFourTermsPerStep is 017 §1's model on the batched path:
// the four are exhaustive, so a batched step must not report a device cost as
// host time.
func TestRunnerRecordsTheFourTermsPerStep(t *testing.T) {
	t.Parallel()
	rec := bench.NewRecorder(64)
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock, Recorder: rec})

	st, err := r.Complete(context.Background(), RunRequest{}, "measure me", greedy(3))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	drainSlot(t, st)

	steps := rec.Steps()
	if len(steps) == 0 {
		t.Fatal("the runner recorded no steps")
	}
	for i, s := range steps {
		if s.Device == 0 {
			t.Errorf("step %d recorded no device time; a batched step's fence wait "+
				"cannot be zero", i)
		}
		// The claim, stated as a comparison rather than as a floor on each
		// term: a batched loop must not report a device cost as host time. A
		// floor on the readback is a claim about the clock -- Windows resolves
		// a short interval to zero -- and this is a claim about the
		// attribution.
		if s.Host >= s.Total() {
			t.Errorf("step %d puts all %s of the step on the host; the three device "+
				"terms were not measured", i, s.Total())
		}
		if s.Batch < 1 {
			t.Errorf("step %d reports a batch of %d", i, s.Batch)
		}
	}
}

// TestRunnerQueuesPastItsSlots is what 021 bought: a request arriving at a full
// batch waits rather than being refused.
func TestRunnerQueuesPastItsSlots(t *testing.T) {
	t.Parallel()
	// A budget far past the default, because what is under test is that a
	// request waits and is admitted, and a fixture whose forward passes are the
	// CPU oracle's would otherwise measure the machine.
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock, Queue: QueueOptions{Wait: 5 * time.Minute}})

	const n = 4
	var wg sync.WaitGroup
	texts := make([]string, n)
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			st, err := r.Complete(context.Background(), RunRequest{},
				strings.Repeat("q ", 2+i), greedy(2))
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				return
			}
			texts[i], _ = drainSlot(t, st)
		}(i)
	}
	wg.Wait()
	for i, text := range texts {
		if text == "" {
			t.Errorf("request %d produced nothing", i)
		}
	}
	if d := r.Queue().Depth(); d != 0 {
		t.Errorf("Queue.Depth = %d once every request finished, want 0", d)
	}
}

// TestPerSlotGrammarMask is what "per-slot masking" means: two constrained
// requests with **different** schemas in one step, each answering its own.
//
// Two different schemas and not one constrained request beside a free one,
// because a shared grammar state would still let a single mask look correct.
// The mask is a per-request, per-position write over the vocabulary derived
// from that request's position in its own grammar (022 §5), so a state that
// leaked would put one request's admissible set on the other's row and the
// document it produced would not parse.
//
// The completions are not compared against the same requests run alone. A
// reused prefix was computed under a different prefill shape and floating point
// is not associative, so a warm answer matches a cold one in distribution
// rather than bit for bit (016-D6) -- which 022 §9 names as the one real cost
// of batching by default.
func TestPerSlotGrammarMask(t *testing.T) {
	t.Parallel()
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})

	kinds := []struct {
		name   string
		schema string
		ok     func(string) bool
	}{
		{"boolean", `{"type":"boolean"}`, func(s string) bool {
			return s == "true" || s == "false"
		}},
		{"integer", `{"type":"integer"}`, func(s string) bool {
			if s == "" {
				return false
			}
			for i, c := range s {
				if c == '-' && i == 0 {
					continue
				}
				if c < '0' || c > '9' {
					return false
				}
			}
			return true
		}},
	}
	out := make([]string, len(kinds))
	var wg sync.WaitGroup
	for i, k := range kinds {
		wg.Add(1)
		go func(i int, schema string) {
			defer wg.Done()
			st, err := r.Complete(context.Background(), RunRequest{}, "answer",
				Policy{MaxTokens: 12, Seed: 3, Schema: []byte(schema)})
			if err != nil {
				t.Errorf("the %s request: %v", kinds[i].name, err)
				return
			}
			out[i], _ = drainSlot(t, st)
		}(i, k.schema)
	}
	wg.Wait()

	for i, k := range kinds {
		if got := strings.TrimSpace(out[i]); !k.ok(got) {
			t.Errorf("the %s request produced %q, which its own schema does not "+
				"admit; the masks did not stay on their own slots", k.name, got)
		}
	}
}

// TestPenaltiesReadOnlyTheirOwnSlot: two requests alike in everything but their
// repetition penalty, in one step. The penalties read that slot's own history
// (022 §5), so a penalty state shared across the batch would make the two
// agree.
func TestPenaltiesReadOnlyTheirOwnSlot(t *testing.T) {
	t.Parallel()
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})

	base := Policy{MaxTokens: 6, Temperature: 0.9, Seed: 11}
	penalised := base
	penalised.RepetitionPenalty = 1.8

	out := make([]string, 2)
	var wg sync.WaitGroup
	for i, pol := range []Policy{base, penalised} {
		wg.Add(1)
		go func(i int, pol Policy) {
			defer wg.Done()
			st, err := r.Complete(context.Background(), RunRequest{}, "repeat", pol)
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				return
			}
			out[i], _ = drainSlot(t, st)
		}(i, pol)
	}
	wg.Wait()

	if out[0] == "" || out[1] == "" {
		t.Fatalf("a request produced nothing: %q and %q", out[0], out[1])
	}
	if out[0] == out[1] {
		t.Errorf("the penalised and the unpenalised request both produced %q from "+
			"the same seed, so the penalty did not reach one slot alone", out[0])
	}
}

// longPrompt is more than one CacheBlock of tokens, because reuse is
// block-aligned: a prompt that fits in one block has no whole block to find
// again (016-D4).
func longPrompt() string { return strings.Repeat("shared opening words ", 12) }

// TestUnsaltedRequestsDoNotShareBlocks is 022 §7, and it is the half that is a
// correctness claim rather than a performance one.
//
// [016-D7](../specs/016-prefix-cache.md) and 019-D3 both say an unkeyed request
// shares with nobody rather than with everybody. Under a pooled session that is
// true by construction, because routing compares the key. Under a shared block
// pool the seed's domain is empty, so without a minted salt every unsalted
// request hashes into one domain — and the second one's first token arrives
// fast, which is a membership test over the first one's prompt.
func TestUnsaltedRequestsDoNotShareBlocks(t *testing.T) {
	t.Parallel()
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})

	reused := make([]int, 2)
	for i := range reused {
		st, err := r.Complete(context.Background(), RunRequest{}, longPrompt(),
			greedy(2))
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		drainSlot(t, st)
		reused[i] = st.Usage().CachedPromptTokens
	}
	if reused[1] != 0 {
		t.Errorf("the second unsalted request reused %d positions of the first "+
			"one's prompt; an unkeyed request shares with nobody", reused[1])
	}
}

// TestSaltedRequestsShareBlocks is the other half, and without it the row above
// passes by disabling the cache.
func TestSaltedRequestsShareBlocks(t *testing.T) {
	t.Parallel()
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})

	req := RunRequest{Key: "tenant-a"}
	reused := make([]int, 2)
	for i := range reused {
		st, err := r.Complete(context.Background(), req, longPrompt(), greedy(2))
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		drainSlot(t, st)
		reused[i] = st.Usage().CachedPromptTokens
	}
	if reused[0] != 0 {
		t.Errorf("the first request reused %d positions and there was nothing to "+
			"reuse", reused[0])
	}
	if reused[1] == 0 {
		t.Error("the second request with the same cache_salt and the same prompt " +
			"reused nothing, which is what the field is for")
	}
}

// TestADifferentSaltIsADifferentDomain is the isolation the minted salt buys,
// stated over two callers who both named one: the same prompt under two
// cache_salts shares nothing.
func TestADifferentSaltIsADifferentDomain(t *testing.T) {
	t.Parallel()
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})

	reused := make([]int, 2)
	for i, key := range []string{"tenant-a", "tenant-b"} {
		st, err := r.Complete(context.Background(), RunRequest{Key: key},
			longPrompt(), greedy(2))
		if err != nil {
			t.Fatalf("%s: %v", key, err)
		}
		drainSlot(t, st)
		reused[i] = st.Usage().CachedPromptTokens
	}
	if reused[1] != 0 {
		t.Errorf("tenant-b reused %d positions of tenant-a's prompt", reused[1])
	}
}

// TestAMintedSaltCannotBeNamed is why the minted salt carries sixteen random
// bytes rather than a counter: a caller's salt is used verbatim, so a
// predictable namespace is one a request can name and share into.
func TestAMintedSaltCannotBeNamed(t *testing.T) {
	t.Parallel()
	r := newRunner(t, batchModel(t), RunnerOptions{Slots: 2, Chunk: 8,
		Reserve: CacheBlock})

	first, second := r.salt(""), r.salt("")
	if first == second {
		t.Fatalf("two minted salts are both %q", first)
	}
	for _, guess := range []string{"1", "0", "-1", "tgo-1", first[:8]} {
		if r.salt(guess) == first || r.salt(guess) == second {
			t.Errorf("a caller naming %q reaches a minted domain", guess)
		}
	}
	if got := r.salt("tenant-a"); got != "tenant-a" {
		t.Errorf("a named salt became %q; it is used verbatim", got)
	}
}
