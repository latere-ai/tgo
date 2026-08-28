// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/server"
)

// specs/022-batched-serving.md §11, over the real handler and a real model.
//
// The fake engine cannot answer any of these: what is under test is that the
// engine behind [server.Engine] can put several requests in one forward pass
// and that nothing in front of it noticed.

// openShared opens the synthetic model with the shared block pool a batch needs
// (022-D1).
func openShared(t *testing.T) *tgo.Model {
	t.Helper()
	m, err := tgo.Open(writeCheckpoint(t), tgo.WithDevice(tgo.CPU),
		tgo.WithContext(96), tgo.WithPrefixCache(tgo.CacheProcess, 4*96))
	if err != nil {
		t.Fatalf("tgo.Open: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("Model.Close: %v", err)
		}
	})
	return m
}

// batchedEngine builds a batched engine of n slots and closes it with the test.
// A zero wait takes the queue's default.
func batchedEngine(t *testing.T, m *tgo.Model, n int, rec *bench.Recorder,
	wait time.Duration) *server.RunnerEngine {

	t.Helper()
	e, err := server.WrapRunner(m, synthName, tgo.RunnerOptions{
		Slots: n, Chunk: 16, Reserve: tgo.CacheBlock, Recorder: rec,
		Queue: tgo.QueueOptions{Wait: wait},
	})
	if err != nil {
		t.Fatalf("server.WrapRunner: %v", err)
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Errorf("RunnerEngine.Close: %v", err)
		}
	})
	return e
}

// batchedServer builds a server over a batched engine of n slots.
func batchedServer(t *testing.T, m *tgo.Model, n int, rec *bench.Recorder) *server.Server {
	t.Helper()
	s, err := server.New(batchedEngine(t, m, n, rec, 0),
		server.WithNotice(&strings.Builder{}), server.WithConcurrency(n))
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s
}

// TestTheFourDialectsAnswerThroughTheBatchedEngine is §10's preservation list
// where it is cheapest to check: the four routes and their streaming forms are
// what they were, over an engine that batches behind them.
func TestTheFourDialectsAnswerThroughTheBatchedEngine(t *testing.T) {
	s := batchedServer(t, openShared(t), 2, nil)

	for _, c := range []struct{ name, path, body string }{
		{"chat", "/v1/chat/completions", `{"model":"` + synthName + `","max_tokens":2,` +
			`"messages":[{"role":"user","content":"hi"}]}`},
		{"messages", "/v1/messages", `{"model":"` + synthName + `","max_tokens":2,` +
			`"messages":[{"role":"user","content":"hi"}]}`},
		{"responses", "/v1/responses", `{"model":"` + synthName + `","max_output_tokens":2,` +
			`"input":"hi"}`},
		{"completions", "/v1/completions", `{"model":"` + synthName + `","max_tokens":2,` +
			`"prompt":"hi"}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			w := do(t, s, http.MethodPost, c.path, c.body)
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", w.Code, w.Body.String())
			}
			var body map[string]any
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
				t.Fatalf("the answer is not JSON: %v: %s", err, w.Body.String())
			}
		})
		t.Run(c.name+" streaming", func(t *testing.T) {
			body := strings.TrimSuffix(c.body, "}") + `,"stream":true}`
			w := do(t, s, http.MethodPost, c.path, body)
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", w.Code, w.Body.String())
			}
			if !strings.Contains(w.Body.String(), "data: ") {
				t.Fatalf("the stream carried no frames: %q", w.Body.String())
			}
		})
	}
}

// TestTwoRequestsShareOneForwardPass is 022's gate over the real handler.
//
// Read from the recorder rather than from a clock (010-D7). A clock measures
// the machine; the batch width is the claim, and it is the field a session per
// request sets to 1 because that is what it is.
func TestTwoRequestsShareOneForwardPass(t *testing.T) {
	rec := bench.NewRecorder(512)
	s := batchedServer(t, openShared(t), 2, rec)

	// Sixteen tokens and not four: the two requests have to overlap, and on a
	// device where a completion is milliseconds a short one can finish before
	// the other is admitted.
	const body = `{"model":"` + synthName + `","max_tokens":16,"prompt":"hi"}`
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := do(t, s, http.MethodPost, "/v1/completions", body)
			if w.Code != http.StatusOK {
				t.Errorf("status = %d: %s", w.Code, w.Body.String())
			}
		}()
	}
	wg.Wait()

	var shared, total int
	for _, st := range rec.Steps() {
		total++
		if st.Batch >= 2 {
			shared++
		}
	}
	if total == 0 {
		t.Fatal("the runner recorded no steps")
	}
	if shared == 0 {
		t.Errorf("none of %d steps carried two sequences, so the two requests each "+
			"read the weights for themselves", total)
	}
}

// TestBatchedEngineRefusesWithoutABlockPool is 022-D1: the batched path and the
// process scope are one configuration, and the refusal names the option rather
// than an allocator's error.
func TestBatchedEngineRefusesWithoutABlockPool(t *testing.T) {
	_, err := server.WrapRunner(openSynthetic(t), synthName, tgo.RunnerOptions{Slots: 2})
	if !errors.Is(err, tgo.ErrNoBlockPool) {
		t.Fatalf("WrapRunner over a model with no shared pool = %v, want "+
			"tgo.ErrNoBlockPool", err)
	}
}

// TestAdmissionAboveTheSlotsIsQueuedNotRefused is [019 §8.6]'s refusal turned
// off for the engine that no longer needs it, and still on for the one that
// does.
//
// The refusal exists because a pooled engine makes the surplus above its pool
// wait inside NewSession, where this package neither counts it nor times it out
// -- so the Retry-After stops describing what that request waits. A batched
// engine answers for its own wait: it bounds the depth, it bounds the budget,
// and it reports both, so the surplus is a wait the deployment can see.
func TestAdmissionAboveTheSlotsIsQueuedNotRefused(t *testing.T) {
	e := batchedEngine(t, openShared(t), 2, nil, 0)
	for _, n := range []int{2, 3, 16} {
		if _, err := server.New(e, server.WithNotice(&strings.Builder{}),
			server.WithConcurrency(n)); err != nil {
			t.Errorf("a concurrency of %d over a batch of 2 slots was refused: %v",
				n, err)
		}
	}

	// And the pooled engine, whose wait this package cannot see, still is.
	p, err := server.WrapPool(openSynthetic(t), synthName, 2)
	if err != nil {
		t.Fatalf("WrapPool: %v", err)
	}
	t.Cleanup(func() {
		if err := p.Close(); err != nil {
			t.Errorf("PoolEngine.Close: %v", err)
		}
	})
	if _, err := server.New(p, server.WithNotice(&strings.Builder{}),
		server.WithConcurrency(3)); err == nil {
		t.Fatal("a concurrency of 3 over a pool of 2 sessions was accepted")
	}
}

// TestAFullBatchAnswers429WithTheEnginesOwnBudget is what makes the row above
// safe: the wait moved into the engine, and the deployment's answer must not
// change with it.
//
// The slots are held outside the handler so the refusal is the queue's budget
// rather than a race with two completions finishing.
func TestAFullBatchAnswers429WithTheEnginesOwnBudget(t *testing.T) {
	const budget = 100 * time.Millisecond
	e := batchedEngine(t, openShared(t), 2, nil, budget)
	s, err := server.New(e, server.WithNotice(&strings.Builder{}),
		server.WithConcurrency(4))
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}

	// Both slots, with completions long enough that neither ends inside the
	// budget, and which nobody reads.
	for i := 0; i < 2; i++ {
		sess, err := e.NewSession(context.Background(), server.SessionSpec{})
		if err != nil {
			t.Fatalf("holding session %d: %v", i, err)
		}
		if _, err := sess.Complete(context.Background(), strings.Repeat("hold ", 2+i),
			tgo.Policy{MaxTokens: 80}); err != nil {
			t.Fatalf("holding request %d: %v", i, err)
		}
	}

	w := do(t, s, http.MethodPost, "/v1/completions",
		`{"model":"`+synthName+`","max_tokens":2,"prompt":"hi"}`)
	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want %q: it is the engine's budget rounded up "+
			"with a floor of one, and not an estimated service time", got, "1")
	}
	if body := w.Body.String(); !strings.Contains(body, "wait budget elapsed") {
		t.Errorf("the 429 does not say what the caller waited on: %s", body)
	} else if strings.Contains(body, "tgo: tgo:") {
		t.Errorf("the message is prefixed twice: %s", body)
	}
	// The reason is a label rather than a body field, which is where the two
	// refusals are told apart (021-D9).
	if m := do(t, s, http.MethodGet, "/metrics", "").Body.String(); !strings.Contains(
		m, `tgo_sessions_rejected_total{reason="queue_timeout"} 1`) {
		t.Errorf("the refusal was not counted as a queue timeout:\n%s", m)
	}
}

// TestABatchedRequestReportsWhatItReused is the number that says whether a
// request was isolated from another one's work, over the engine that shares a
// pool between every slot.
func TestABatchedRequestReportsWhatItReused(t *testing.T) {
	s := batchedServer(t, openShared(t), 2, nil)
	const body = `{"model":"` + synthName + `","max_tokens":2,` +
		`"prompt":"a prompt long enough to fill a block or two, twice over",` +
		`"cache_salt":"tenant-a"}`

	for i := 0; i < 2; i++ {
		w := do(t, s, http.MethodPost, "/v1/completions", body)
		if w.Code != http.StatusOK {
			t.Fatalf("request %d: status = %d: %s", i, w.Code, w.Body.String())
		}
	}
	// The second request shares the first's salt and its prompt, so the pool
	// holds its leading blocks. What is asserted is that the request answered
	// and that the loss header did not claim cache_salt was dropped: the reuse
	// itself is 016's, measured there.
	w := do(t, s, http.MethodPost, "/v1/completions", body)
	if got := w.Header().Get("X-Tgo-Loss"); strings.Contains(got, "cache_salt") {
		t.Errorf("X-Tgo-Loss = %q, and the batched engine does honour cache_salt", got)
	}
}

// TestABatchedSessionReportsItsReuseAndItsQueue is the two numbers a
// deployment reads off this engine that the Engine interface has no method
// for: what a request took from the shared pool, and what the queue in front of
// admission measured (021 §7).
func TestABatchedSessionReportsItsReuseAndItsQueue(t *testing.T) {
	e := batchedEngine(t, openShared(t), 2, nil, 0)

	sess, err := e.NewSession(context.Background(), server.SessionSpec{Key: "tenant-a"})
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	reused, ok := sess.(interface{ Reused() int })
	if !ok {
		t.Fatal("a batched session does not report what it reused")
	}
	if got := reused.Reused(); got != 0 {
		t.Errorf("Reused = %d before anything generated, want 0", got)
	}
	st, err := sess.Complete(context.Background(), "a prompt of several tokens",
		tgo.Policy{MaxTokens: 2})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	for st.Next() {
	}
	if err := st.Err(); err != nil {
		t.Fatalf("the stream failed: %v", err)
	}
	if got, want := reused.Reused(), st.Usage().CachedPromptTokens; got != want {
		t.Errorf("Reused = %d, want the stream's CachedPromptTokens of %d", got, want)
	}
	if err := sess.Close(); err != nil {
		t.Errorf("Session.Close: %v", err)
	}

	// The queue is reachable, which is what lets a deployment expose 008 §3's
	// two deferral reasons without this package owning the series.
	q := e.Runner().Queue()
	if q == nil {
		t.Fatal("the engine does not expose its admission queue")
	}
	if got, want := q.MaxDepth(), tgo.DefaultQueueDepth*e.Sessions(); got != want {
		t.Errorf("the queue holds %d waiters, want %d", got, want)
	}
	if s := q.Stats(); s.Depth != 0 {
		t.Errorf("Depth = %d with nothing in flight, want 0", s.Depth)
	}
}
