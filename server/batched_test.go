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

// batchedServer builds a server over a batched engine of n slots.
func batchedServer(t *testing.T, m *tgo.Model, n int, rec *bench.Recorder) *server.Server {
	t.Helper()
	e, err := server.WrapRunner(m, synthName, tgo.RunnerOptions{
		Slots: n, Chunk: 16, Reserve: tgo.CacheBlock, Recorder: rec,
	})
	if err != nil {
		t.Fatalf("server.WrapRunner: %v", err)
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Errorf("RunnerEngine.Close: %v", err)
		}
	})
	s, err := server.New(e, server.WithNotice(&strings.Builder{}),
		server.WithConcurrency(n))
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

// TestAdmissionAboveTheSlotsIsRefused is 019 §8.6's refusal, unchanged, with
// the slot count in place of the pool size: a concurrency above what the engine
// runs at once would make requests wait where the queue neither counts nor
// times them out, so the Retry-After would stop describing what they wait.
func TestAdmissionAboveTheSlotsIsRefused(t *testing.T) {
	m := openShared(t)
	e, err := server.WrapRunner(m, synthName, tgo.RunnerOptions{Slots: 2})
	if err != nil {
		t.Fatalf("WrapRunner: %v", err)
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Errorf("RunnerEngine.Close: %v", err)
		}
	})
	if _, err := server.New(e, server.WithNotice(&strings.Builder{}),
		server.WithConcurrency(3)); err == nil {
		t.Fatal("a concurrency of 3 over a batch of 2 slots was accepted")
	}
	if _, err := server.New(e, server.WithNotice(&strings.Builder{}),
		server.WithConcurrency(2)); err != nil {
		t.Fatalf("a concurrency equal to the slot count was refused: %v", err)
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
	m := openShared(t)
	e, err := server.WrapRunner(m, synthName, tgo.RunnerOptions{
		Slots: 2, Chunk: 16, Reserve: tgo.CacheBlock,
	})
	if err != nil {
		t.Fatalf("WrapRunner: %v", err)
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Errorf("RunnerEngine.Close: %v", err)
		}
	})

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
