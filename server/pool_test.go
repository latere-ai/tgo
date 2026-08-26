// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/server"
)

// specs/019-session-affinity.md through the real handler.
//
// Every other pooling test lives in the tgo package, where the routing and the
// token ids are. What is here is the half that only the server can be wrong
// about: that a request's session is borrowed rather than allocated, that it is
// given back with its history, and that cache_salt reaches the affinity key
// instead of being reported as a knob nobody read.

// poolServer opens the fixture with the prefix cache on and serves it from a
// pool of n sessions.
//
// The prefix cache is what a pooled session has to reuse: pooling keeps the
// history and [tgo.WithPrefixCache] is what is allowed to read it. Without both
// the pool is only an allocator.
func poolServer(t *testing.T, n int) (*server.Server, *recordingEngine) {
	t.Helper()
	m, err := tgo.Open(writeCheckpoint(t), tgo.WithDevice(tgo.CPU), tgo.WithContext(96),
		tgo.WithPrefixCache(tgo.CacheSession, 96))
	if err != nil {
		t.Fatalf("tgo.Open: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("Model.Close: %v", err)
		}
	})
	pe, err := server.WrapPool(m, synthName, n)
	if err != nil {
		t.Fatalf("server.WrapPool: %v", err)
	}
	// Registered after the model's, and cleanups run last in first out, so the
	// pool closes before the model. That is the order accel requires.
	t.Cleanup(func() {
		if err := pe.Close(); err != nil {
			t.Errorf("PoolEngine.Close: %v", err)
		}
	})
	if got := pe.Sessions(); got != n {
		t.Fatalf("the pool holds %d sessions, want %d", got, n)
	}
	eng := &recordingEngine{PoolEngine: pe}
	// The concurrency is the pool size and is not divided out a second time:
	// two numbers that must agree is the bug shape 019 §4 is about.
	s, err := server.New(eng, server.WithNotice(&strings.Builder{}),
		server.WithConcurrency(pe.Sessions()))
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s, eng
}

// recordingEngine is the pooled engine with one number kept: how many leading
// prompt positions each request took from the session it was routed to.
//
// It is the number an isolation test reads, because a cache hit is faster than
// a miss and the reuse count is what a timing oracle would be measuring
// (019 §5). The answer is the same either way, so asserting the answer would
// pass on a pool with no isolation at all.
type recordingEngine struct {
	*server.PoolEngine

	mu     sync.Mutex
	reused []int
}

func (e *recordingEngine) NewSession(ctx context.Context, spec server.SessionSpec) (server.Session, error) {
	s, err := e.PoolEngine.NewSession(ctx, spec)
	if err != nil {
		return nil, err
	}
	return &recordingSession{Session: s, eng: e}, nil
}

func (e *recordingEngine) counts() []int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]int(nil), e.reused...)
}

type recordingSession struct {
	server.Session
	eng *recordingEngine
}

// Close reads the reuse count on the way out, which is the last moment it
// exists: the lease is another request's after this.
func (s *recordingSession) Close() error {
	n := 0
	if r, ok := s.Session.(interface{ Reused() int }); ok {
		n = r.Reused()
	}
	s.eng.mu.Lock()
	s.eng.reused = append(s.eng.reused, n)
	s.eng.mu.Unlock()
	return s.Session.Close()
}

// chatBody is one OpenAI Chat request carrying a whole conversation.
func chatBody(extra string, turns ...string) string {
	var b strings.Builder
	b.WriteString(`{"model":"` + synthName + `","max_tokens":3`)
	b.WriteString(extra)
	b.WriteString(`,"messages":[`)
	for i, turn := range turns {
		if i > 0 {
			b.WriteString(",")
		}
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		raw, _ := json.Marshal(turn)
		b.WriteString(`{"role":"` + role + `","content":` + string(raw) + `}`)
	}
	b.WriteString(`]}`)
	return b.String()
}

// TestAPooledServerReusesTheConversation is 016 §7.2's empty row filled in.
//
// Two requests continuing one conversation. The second is routed to the
// session the first left in the pool, and reuses the run the two renders share.
// The comparison is against a server whose pool never saw the first turn, so
// the number means something rather than merely being non-zero.
func TestAPooledServerReusesTheConversation(t *testing.T) {
	s, eng := poolServer(t, 2)

	first := do(t, s, http.MethodPost, "/v1/chat/completions", chatBody("", "hi"))
	if first.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", first.Code, first.Body.String())
	}
	answer := answerOf(t, first)

	second := do(t, s, http.MethodPost, "/v1/chat/completions",
		chatBody("", "hi", answer, "and again"))
	if second.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", second.Code, second.Body.String())
	}

	got := eng.counts()
	if len(got) != 2 {
		t.Fatalf("%d requests were recorded, want 2", len(got))
	}
	if got[0] != 0 {
		t.Fatalf("the first request reused %d positions from an empty pool", got[0])
	}
	if got[1] == 0 {
		t.Fatal("the second turn reused nothing; turn n's render begins with turn n-1's, " +
			"so a session that kept its history has at least the first turn's prompt")
	}

	// The same second turn on a pool that never saw the first, which is what
	// the unpooled server does on every request.
	cold, coldEng := poolServer(t, 1)
	w := do(t, cold, http.MethodPost, "/v1/chat/completions",
		chatBody("", "hi", answer, "and again"))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if n := coldEng.counts()[0]; n != 0 {
		t.Fatalf("a pool that never saw the conversation reused %d positions", n)
	}
	// And what was saved was arithmetic, not an answer (016-D6, §6).
	if warm, coldAnswer := answerOf(t, second), answerOf(t, w); warm != coldAnswer {
		t.Fatalf("the pooled second turn answered %q and a cold one answers %q",
			warm, coldAnswer)
	}
}

// TestCacheSaltIsTheAffinityKey is 019-D3 over the wire.
//
// Asserted on the reuse count: a salted request must not read an unsalted
// session's history. Two sessions, so each request has somewhere cold to go and
// the miss is a routing decision rather than an empty pool.
//
// The other direction -- an unsalted request against a salted session -- is not
// reachable from this sequence, because the unsalted conversation is
// established first and the unsalted request always has its own session to hit.
// [TestAnUnsaltedRequestDoesNotReadASaltedSession] is that direction.
func TestCacheSaltIsTheAffinityKey(t *testing.T) {
	s, eng := poolServer(t, 2)

	body := func(salt string) string {
		if salt == "" {
			return chatBody("", "hello there")
		}
		return chatBody(`,"cache_salt":"`+salt+`"`, "hello there")
	}
	for _, c := range []struct {
		what string
		salt string
		want string // "cold" or "warm"
	}{
		{"the unsalted conversation", "", "cold"},
		{"a salted request against an unsalted session", "tenant-a", "cold"},
		{"the salted conversation again", "tenant-a", "warm"},
		{"the unsalted conversation again", "", "warm"},
		{"a second salt against both", "tenant-b", "cold"},
	} {
		w := do(t, s, http.MethodPost, "/v1/chat/completions", body(c.salt))
		if w.Code != http.StatusOK {
			t.Fatalf("%s: status = %d: %s", c.what, w.Code, w.Body.String())
		}
		got := eng.counts()
		n := got[len(got)-1]
		if c.want == "cold" && n != 0 {
			t.Fatalf("%s reused %d positions; affinity fails closed, so it shares with "+
				"nobody rather than with everybody", c.what, n)
		}
		if c.want == "warm" && n == 0 {
			t.Fatalf("%s reused nothing, so its own key's session was not found", c.what)
		}
	}
}

// TestAnUnsaltedRequestDoesNotReadASaltedSession is the second row of
// specs/019-session-affinity.md §5's table over the wire: a request with no
// cache_salt matches a session with no salt, "and never one with a salt".
//
// The salted conversation is established first and nothing unsalted has run, so
// the only session holding the prompt is the salted one and the pool's other
// session has never served anything. A reuse count above zero is therefore the
// salted caller's history being read by somebody who supplied no salt, which is
// the fail-open default 019-D3 exists to refuse.
func TestAnUnsaltedRequestDoesNotReadASaltedSession(t *testing.T) {
	s, eng := poolServer(t, 2)

	body := chatBody(`,"cache_salt":"tenant-a"`, "hello there")
	if w := do(t, s, http.MethodPost, "/v1/chat/completions", body); w.Code != http.StatusOK {
		t.Fatalf("the salted request: status = %d: %s", w.Code, w.Body.String())
	}
	if n := eng.counts()[0]; n != 0 {
		t.Fatalf("the salted request reused %d positions of an empty pool", n)
	}
	// The same conversation with the salt removed.
	if w := do(t, s, http.MethodPost, "/v1/chat/completions",
		chatBody("", "hello there")); w.Code != http.StatusOK {
		t.Fatalf("the unsalted request: status = %d: %s", w.Code, w.Body.String())
	}
	if n := eng.counts()[1]; n != 0 {
		t.Fatalf("an unsalted request reused %d positions of a salted session's history; "+
			"affinity fails closed, so a caller who supplies no salt shares with nobody "+
			"rather than with everybody", n)
	}
	// The salted caller still has its own session, so the miss above was
	// isolation and not an emptied pool.
	if w := do(t, s, http.MethodPost, "/v1/chat/completions", body); w.Code != http.StatusOK {
		t.Fatalf("the salted request again: status = %d: %s", w.Code, w.Body.String())
	}
	if n := eng.counts()[2]; n == 0 {
		t.Fatal("the salted conversation reused nothing on its second turn, so its own " +
			"session was taken rather than isolated")
	}
}

// TestCacheSaltIsNotReportedAsLost is the other half of honouring it.
//
// llmdialect files every member its IR cannot carry, and cache_salt is one, so
// without the subtraction a caller who isolated their cache would be told the
// field was dropped and would reasonably assume they had not.
func TestCacheSaltIsNotReportedAsLost(t *testing.T) {
	s, _ := poolServer(t, 1)
	w := do(t, s, http.MethodPost, "/v1/chat/completions",
		chatBody(`,"cache_salt":"tenant-a"`, "hi"))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	for _, f := range strings.Split(w.Header().Get("X-Tgo-Loss"), ", ") {
		if f == "cache_salt" {
			t.Fatalf("X-Tgo-Loss = %q reports cache_salt, which bounded what this request "+
				"could reuse", w.Header().Get("X-Tgo-Loss"))
		}
	}
}

// TestACacheSaltOfTheWrongTypeIsRefused: a salt that did not parse is a request
// whose caller believes their cache is isolated and is not, so it is an error
// naming the field rather than a silent empty key.
func TestACacheSaltOfTheWrongTypeIsRefused(t *testing.T) {
	s, _ := poolServer(t, 1)
	w := do(t, s, http.MethodPost, "/v1/chat/completions",
		chatBody(`,"cache_salt":17`, "hi"))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "cache_salt") {
		t.Fatalf("the refusal does not name the field: %s", w.Body.String())
	}
}

// TestAPooledSessionSurvivesACancelledRequest is 019-D5 through the handler.
//
// The request is cancelled before it is admitted, so the session it would have
// taken is returned untouched and the request after it still finds a pool of
// the size it started with.
func TestAPooledSessionSurvivesACancelledRequest(t *testing.T) {
	s, eng := poolServer(t, 1)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(chatBody("", "hi"))).WithContext(ctx)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	if w.Code == http.StatusOK {
		t.Fatalf("a cancelled request answered 200 with %q", w.Body.String())
	}

	// The session was never taken, so nothing was recorded and the next
	// request still gets one.
	if got := eng.counts(); len(got) != 0 {
		t.Fatalf("%d sessions were leased by a request that had already gone", len(got))
	}
	if ok := do(t, s, http.MethodPost, "/v1/chat/completions", chatBody("", "hi")); ok.Code != http.StatusOK {
		t.Fatalf("the request after a cancelled one got %d: %s", ok.Code, ok.Body.String())
	}
}

// TestWrapPoolRefusesAPoolThatCannotBeReserved is 019-D2 at startup: a device
// that cannot hold N sessions' cache says so before the server binds, rather
// than under load.
func TestWrapPoolRefusesAPoolThatCannotBeReserved(t *testing.T) {
	m, err := tgo.Open(writeCheckpoint(t), tgo.WithDevice(tgo.CPU), tgo.WithContext(96))
	if err != nil {
		t.Fatalf("tgo.Open: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if _, err := server.WrapPool(m, synthName, 0); err == nil {
		t.Fatal("a pool of no sessions was accepted")
	}
}

// TestAPooledEngineForwardsTheModelsNumbers pins that WrapPool is Wrap with a
// pool behind it rather than a second engine: the id, the context, the
// vocabulary and the cache size are the model's.
func TestAPooledEngineForwardsTheModelsNumbers(t *testing.T) {
	m := openSynthetic(t)
	pe, err := server.WrapPool(m, synthName, 2)
	if err != nil {
		t.Fatalf("server.WrapPool: %v", err)
	}
	t.Cleanup(func() {
		if err := pe.Close(); err != nil {
			t.Errorf("PoolEngine.Close: %v", err)
		}
	})
	i := m.Info()
	if pe.Name() != synthName || pe.Context() != i.Context || pe.VocabSize() != i.VocabSize ||
		pe.CacheBytesPerSession() != i.CacheBytesPerSession {
		t.Fatalf("the pooled engine reports %q/%d/%d/%d and the model says %q/%d/%d/%d",
			pe.Name(), pe.Context(), pe.VocabSize(), pe.CacheBytesPerSession(),
			synthName, i.Context, i.VocabSize, i.CacheBytesPerSession)
	}
	if err := pe.CheckSchema([]byte(`{"type":"object"}`)); err != nil {
		t.Fatalf("CheckSchema through the pooled engine: %v", err)
	}
	// Idempotent, because a shutdown that closed the pool and then failed on
	// its way out must not close every buffer twice.
	if err := pe.Close(); err != nil {
		t.Fatalf("PoolEngine.Close: %v", err)
	}
}

// TestAPooledServerServesEveryRoute: the four request routes reach a lease the
// same way they reached a session.
func TestAPooledServerServesEveryRoute(t *testing.T) {
	s, eng := poolServer(t, 1)
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
		})
	}
	if n := len(eng.counts()); n != 4 {
		t.Fatalf("%d requests took a lease from a pool of one; each must return it", n)
	}
}

// TestAdmissionAboveThePoolIsRefused is 019 §8.6: the pool is a semaphore too,
// so two of them with different sizes is a misconfiguration rather than a
// larger limit.
//
// The admitter's slots are taken first, so a concurrency above the pool admits
// the surplus and then blocks it inside NewSession waiting for a session. That
// is bounded, and the requests behind it still queue and still get their 429.
// What breaks is narrower: those waiters are not in the queue, so
// [server.WithQueueWait] does not time them out and the queue depth does not
// count them, and their caller waits past the Retry-After instead of being
// refused by it. Both numbers are known at startup, so the disagreement is
// refused there.
//
// Below the pool is allowed, because that is a deployment asking for a reuse
// depth larger than its concurrency, which is a configuration and not a
// mistake.
func TestAdmissionAboveThePoolIsRefused(t *testing.T) {
	m := openSynthetic(t)
	pe, err := server.WrapPool(m, synthName, 2)
	if err != nil {
		t.Fatalf("server.WrapPool: %v", err)
	}
	t.Cleanup(func() {
		if err := pe.Close(); err != nil {
			t.Errorf("PoolEngine.Close: %v", err)
		}
	})
	quiet := server.WithNotice(&strings.Builder{})

	if _, err := server.New(pe, quiet, server.WithConcurrency(3)); err == nil {
		t.Fatal("a concurrency of 3 over a pool of 2 was accepted")
	} else if !strings.Contains(err.Error(), "pools 2 session") {
		t.Fatalf("the refusal does not name the pool's size: %v", err)
	}
	// The same number arrived at from memory rather than from a flag, which is
	// the way a deployment reaches it by accident: WithKVBudget divides the
	// device's free memory and knows nothing about how many sessions were
	// actually reserved.
	budget := 4*m.Info().CacheBytesPerSession + m.Info().CacheBytesPerSession/2
	if _, err := server.New(pe, quiet, server.WithKVBudget(budget)); err == nil {
		t.Fatal("a KV budget holding 4 sessions over a pool of 2 was accepted")
	}

	for _, n := range []int{1, 2} {
		s, err := server.New(pe, quiet, server.WithConcurrency(n))
		if err != nil {
			t.Fatalf("a concurrency of %d over a pool of 2: %v", n, err)
		}
		if s.Concurrency() != n {
			t.Fatalf("the server admits %d and was built for %d", s.Concurrency(), n)
		}
	}
}
