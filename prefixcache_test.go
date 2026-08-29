// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
)

// The capacity every test below runs at.
//
// 96 rather than a power of two, so bucketsFor gives {32, 64, 96} and a prompt,
// its suffix and its cold twin can land in three different plans. A capacity
// that was itself a bucket would make "the warm run used a different prefill
// shape" untestable, which is the one thing §6 is about.
const cacheCap = 96

// promptIDs is n deterministic token ids, distinct per seed.
//
// Ids from 1 upward and never 0: id 0 is a real token of the fixture
// vocabulary, but a slice of zeros is the value a bug leaves behind, and a
// prompt that is indistinguishable from an uninitialised one is a fixture that
// cannot fail.
func promptIDs(seed, n int) []int {
	out := make([]int, n)
	x := uint32(seed)*2654435761 + 1
	for i := range out {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		out[i] = 1 + int(x%uint32(synthVocab-1))
	}
	return out
}

// extend is prefix followed by n fresh ids, the first of which is not avoid.
//
// The avoidance is what makes a reuse assertion exact rather than probabilistic.
// A session that generated token g at position len(prefix) holds g there, so a
// continuation that happened to begin with g would share len(prefix)+1 positions
// with the cache and the test would read a correct engine as a broken one.
func extend(prefix []int, seed, n, avoid int) []int {
	tail := promptIDs(seed, n)
	if len(tail) > 0 && tail[0] == avoid {
		tail[0] = 1 + tail[0]%(synthVocab-1)
	}
	out := make([]int, 0, len(prefix)+len(tail))
	return append(append(out, prefix...), tail...)
}

// at is the token the session holds at position p, or -1 if it holds none.
func at(s *Session, p int) int {
	if p < 0 || p >= len(s.history) {
		return -1
	}
	return s.history[p]
}

// generation is one request's result, in the three forms a test compares.
type generation struct {
	text  string
	toks  []int
	usage Usage
}

// generate runs one request through [Session.start], which is the seam
// [Session.Chat] and [Session.Complete] both reach after tokenizing, and drains
// it.
//
// toks comes from the session's own history rather than from the events, so it
// is token ids and not decoded text: two completions that differ in one token
// can decode to the same string when the difference is a byte-level piece the
// decoder held back. It is every generated token but the last, because the last
// one sampled is emitted and never fed back; the text and
// [Usage.CompletionTokens] cover that one.
func request(t *testing.T, s *Session, ids []int, p Policy) generation {
	t.Helper()
	st, err := s.start(context.Background(), ids, p)
	if err != nil {
		t.Fatalf("start on a %d-token prompt: %v", len(ids), err)
	}
	var b strings.Builder
	for st.Next() {
		b.WriteString(st.Text())
	}
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	return generation{
		text:  b.String(),
		toks:  append([]int(nil), s.history[len(ids):]...),
		usage: st.Usage(),
	}
}

// sameGeneration reports the first place two runs differ, or nothing.
//
// It names the index rather than dumping both, because 016-D6 declines to claim
// bit-exactness and the number a divergence report needs is where it first
// showed — 010 §3's measurement, not a diff.
func sameGeneration(t *testing.T, what string, cold, warm generation) {
	t.Helper()
	for i := range min(len(cold.toks), len(warm.toks)) {
		if cold.toks[i] != warm.toks[i] {
			t.Fatalf("%s: cold and warm diverge at generated token %d: cold %d, warm %d\n"+
				"cold %v\nwarm %v", what, i, cold.toks[i], warm.toks[i], cold.toks, warm.toks)
		}
	}
	if len(cold.toks) != len(warm.toks) {
		t.Fatalf("%s: cold generated %d tokens and warm generated %d: %v against %v",
			what, len(cold.toks), len(warm.toks), cold.toks, warm.toks)
	}
	if cold.usage.CompletionTokens != warm.usage.CompletionTokens {
		t.Fatalf("%s: cold generated %d tokens and warm %d", what,
			cold.usage.CompletionTokens, warm.usage.CompletionTokens)
	}
	if cold.text != warm.text {
		t.Fatalf("%s: the token ids agree and the text does not: cold %q, warm %q",
			what, cold.text, warm.text)
	}
}

// warmModel is the fixture with the prefix cache on.
func warmModel(t *testing.T) *Model {
	t.Helper()
	return openSynthetic(t, WithPrefixCache(CacheSession, cacheCap))
}

// TestPrefixCacheWarmEqualsCold is specs/016-prefix-cache.md §6's first row and
// the whole point of the feature.
//
// The two runs answer the same prompt from different arithmetic. The cold one
// prefills 50 tokens in one 64-row plan. The warm one holds the first 20
// positions from a 32-row plan it ran for an earlier turn and prefills the
// remaining 30 in a 32-row plan at base 20, so every shared position's
// key/value state was produced by a differently shaped GEMM at a different
// point in the run.
//
// 016-D6 declines to claim the bits are equal and 010 §5.1 says why: floating
// point is not associative and the storage format sets the tolerance. What is
// asserted is the property a caller checks — the same greedy tokens — with the
// first differing index reported if it ever stops holding, which is 010 §3's
// measurement rather than a raised tolerance.
func TestPrefixCacheWarmEqualsCold(t *testing.T) {
	t.Parallel()
	m := warmModel(t)

	first := promptIDs(1, 20)
	warm := session(t, m, WithSessionContext(cacheCap))
	turn1 := request(t, warm, first, greedy(8))
	if turn1.usage.CachedPromptTokens != 0 {
		t.Fatalf("a fresh session reused %d positions; it holds none",
			turn1.usage.CachedPromptTokens)
	}

	// The continuation must not begin with the token turn 1 generated at
	// position 20, or the shared run would be 21 and not 20.
	full := extend(first, 2, 30, at(warm, len(first)))

	cold := request(t, session(t, m, WithSessionContext(cacheCap)), full, greedy(8))
	if cold.usage.CachedPromptTokens != 0 {
		t.Fatalf("the cold session reused %d positions", cold.usage.CachedPromptTokens)
	}

	turn2 := request(t, warm, full, greedy(8))
	if turn2.usage.CachedPromptTokens != len(first) {
		t.Fatalf("the warm request reused %d positions and the prompts share %d",
			turn2.usage.CachedPromptTokens, len(first))
	}
	if turn2.usage.PromptTokens != len(full) {
		t.Fatalf("PromptTokens is %d on a %d-token prompt; reuse is not a discount on "+
			"what the caller sent", turn2.usage.PromptTokens, len(full))
	}
	sameGeneration(t, "a 20-position hit", cold, turn2)
}

// TestPrefixCacheIdenticalPromptPrefillsOneToken is specs/016-prefix-cache.md
// §3.1 and 016-D10, tested the only way it can be.
//
// The cache holds key/value state and not logits. Sampling needs the logits at
// the last prompt position, which come from a forward pass over it, so reuse is
// capped at T-1 and a full hit still prefills one token. Reuse T and the
// request has a warm cache and nothing to sample from.
//
// It hides in chat: 003 §3's rendered prompt always ends with a fresh assistant
// opener, so the suffix is never empty and the bug never fires. An identical
// prompt submitted twice is the case the chat path cannot produce.
func TestPrefixCacheIdenticalPromptPrefillsOneToken(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	rec := bench.NewRecorder(64)
	s := session(t, m, WithSessionContext(cacheCap), WithRecorder(rec))

	ids := promptIDs(3, 40)
	cold := request(t, s, ids, greedy(6))
	if got := rec.Report().Prefill.Tokens; got != len(ids) {
		t.Fatalf("the cold prefill scored %d tokens and the prompt is %d", got, len(ids))
	}

	rec.Reset()
	step := capturePrefill(s)
	warm := request(t, s, ids, greedy(6))

	// The one-token suffix must still take a *prefill* plan. Session.plan and
	// Session.bindings both branch on rows == 1, and a decode plan drops
	// BaseName -- accel refuses one, since a decode attends over the whole
	// cache its Lengths names. A single token routed there would attend to the
	// reused prefix as though it began the sequence, with nothing refused.
	if step.rows <= 1 {
		t.Fatalf("the one-token suffix ran a %d-row step; a decode plan declares no "+
			"base scalar and would let the last prompt token see the whole cache",
			step.rows)
	}
	if want := uint32(len(ids) - 1); step.base != want {
		t.Fatalf("the one-token suffix prefilled at base %d, want %d", step.base, want)
	}
	if int(step.lengths) != len(ids) || step.slots[0] != uint32(len(ids)-1) {
		t.Fatalf("the one-token suffix bound lengths %d and slot %d on a %d-token prompt",
			step.lengths, step.slots[0], len(ids))
	}

	if want := len(ids) - 1; warm.usage.CachedPromptTokens != want {
		t.Fatalf("an identical prompt reused %d of %d positions; the cap is T-1 = %d",
			warm.usage.CachedPromptTokens, len(ids), want)
	}
	r := rec.Report()
	if r.Prefill.Steps != 1 || r.Prefill.Tokens != 1 {
		t.Fatalf("the warm prefill ran %d steps over %d tokens; a full hit prefills "+
			"exactly the last prompt position", r.Prefill.Steps, r.Prefill.Tokens)
	}
	sameGeneration(t, "an identical prompt", cold, warm)
}

// TestPrefixCachePartialHitPrefillsExactlyTheSuffix is §3: what is recomputed is
// the suffix and nothing else.
//
// Asserted against the recorder's token count rather than against a duration.
// A timing assertion would pass on a session that prefilled the whole prompt on
// a fast machine, which is the measurement that reads as working right up to
// the point somebody trusts it.
func TestPrefixCachePartialHitPrefillsExactlyTheSuffix(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	rec := bench.NewRecorder(64)
	s := session(t, m, WithSessionContext(cacheCap), WithRecorder(rec))

	first := promptIDs(4, 24)
	request(t, s, first, greedy(4))

	full := extend(first, 5, 26, at(s, len(first)))
	rec.Reset()
	got := request(t, s, full, greedy(4))

	if got.usage.CachedPromptTokens != len(first) {
		t.Fatalf("reused %d positions and the prompts share %d",
			got.usage.CachedPromptTokens, len(first))
	}
	suffix := len(full) - len(first)
	r := rec.Report()
	if r.Prefill.Steps != 1 || r.Prefill.Tokens != suffix {
		t.Fatalf("the prefill ran %d steps over %d tokens; the suffix is %d",
			r.Prefill.Steps, r.Prefill.Tokens, suffix)
	}
	// One decode step per generated token after the first: the prefill's
	// logits produce token one, and every step after it feeds the token
	// before. A prefill that had also been counted as a decode, or a suffix
	// re-scored one token at a time, shows up here.
	if want := got.usage.CompletionTokens - 1; r.Decode.Steps != want ||
		r.Decode.Tokens != want {
		t.Fatalf("decode ran %d steps over %d tokens for %d generated tokens",
			r.Decode.Steps, r.Decode.Tokens, got.usage.CompletionTokens)
	}
}

// TestPrefixCacheDivergentPromptReusesNothing is the other end of §3: a prompt
// that shares no first token shares no position.
func TestPrefixCacheDivergentPromptReusesNothing(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	s := session(t, m, WithSessionContext(cacheCap))

	a := promptIDs(6, 20)
	request(t, s, a, greedy(4))

	b := promptIDs(7, 20)
	if b[0] == a[0] {
		b[0] = 1 + b[0]%(synthVocab-1)
	}
	got := request(t, s, b, greedy(4))
	if got.usage.CachedPromptTokens != 0 {
		t.Fatalf("a prompt with a different first token reused %d positions",
			got.usage.CachedPromptTokens)
	}
	cold := request(t, session(t, m, WithSessionContext(cacheCap)), b, greedy(4))
	sameGeneration(t, "a prompt that shares nothing", cold, got)
}

// TestPrefixCacheOffPrefillsEverything pins the default.
//
// A Model opened with no option reuses nothing, whatever the session has
// already scored, because turning the cache on changes what a request produces
// (016-D6) and that is not a default anyone should get without asking.
// It is run twice: once on the default, and once on a Model that asked for
// [CacheOff] with a budget. The second is the configuration Open accepts and
// nothing else covers -- checkCache lets any positions through on an off scope,
// so a budget survives into the Model and only Session.reusable's scope guard
// stops it being spent. Without this case that guard reads as dead code, and
// deleting it turns reuse on for a caller who asked for off.
func TestPrefixCacheOffPrefillsEverything(t *testing.T) {
	t.Parallel()
	for _, c := range []struct {
		name string
		opts []Option
	}{
		{"no option at all", nil},
		{"off with a budget", []Option{WithPrefixCache(CacheOff, cacheCap)}},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			m := openSynthetic(t, c.opts...)
			if m.cacheScope != CacheOff {
				t.Fatalf("the scope is %v", m.cacheScope)
			}
			rec := bench.NewRecorder(64)
			s := session(t, m, WithSessionContext(cacheCap), WithRecorder(rec))

			ids := promptIDs(8, 30)
			request(t, s, ids, greedy(4))
			rec.Reset()
			got := request(t, s, ids, greedy(4))

			if got.usage.CachedPromptTokens != 0 {
				t.Fatalf("the cache is off and the request reused %d positions",
					got.usage.CachedPromptTokens)
			}
			if n := rec.Report().Prefill.Tokens; n != len(ids) {
				t.Fatalf("the cold prefill scored %d of %d tokens", n, len(ids))
			}
		})
	}
}

// TestPrefixCacheOffMatchesWarm is §7's `off` row doing its job: the cold
// baseline a measurement compares against is the same answer, not another one.
func TestPrefixCacheOffMatchesWarm(t *testing.T) {
	t.Parallel()
	ids := promptIDs(9, 34)

	off := openSynthetic(t)
	cold := request(t, session(t, off, WithSessionContext(cacheCap)), ids, greedy(6))

	on := warmModel(t)
	warmSess := session(t, on, WithSessionContext(cacheCap))
	request(t, warmSess, ids[:16], greedy(3))
	warm := request(t, warmSess, ids, greedy(6))
	if warm.usage.CachedPromptTokens == 0 {
		t.Fatal("the warm session reused nothing, so this compares two cold runs")
	}
	sameGeneration(t, "off against session scope", cold, warm)
}

// TestPrefixCacheResetDropsTheReuse is the release the spec asks for, in the
// shape this engine has one.
//
// [Session.Reset] rewinds to zero, so the next request prefills its whole
// prompt, and [Session.Close] gives the cache's memory back. A session that
// kept reusing after a reset would be attending to a conversation the caller
// declared over.
func TestPrefixCacheResetDropsTheReuse(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	s := session(t, m, WithSessionContext(cacheCap))

	ids := promptIDs(10, 28)
	request(t, s, ids, greedy(4))
	if got := request(t, s, ids, greedy(4)); got.usage.CachedPromptTokens == 0 {
		t.Fatal("the second identical prompt reused nothing")
	}

	s.Reset()
	if s.length != 0 || len(s.history) != 0 {
		t.Fatalf("Reset left length %d and %d history entries", s.length, len(s.history))
	}
	if got := request(t, s, ids, greedy(4)); got.usage.CachedPromptTokens != 0 {
		t.Fatalf("a request after Reset reused %d positions", got.usage.CachedPromptTokens)
	}
}

// TestPrefixCacheClosedSessionHoldsNothing is the rest of the release: a closed
// session refuses work and a new one starts cold, so nothing outlives the
// conversation it belonged to.
func TestPrefixCacheClosedSessionHoldsNothing(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	s, err := m.NewSession(WithSessionContext(cacheCap))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	ids := promptIDs(11, 26)
	request(t, s, ids, greedy(4))
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := s.Complete(context.Background(), "anything", greedy(4)); err == nil {
		t.Fatal("a closed session accepted a request")
	}
	next := session(t, m, WithSessionContext(cacheCap))
	if got := request(t, next, ids, greedy(4)); got.usage.CachedPromptTokens != 0 {
		t.Fatalf("a new session reused %d positions from a closed one's conversation",
			got.usage.CachedPromptTokens)
	}
}

// TestPrefixCacheRefusedRequestLeavesTheCacheIntact is the ordering property
// that has no second chance.
//
// Every refusal in Session.start runs before the rewind, so a request that does
// not fit leaves the conversation exactly as it found it. A rewind that ran
// first would shorten a cache the caller can still generate from, and the
// damage would show as a later turn silently answering from half its context.
func TestPrefixCacheRefusedRequestLeavesTheCacheIntact(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	s := session(t, m, WithSessionContext(cacheCap))

	ids := promptIDs(12, 30)
	request(t, s, ids, greedy(4))
	length, history := s.length, append([]int(nil), s.history...)

	// A prompt that shares the whole of this conversation and does not fit.
	tooLong := extend(ids, 13, cacheCap, at(s, len(ids)))
	if _, err := s.start(context.Background(), tooLong, greedy(1)); err == nil {
		t.Fatal("a prompt longer than the cache was accepted")
	}
	if s.length != length || fmtIDs(s.history) != fmtIDs(history) {
		t.Fatalf("a refused request rewound the cache: length %d -> %d", length, s.length)
	}
	if got := request(t, s, ids, greedy(4)); got.usage.CachedPromptTokens != len(ids)-1 {
		t.Fatalf("after the refusal the conversation reused %d of %d positions",
			got.usage.CachedPromptTokens, len(ids)-1)
	}
}

// TestPrefixCacheSeededCompletionIsIdenticalColdAndWarm is §6's second
// subtlety, which is the one that is not real and is worth pinning anyway.
//
// A prefill consumes no draws — 006-D2 draws once per step — so a seeded stream
// produces the same completion whether or not the prompt was cached. It would
// be easy to break by threading the cache through the sampler, and it is the
// property a user checks.
func TestPrefixCacheSeededCompletionIsIdenticalColdAndWarm(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	ids := promptIDs(14, 32)
	p := Policy{MaxTokens: 8, Temperature: 0.8, TopK: 20, Seed: 0x5eed}

	cold := request(t, session(t, m, WithSessionContext(cacheCap)), ids, p)

	warm := session(t, m, WithSessionContext(cacheCap))
	request(t, warm, ids[:15], p)
	got := request(t, warm, ids, p)
	if got.usage.CachedPromptTokens == 0 {
		t.Fatal("the warm run reused nothing")
	}
	sameGeneration(t, "a seeded stream", cold, got)
}

// TestPrefixCacheChatReusesTheConversation is the multi-turn win end to end,
// through the public surface.
//
// Turn n's rendered prompt begins with turn n-1's, so the shared run is
// everything the session already scored: specs/016-prefix-cache.md §1's 1-1/n,
// and 003 §3's structural guarantee is what makes it so rather than a radix
// trie's discovery.
func TestPrefixCacheChatReusesTheConversation(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	rec := bench.NewRecorder(256)
	s := session(t, m, WithSessionContext(cacheCap), WithRecorder(rec))

	text := func(role chat.Role, body string) chat.Message {
		return chat.Message{Role: role,
			Blocks: []chat.Block{{Type: chat.BlockText, Text: body}}}
	}
	msgs := []chat.Message{text(chat.User, "hi")}
	first := chatTurn(t, s, msgs, greedy(4))
	if first.usage.CachedPromptTokens != 0 {
		t.Fatalf("the first turn reused %d positions", first.usage.CachedPromptTokens)
	}

	msgs = append(msgs,
		text(chat.Assistant, first.text),
		text(chat.User, "and again"))
	rec.Reset()
	second := chatTurn(t, s, msgs, greedy(4))

	if second.usage.CachedPromptTokens == 0 {
		t.Fatal("the second chat turn reused nothing; turn n's render begins with " +
			"turn n-1's, so the shared run is at least the first turn's prompt")
	}
	if second.usage.CachedPromptTokens >= second.usage.PromptTokens {
		t.Fatalf("reused %d of a %d-token prompt; the cap is T-1",
			second.usage.CachedPromptTokens, second.usage.PromptTokens)
	}
	want := second.usage.PromptTokens - second.usage.CachedPromptTokens
	if n := rec.Report().Prefill.Tokens; n != want {
		t.Fatalf("the turn prefilled %d tokens and the unshared suffix is %d", n, want)
	}
}

// chatTurn runs one [Session.Chat] turn and drains it.
func chatTurn(t *testing.T, s *Session, msgs []chat.Message, p Policy) generation {
	t.Helper()
	st, err := s.Chat(context.Background(), msgs, p)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	var b strings.Builder
	for st.Next() {
		if st.Event().Kind == TextDelta {
			b.WriteString(st.Text())
		}
	}
	if err := st.Err(); err != nil {
		t.Fatalf("Chat stream: %v", err)
	}
	return generation{text: b.String(), usage: st.Usage()}
}

// TestPrefixCacheCompleteReusesThePrompt is the same through [Session.Complete],
// which has no template and therefore no assistant opener to hide §3.1's cap
// behind.
func TestPrefixCacheCompleteReusesThePrompt(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	s := session(t, m, WithSessionContext(cacheCap))

	drain := func(prompt string) Usage {
		t.Helper()
		st, err := s.Complete(context.Background(), prompt, greedy(4))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		for st.Next() {
		}
		if err := st.Err(); err != nil {
			t.Fatalf("Complete stream: %v", err)
		}
		return st.Usage()
	}
	const text = "the quick brown fox jumps over the lazy dog"
	if u := drain(text); u.CachedPromptTokens != 0 {
		t.Fatalf("the first Complete reused %d positions", u.CachedPromptTokens)
	}
	u := drain(text)
	if u.CachedPromptTokens != u.PromptTokens-1 {
		t.Fatalf("an identical Complete reused %d of %d positions; the cap is T-1",
			u.CachedPromptTokens, u.PromptTokens)
	}
}

// TestPrefixCachePositionsCap is WithPrefixCache's second argument doing
// something: a smaller budget reuses less and still answers the same.
func TestPrefixCachePositionsCap(t *testing.T) {
	t.Parallel()
	const cap = 12
	m := openSynthetic(t, WithPrefixCache(CacheSession, cap))
	s := session(t, m, WithSessionContext(cacheCap))

	ids := promptIDs(15, 30)
	cold := request(t, s, ids, greedy(5))
	warm := request(t, s, ids, greedy(5))
	if warm.usage.CachedPromptTokens != cap {
		t.Fatalf("reused %d positions against a budget of %d",
			warm.usage.CachedPromptTokens, cap)
	}
	sameGeneration(t, "a capped budget", cold, warm)
}

// TestPrefixCacheConcurrentSessions is specs/016-prefix-cache.md §10.4 in the
// form this engine can reach it.
//
// 007-D1 leaves a Session unlocked and serves sessions from independent
// goroutines, so the reuse bookkeeping runs concurrently with every other
// session's. Under session scope no block crosses a session, so what this
// proves is that the per-session state is per-session: every goroutine's warm
// answer equals the serial cold one, and -race sees the Model's submission lock
// and the shared plan cache underneath.
//
// The stronger case the section is actually about — many sessions sharing one
// block pool — is not reachable; see this package's reported discrepancies.
func TestPrefixCacheConcurrentSessions(t *testing.T) {
	t.Parallel()
	m := warmModel(t)

	shared := promptIDs(16, 20)
	prompts := make([][]int, 4)
	want := make([]generation, len(prompts))
	for i := range prompts {
		prompts[i] = extend(shared, 100+i, 14, -1)
		want[i] = request(t, session(t, m, WithSessionContext(cacheCap)), prompts[i],
			greedy(5))
	}

	got := make([]generation, len(prompts))
	var wg sync.WaitGroup
	for i := range prompts {
		wg.Go(func() {
			s, err := m.NewSession(WithSessionContext(cacheCap))
			if err != nil {
				t.Errorf("NewSession: %v", err)
				return
			}
			defer func() {
				if err := s.Close(); err != nil {
					t.Errorf("Session.Close: %v", err)
				}
			}()
			// The shared run first, so every session is warm on the same
			// prefix when the second request goes in.
			request(t, s, shared, greedy(3))
			got[i] = request(t, s, prompts[i], greedy(5))
		})
	}
	wg.Wait()
	if t.Failed() {
		return
	}
	for i := range prompts {
		if got[i].usage.CachedPromptTokens == 0 {
			t.Errorf("session %d reused nothing", i)
		}
		sameGeneration(t, fmt.Sprintf("concurrent session %d", i), want[i], got[i])
	}
}

// TestPrefixCacheRefusals is every configuration the option declines, which is
// where the page-table gap is visible from the outside.
func TestPrefixCacheRefusals(t *testing.T) {
	t.Parallel()
	dir := checkpoint{tie: true}.write(t)
	for _, c := range []struct {
		name string
		opt  Option
		want string
	}{
		{"a process pool below one block", WithPrefixCache(CacheProcess, CacheBlock-1),
			"at least one block"},
		{"an empty session budget", WithPrefixCache(CacheSession, 0),
			"holds at least one"},
		{"a negative session budget", WithPrefixCache(CacheSession, -1),
			"holds at least one"},
		{"a scope that is not one of the three", WithPrefixCache(CacheScope(9), 512),
			"CacheScope(9)"},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			m, err := Open(dir, WithDevice(CPU), c.opt)
			if err == nil {
				_ = m.Close()
				t.Fatal("the configuration was accepted")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("the refusal does not say %q: %v", c.want, err)
			}
		})
	}
}

// TestCacheScopeString covers the names the refusals print.
func TestCacheScopeString(t *testing.T) {
	t.Parallel()
	for _, c := range []struct {
		s    CacheScope
		want string
	}{
		{CacheOff, "off"},
		{CacheSession, "session"},
		{CacheProcess, "process"},
		{CacheScope(-3), "CacheScope(-3)"},
	} {
		if got := c.s.String(); got != c.want {
			t.Errorf("CacheScope(%d).String() = %q, want %q", int8(c.s), got, c.want)
		}
	}
}

// TestPrefixCacheSuffixCarriesTheReusedPositions is specs/016-prefix-cache.md
// §4, asserted on the values the step binds rather than on the answer it
// produces.
//
// A suffix prefilled at position 0 instead of at the end of the reused run
// rotates every query by the wrong angle and scatters every key over the prefix
// it was supposed to keep. That is specs/004-model-graph.md §2.5.1's failure
// reached from the other direction: fluent text, coherence lost, nothing
// refused. An end-to-end comparison catches it only when the model is good
// enough for the difference to reach a token, so this reads the four ports.
func TestPrefixCacheSuffixCarriesTheReusedPositions(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	s := session(t, m, WithSessionContext(cacheCap))

	first := promptIDs(17, 24)
	request(t, s, first, greedy(4))
	full := extend(first, 18, 20, at(s, len(first)))

	first0 := capturePrefill(s)
	got := request(t, s, full, greedy(4))
	base, lengths, rows := first0.base, first0.lengths, first0.rows
	slots, posq, pos := first0.slots, first0.posq, first0.posk
	if got.usage.CachedPromptTokens != len(first) {
		t.Fatalf("reused %d positions and the prompts share %d",
			got.usage.CachedPromptTokens, len(first))
	}

	reused, suffix := len(first), len(full)-len(first)
	if int(base) != reused {
		t.Fatalf("the prefill's base scalar is %d and the reused run ends at %d; the "+
			"causal mask hides pos > base+s, so a base of zero lets the suffix's first "+
			"token see nothing behind it", base, reused)
	}
	if int(lengths) != len(full) {
		t.Fatalf("lengths is %d on a %d-token prompt; the reused prefix is part of the "+
			"cache the suffix attends over", lengths, len(full))
	}
	c := m.cfg
	for i := range suffix {
		if want := uint32(reused + i); slots[i] != want {
			t.Fatalf("suffix token %d scatters to row %d, want %d; a slot below %d "+
				"overwrites the prefix this request is reusing", i, slots[i], want, reused)
		}
		for h := range c.NumHeads {
			if want := uint32(reused + i); posq[i*c.NumHeads+h] != want {
				t.Fatalf("query row %d head %d rotates at position %d, want %d",
					i, h, posq[i*c.NumHeads+h], want)
			}
		}
		for h := range c.NumKVHeads {
			if want := uint32(reused + i); pos[i*c.NumKVHeads+h] != want {
				t.Fatalf("key row %d head %d rotates at position %d, want %d",
					i, h, pos[i*c.NumKVHeads+h], want)
			}
		}
	}
	// The dispatch is the suffix's bucket and not the prompt's. Nothing else
	// in this file reads the shape: fill accepts any rows >= t, a pad row's
	// slot is still the capacity, the last real row's logits are still the
	// ones sliced, and Usage.CachedPromptTokens and the recorder's token count
	// are both host arithmetic over st.reused. A prefill bucketed on the whole
	// prompt therefore answers identically while paying for every position the
	// cache already held -- the win disappears and every other assertion here
	// still passes.
	if want, err := s.buckets.For(suffix); err != nil || rows != want {
		t.Fatalf("a %d-token suffix ran a %d-row plan and its bucket is %d (err %v); "+
			"reuse must shrink the dispatch, not only the count the recorder reports",
			suffix, rows, want, err)
	}
	// The bucket's pad rows still write nothing (007-D3), which reuse must not
	// have quietly turned into a write at the reused offset.
	for i := suffix; i < rows; i++ {
		if int(slots[i]) != s.capacity {
			t.Fatalf("pad row %d scatters to %d and the capacity is %d; an in-range pad "+
				"slot corrupts the cache with a row nobody reads", i, slots[i], s.capacity)
		}
	}
}

// prefillStep is what one submission bound, captured through [Session.submit].
type prefillStep struct {
	base, lengths     uint32
	rows              int
	slots, posq, posk []uint32
}

// capturePrefill wraps the session's submission seam and keeps the first step's
// port data.
//
// The seam exists so a test can inject a device failure; reading a step through
// it is the same access from the other side, and it is the only way to assert
// what a request bound rather than what it produced. The slices are copied
// because the session reuses its own.
func capturePrefill(s *Session) *prefillStep {
	got := &prefillStep{}
	inner := s.submit
	s.submit = func(p *tensor.Plan, b tensor.Bindings) error {
		if got.rows == 0 {
			got.base, got.lengths, got.rows = s.step.base, s.step.lengths[0], len(s.step.slots)
			got.slots = append([]uint32(nil), s.step.slots...)
			got.posq = append([]uint32(nil), s.step.posq...)
			got.posk = append([]uint32(nil), s.step.posk...)
		}
		return inner(p, b)
	}
	return got
}

// TestPrefixCacheAbandonedStreamLeavesTheCacheReadable is the state between
// [Session.start] and the first [Stream.Next].
//
// start rewinds the conversation to the run the new prompt shares, and
// Stream.advance is what refills it. A caller who abandons the stream in
// between leaves a session whose length is the shared run while the cache
// physically holds a longer conversation. That is conservative rather than
// wrong -- every row below the length still holds the token the history names,
// and every row above it is masked by the next step's lengths -- and it is a
// state nothing else in this file enters.
func TestPrefixCacheAbandonedStreamLeavesTheCacheReadable(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	s := session(t, m, WithSessionContext(cacheCap))

	a := promptIDs(19, 30)
	request(t, s, a, greedy(4))

	// Built and dropped: the rewind ran, the prefill did not.
	if _, err := s.start(context.Background(), extend(a, 20, 16, at(s, len(a))),
		greedy(4)); err != nil {
		t.Fatalf("start: %v", err)
	}
	if s.length != len(a) {
		t.Fatalf("the abandoned request left the length at %d, want the %d it shares",
			s.length, len(a))
	}

	// A third prompt that shares only the first ten positions, so the reuse
	// crosses the abandoned request's rewind.
	c := extend(a[:10], 21, 24, a[10])
	got := request(t, s, c, greedy(4))
	if got.usage.CachedPromptTokens != 10 {
		t.Fatalf("reused %d positions and the prompts share 10",
			got.usage.CachedPromptTokens)
	}
	cold := request(t, session(t, m, WithSessionContext(cacheCap)), c, greedy(4))
	sameGeneration(t, "after an abandoned stream", cold, got)
}

// TestPrefixCacheDoesNotWidenTheContext is the refusal boundary, warm and cold.
//
// A reused prefix still occupies its rows, so reuse buys arithmetic and not
// capacity. specs/007-engine.md §7 refuses at the request rather than partway
// through it, and a warm path that admitted a prompt a cold one refuses would
// have moved the refusal into the loop, where 006 §4 says it must not be.
func TestPrefixCacheDoesNotWidenTheContext(t *testing.T) {
	t.Parallel()
	m := warmModel(t)
	warm := session(t, m, WithSessionContext(cacheCap))
	cold := session(t, m, WithSessionContext(cacheCap))

	shared := promptIDs(22, 40)
	request(t, warm, shared, greedy(4))

	for _, c := range []struct {
		name string
		ids  []int
		p    Policy
	}{
		{"a prompt the length of the cache", extend(shared, 23, cacheCap-len(shared), -1),
			greedy(1)},
		{"a prompt and a budget that do not both fit",
			extend(shared, 24, 20, -1), greedy(cacheCap - 59)},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, cerr := cold.start(context.Background(), c.ids, c.p)
			_, werr := warm.start(context.Background(), c.ids, c.p)
			if cerr == nil {
				t.Fatalf("the cold session accepted a %d-token prompt with MaxTokens %d "+
					"in a %d-position cache", len(c.ids), c.p.MaxTokens, cacheCap)
			}
			if werr == nil {
				t.Fatal("reuse admitted a request the cold path refuses; it buys " +
					"arithmetic, not capacity")
			}
			if cerr.Error() != werr.Error() {
				t.Fatalf("the two refusals differ:\ncold %v\nwarm %v", cerr, werr)
			}
		})
	}
	// The warm session is still usable: a refusal changes nothing.
	if got := request(t, warm, shared, greedy(4)); got.usage.CachedPromptTokens !=
		len(shared)-1 {
		t.Fatalf("after two refusals the conversation reused %d of %d positions",
			got.usage.CachedPromptTokens, len(shared)-1)
	}
}
