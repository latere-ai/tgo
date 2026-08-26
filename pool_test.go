// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/bench"
	"github.com/latere-ai/tgo/chat"
)

// The capacity every pooled session in this file runs at.
//
// 80 rather than 96, so it is not [cacheCap] and a test that reached for the
// wrong constant does not still fit. bucketsFor(80) gives {32, 64, 80}, so a
// prompt, its suffix and its cold twin can still land in three different
// prefill shapes, which is what specs/016-prefix-cache.md §6 needs of a
// capacity.
const poolCap = 80

// poolModel is the fixture with the prefix cache on and a context small enough
// that N sessions' worth of it is cheap to allocate.
func poolModel(t *testing.T) *Model {
	t.Helper()
	return openSynthetic(t, WithContext(poolCap), WithPrefixCache(CacheSession, poolCap))
}

// pool builds a pool of n sessions and closes it before the model.
//
// The cleanup is registered after the model's, and t.Cleanup runs last in
// first out, so the pool closes first. That is the order accel requires: it
// closes in order rather than recursively, so a buffer outliving its device is
// a leak the runtime reports rather than one it repairs.
func pool(t *testing.T, m *Model, n int) *Pool {
	t.Helper()
	p, err := m.NewPool(n)
	if err != nil {
		t.Fatalf("NewPool(%d): %v", n, err)
	}
	t.Cleanup(func() {
		if err := p.Close(); err != nil {
			t.Errorf("Pool.Close: %v", err)
		}
	})
	return p
}

// leaseRun runs one request through the pool and releases the lease.
//
// It goes through [Lease.generate], which is the seam [Lease.Chat] and
// [Lease.Complete] both reach once the prompt is tokenized: a test that wants
// to control the token ids exactly cannot go through a template, and §3's
// routing is defined over ids.
func leaseRun(t *testing.T, p *Pool, req PoolRequest, ids []int, pol Policy) generation {
	t.Helper()
	g, _ := leaseRunOn(t, p, req, ids, pol)
	return g
}

// leaseRunOn is [leaseRun] and also the entry the request routed to, which is
// what an eviction assertion reads.
func leaseRunOn(t *testing.T, p *Pool, req PoolRequest, ids []int,
	pol Policy) (generation, *poolEntry) {

	t.Helper()
	l, err := p.Acquire(context.Background(), req)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer l.Release()
	st, err := l.generate(context.Background(), ids, pol)
	if err != nil {
		t.Fatalf("generate on a %d-token prompt: %v", len(ids), err)
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
		toks:  append([]int(nil), l.e.s.history[len(ids):]...),
		usage: st.Usage(),
	}, l.e
}

// coldRun answers the same prompt on a session that has never seen anything,
// which is what a warm answer is compared against (016-D6).
func coldRun(t *testing.T, m *Model, ids []int, pol Policy) generation {
	t.Helper()
	return request(t, session(t, m, WithSessionContext(poolCap)), ids, pol)
}

// continuation is the prompt of the next turn of a conversation: what the
// session already holds, followed by n fresh ids.
//
// The session holds the first turn's prompt and every token it generated but
// the last, so this is exactly the render a caller would send back — and the
// match against it is the whole history rather than only the first prompt,
// which is the multi-turn win specs/019-session-affinity.md §1 is about.
func continuation(prompt []int, g generation, seed, n int) []int {
	out := append([]int(nil), prompt...)
	out = append(out, g.toks...)
	return append(out, promptIDs(seed, n)...)
}

// TestPoolSecondTurnPrefillsOnlyTheSuffix is specs/019-session-affinity.md §7's
// first row and the reason the spec exists.
//
// It is measured rather than asserted structurally, which is what 010-D7 asks
// for: a probe that only checks the code path was taken is what let 016 §9 be
// confidently wrong. The number read is the recorder's prefill token count, and
// the same prompt is run against a pool that has never seen the conversation so
// that the number means something — a suffix of 9 tokens is only a win beside
// the 42 a cold request pays.
func TestPoolSecondTurnPrefillsOnlyTheSuffix(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 2)

	rec := bench.NewRecorder(128)
	req := PoolRequest{Recorder: rec}

	first := promptIDs(31, 18)
	turn1 := leaseRun(t, p, req, first, greedy(6))
	if turn1.usage.CachedPromptTokens != 0 {
		t.Fatalf("the first turn of a conversation reused %d positions",
			turn1.usage.CachedPromptTokens)
	}
	// The first turn is read from the recorder as well, and not only assumed
	// cold from its reuse count. specs/019-session-affinity.md §8.1 prints it
	// as a measured row beside the second turn's, and a table measured on one
	// row and reasoned on the next is the half-probe 010-D7 is about.
	if r := rec.Report(); r.Prefill.Steps != 1 || r.Prefill.Tokens != len(first) {
		t.Fatalf("the first turn prefilled %d tokens in %d step(s); the prompt is %d",
			r.Prefill.Tokens, r.Prefill.Steps, len(first))
	}

	second := continuation(first, turn1, 32, 9)
	held := len(first) + len(turn1.toks)
	rec.Reset()
	turn2 := leaseRun(t, p, req, second, greedy(6))

	if turn2.usage.CachedPromptTokens != held {
		t.Fatalf("the second turn reused %d positions and the session holds %d",
			turn2.usage.CachedPromptTokens, held)
	}
	suffix := len(second) - held
	r := rec.Report()
	// Logged, not derived twice: specs/019-session-affinity.md §8.1 prints
	// these numbers, and a table nothing recomputes is a table that can be
	// wrong with nothing red.
	t.Logf("turn 1 prompt %d, generated %d; turn 2 prompt %d, held %d, prefilled %d in %d step(s)",
		len(first), len(turn1.toks), len(second), held, r.Prefill.Tokens, r.Prefill.Steps)
	if r.Prefill.Steps != 1 || r.Prefill.Tokens != suffix {
		t.Fatalf("the second turn prefilled %d tokens in %d steps; the suffix is %d",
			r.Prefill.Tokens, r.Prefill.Steps, suffix)
	}

	// The same prompt on a pool that never saw the first turn, so the saving
	// is a difference between two measurements rather than one number that
	// looks small.
	cold := bench.NewRecorder(128)
	fresh := pool(t, m, 1)
	coldTurn := leaseRun(t, fresh, PoolRequest{Recorder: cold}, second, greedy(6))
	if n := cold.Report().Prefill.Tokens; n != len(second) {
		t.Fatalf("a cold pool prefilled %d tokens of a %d-token prompt", n, len(second))
	}
	if suffix >= len(second) {
		t.Fatalf("the warm turn prefilled %d tokens and the cold one %d; there is no win "+
			"to measure", suffix, len(second))
	}

	// §6: what was saved was arithmetic, not an answer. The warm turn and the
	// cold one answer the same prompt.
	sameGeneration(t, "a pooled second turn", coldTurn, turn2)
}

// TestPoolWarmAnswerMatchesAnUnpooledSession is §7's second row: the pool
// changes what a request costs and not what it says.
//
// The comparison is against a [Session] the pool never touched, which is the
// unpooled server's path.
func TestPoolWarmAnswerMatchesAnUnpooledSession(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	first := promptIDs(41, 14)
	turn1 := leaseRun(t, p, PoolRequest{}, first, greedy(5))
	second := continuation(first, turn1, 42, 11)
	warm := leaseRun(t, p, PoolRequest{}, second, greedy(5))
	if warm.usage.CachedPromptTokens == 0 {
		t.Fatal("the second turn reused nothing, so this compares two cold runs")
	}
	sameGeneration(t, "a warm pooled turn", coldRun(t, m, second, greedy(5)), warm)
}

// TestPoolEvictsTheColdestConversation is §3.1's reuse distance: a turn hits if
// the number of distinct other conversations served since that conversation's
// last turn is below N.
//
// Two sessions and three conversations, so the third one served evicts the
// first. What is asserted is both halves: the evicted conversation prefills
// whole on its next turn, and the one still resident does not.
func TestPoolEvictsTheColdestConversation(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 2)

	a := promptIDs(51, 12)
	b := promptIDs(52, 13)
	c := promptIDs(53, 15)
	ga := leaseRun(t, p, PoolRequest{}, a, greedy(4))
	gb := leaseRun(t, p, PoolRequest{}, b, greedy(4))
	// The third distinct conversation takes the session A was on: it matches
	// nothing, and A is the coldest by last use.
	gc := leaseRun(t, p, PoolRequest{}, c, greedy(4))
	if gc.usage.CachedPromptTokens != 0 {
		t.Fatalf("a conversation nothing has seen reused %d positions",
			gc.usage.CachedPromptTokens)
	}

	// A's reuse distance is 2, which is not below N, so it misses.
	nextA := continuation(a, ga, 54, 7)
	if got := leaseRun(t, p, PoolRequest{}, nextA, greedy(4)); got.usage.CachedPromptTokens != 0 {
		t.Fatalf("the evicted conversation reused %d positions; two other conversations "+
			"were served since its last turn and the pool holds two sessions",
			got.usage.CachedPromptTokens)
	}
	// C's reuse distance at this point is 1, which is below N, so it hits.
	// Asserted after A's miss on purpose: A's miss is what would have taken
	// C's session if the victim were chosen by history length rather than by
	// last use.
	nextC := continuation(c, gc, 55, 7)
	want := len(c) + len(gc.toks)
	if got := leaseRun(t, p, PoolRequest{}, nextC, greedy(4)); got.usage.CachedPromptTokens != want {
		t.Fatalf("the resident conversation reused %d positions of the %d it holds",
			got.usage.CachedPromptTokens, want)
	}
	// B was served between A and C, so it is neither the victim nor stale.
	nextB := continuation(b, gb, 56, 7)
	if got := leaseRun(t, p, PoolRequest{}, nextB, greedy(4)); got.usage.CachedPromptTokens != 0 {
		t.Fatalf("B reused %d positions after A's miss and C's hit each took a session",
			got.usage.CachedPromptTokens)
	}
}

// TestPoolAMissTakesTheColdestAndNotTheEmptiest is the second half of §3.2.
//
// A request that matches nothing goes to the coldest session by last use, not
// to the one holding least. An empty session is usually empty because it was
// just evicted, so the emptiest is the session most recently claimed by
// somebody who may still be talking; the coldest is the one whose owner is
// least likely to come back.
//
// The two rules disagree here on purpose: the coldest session holds 40
// positions and the warmest holds 10, so a router reading emptiness would take
// the wrong one and this test would read it.
func TestPoolAMissTakesTheColdestAndNotTheEmptiest(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 2)

	long := promptIDs(141, 40)
	short := promptIDs(142, 10)
	fresh := promptIDs(143, 14)
	for _, other := range [][]int{short, fresh} {
		if other[0] == long[0] {
			t.Fatalf("two prompts share a first token, so one would match the other")
		}
	}
	if short[0] == fresh[0] {
		t.Fatal("the short conversation and the new one share a first token")
	}

	gLong := leaseRun(t, p, PoolRequest{}, long, greedy(4))
	gShort := leaseRun(t, p, PoolRequest{}, short, greedy(4))

	// Matches neither, so it lands on the coldest, which is the long one.
	if g := leaseRun(t, p, PoolRequest{}, fresh, greedy(4)); g.usage.CachedPromptTokens != 0 {
		t.Fatalf("a prompt sharing nothing reused %d positions", g.usage.CachedPromptTokens)
	}
	// The warmest conversation is untouched, which is what a router reading
	// emptiness would have got wrong: it holds 10 positions against the other's
	// 40.
	want := len(short) + len(gShort.toks)
	if g := leaseRun(t, p, PoolRequest{}, continuation(short, gShort, 145, 6), greedy(4)); g.usage.CachedPromptTokens != want {
		t.Fatalf("the most recently used conversation reused %d of the %d positions it "+
			"holds; a miss took its session because it held least",
			g.usage.CachedPromptTokens, want)
	}
	// And the coldest is the one that paid.
	if g := leaseRun(t, p, PoolRequest{}, continuation(long, gLong, 144, 6), greedy(4)); g.usage.CachedPromptTokens != 0 {
		t.Fatalf("the coldest conversation kept %d positions; the miss should have taken "+
			"its session", g.usage.CachedPromptTokens)
	}
}

// leaseHold runs one request through the pool and keeps the lease.
//
// It is what a setup step needs when two conversations must end up on two
// different sessions: routing skips a busy entry, so holding the first lease is
// the only thing that stops the second request from landing on it. The caller
// releases.
func leaseHold(t *testing.T, p *Pool, req PoolRequest, ids []int, pol Policy) (*Lease, generation) {
	t.Helper()
	l, err := p.Acquire(context.Background(), req)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	st, err := l.generate(context.Background(), ids, pol)
	if err != nil {
		l.Release()
		t.Fatalf("generate on a %d-token prompt: %v", len(ids), err)
	}
	var b strings.Builder
	for st.Next() {
		b.WriteString(st.Text())
	}
	if err := st.Err(); err != nil {
		l.Release()
		t.Fatalf("stream: %v", err)
	}
	return l, generation{
		text:  b.String(),
		toks:  append([]int(nil), l.e.s.history[len(ids):]...),
		usage: st.Usage(),
	}
}

// TestPoolAShortMatchDoesNotDiscardALongHistory is 019-D4.
//
// Routing chooses what to destroy as much as what to reuse. Two sessions agree
// with the incoming prompt on the same 12 tokens; one of them holds 45 more
// after that and the other holds 12. Taking the long one would win the same 12
// positions and throw away a history another conversation would have hit on, so
// the tie-break is the shorter history.
//
// # Why the setup holds a lease
//
// The two conversations share their own first 12 tokens, because a prompt that
// agrees with two sessions on a run means those two sessions agree with each
// other on it. So the second one, routed normally, would land on the first and
// there would be one session to choose from. Holding the first lease across the
// second request is what puts them on two sessions: routing skips a busy entry.
func TestPoolAShortMatchDoesNotDiscardALongHistory(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 2)

	shared := promptIDs(61, 12)
	// The three tails must disagree at their first token, or the "equal match"
	// this test is built on is not equal.
	long := append(append([]int(nil), shared...), promptIDs(62, 45)...)
	short := append(append([]int(nil), shared...), promptIDs(63, 12)...)
	incoming := append(append([]int(nil), shared...), promptIDs(64, 14)...)
	if long[len(shared)] == incoming[len(shared)] || short[len(shared)] == incoming[len(shared)] ||
		long[len(shared)] == short[len(shared)] {
		t.Fatalf("the three prompts do not diverge at position %d: %d, %d, %d",
			len(shared), long[len(shared)], short[len(shared)], incoming[len(shared)])
	}

	lLong, gLong := leaseHold(t, p, PoolRequest{}, long, greedy(4))
	shortEntry := func() *poolEntry {
		l, _ := leaseHold(t, p, PoolRequest{}, short, greedy(4))
		defer l.Release()
		return l.e
	}()
	longEntry := lLong.e
	lLong.Release()
	if shortEntry == longEntry {
		t.Fatal("both conversations were established on one session, so there is no " +
			"choice to make")
	}

	got, on := leaseRunOn(t, p, PoolRequest{}, incoming, greedy(4))
	if got.usage.CachedPromptTokens != len(shared) {
		t.Fatalf("the incoming prompt reused %d positions and it agrees with both sessions "+
			"on %d", got.usage.CachedPromptTokens, len(shared))
	}
	if on != shortEntry {
		t.Fatalf("a %d-token match landed on the session holding %d positions rather than "+
			"the one holding %d", len(shared), len(longEntry.s.history),
			len(shortEntry.s.history))
	}

	// The long conversation is intact: its next turn still reuses everything
	// it had. Without the tie-break it would have been rewound to 12.
	nextLong := continuation(long, gLong, 65, 6)
	want := len(long) + len(gLong.toks)
	if again := leaseRun(t, p, PoolRequest{}, nextLong, greedy(4)); again.usage.CachedPromptTokens != want {
		t.Fatalf("the long conversation reused %d of the %d positions it held before a "+
			"%d-token match arrived", again.usage.CachedPromptTokens, want, len(shared))
	}
}

// TestPoolAffinityKeyFailsClosed is 019-D3 and §7's fifth row.
//
// The assertion is on the reuse count and never on the answer. A keyed request
// and an unkeyed one produce the same tokens either way, because the prompt is
// the same prompt; what differs is whether the second one paid for the prefill,
// and that difference is what a timing oracle reads. Asserting the answer would
// be a test that passes on a pool with no isolation at all.
func TestPoolAffinityKeyFailsClosed(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 2)

	prompt := promptIDs(71, 16)
	// The unkeyed conversation, on one of the two sessions.
	if g := leaseRun(t, p, PoolRequest{}, prompt, greedy(4)); g.usage.CachedPromptTokens != 0 {
		t.Fatalf("the first request reused %d positions", g.usage.CachedPromptTokens)
	}
	// The same prompt under a key. It must not read the unkeyed session's
	// history, so it goes to the other session cold.
	keyed := PoolRequest{Key: "tenant-a"}
	if g := leaseRun(t, p, keyed, prompt, greedy(4)); g.usage.CachedPromptTokens != 0 {
		t.Fatalf("a keyed request reused %d positions of an unkeyed session's history; "+
			"that is a membership oracle over another caller's prompts",
			g.usage.CachedPromptTokens)
	}
	// Both sessions now hold the same tokens, one keyed and one not, so the
	// two directions below are each measured against a session that would
	// have matched if the key were ignored.
	want := len(prompt) - 1
	if g := leaseRun(t, p, keyed, prompt, greedy(4)); g.usage.CachedPromptTokens != want {
		t.Fatalf("a keyed request reused %d positions of its own key's session, and the "+
			"cap is %d", g.usage.CachedPromptTokens, want)
	}
	if g := leaseRun(t, p, PoolRequest{}, prompt, greedy(4)); g.usage.CachedPromptTokens != want {
		t.Fatalf("an unkeyed request reused %d positions of the unkeyed session, and the "+
			"cap is %d", g.usage.CachedPromptTokens, want)
	}
	// And the other direction: a second key matches neither of the two.
	other := PoolRequest{Key: "tenant-b"}
	if g := leaseRun(t, p, other, prompt, greedy(4)); g.usage.CachedPromptTokens != 0 {
		t.Fatalf("a request under one key reused %d positions of another key's session",
			g.usage.CachedPromptTokens)
	}
}

// TestPoolAnUnkeyedRequestNeverReadsAKeyedSession is the second direction of
// specs/019-session-affinity.md §5's table, and the one
// [TestPoolAffinityKeyFailsClosed] cannot reach.
//
// §5 binds both rows: a keyed request may match a session whose last key is
// equal, and an unkeyed one may match a session with no key "and never one
// with a key". The table test above establishes an unkeyed conversation first,
// so by the time its unkeyed request runs again there is an unkeyed session
// holding the same prompt and the request hits it whichever way the comparison
// is written. The direction is only observable when the *only* session holding
// the prompt is a keyed one, which is what this sets up.
//
// A pool of two, so the miss is a routing decision and not an empty pool: the
// unkeyed request has a never-used session to be routed to, and reuses nothing
// because the one session that holds its prompt is somebody else's key's.
//
// Asserted on the reuse count and never on the answer: both requests send the
// same prompt, so the tokens are the same either way, and what leaks is which
// of them paid for the prefill (019 §5).
func TestPoolAnUnkeyedRequestNeverReadsAKeyedSession(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 2)

	prompt := promptIDs(171, 16)
	keyed := PoolRequest{Key: "tenant-a"}
	if g := leaseRun(t, p, keyed, prompt, greedy(4)); g.usage.CachedPromptTokens != 0 {
		t.Fatalf("the first request reused %d positions of an empty pool",
			g.usage.CachedPromptTokens)
	}
	// The same prompt with no key. One session holds it under "tenant-a" and
	// the other has never served anything, so a reuse count above zero is the
	// keyed session's history being read.
	if g := leaseRun(t, p, PoolRequest{}, prompt, greedy(4)); g.usage.CachedPromptTokens != 0 {
		t.Fatalf("an unkeyed request reused %d positions of a keyed session's history; "+
			"that is a membership oracle over the keyed caller's prompts",
			g.usage.CachedPromptTokens)
	}
	// And the keyed conversation still has its own session to come back to,
	// so the miss above was isolation rather than the pool having been emptied.
	want := len(prompt) - 1
	if g := leaseRun(t, p, keyed, prompt, greedy(4)); g.usage.CachedPromptTokens != want {
		t.Fatalf("the keyed conversation reused %d positions of its own session, and the "+
			"cap is %d", g.usage.CachedPromptTokens, want)
	}
}

// TestPoolReleaseLeavesTheHistoryAtTheValidLength is the postcondition
// [Lease.Release]'s truncation exists for (019-D5).
//
// §8.3 records that the truncation moves nothing today, because
// [Stream.advance] appends to the history and advances the length in the same
// branch. That makes the truncation unobservable through any request:
// [Session.reusable] is bounded by the length as well as by the history, so a
// history longer than the length cannot be routed against. What it does reach
// is the sampler, which reads s.history for the repetition penalties, and any
// later reader that trusts the two to agree.
//
// So this asserts the postcondition directly rather than through a request. The
// state is desynchronised the only way a caller cannot -- by appending to the
// history without advancing the length, which is what a refactor of
// [Stream.advance] would do by accident -- and Release must put it back.
func TestPoolReleaseLeavesTheHistoryAtTheValidLength(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	l, gen := leaseHold(t, p, PoolRequest{}, promptIDs(181, 14), greedy(4))
	s := l.e.s
	valid := s.length
	if valid != len(s.history) {
		t.Fatalf("the session holds %d positions and %d history entries before anything "+
			"is desynchronised", valid, len(s.history))
	}
	if len(gen.toks) == 0 {
		t.Fatal("the request generated nothing, so the history is only the prompt")
	}
	// A position whose key/value state was never written: the row above the
	// length, which is what a step that failed on the device would have left
	// if it had appended before it submitted.
	s.history = append(s.history, s.history[0])
	l.Release()

	if len(s.history) != valid || s.length != valid {
		t.Fatalf("the session went back to the pool holding %d positions and %d history "+
			"entries; the key/value state ends at %d, and every entry above it "+
			"advertises a row no step wrote", s.length, len(s.history), valid)
	}
}

// TestPoolCancelledRequestLeavesOnlyValidPositions is §7's sixth row.
//
// A conversation is cancelled part way through generating. The session goes
// back to the pool, the next request matches it, and the answer must be the one
// a cold session gives.
//
// This holds by an invariant rather than by the truncation: [Stream.advance]
// appends to the history and advances the length in the same branch and only
// after the step returned, so a stream that stopped between steps left the two
// agreeing. The test is here because that invariant is one refactor away from
// being untrue, and what it would break is the *next* request's output rather
// than this one's.
func TestPoolCancelledRequestLeavesOnlyValidPositions(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	prompt := promptIDs(81, 15)
	ctx, cancel := context.WithCancel(context.Background())
	l, err := p.Acquire(ctx, PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	st, err := l.generate(ctx, prompt, greedy(12))
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	// Two events, then the client hangs up. Driven until the events happen
	// rather than until a timer expires: a stream yields nothing between
	// steps, so a test that stopped nudging would park.
	for range 2 {
		if !st.Next() {
			t.Fatalf("the stream ended after %d events with %v", 2, st.Err())
		}
	}
	cancel()
	for st.Next() {
	}
	if !errors.Is(st.Err(), context.Canceled) {
		t.Fatalf("the stream ended with %v and the context was cancelled", st.Err())
	}
	l.Release()

	// The next turn continues from whatever the cancelled one completed, which
	// is what a client resending its transcript would send.
	held := p.entries[0].s
	transcript := append([]int(nil), held.history...)
	next := append(transcript, promptIDs(82, 6)...)
	warm := leaseRun(t, p, PoolRequest{}, next, greedy(5))
	if warm.usage.CachedPromptTokens != len(transcript) {
		t.Fatalf("the turn after a cancellation reused %d of the %d positions the session "+
			"holds", warm.usage.CachedPromptTokens, len(transcript))
	}
	sameGeneration(t, "the turn after a cancellation", coldRun(t, m, next, greedy(5)), warm)
}

// TestPoolFailedRequestReturnsAUsableSession is 019-D5's other half, and the
// one the truncation in [Lease.Release] is actually for.
//
// 007-D5 poisons a session whose step failed: the cache holds a partial write
// of unknown extent, and the session refuses further work until it is reset.
// That is right for a session a caller owns and wrong for one the pool is about
// to hand to somebody else, where it would take a session out of the pool for
// the life of the process.
//
// The extent is not unknown here. [stepData.fill] gives real row i the slot
// first+i and every pad row a slot at the capacity, which tensor.ScatterRows
// drops, so a failed step wrote nothing below the length the step started from.
// Rewinding to that length is therefore exactly the valid prefix, and clearing
// the failure with it is sound.
//
// The failure is injected through the session's submit seam, which is the only
// way to fake a fault that only a driver produces.
func TestPoolFailedRequestReturnsAUsableSession(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	prompt := promptIDs(91, 13)
	// The turn whose positions the failure must not touch. What it answered
	// does not matter; what the session holds afterwards is the whole point,
	// so the history is what is kept.
	leaseRun(t, p, PoolRequest{}, prompt, greedy(4))
	valid := append([]int(nil), p.entries[0].s.history...)

	// The next request fails on its first submission, which is a prefill at
	// the position the valid prefix ends at.
	errDevice := errors.New("the device lost the queue")
	s := p.entries[0].s
	pass := s.submit
	s.submit = func(pl *tensor.Plan, b tensor.Bindings) error { return errDevice }
	failing := append(append([]int(nil), valid...), promptIDs(92, 8)...)
	l, err := p.Acquire(context.Background(), PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	st, err := l.generate(context.Background(), failing, greedy(4))
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	for st.Next() {
	}
	if !errors.Is(st.Err(), errDevice) {
		t.Fatalf("the stream ended with %v and the device failed with %v", st.Err(), errDevice)
	}
	l.Release()
	s.submit = pass

	// Without the rewind in Release the session is still poisoned and this
	// request is refused with ErrSessionFailed, which takes one of the pool's
	// N sessions out of service for the life of the process.
	next := append(append([]int(nil), valid...), promptIDs(93, 9)...)
	l2, err := p.Acquire(context.Background(), PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire after a device failure: %v", err)
	}
	st2, err := l2.generate(context.Background(), next, greedy(4))
	if err != nil {
		l2.Release()
		t.Fatalf("the request after a device failure was refused: %v", err)
	}
	var b strings.Builder
	for st2.Next() {
		b.WriteString(st2.Text())
	}
	if err := st2.Err(); err != nil {
		l2.Release()
		t.Fatalf("the request after a device failure: %v", err)
	}
	warm := generation{text: b.String(),
		toks:  append([]int(nil), l2.e.s.history[len(next):]...),
		usage: st2.Usage()}
	l2.Release()

	if warm.usage.CachedPromptTokens != len(valid) {
		t.Fatalf("the request after a device failure reused %d positions and the valid "+
			"prefix is %d", warm.usage.CachedPromptTokens, len(valid))
	}
	// The reused prefix is the one the failure did not touch, so the answer is
	// the cold one. A failed step that had written below the length would show
	// here and nowhere else.
	sameGeneration(t, "the request after a device failure", coldRun(t, m, next, greedy(4)), warm)
}

// TestPoolTheExtraRequestWaitsRatherThanFailing is §7's seventh row and §4's
// semaphore.
//
// The wait is proved by ordering rather than by a timer. With every token
// taken, nothing can be acquired until one is returned, so the waiter can only
// observe the flag set immediately before the release.
func TestPoolTheExtraRequestWaitsRatherThanFailing(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	const n = 3
	p := pool(t, m, n)

	held := make([]*Lease, 0, n)
	for range n {
		l, err := p.Acquire(context.Background(), PoolRequest{})
		if err != nil {
			t.Fatalf("Acquire: %v", err)
		}
		held = append(held, l)
	}
	// Structural, and the reason the ordering below is not a race: with every
	// token taken there is no path through Acquire that does not block.
	if len(p.sem) != cap(p.sem) {
		t.Fatalf("%d of %d tokens are taken with %d leases held", len(p.sem), cap(p.sem), n)
	}

	var releasing atomic.Bool
	waited := make(chan bool, 1)
	got := make(chan *Lease, 1)
	go func() {
		l, err := p.Acquire(context.Background(), PoolRequest{})
		waited <- releasing.Load()
		if err != nil {
			got <- nil
			return
		}
		got <- l
	}()

	releasing.Store(true)
	held[0].Release()

	l := <-got
	if l == nil {
		t.Fatal("the extra request failed rather than waiting for a session")
	}
	if !<-waited {
		t.Fatal("the extra request was admitted before a session was returned; the pool " +
			"handed out more leases than it holds sessions")
	}
	l.Release()
	for _, h := range held[1:] {
		h.Release()
	}
}

// TestPoolAcquireStopsWaitingWhenTheCallerGivesUp is the other end of the
// semaphore: a client that hangs up while queued does not hold a token.
func TestPoolAcquireStopsWaitingWhenTheCallerGivesUp(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	l, err := p.Acquire(context.Background(), PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	errs := make(chan error, 1)
	go func() {
		_, err := p.Acquire(ctx, PoolRequest{})
		errs <- err
	}()
	cancel()
	if err := <-errs; !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled waiter got %v", err)
	}
	l.Release()
	// The token the abandoned waiter did not take is still there.
	l2, err := p.Acquire(context.Background(), PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire after an abandoned wait: %v", err)
	}
	l2.Release()
}

// TestPoolConcurrentRoutingKeepsOneOwnerPerSession is §7's last row.
//
// Under -race, what it covers is the boundary between the pool's lock and the
// session's absence of one: [Pool.route] reads a session's history under p.mu,
// and [Stream.advance] writes that history under no lock at all. The two do not
// meet because route skips a busy entry, and busy is set inside route and
// cleared inside Release, both under p.mu. If that ever stops being true this
// is where it shows.
func TestPoolConcurrentRoutingKeepsOneOwnerPerSession(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 3)

	var mu sync.Mutex
	owner := map[*poolEntry]int{}

	var wg sync.WaitGroup
	for g := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for turn := range 2 {
				l, err := p.Acquire(context.Background(), PoolRequest{})
				if err != nil {
					t.Errorf("goroutine %d: Acquire: %v", g, err)
					return
				}
				st, err := l.generate(context.Background(),
					promptIDs(100+g*10+turn, 6), greedy(2))
				if err != nil {
					l.Release()
					t.Errorf("goroutine %d: generate: %v", g, err)
					return
				}
				mu.Lock()
				if prev, busy := owner[l.e]; busy {
					mu.Unlock()
					l.Release()
					t.Errorf("goroutine %d took a session goroutine %d is using", g, prev)
					return
				}
				owner[l.e] = g
				mu.Unlock()

				for st.Next() {
				}
				err = st.Err()

				// The release is inside the lock: an entry is busy until
				// Release clears the flag, so deleting first would leave a
				// window in which a broken busy flag could hand this session
				// to another goroutine with nothing watching. Release takes
				// p.mu briefly and returns a token to a buffered channel that
				// this lease's own token left room in, so it cannot block.
				mu.Lock()
				delete(owner, l.e)
				l.Release()
				mu.Unlock()
				if err != nil {
					t.Errorf("goroutine %d: stream: %v", g, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// TestPoolARenderThatDivergesEarlyCollapsesTheMatch is why a lease may
// reassign a session's tools and thinking flag in place.
//
// [Model.NewSession] fixes both, and both are rendered into the prompt, so a
// pooled session configured for one tool set cannot serve a request with
// another without a reset. Reassigning them on the lease is sound only because
// routing compares *rendered token ids*: tools go into the system turn, which
// is the head of the prompt, so a changed tool set diverges at the head and the
// match collapses on its own. Nothing has to notice the change.
//
// This asserts the mechanism at the ids, because the fixture tokenizer carries
// no <tool_call> control token and so cannot render a tool spec at all
// (specs/003-chat-template.md's tool turn is the one part of the template this
// fixture cannot reach). What is asserted is the property the tool set relies
// on: a prompt that diverges at position 4 reuses 4 positions and no more, and
// its answer is the cold one. [TestPoolThinkingIsReassignedOnTheLease] carries
// the same argument through a real render.
func TestPoolARenderThatDivergesEarlyCollapsesTheMatch(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	// A rendered prompt is a system turn and then the conversation, so a
	// changed system turn is a divergence a few tokens in with a long tail
	// behind it. That is the shape, and 4 is the head the two renders share.
	const head = 4
	first := promptIDs(121, 30)
	other := append(append([]int(nil), first[:head]...), promptIDs(122, 26)...)
	if other[head] == first[head] {
		t.Fatalf("the two prompts do not diverge at position %d", head)
	}

	if g := leaseRun(t, p, PoolRequest{}, first, greedy(4)); g.usage.CachedPromptTokens != 0 {
		t.Fatalf("the first request reused %d positions", g.usage.CachedPromptTokens)
	}
	got := leaseRun(t, p, PoolRequest{}, other, greedy(4))
	if got.usage.CachedPromptTokens != head {
		t.Fatalf("a prompt that diverges at position %d reused %d positions",
			head, got.usage.CachedPromptTokens)
	}
	sameGeneration(t, "a prompt that diverges early",
		coldRun(t, m, other, greedy(4)), got)
}

// TestPoolALeaseReassignsTheSessionsRenderOptions is the other half of that
// argument: the reassignment happens, so the session's own fields cannot say
// one thing while its history says another.
func TestPoolALeaseReassignsTheSessionsRenderOptions(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	tools := []chat.ToolSpec{{Name: "lookup", Description: "look a thing up",
		InputSchema: []byte(`{"type":"object"}`)}}
	l, _ := leaseHold(t, p, PoolRequest{Thinking: true, Tools: tools},
		promptIDs(131, 10), greedy(3))
	if !l.e.s.thinking || len(l.e.s.tools) != 1 {
		t.Fatalf("the lease left the session with thinking=%v and %d tools",
			l.e.s.thinking, len(l.e.s.tools))
	}
	l.Release()

	l2, _ := leaseHold(t, p, PoolRequest{}, promptIDs(132, 10), greedy(3))
	if l2.e.s.thinking || len(l2.e.s.tools) != 0 {
		t.Fatalf("a request that declared no tools left the session with thinking=%v and "+
			"%d tools; the next Session.Chat on it would render a turn this request "+
			"never asked for", l2.e.s.thinking, len(l2.e.s.tools))
	}
	l2.Release()
}

// TestPoolThinkingIsReassignedOnTheLease is the same argument for the other
// render option, which lands at the other end of the prompt.
//
// The thinking flag emits a pre-closed block at the generation hint, which is
// the tail of the render, so the two prompts share a long head and the reuse is
// large. That is correct, and it is the case where reassigning the session's
// flag in place could be wrong without the ids saying so: the assertion is that
// the answer is the one a session built with that flag gives.
func TestPoolThinkingIsReassignedOnTheLease(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	msgs := []chat.Message{{Role: chat.User,
		Blocks: []chat.Block{{Type: chat.BlockText, Text: "think about it"}}}}

	leaseChat(t, p, PoolRequest{Thinking: true}, msgs, greedy(4))
	off := leaseChat(t, p, PoolRequest{Thinking: false}, msgs, greedy(4))
	if off.usage.CachedPromptTokens == 0 {
		t.Fatal("turning thinking off shared nothing with the turn before it; the flag " +
			"renders at the generation hint, so the system and user turns are identical")
	}
	s := session(t, m, WithSessionContext(poolCap), WithThinking(false))
	if cold := chatTurn(t, s, msgs, greedy(4)); cold.text != off.text {
		t.Fatalf("a pooled request with thinking off answered %q and a session built with "+
			"thinking off answers %q", off.text, cold.text)
	}
}

// leaseChat runs one [Lease.Chat] turn through the pool and releases the lease.
func leaseChat(t *testing.T, p *Pool, req PoolRequest, msgs []chat.Message,
	pol Policy) generation {

	t.Helper()
	l, err := p.Acquire(context.Background(), req)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer l.Release()
	st, err := l.Chat(context.Background(), msgs, pol)
	if err != nil {
		t.Fatalf("Lease.Chat: %v", err)
	}
	var b strings.Builder
	for st.Next() {
		if st.Event().Kind == TextDelta {
			b.WriteString(st.Text())
		}
	}
	if err := st.Err(); err != nil {
		t.Fatalf("Lease.Chat stream: %v", err)
	}
	if l.Reused() != st.Usage().CachedPromptTokens {
		t.Fatalf("Lease.Reused is %d and the usage says %d positions were reused",
			l.Reused(), st.Usage().CachedPromptTokens)
	}
	return generation{text: b.String(), usage: st.Usage()}
}

// TestPoolCompleteRoutesLikeChat covers the raw-text entry, which is what
// /v1/completions reaches.
func TestPoolCompleteRoutesLikeChat(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	run := func(prompt string) generation {
		t.Helper()
		l, err := p.Acquire(context.Background(), PoolRequest{})
		if err != nil {
			t.Fatalf("Acquire: %v", err)
		}
		defer l.Release()
		st, err := l.Complete(context.Background(), prompt, greedy(4))
		if err != nil {
			t.Fatalf("Lease.Complete: %v", err)
		}
		var b strings.Builder
		for st.Next() {
			b.WriteString(st.Text())
		}
		if err := st.Err(); err != nil {
			t.Fatalf("Lease.Complete stream: %v", err)
		}
		return generation{text: b.String(), usage: st.Usage()}
	}

	first := run("the quick brown fox jumps over")
	if first.usage.CachedPromptTokens != 0 {
		t.Fatalf("the first completion reused %d positions", first.usage.CachedPromptTokens)
	}
	again := run("the quick brown fox jumps over")
	if want := first.usage.PromptTokens - 1; again.usage.CachedPromptTokens != want {
		t.Fatalf("an identical completion reused %d positions and the cap is %d",
			again.usage.CachedPromptTokens, want)
	}
}

// TestPoolReusedIsZeroBeforeARequestRuns pins the number an isolation test
// reads on a lease that has not generated. It is a real case: a request refused
// by [Lease.generate] releases without having routed.
func TestPoolReusedIsZeroBeforeARequestRuns(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)
	l, err := p.Acquire(context.Background(), PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if n := l.Reused(); n != 0 {
		t.Fatalf("a lease that has not generated reports %d reused positions", n)
	}
	if _, err := l.generate(context.Background(), nil, greedy(2)); err == nil {
		t.Fatal("an empty prompt was accepted; there is nothing to condition on")
	}
	if n := l.Reused(); n != 0 {
		t.Fatalf("a lease whose request was refused reports %d reused positions", n)
	}
	l.Release()
}

// TestPoolRefusals is every way the pool says no.
func TestPoolRefusals(t *testing.T) {
	t.Parallel()
	m := poolModel(t)

	if _, err := m.NewPool(0); err == nil {
		t.Fatal("a pool of no sessions was accepted; it holds no conversation")
	}

	p := pool(t, m, 1)
	if got := p.Size(); got != 1 {
		t.Fatalf("Size is %d for a pool of one", got)
	}
	if _, err := p.Acquire(nil, PoolRequest{}); err == nil { //nolint:staticcheck // the nil is the case
		t.Fatal("a nil context was accepted")
	}
	done, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Acquire(done, PoolRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled context got %v", err)
	}

	l, err := p.Acquire(context.Background(), PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	l.Release()
	// A second release is a no-op rather than a second token: without the
	// guard, a handler with two defers would grow the pool past its size.
	l.Release()
	if len(p.sem) != 0 {
		t.Fatalf("%d tokens are still taken after one lease was released twice", len(p.sem))
	}
	if _, err := l.Chat(context.Background(), nil, greedy(1)); err == nil {
		t.Fatal("a released lease accepted a request; its session is another request's")
	}
	if _, err := l.Complete(context.Background(), "hi", greedy(1)); err == nil {
		t.Fatal("a released lease accepted a completion")
	}
}

// TestPoolClosedPoolRefusesAndClosesOnce covers the shutdown path.
func TestPoolClosedPoolRefusesAndClosesOnce(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p, err := m.NewPool(2)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("Pool.Close: %v", err)
	}
	// Idempotent: a shutdown that closed the pool and then failed on its way
	// out must not close every buffer twice.
	if err := p.Close(); err != nil {
		t.Fatalf("a second Pool.Close: %v", err)
	}
	if _, err := p.Acquire(context.Background(), PoolRequest{}); err == nil {
		t.Fatal("a closed pool handed out a lease")
	}
	if len(p.sem) != 0 {
		t.Fatalf("%d tokens are held after a closed pool refused a lease", len(p.sem))
	}
}

// TestPoolClosedWithALeaseOutReportsIt is the shutdown that ran out of grace.
//
// The memory is released either way, because leaking N sessions' key/value
// cache is the worse answer, and the operator is told that a request was still
// generating.
func TestPoolClosedWithALeaseOutReportsIt(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p, err := m.NewPool(1)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	l, err := p.Acquire(context.Background(), PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	// Routed but not generating: the entry is marked busy and nothing is in
	// flight on the device, which is what makes this safe to close under.
	if _, err := l.generate(context.Background(), promptIDs(111, 8), greedy(1)); err != nil {
		t.Fatalf("generate: %v", err)
	}
	err = p.Close()
	if err == nil {
		t.Fatal("a pool closed with a lease outstanding reported nothing")
	}
	if !strings.Contains(err.Error(), "still") {
		t.Fatalf("Pool.Close reported %v, which does not name the outstanding lease", err)
	}
}

// TestNewPoolReportsWhichSessionItCouldNotReserve is 019-D2 at startup: N
// sessions' cache is what the process will hold for its life, so a device that
// cannot hold it says so now rather than under load.
//
// The device is closed before the pool is built, which is the only way to make
// a buffer allocation fail on the CPU backend without a fake device.
func TestNewPoolReportsWhichSessionItCouldNotReserve(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	if err := m.Close(); err != nil {
		t.Fatalf("Model.Close: %v", err)
	}
	p, err := m.NewPool(4)
	if err == nil {
		_ = p.Close()
		t.Fatal("a pool was reserved on a closed device")
	}
	if !strings.Contains(err.Error(), "of 4") {
		t.Fatalf("NewPool failed with %v, which does not say which of the four sessions "+
			"could not be reserved", err)
	}
}

// TestPoolAnAbandonedStreamIsOverWhenTheLeaseIsReleased.
//
// A caller who stopped reading has stopped reading, and the session is another
// request's the moment the lease goes back. A Next() on the old stream would
// run a step on somebody else's conversation, so releasing ends it.
func TestPoolAnAbandonedStreamIsOverWhenTheLeaseIsReleased(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	prompt := promptIDs(151, 12)
	l, err := p.Acquire(context.Background(), PoolRequest{})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	st, err := l.generate(context.Background(), prompt, greedy(10))
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if !st.Next() {
		t.Fatalf("the stream produced nothing: %v", st.Err())
	}
	held := len(p.entries[0].s.history)
	l.Release()

	if st.Next() {
		t.Fatalf("an abandoned stream produced %q after its lease went back to the pool",
			st.Text())
	}
	// And the session went back holding what it actually computed, which the
	// next request reuses.
	next := append(append([]int(nil), p.entries[0].s.history...), promptIDs(152, 6)...)
	got := leaseRun(t, p, PoolRequest{}, next, greedy(4))
	if got.usage.CachedPromptTokens != held {
		t.Fatalf("the request after an abandoned one reused %d of the %d positions the "+
			"session holds", got.usage.CachedPromptTokens, held)
	}
}

// TestPoolChatReportsARenderThisTokenizerCannotEncode.
//
// The fixture's vocabulary carries no <tool_call> marker, so a request that
// declares a tool renders a control token the tokenizer does not hold. That is
// [Model.encode]'s refusal (003-D4) reached through a lease, and the point here
// is that it is a refusal rather than characters: text that reads like a turn
// marker must never encode as one.
func TestPoolChatReportsARenderThisTokenizerCannotEncode(t *testing.T) {
	t.Parallel()
	m := poolModel(t)
	p := pool(t, m, 1)

	l, err := p.Acquire(context.Background(), PoolRequest{Tools: []chat.ToolSpec{{
		Name: "lookup", InputSchema: []byte(`{"type":"object"}`)}}})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer l.Release()
	_, err = l.Chat(context.Background(), []chat.Message{{Role: chat.User,
		Blocks: []chat.Block{{Type: chat.BlockText, Text: "hello"}}}}, greedy(2))
	if err == nil || !strings.Contains(err.Error(), "control token") {
		t.Fatalf("Lease.Chat = %v; it should name the control token the vocabulary lacks", err)
	}
}
