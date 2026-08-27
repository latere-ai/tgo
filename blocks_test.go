// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"strings"
	"testing"

	"golang.design/x/accel"
)

// blockPoolCap is the shared pool's size, in positions. Whole blocks, and large
// enough that two conversations fit beside each other.
const blockPoolCap = 8 * CacheBlock

// sharedCap is a session's context in the shared tests: room for a system
// prompt of two whole blocks and a continuation after it.
//
// Both constants are as small as the properties allow. Every test here runs a
// real forward pass on the CPU backend, the race gate costs about an order of
// magnitude on that arithmetic, and CONTRIBUTING measures that in CPU seconds
// rather than wall -- so a fixture larger than the assertion needs is a gate
// that goes red on a slower runner.
const sharedCap = 4 * CacheBlock

// sharedModel is the fixture with a process-scoped block pool.
func sharedModel(t *testing.T, opts ...Option) *Model {
	t.Helper()
	return openSynthetic(t, append([]Option{
		WithPrefixCache(CacheProcess, blockPoolCap)}, opts...)...)
}

// TestABlockPoolGeneratesWhatAContiguousCacheDoes is the first half of this
// wave, and it deliberately asserts nothing about sharing.
//
// A session that leases blocks from a pool addresses its cache through a page
// table, writes its key and value state into blocks the pool chose, and reads
// them back through that table. Every one of those is a chance to be off by a
// block. So before any prefix is matched, the question is whether the plain
// forward pass still produces the same tokens -- and the answer has to come
// from a run, because an addressing error produces fluent text rather than an
// error.
func TestABlockPoolGeneratesWhatAContiguousCacheDoes(t *testing.T) {
	t.Parallel()
	ids := promptIDs(1, CacheBlock+2)

	plain := openSynthetic(t)
	want := request(t, session(t, plain, WithSessionContext(cacheCap)), ids, greedy(4))

	paged := sharedModel(t)
	got := request(t, session(t, paged, WithSessionContext(cacheCap)), ids, greedy(4))

	sameGeneration(t, "a pooled session against a contiguous one", want, got)
}

// TestTwoConversationsShareOneSystemPrompt is what the pool is for, and what no
// session-scoped cache can do at any size.
//
// Two sessions, the same leading tokens, different continuations. The second
// must reuse what the first computed, and the reuse is block-aligned, so what
// is asserted is that it reused a whole number of blocks covering the common
// run rather than an exact token count (016 §3).
func TestTwoConversationsShareOneSystemPrompt(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)

	system := promptIDs(1, 2*CacheBlock)

	first := session(t, m, WithSessionContext(sharedCap))
	one := request(t, first, extend(system, 7, 10, 0), greedy(1))
	if one.usage.CachedPromptTokens != 0 {
		t.Fatalf("the first conversation reused %d positions from an empty pool",
			one.usage.CachedPromptTokens)
	}

	// A different conversation, in a session that has never seen these tokens.
	second := session(t, m, WithSessionContext(sharedCap))
	two := request(t, second, extend(system, 11, 10, 0), greedy(1))

	if two.usage.CachedPromptTokens == 0 {
		t.Fatalf("a second conversation sharing %d leading tokens reused none of "+
			"them; a process-scoped pool that shares nothing is a session-scoped "+
			"one with a page table attached", len(system))
	}
	if two.usage.CachedPromptTokens%CacheBlock != 0 {
		t.Fatalf("the reuse is %d positions and sharing is block-aligned at %d",
			two.usage.CachedPromptTokens, CacheBlock)
	}
	if two.usage.CachedPromptTokens > len(system) {
		t.Fatalf("the second conversation reused %d positions and the two share %d",
			two.usage.CachedPromptTokens, len(system))
	}
	// The blocks it did not reuse are the ones the shared run does not cover.
	if want := len(system) / CacheBlock * CacheBlock; two.usage.CachedPromptTokens != want {
		t.Fatalf("the two share %d leading tokens, which is %d whole blocks, and the "+
			"second reused %d", len(system), want, two.usage.CachedPromptTokens)
	}
	t.Logf("a second conversation reused %d of %d prompt tokens it never computed",
		two.usage.CachedPromptTokens, two.usage.PromptTokens)
}

// TestASharedHitGeneratesWhatAColdRunDoes is the correctness half of the test
// above, and the one that would catch a page table read a block out.
//
// Reusing another conversation's blocks means attending to rows this session
// never wrote. If the table were wrong the answer would still be fluent, so the
// only check that means anything is against a run that computed everything
// itself.
func TestASharedHitGeneratesWhatAColdRunDoes(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	system := promptIDs(1, 2*CacheBlock)
	prompt := extend(system, 11, 10, 0)

	// Cold: a pool that has never seen the system prompt.
	cold := request(t, session(t, sharedModel(t), WithSessionContext(sharedCap)),
		prompt, greedy(4))
	if cold.usage.CachedPromptTokens != 0 {
		t.Fatalf("the cold run reused %d positions", cold.usage.CachedPromptTokens)
	}

	// Warm: another conversation computed the system prompt first.
	request(t, session(t, m, WithSessionContext(sharedCap)),
		extend(system, 7, 10, 0), greedy(4))
	warm := request(t, session(t, m, WithSessionContext(sharedCap)), prompt, greedy(4))
	if warm.usage.CachedPromptTokens == 0 {
		t.Fatal("the warm run reused nothing, so this measures no sharing")
	}
	sameGeneration(t, "a hit on another conversation's blocks", cold, warm)
}

// TestASaltKeepsTwoConversationsApart is 016 §7's oracle, closed.
//
// A hit is faster than a miss and that timing is observable, so a shared pool
// is a membership test over other conversations' prompts. The salt is mixed
// into every block hash, so a conversation under one salt cannot see a block
// another salt wrote -- and this asserts the direction that matters, which is
// the unsalted one: a probe with no salt must not be able to tell whether a
// salted caller's prompt exists.
func TestASaltKeepsTwoConversationsApart(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	system := promptIDs(1, 2*CacheBlock)

	salted := session(t, m, WithSessionContext(sharedCap), WithCacheSalt("tenant-a"))
	request(t, salted, extend(system, 7, 10, 0), greedy(1))

	for _, c := range []struct {
		name string
		opts []SessionOption
	}{
		{"an unsalted probe", nil},
		{"a differently salted probe", []SessionOption{WithCacheSalt("tenant-b")}},
	} {
		t.Run(c.name, func(t *testing.T) {
			probe := session(t, m, append([]SessionOption{
				WithSessionContext(sharedCap)}, c.opts...)...)
			got := request(t, probe, extend(system, 11, 10, 0), greedy(1))
			if got.usage.CachedPromptTokens != 0 {
				t.Fatalf("%s reused %d positions of a prompt computed under another "+
					"salt; the timing of that hit is a membership oracle over it",
					c.name, got.usage.CachedPromptTokens)
			}
		})
	}

	// And the same salt still shares, or the test above passes because nothing
	// shares at all.
	same := session(t, m, WithSessionContext(sharedCap), WithCacheSalt("tenant-a"))
	got := request(t, same, extend(system, 13, 10, 0), greedy(1))
	if got.usage.CachedPromptTokens == 0 {
		t.Fatal("a conversation under the same salt reused nothing, so the two " +
			"refusals above prove no separation")
	}
}

// TestAPooledSessionReleasesItsBlocks: a session that closed while holding a
// lease would take blocks out of a pool every other conversation draws from.
func TestAPooledSessionReleasesItsBlocks(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	before := m.blocks.pool.Stats()

	s := session(t, m, WithSessionContext(sharedCap))
	request(t, s, promptIDs(1, CacheBlock+2), greedy(2))
	if err := s.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	after := m.blocks.pool.Stats()
	if after.InUse != before.InUse {
		t.Fatalf("%d blocks are still leased after the session closed, and %d were "+
			"before; a leaked block is one no other conversation can have",
			after.InUse, before.InUse)
	}
	// The blocks are not free either: a block at refcount zero that a hash
	// entry still names is the cache, and it stays until the pool needs it.
	if after.Cached == 0 {
		t.Fatal("closing the session freed every block outright; a released block " +
			"whose hash entry still names it is what the next conversation matches")
	}
}

// TestAPromptLargerThanThePoolIsRefused, and named rather than silently
// truncated.
//
// The session's own context is larger than the pool here on purpose: the two
// are different limits now, and a request that fits one and not the other has
// to be refused by the one it does not fit rather than reaching a lease that
// covers part of it.
func TestAPromptLargerThanThePoolIsRefused(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	// Past the pool and inside the context, so the pool is the limit that
	// answers rather than the context reaching it first.
	s := session(t, m, WithSessionContext(blockPoolCap*2))
	ids := promptIDs(1, blockPoolCap+CacheBlock)
	if _, err := s.start(context.Background(), ids, greedy(1)); err == nil {
		t.Fatal("a prompt longer than the pool was accepted")
	} else if !strings.Contains(err.Error(), "pool holds") {
		t.Fatalf("the refusal does not name the pool: %v", err)
	}
}

// TestAnUnsizedPageTableIsRefused is the regression for the defect that made
// TestABlockPoolGeneratesWhatAContiguousCacheDoes red the first time it ran.
//
// NewSession built its host-side step slices twice: the paged branch sized the
// page table, and the shared initialisation below it replaced the whole struct
// and dropped it. WriteBuffer over an empty slice writes nothing and reports
// nothing, so the port kept whatever the allocation held and every step
// attended to blocks nobody chose. Nothing failed. The output stayed fluent and
// simply was not the answer.
//
// The fix was one line in the constructor and would have been invisible without
// a check at the seam, so this asserts the check rather than the constructor: a
// short table is a named refusal at the step that would have used it.
func TestAnUnsizedPageTableIsRefused(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	s := session(t, m, WithSessionContext(sharedCap))
	if len(s.step.pages) != m.blocks.maxPages() {
		t.Fatalf("a new pooled session sized its page table %d, and the port declares %d",
			len(s.step.pages), m.blocks.maxPages())
	}

	// The state the constructor used to leave behind.
	s.step.pages = nil
	_, _, err := s.run(1, []int{1}, 0)
	if err == nil {
		t.Fatal("a step with no page table binding ran; it would have attended to " +
			"whatever the port happened to hold, and produced text rather than an error")
	}
	if !strings.Contains(err.Error(), "page table binding") {
		t.Fatalf("the refusal does not name the binding: %v", err)
	}
}

// TestAPooledRequestCarriesItsKeyIntoTheBlockPool: 019's affinity key and 016's
// block salt are the same string, and this is why they have to be.
//
// The pool routes a request only to a session whose last request carried the
// same key, and a request that matched nothing gets a session emptied of its
// history. That bounds what one request can see of another *through a session*.
// A process-scoped block pool is a second way to see it, one layer down, so a
// key that stopped at the session boundary would exclude a request from a
// conversation's history and hand it the same tokens through the blocks.
func TestAPooledRequestCarriesItsKeyIntoTheBlockPool(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	p, err := m.NewPool(2)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	t.Cleanup(func() {
		if err := p.Close(); err != nil {
			t.Errorf("Pool.Close: %v", err)
		}
	})

	l, err := p.Acquire(context.Background(), PoolRequest{Key: "tenant-a"})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	st, err := l.generate(context.Background(), promptIDs(1, CacheBlock+2), greedy(1))
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	got := l.e.s.salt
	for st.Next() {
	}
	l.Release()

	if got != "tenant-a" {
		t.Fatalf("the session's block salt is %q and the request's key is %q; the two "+
			"bound the same thing and a request excluded from a session's history "+
			"would reach the same tokens through the pool", got, "tenant-a")
	}
}

// TestAnIdleConversationPinsNoBlocks is the lease lifetime, and it is the one
// thing about a shared pool that a per-session cache never had to decide.
//
// A lease is a refcount and not the key/value state. Every complete block is
// published as it is computed, so giving the lease back at the end of a stream
// keeps the blocks in the pool and the next request that shares the prefix --
// including this conversation's own next turn -- finds them by hash. Holding it
// between requests would instead make an idle conversation compete with running
// ones for the single resource the process shares, which with B blocks over N
// sessions is how a pool deadlocks.
//
// Both halves are asserted, because either alone passes on a broken pool: an
// idle session holds nothing, *and* its next turn still reuses.
func TestAnIdleConversationPinsNoBlocks(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	s := session(t, m, WithSessionContext(sharedCap))

	first := promptIDs(1, 2*CacheBlock)
	request(t, s, first, greedy(2))

	if got := m.blocks.pool.Stats().InUse; got != 0 {
		t.Fatalf("a conversation that finished its request holds %d blocks; a "+
			"reference held by an idle session is a block no live conversation "+
			"can have", got)
	}
	if got := m.blocks.pool.Stats().Cached; got == 0 {
		t.Fatal("releasing the lease freed every block outright; a released block " +
			"whose hash entry still names it is what the next request matches")
	}

	two := request(t, s, extend(first, 5, 8, 0), greedy(2))
	if two.usage.CachedPromptTokens == 0 {
		t.Fatal("the next turn reused nothing, so the blocks were released and lost " +
			"rather than released and kept")
	}
	if want := len(first); two.usage.CachedPromptTokens != want {
		t.Fatalf("the next turn reused %d positions of a %d-token opening it had "+
			"already paid for", two.usage.CachedPromptTokens, want)
	}
}

// TestARefusedSharedRequestLeavesTheConversationReusable: a request that does
// not fit the pool leaves the conversation exactly as it found it, which is the
// rule [Session.start] states above the rewind and which the lease sits below.
func TestARefusedSharedRequestLeavesTheConversationReusable(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	s := session(t, m, WithSessionContext(blockPoolCap*2))

	first := promptIDs(1, 2*CacheBlock)
	request(t, s, first, greedy(2))

	if _, err := s.start(context.Background(), promptIDs(2, blockPoolCap+CacheBlock),
		greedy(1)); err == nil {
		t.Fatal("a prompt longer than the pool was accepted")
	}

	two := request(t, s, extend(first, 5, 8, 0), greedy(2))
	if two.usage.CachedPromptTokens != len(first) {
		t.Fatalf("the turn after a refused request reused %d positions of a %d-token "+
			"opening; the refusal cost the conversation what it had already paid for",
			two.usage.CachedPromptTokens, len(first))
	}
}

// TestTheSharedPoolIsNarrow is C5's whole argument, asserted rather than
// commented.
//
// The key and value states are the largest allocation a serving process has
// after the weights, and the only one that scales with *both* concurrency and
// context. Halving them is twice the blocks, twice the prefixes worth keeping,
// and — by 008 §1, where the throughput ceiling is proportional to 1/A — twice
// the batch size worth reaching.
//
// It cost six hours rather than a wave: C24 was accel selecting the f16 prefill
// kernel and then overwriting the selection whenever a page table was supplied,
// and a pool is paged by construction.
func TestTheSharedPoolIsNarrow(t *testing.T) {
	t.Parallel()
	m := sharedModel(t)
	if got := m.blocks.dtype; got != accel.F16 {
		t.Fatalf("the shared pool holds %v; a narrow cache is twice the blocks for "+
			"the same bytes, which is what C5 closed for", got)
	}
	for _, b := range []*accel.Buffer{m.blocks.keys, m.blocks.values} {
		if got := b.DType(); got != accel.F16 {
			t.Errorf("a pool state is %v", got)
		}
	}
	// And the graph is told, or ScatterRows would refuse the pair split apart:
	// one kernel reads the rows and writes the state, so an f16 state needs
	// f16 rows.
	s := session(t, m, WithSessionContext(sharedCap))
	if got := s.cacheDType(); got != accel.F16 {
		t.Fatalf("a pooled session records a %v cache while the pool holds f16", got)
	}
	// A session that owns its cache keeps f32: it is sized to one conversation,
	// so halving it buys one conversation's memory rather than the allocation
	// that grows with concurrency.
	own := session(t, openSynthetic(t), WithSessionContext(cacheCap))
	if got := own.cacheDType(); got != accel.F32 {
		t.Fatalf("an unshared session records a %v cache", got)
	}
}
