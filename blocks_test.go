// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"strings"
	"testing"
)

// blockPoolCap is the shared pool's size, in positions. Whole blocks, and large
// enough that two conversations fit beside each other.
const blockPoolCap = 16 * CacheBlock

// sharedCap is a session's context in the shared tests: room for a system
// prompt of several whole blocks and a continuation after it.
const sharedCap = 8 * CacheBlock

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
	ids := promptIDs(1, 40)

	plain := openSynthetic(t)
	want := request(t, session(t, plain, WithSessionContext(cacheCap)), ids, greedy(8))

	paged := sharedModel(t)
	got := request(t, session(t, paged, WithSessionContext(cacheCap)), ids, greedy(8))

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

	system := promptIDs(1, 3*CacheBlock)

	first := session(t, m, WithSessionContext(sharedCap))
	one := request(t, first, extend(system, 7, 10, 0), greedy(4))
	if one.usage.CachedPromptTokens != 0 {
		t.Fatalf("the first conversation reused %d positions from an empty pool",
			one.usage.CachedPromptTokens)
	}

	// A different conversation, in a session that has never seen these tokens.
	second := session(t, m, WithSessionContext(sharedCap))
	two := request(t, second, extend(system, 11, 10, 0), greedy(4))

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
	system := promptIDs(1, 3*CacheBlock)
	prompt := extend(system, 11, 10, 0)

	// Cold: a pool that has never seen the system prompt.
	cold := request(t, session(t, sharedModel(t), WithSessionContext(sharedCap)),
		prompt, greedy(8))
	if cold.usage.CachedPromptTokens != 0 {
		t.Fatalf("the cold run reused %d positions", cold.usage.CachedPromptTokens)
	}

	// Warm: another conversation computed the system prompt first.
	request(t, session(t, m, WithSessionContext(sharedCap)),
		extend(system, 7, 10, 0), greedy(4))
	warm := request(t, session(t, m, WithSessionContext(sharedCap)), prompt, greedy(8))
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
	system := promptIDs(1, 3*CacheBlock)

	salted := session(t, m, WithSessionContext(sharedCap), WithCacheSalt("tenant-a"))
	request(t, salted, extend(system, 7, 10, 0), greedy(4))

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
			got := request(t, probe, extend(system, 11, 10, 0), greedy(4))
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
	got := request(t, same, extend(system, 13, 10, 0), greedy(4))
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
	request(t, s, promptIDs(1, 40), greedy(4))
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
