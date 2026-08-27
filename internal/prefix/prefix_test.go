// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package prefix

import (
	"errors"
	"strings"
	"testing"
)

// The fixtures keep every dimension distinct on purpose: a block size that
// equals the pool size, or a prompt that is exactly one block, is the identity
// for a confusion between blocks and positions.
const (
	testBlock  = 4
	testBlocks = 7
)

// run returns n ids starting at start. Distinct starts give prompts that share
// no token, which is what a "these must not match" test needs.
func run(start, n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = start + i
	}
	return out
}

func newPool(t *testing.T, cfg Config) *Pool {
	t.Helper()
	p, err := New(cfg)
	if err != nil {
		t.Fatalf("New(%+v) = %v", cfg, err)
	}
	return p
}

func testPool(t *testing.T) *Pool {
	t.Helper()
	return newPool(t, Config{Block: testBlock, Blocks: testBlocks})
}

func acquire(t *testing.T, p *Pool, r Request) *Lease {
	t.Helper()
	l, err := p.Acquire(r)
	if err != nil {
		t.Fatalf("Acquire(%d ids) = %v", len(r.IDs), err)
	}
	return l
}

// warm prefills and publishes a prompt, then releases it, leaving its complete
// blocks in the cache. It is what "somebody already paid for this prefix"
// means, in one line.
func warm(t *testing.T, p *Pool, r Request) {
	t.Helper()
	l := acquire(t, p, r)
	l.Publish(l.Len())
	l.Release()
}

// checkPublished asserts the agreement a swap could break in either direction:
// every block this lease has published is the block the map names for that
// hash. A lease updated without the map, or a map updated without the lease,
// is exactly how one request comes to attend to another's KV.
func checkPublished(t *testing.T, p *Pool, l *Lease) {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := range l.published {
		id, ok := p.entry[l.hashes[i]]
		if !ok {
			t.Fatalf("block %d is published and no hash entry names it", i)
		}
		if id != l.blocks[i] {
			t.Fatalf("the lease holds block %d at index %d and the map names %d",
				l.blocks[i], i, id)
		}
		if !p.blk[id].published || p.blk[id].hash != l.hashes[i] {
			t.Fatalf("block %d does not carry the hash that names it", id)
		}
	}
}

func TestAWarmPrefixIsReusedAndAColdOneIsNot(t *testing.T) {
	// The whole point, at its smallest: a prompt whose blocks the pool holds
	// starts its prefill part way in, and one whose blocks it does not starts
	// at zero.
	p := testPool(t)
	prompt := run(100, 9)

	cold := acquire(t, p, Request{IDs: prompt})
	if cold.Reused() != 0 {
		t.Fatalf("a cold request reused %d positions, want 0", cold.Reused())
	}
	cold.Publish(cold.Len())
	cold.Release()

	warmed := acquire(t, p, Request{IDs: prompt})
	defer warmed.Release()
	if got, want := warmed.Reused(), 8; got != want {
		t.Fatalf("a warm request reused %d positions, want %d", got, want)
	}
	if got, want := warmed.Matched(), 2; got != want {
		t.Fatalf("matched %d blocks, want %d", got, want)
	}
	// The reused blocks are the same physical blocks, or nothing was shared.
	if a, b := cold.Blocks(), warmed.Blocks(); a[0] != b[0] || a[1] != b[1] {
		t.Fatalf("warm blocks %v do not reuse the cold ones %v", b, a)
	}
}

func TestAnUnrelatedPromptSharesNothing(t *testing.T) {
	p := testPool(t)
	warm(t, p, Request{IDs: run(100, 9)})

	l := acquire(t, p, Request{IDs: run(500, 9)})
	defer l.Release()
	if l.Reused() != 0 {
		t.Fatalf("an unrelated prompt reused %d positions, want 0", l.Reused())
	}
}

func TestAnInteriorRunSharedByTwoPromptsIsNotSharedByTheirBlocks(t *testing.T) {
	// 016-D2, and the reason the hash is chained. Two prompts carry the same
	// 32-token run at the same block index, behind different leading blocks.
	// An unchained content hash gives that run one identity, and the second
	// prompt attends to the first's context -- silently, because the output
	// stays fluent.
	const block = 32
	p := newPool(t, Config{Block: block, Blocks: 7})
	shared := run(9000, block)

	first := append(run(100, block), shared...)
	first = append(first, run(700, 1)...)
	second := append(run(500, block), shared...)
	second = append(second, run(700, 1)...)

	one := acquire(t, p, Request{IDs: first})
	theirs := one.Publish(one.Len())
	one.Release()

	l := acquire(t, p, Request{IDs: second})
	defer l.Release()
	if l.Matched() != 0 {
		t.Fatalf("a prompt sharing only an interior run matched %d blocks, want 0",
			l.Matched())
	}

	// Acquire alone cannot see this bug: the match loop stops at the first
	// miss, so the interior block is never looked up. PUBLISH is where a
	// content key does its damage -- the second prompt offers the same 32
	// tokens, the map already names them, and this lease ADOPTS the first
	// prompt's block. From then on it attends to KV computed behind a
	// different predecessor, and the output stays fluent.
	mine := l.Blocks()
	page := l.Publish(l.Len())
	if got := p.Stats().Adoptions; got != 0 {
		t.Fatalf("the second prompt adopted %d blocks; the interior run was "+
			"given one identity across two prompts", got)
	}
	for i := range page {
		if page[i] != mine[i] {
			t.Fatalf("logical block %d moved from %d to %d at publish", i,
				mine[i], page[i])
		}
		// theirs[0] and theirs[1] are the first prompt's published blocks.
		// Its partial tail went back to the free list and may legitimately
		// be handed out again.
		if page[i] == theirs[0] || page[i] == theirs[1] {
			t.Fatalf("the second prompt holds block %d at index %d, which is "+
				"the first prompt's; the chain did not separate them", page[i], i)
		}
	}

	// A positive control. Without it a hash that returned random bytes would
	// pass the assertion above.
	same := acquire(t, p, Request{IDs: first})
	defer same.Release()
	if got, want := same.Matched(), 2; got != want {
		t.Fatalf("the identical prompt matched %d blocks, want %d; the chain "+
			"separated what it should share", got, want)
	}
}

func TestTheIdenticalPromptResubmittedPrefillsOneToken(t *testing.T) {
	// 016-D10 and specs/016 §3.1. The cache holds KV, not logits: reuse the
	// whole match and the request has a warm cache and nothing to sample from.
	// The chat path cannot produce this case, because a rendered prompt always
	// ends with a fresh assistant opener.
	p := testPool(t)
	prompt := run(100, 2*testBlock+1) // 9 tokens: two whole blocks and one
	warm(t, p, Request{IDs: prompt})

	l := acquire(t, p, Request{IDs: prompt})
	defer l.Release()
	if got, want := l.Reused(), len(prompt)-1; got != want {
		t.Fatalf("reused %d of %d positions, want %d -- the cap is T-1, never T",
			got, len(prompt), want)
	}
	if got := len(prompt) - l.Reused(); got != 1 {
		t.Fatalf("the resubmission prefills %d tokens, want exactly 1", got)
	}
}

func TestAFullHitOnAWholeNumberOfBlocksGivesUpOneBlock(t *testing.T) {
	// The same cap, at the residue §8's "exactly one token" does not cover.
	// Sharing is block-aligned, so capping at T-1 positions declines a whole
	// block when T is a multiple of the block size: the suffix is Block
	// tokens, not one.
	p := testPool(t)
	prompt := run(100, 2*testBlock) // 8 tokens: two whole blocks, no remainder
	warm(t, p, Request{IDs: prompt})

	l := acquire(t, p, Request{IDs: prompt})
	if got, want := l.Reused(), testBlock; got != want {
		t.Fatalf("reused %d positions, want %d: the cap declines the last "+
			"complete block", got, want)
	}
	if got := len(prompt) - l.Reused(); got != testBlock {
		t.Fatalf("the resubmission prefills %d tokens, want %d", got, testBlock)
	}

	// Publishing the recomputed block finds the cached one already there. The
	// pool must keep one block, not two.
	before := l.Blocks()
	after := l.Publish(l.Len())
	if before[1] == after[1] {
		t.Fatalf("the recomputed block %d was not swapped for the cached one",
			before[1])
	}
	if got := p.Stats().Adoptions; got != 1 {
		t.Fatalf("Adoptions = %d, want 1", got)
	}
	// The block that survived must be the block the map names. A swap that
	// updated one and not the other is how a request comes to attend to
	// another sequence's KV.
	checkPublished(t, p, l)
	l.Release()
	if s := p.Stats(); s.Cached != 2 || s.Free != testBlocks-2 {
		t.Fatalf("after release the pool holds %d cached and %d free blocks, "+
			"want 2 and %d", s.Cached, s.Free, testBlocks-2)
	}
}

func TestTheHitIsBlockAlignedAndNeverExceedsTheCommonPrefix(t *testing.T) {
	// 016 §3: up to Block-1 tokens of a genuine common prefix are re-prefilled,
	// and never one token more than the prefix is shared.
	p := testPool(t)
	common := 7 // one whole block and three tokens
	first := append(run(100, common), run(300, 4)...)
	second := append(run(100, common), run(800, 4)...)
	warm(t, p, Request{IDs: first})

	l := acquire(t, p, Request{IDs: second})
	defer l.Release()
	if got := l.Reused(); got > common {
		t.Fatalf("reused %d positions of a %d-token common prefix", got, common)
	}
	if got := l.Reused() % testBlock; got != 0 {
		t.Fatalf("reused %d positions, which is not block-aligned", l.Reused())
	}
	if got, want := l.Reused(), testBlock; got != want {
		t.Fatalf("reused %d positions, want %d", got, want)
	}
}

func TestAnEvictedPrefixIsAMissRatherThanAStaleBlock(t *testing.T) {
	// 016-D5's invariant, tested directly rather than through the paths that
	// would happen to produce it: a hash entry must never outlive the block it
	// names, or a later request maps a hash to a block that now holds another
	// sequence's KV.
	p := newPool(t, Config{Block: testBlock, Blocks: 3})
	victim := run(100, 5) // two blocks: one whole, one partial
	warm(t, p, Request{IDs: victim})
	if got := p.Stats().Cached; got != 1 {
		t.Fatalf("Cached = %d after warming, want 1", got)
	}

	// Three blocks of another prompt evict everything reclaimable.
	pressure := acquire(t, p, Request{IDs: run(500, 3*testBlock)})
	if got := p.Stats().Evictions; got == 0 {
		t.Fatal("the pool served three blocks under pressure without evicting one")
	}
	pressure.Publish(pressure.Len())
	pressure.Release()

	l := acquire(t, p, Request{IDs: victim})
	defer l.Release()
	if l.Matched() != 0 {
		t.Fatalf("the evicted prefix matched %d blocks; its hash entry outlived "+
			"the block", l.Matched())
	}
}

func TestABlockSharedByTwoSequencesSurvivesOneOfThemFinishing(t *testing.T) {
	p := testPool(t)
	prompt := run(100, 9)

	a := acquire(t, p, Request{IDs: prompt})
	a.Publish(a.Len())

	b := acquire(t, p, Request{IDs: prompt})
	if b.Matched() != 2 {
		t.Fatalf("the second sequence matched %d blocks, want 2", b.Matched())
	}
	shared := b.Blocks()[0]

	a.Release()
	if got := p.blk[shared].refs; got != 1 {
		t.Fatalf("the shared block has %d references after one sequence "+
			"finished, want 1", got)
	}

	// Pool pressure must not take it: it is being attended to.
	hog, err := p.Acquire(Request{IDs: run(900, testBlocks*testBlock)})
	if !errors.Is(err, ErrExhausted) {
		t.Fatalf("a request for the whole pool = %v, want ErrExhausted", err)
	}
	if hog != nil {
		t.Fatal("a refused Acquire returned a lease")
	}
	if got := p.blk[shared].refs; got != 1 {
		t.Fatalf("the shared block has %d references after pool pressure, want 1", got)
	}
	b.Release()
}

func TestEvictionNeverFreesAReferencedBlock(t *testing.T) {
	p := newPool(t, Config{Block: testBlock, Blocks: 3})
	held := acquire(t, p, Request{IDs: run(100, 3*testBlock)})
	defer held.Release()

	if _, err := p.Acquire(Request{IDs: run(500, 1)}); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Acquire with every block referenced = %v, want ErrExhausted", err)
	}
	if s := p.Stats(); s.InUse != 3 || s.Evictions != 0 {
		t.Fatalf("InUse = %d and Evictions = %d, want 3 and 0", s.InUse, s.Evictions)
	}
	// The refused request must not have kept a block it could not complete.
	if s := p.Stats(); s.Free != 0 {
		t.Fatalf("Free = %d after a refusal, want 0", s.Free)
	}
}

func TestTheLeastRecentlyUsedPrefixIsEvictedFirst(t *testing.T) {
	// The order comes from the sequence of operations and never from a clock:
	// a coarse timer ties two entries and makes the victim arbitrary.
	p := newPool(t, Config{Block: testBlock, Blocks: 3})
	older := run(100, testBlock)
	newer := run(500, testBlock)
	warm(t, p, Request{IDs: older})
	warm(t, p, Request{IDs: newer})
	if got := p.Stats().Cached; got != 2 {
		t.Fatalf("Cached = %d, want 2", got)
	}

	// One block of pressure takes exactly one victim.
	l := acquire(t, p, Request{IDs: run(900, testBlock+1)})
	l.Release()
	if got := p.Stats().Evictions; got != 1 {
		t.Fatalf("Evictions = %d, want 1", got)
	}

	kept := acquire(t, p, Request{IDs: append(run(500, testBlock), 1)})
	if kept.Matched() != 1 {
		t.Fatalf("the more recently used prefix matched %d blocks, want 1; the "+
			"wrong victim was chosen", kept.Matched())
	}
	kept.Release()

	gone := acquire(t, p, Request{IDs: append(run(100, testBlock), 1)})
	defer gone.Release()
	if gone.Matched() != 0 {
		t.Fatalf("the least recently used prefix matched %d blocks, want 0", gone.Matched())
	}
}

func TestAReusedBlockIsMoreRecentlyUsedThanOneOnlyPublished(t *testing.T) {
	// A hit takes the block out of the eviction list and puts it back at the
	// far end when it is released. Reuse therefore protects a prefix, which is
	// the behaviour "least recently used" promises.
	p := newPool(t, Config{Block: testBlock, Blocks: 3})
	first := run(100, testBlock)
	second := run(500, testBlock)
	warm(t, p, Request{IDs: first})
	warm(t, p, Request{IDs: second})
	warm(t, p, Request{IDs: first}) // first is now the most recent

	l := acquire(t, p, Request{IDs: run(900, testBlock+1)})
	l.Release()

	hit := acquire(t, p, Request{IDs: append(run(100, testBlock), 1)})
	defer hit.Release()
	if hit.Matched() != 1 {
		t.Fatalf("the reused prefix matched %d blocks, want 1", hit.Matched())
	}
}

func TestStatsCountWhatThePoolDid(t *testing.T) {
	p := testPool(t)
	prompt := run(100, 9)
	warm(t, p, Request{IDs: prompt})
	l := acquire(t, p, Request{IDs: prompt})
	l.Release()

	s := p.Stats()
	if s.Acquires != 2 || s.Hits != 1 {
		t.Fatalf("Acquires = %d and Hits = %d, want 2 and 1", s.Acquires, s.Hits)
	}
	if s.PromptTokens != 18 || s.ReusedTokens != 8 {
		t.Fatalf("PromptTokens = %d and ReusedTokens = %d, want 18 and 8",
			s.PromptTokens, s.ReusedTokens)
	}
	if s.Publishes != 2 {
		t.Fatalf("Publishes = %d, want 2", s.Publishes)
	}
	if s.InUse != 0 || s.Cached != 2 || s.Free != testBlocks-2 {
		t.Fatalf("InUse = %d, Cached = %d, Free = %d; want 0, 2 and %d",
			s.InUse, s.Cached, s.Free, testBlocks-2)
	}
}

func TestTheAccessorsReportTheConfiguration(t *testing.T) {
	p := testPool(t)
	if p.Block() != testBlock || p.Blocks() != testBlocks {
		t.Fatalf("Block() = %d and Blocks() = %d, want %d and %d",
			p.Block(), p.Blocks(), testBlock, testBlocks)
	}
	if got, want := p.Capacity(), testBlock*testBlocks; got != want {
		t.Fatalf("Capacity() = %d, want %d", got, want)
	}
	if got := p.Scope(); got != ScopeProcess {
		t.Fatalf("Scope() = %v, want process", got)
	}
}

func TestNewRefusesAPoolThatCannotHoldAnything(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
		want string
	}{
		{"no block size", Config{Block: 0, Blocks: 3}, "at least one position"},
		{"a negative block size", Config{Block: -8, Blocks: 3}, "at least one position"},
		{"no blocks", Config{Block: 4, Blocks: 0}, "at least one"},
		{"an unknown scope", Config{Block: 4, Blocks: 3, Scope: Scope(9)}, "is not a scope"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := New(tc.cfg)
			if err == nil {
				t.Fatalf("New(%+v) = %v, want an error", tc.cfg, p)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("New(%+v) = %q, want it to mention %q", tc.cfg, err, tc.want)
			}
		})
	}
}

func TestAcquireRefusesWhatItCannotKey(t *testing.T) {
	t.Run("an empty prompt", func(t *testing.T) {
		p := testPool(t)
		if _, err := p.Acquire(Request{}); err == nil ||
			!strings.Contains(err.Error(), "empty") {
			t.Fatalf("Acquire with no ids = %v, want an error naming the empty prompt", err)
		}
	})
	t.Run("a session scope with no session", func(t *testing.T) {
		// Failing open here would put every unnamed session in one shared
		// domain, which is the leak the scope exists to close.
		p := newPool(t, Config{Block: testBlock, Blocks: testBlocks, Scope: ScopeSession})
		if _, err := p.Acquire(Request{IDs: run(100, 5)}); err == nil ||
			!strings.Contains(err.Error(), "session") {
			t.Fatalf("Acquire with no session = %v, want an error naming the session", err)
		}
	})
	t.Run("a prompt larger than the pool", func(t *testing.T) {
		p := newPool(t, Config{Block: testBlock, Blocks: 3})
		_, err := p.Acquire(Request{IDs: run(100, 3*testBlock+1)})
		if !errors.Is(err, ErrExhausted) {
			t.Fatalf("Acquire of an oversized prompt = %v, want ErrExhausted", err)
		}
		if !strings.Contains(err.Error(), "the pool holds 3") {
			t.Fatalf("Acquire of an oversized prompt = %q, want the pool size", err)
		}
	})
}

func TestScopeNamesItself(t *testing.T) {
	for _, tc := range []struct {
		s    Scope
		want string
	}{
		{ScopeProcess, "process"},
		{ScopeSession, "session"},
		{ScopeOff, "off"},
		{Scope(9), "scope(9)"},
	} {
		if got := tc.s.String(); got != tc.want {
			t.Fatalf("Scope(%d).String() = %q, want %q", int8(tc.s), got, tc.want)
		}
	}
}

func TestARefusedAcquireGivesBackTheBlocksItMatched(t *testing.T) {
	// A request that hits and then cannot allocate the rest of what it needs
	// must leave the pool as it found it -- and the block it matched goes back
	// to the CACHE, not to the free list. Freeing it would throw away a prefix
	// on a failure that has nothing to do with it.
	p := newPool(t, Config{Block: testBlock, Blocks: 3})
	prefix := run(100, testBlock+1) // one complete block and a partial
	warm(t, p, Request{IDs: prefix})

	hog := acquire(t, p, Request{IDs: run(500, 2*testBlock)})
	if s := p.Stats(); s.Free != 0 || s.Cached != 1 {
		t.Fatalf("Free = %d and Cached = %d before the refusal, want 0 and 1",
			s.Free, s.Cached)
	}

	long := append(run(100, testBlock), run(900, 2*testBlock)...)
	if _, err := p.Acquire(Request{IDs: long}); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Acquire = %v, want ErrExhausted", err)
	}
	if s := p.Stats(); s.Cached != 1 || s.InUse != 2 || s.Free != 0 {
		t.Fatalf("Cached = %d, InUse = %d, Free = %d after the refusal; want 1, 2 and 0",
			s.Cached, s.InUse, s.Free)
	}
	// And the prefix is still there to hit.
	hog.Release()
	hit := acquire(t, p, Request{IDs: prefix})
	defer hit.Release()
	if hit.Matched() != 1 {
		t.Fatalf("the matched-then-refused prefix now matches %d blocks, want 1",
			hit.Matched())
	}
}

// TestAReserveIsHeldFromAdmission is specs/008-scheduler.md §3.
//
// A sequence admitted on its prompt alone is a sequence that may not be able to
// grow, and a pool of those deadlocks: every slot occupied, the pool empty, and
// nothing able to finish. The reserve is what makes "admitted" mean "shown to
// fit", so it is held from the moment the lease is taken rather than asked for
// later.
func TestAReserveIsHeldFromAdmission(t *testing.T) {
	p, err := New(Config{Block: 4, Blocks: 4, Scope: ScopeProcess})
	if err != nil {
		t.Fatal(err)
	}
	// Four tokens is one block; a reserve of eight takes two more.
	l, err := p.Acquire(Request{IDs: []int{1, 2, 3, 4}, Reserve: 8})
	if err != nil {
		t.Fatalf("Acquire with a reserve: %v", err)
	}
	if got := len(l.Blocks()); got != 3 {
		t.Fatalf("a 4-token prompt with a reserve of 8 holds %d blocks over a block "+
			"size of 4, want 3", got)
	}
	if got := p.Stats().InUse; got != 3 {
		t.Fatalf("the pool reports %d blocks in use, want 3; a reserve that is not "+
			"held is not a reserve", got)
	}

	// Growing into the reserve asks the pool for nothing.
	before := p.Stats().InUse
	if err := l.Append(5, 6, 7, 8); err != nil {
		t.Fatalf("Append into the reserve: %v", err)
	}
	if got := p.Stats().InUse; got != before {
		t.Fatalf("growing into the reserve took %d more blocks; the whole point is "+
			"that it was already taken", got-before)
	}

	// And a second request that would not fit beside it is refused rather than
	// admitted into a pool that cannot hold it.
	if _, err := p.Acquire(Request{IDs: []int{9, 10}, Reserve: 8}); !errors.Is(err, ErrExhausted) {
		t.Fatalf("a request needing 3 blocks of a pool with 1 free = %v, want "+
			"ErrExhausted", err)
	}
	l.Release()
}

// TestAReserveIsRoundedOnceOverTheSum: ceil((T+R)/B), not ceil(T/B)+ceil(R/B).
func TestAReserveIsRoundedOnceOverTheSum(t *testing.T) {
	p, err := New(Config{Block: 4, Blocks: 8, Scope: ScopeProcess})
	if err != nil {
		t.Fatal(err)
	}
	// 2 + 2 = 4 is one block. Rounding the parts would take two.
	l, err := p.Acquire(Request{IDs: []int{1, 2}, Reserve: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := len(l.Blocks()); got != 1 {
		t.Fatalf("a 2-token prompt with a reserve of 2 over blocks of 4 holds %d "+
			"blocks, want 1", got)
	}
	l.Release()
}

// TestANegativeReserveIsRefused.
func TestANegativeReserveIsRefused(t *testing.T) {
	p, err := New(Config{Block: 4, Blocks: 4, Scope: ScopeProcess})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Acquire(Request{IDs: []int{1}, Reserve: -1}); err == nil {
		t.Fatal("a negative reserve was accepted")
	}
}
