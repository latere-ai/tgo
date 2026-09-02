// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Package prefix caches the key/value state of a prompt prefix that some
// earlier request already paid to compute.
//
// The KV at position t is a function of tokens 0..t and the weights alone, so
// two requests with the same leading token ids have the same KV over that run.
// Reusing it is not an approximation; it is declining to recompute a pure
// function. specs/016-prefix-cache.md is the design.
//
// # What this package owns, and what it does not
//
// It owns the *bookkeeping*: which physical block holds which prefix, how many
// sequences reference it, and which block to reclaim when the pool runs dry. It
// touches no device memory and imports nothing from accel.
//
// A physical block is an index in [0, Config.Blocks). The engine allocates one
// key state and one value state of Config.Blocks*Config.Block rows per layer,
// and block b owns rows [b*Block, (b+1)*Block). Logical position t of a
// sequence lives at row [Lease.Row]. A page table row is [Lease.Blocks], bound
// to accel's AttentionOptions.Pages, and the count of positions whose KV is
// already resident is [Lease.Reused], bound as the prefill's BaseName.
//
// Splitting it this way is 016-D8: accel 030 declines to evict because the
// victim is policy, and this package is the policy.
//
// # The two invariants
//
// A hash entry never outlives the block it names. Eviction deletes the entry
// and frees the block in one step under one lock, or a later request maps a
// hash to a block that now holds another sequence's KV and attends to somebody
// else's context — silently, because the output stays fluent (016-D5).
//
// A published block is immutable. Only complete blocks are published, every
// position in one belongs to the shared prefix, and a sequence writes only at
// positions at or after [Lease.Reused]. That is what makes concurrent sharing
// safe without copying.
package prefix

import (
	"errors"
	"fmt"
	"sync"
)

// ErrExhausted is what an allocation is refused with when every block in the
// pool is referenced by a live sequence.
//
// It is not a failure of the cache: a block at refcount 0 is evictable and the
// pool reclaims it without telling anyone. This says the pool is genuinely
// full, and the caller's answer is to run a shorter request, to finish one, or
// to configure more blocks.
var ErrExhausted = errors.New("prefix: the block pool is exhausted")

// ErrReleased reports use of a lease whose blocks have been given back.
var ErrReleased = errors.New("prefix: the lease is released")

// Scope bounds what a request with no salt can reach (016-D7).
//
// A cache hit is faster than a miss and that timing is observable, so
// cross-request reuse is a membership oracle over other requests' prompts. The
// scope is what makes the default safe for the deployment; [Request.Salt] is
// what makes it precise.
type Scope int8

const (
	// ScopeProcess shares blocks across every request in the process. The
	// default, and correct for a single-tenant deployment: a CLI, one team's
	// server, an agent runtime.
	ScopeProcess Scope = iota

	// ScopeSession shares within one conversation only. Safe under
	// multi-tenancy and still captures the multi-turn win, which approaches
	// 1-1/n by turn n and is most of the value.
	ScopeSession

	// ScopeOff looks nothing up and publishes nothing. For measurement
	// against a cold baseline.
	ScopeOff
)

func (s Scope) String() string {
	switch s {
	case ScopeProcess:
		return "process"
	case ScopeSession:
		return "session"
	case ScopeOff:
		return "off"
	}
	return fmt.Sprintf("scope(%d)", int8(s))
}

// Config describes the pool a [New] call builds.
type Config struct {
	// Block is the positions per block, the unit of sharing. Sharing is
	// block-aligned (016-D4), so a genuine common prefix loses at most
	// Block-1 tokens to rounding — a rounding error against a prefix of
	// hundreds, and the alternative is two sequences writing one block at
	// different offsets, which is a correctness problem rather than an
	// optimisation.
	Block int

	// Blocks is the pool capacity, in blocks. Blocks*Block positions of KV
	// exist, shared by every live sequence and by the cache behind them.
	Blocks int

	// Scope defaults to [ScopeProcess], the zero value.
	Scope Scope
}

// Request is one prompt's identity.
type Request struct {
	// IDs is the prompt's token ids, taken after tokenization and never
	// before (016-D1). Text is the wrong key: one string renders to different
	// ids under different template options, and the id sequence is what the
	// model consumed.
	IDs []int

	// Session identifies the conversation. Required under [ScopeSession] and
	// ignored otherwise.
	Session string

	// Salt is the caller-supplied cache_salt, mixed into the chain seed and
	// therefore into every block hash. Blocks match only within one salt. The
	// layer that knows who the caller is supplies it; tgo has no notion of a
	// tenant (009 §7).
	Salt string

	// Reserve is how many positions beyond the prompt to hold blocks for, so
	// that the sequence can grow without asking again.
	//
	// It is specs/008-scheduler.md §3's R, and admitting without it is how a
	// server deadlocks: every slot occupied, the pool empty, and no sequence
	// able to grow -- so nothing finishes and nothing can be evicted into
	// progress. The blocks are held from the moment the lease is taken, which
	// is the point: a request that is admitted has already been shown to fit.
	//
	// Zero is the caller saying it will grow by nothing, which is right for a
	// one-shot scoring pass and wrong for generation.
	Reserve int
}

// Stats is a snapshot of what the pool has done and what it currently holds.
type Stats struct {
	Acquires     int // calls to [Pool.Acquire] that returned a lease
	Hits         int // of those, the ones that reused at least one block
	ReusedTokens int // positions whose KV did not have to be recomputed
	PromptTokens int // positions asked for, over all acquires
	Evictions    int // blocks reclaimed LRU, each with its hash entry
	Publishes    int // complete blocks this pool learned
	Adoptions    int // publishes that lost a race and took the winner's block

	InUse  int // blocks referenced by a live lease
	Cached int // blocks at refcount 0 that a hash entry still names
	Free   int // blocks that name nothing
}

// block is one physical block's bookkeeping.
type block struct {
	hash      [32]byte
	published bool // a hash entry names this block
	refs      int  // live leases holding it

	// prev and next are the LRU list links, meaningful only while the block is
	// cached -- refcount 0 with a hash entry naming it. The list is intrusive
	// and ordered least-recently-used first, so eviction is O(1) and its order
	// comes from a sequence of operations rather than a clock: a coarse clock
	// ties two entries and makes the victim arbitrary.
	prev, next int
}

// Pool is the block pool and the prefix map over it.
//
// It is safe for concurrent use, and it has to be: the default scope is the
// process, so every session's goroutine reaches one pool while a session is
// deliberately unlocked (007-D1, 016 §10.4). Lookup, allocate and insert are
// each atomic against the others.
type Pool struct {
	block  int
	blocks int
	scope  Scope

	mu    sync.Mutex
	blk   []block
	entry map[[32]byte]int // block hash -> physical block
	free  []int            // blocks naming nothing, most recently freed last
	head  int              // least recently used cached block, -1 when none
	tail  int              // most recently used cached block, -1 when none
	stats Stats
}

// New builds a pool of cfg.Blocks blocks of cfg.Block positions each.
func New(cfg Config) (*Pool, error) {
	if cfg.Block <= 0 {
		return nil, fmt.Errorf("prefix: the block size is %d; a block holds at "+
			"least one position", cfg.Block)
	}
	if cfg.Blocks <= 0 {
		return nil, fmt.Errorf("prefix: the pool holds %d blocks; a pool holds at "+
			"least one", cfg.Blocks)
	}
	switch cfg.Scope {
	case ScopeProcess, ScopeSession, ScopeOff:
	default:
		return nil, fmt.Errorf("prefix: %v is not a scope", cfg.Scope)
	}
	p := &Pool{
		block:  cfg.Block,
		blocks: cfg.Blocks,
		scope:  cfg.Scope,
		blk:    make([]block, cfg.Blocks),
		entry:  make(map[[32]byte]int, cfg.Blocks),
		free:   make([]int, cfg.Blocks),
		head:   -1,
		tail:   -1,
	}
	for i := range p.blk {
		p.blk[i].prev, p.blk[i].next = -1, -1
		// Handed out low index first, which makes a test's block ids readable
		// without making them a promise.
		p.free[i] = cfg.Blocks - 1 - i
	}
	p.stats.Free = cfg.Blocks
	return p, nil
}

// Block is the positions per block.
func (p *Pool) Block() int { return p.block }

// Blocks is the pool capacity, in blocks.
func (p *Pool) Blocks() int { return p.blocks }

// Capacity is the pool capacity, in positions.
func (p *Pool) Capacity() int { return p.blocks * p.block }

// Scope is the pool's isolation scope.
func (p *Pool) Scope() Scope { return p.scope }

// Stats snapshots the counters.
func (p *Pool) Stats() Stats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stats
}

// Acquire reserves blocks for r's prompt and reuses every leading block whose
// KV the pool already holds.
//
// The reused run is block-aligned and capped at len(r.IDs)-1 positions, never
// len(r.IDs) (016-D10). The cache holds KV, not logits: sampling the next token
// needs the logits at the last prompt position, and those come from a forward
// pass over it. Reuse everything and the request has a warm cache and nothing
// to sample from.
//
// The caller owns the returned lease until [Lease.Release].
func (p *Pool) Acquire(r Request) (*Lease, error) {
	if len(r.IDs) == 0 {
		return nil, errors.New("prefix: the prompt is empty; there is no prefix to key on")
	}
	if p.scope == ScopeSession && r.Session == "" {
		return nil, errors.New("prefix: the scope is session and the request names no " +
			"session; an unnamed session would share with every other unnamed one")
	}
	if r.Reserve < 0 {
		return nil, fmt.Errorf("prefix: the reserve is %d; it is positions to hold "+
			"beyond the prompt", r.Reserve)
	}
	n := len(r.IDs)
	// The prompt *and* its reserve, taken together or not at all. Rounding once
	// over the sum rather than twice over the parts is what makes the
	// arithmetic §3's ceil((T+R)/B) rather than a block more than it.
	need := (n + r.Reserve + p.block - 1) / p.block
	if need > p.blocks {
		return nil, fmt.Errorf("prefix: a %d-token prompt with a reserve of %d needs "+
			"%d blocks and the pool holds %d: %w", n, r.Reserve, need, p.blocks,
			ErrExhausted)
	}

	l := &Lease{
		pool:  p,
		scope: p.scope,
		ids:   append(make([]int, 0, n), r.IDs...),
	}
	// A block hash is chained over its predecessor, so a block's identity
	// includes every token before it (016-D2). The seed carries the scope
	// domain and the salt, which the chain then propagates to every block.
	domain := ""
	if p.scope == ScopeSession {
		domain = r.Session
	}
	l.seed = seed(p.scope, domain, r.Salt)
	if p.scope != ScopeOff {
		l.hashes = chainAll(l.seed, l.ids, p.block)
	}

	// The whole of lookup, cap, acquire and allocate is one critical section:
	// a block matched under the lock and acquired after it could be evicted in
	// between, and the lease would name a block holding another sequence's KV.
	p.mu.Lock()
	defer p.mu.Unlock()

	matched := 0
	// The cap is what makes this a prefix of at most n-1 positions. It is
	// applied in blocks, because a partial block cannot be shared: for a
	// prompt that is a whole number of blocks it declines the last complete
	// block, which is why an exact resubmission re-prefills Block tokens
	// rather than one.
	limit := (n - 1) / p.block
	for matched < limit && matched < len(l.hashes) {
		id, ok := p.entry[l.hashes[matched]]
		if !ok {
			break
		}
		p.hold(id)
		l.blocks = append(l.blocks, id)
		matched++
	}
	fresh, err := p.alloc(need - matched)
	if err != nil {
		for _, id := range l.blocks {
			p.drop(id)
		}
		return nil, err
	}
	l.blocks = append(l.blocks, fresh...)
	l.matched = matched
	l.reused = matched * p.block
	l.published = matched

	p.stats.Acquires++
	p.stats.PromptTokens += n
	p.stats.ReusedTokens += l.reused
	if matched > 0 {
		p.stats.Hits++
	}
	return l, nil
}

// alloc takes n blocks, evicting the least recently used cached blocks as far
// as it has to. It takes all n or none. The caller holds p.mu.
func (p *Pool) alloc(n int) ([]int, error) {
	out := make([]int, 0, n)
	for len(out) < n {
		id, ok := p.take()
		if !ok {
			for _, id := range out {
				p.release(id)
			}
			return nil, fmt.Errorf("prefix: %d of %d blocks are referenced by a live "+
				"sequence: %w", p.stats.InUse, p.blocks, ErrExhausted)
		}
		out = append(out, id)
	}
	return out, nil
}

// take produces one block at refcount 1, from the free list or by evicting the
// least recently used cached block. The caller holds p.mu.
func (p *Pool) take() (int, bool) {
	if n := len(p.free); n > 0 {
		id := p.free[n-1]
		p.free = p.free[:n-1]
		p.stats.Free--
		p.blk[id].refs = 1
		p.stats.InUse++
		return id, true
	}
	if p.head < 0 {
		// Every block is referenced. Eviction never touches one of those:
		// its KV is being attended to right now.
		return 0, false
	}
	id := p.head
	p.evict(id)
	p.blk[id].refs = 1
	p.stats.InUse++
	return id, true
}

// evict frees a cached block and deletes the hash entry that names it, in one
// step. Splitting the two is the silent bug of 016 §5. The caller holds p.mu.
func (p *Pool) evict(id int) {
	b := &p.blk[id]
	p.lruRemove(id)
	delete(p.entry, b.hash)
	b.hash = [32]byte{}
	b.published = false
	p.stats.Cached--
	p.stats.Evictions++
}

// hold takes a reference to a cached or live block. The caller holds p.mu.
func (p *Pool) hold(id int) {
	b := &p.blk[id]
	b.refs++
	if b.refs == 1 {
		// It was the cache. It is a sequence's now, so it is no longer a
		// candidate for eviction.
		p.lruRemove(id)
		p.stats.Cached--
		p.stats.InUse++
	}
}

// drop gives up a reference. A block that reaches refcount 0 is not freed if a
// hash entry names it — that block IS the cache (016-D5). The caller holds
// p.mu.
func (p *Pool) drop(id int) {
	b := &p.blk[id]
	b.refs--
	if b.refs > 0 {
		return
	}
	p.stats.InUse--
	if b.published {
		p.lruPush(id) // most recently used
		p.stats.Cached++
		return
	}
	p.free = append(p.free, id)
	p.stats.Free++
}

// release returns a block taken by take that was never handed to a lease.
func (p *Pool) release(id int) {
	p.blk[id].refs = 0
	p.stats.InUse--
	p.free = append(p.free, id)
	p.stats.Free++
}

func (p *Pool) lruPush(id int) {
	b := &p.blk[id]
	b.prev, b.next = p.tail, -1
	if p.tail >= 0 {
		p.blk[p.tail].next = id
	} else {
		p.head = id
	}
	p.tail = id
}

// lruRemove takes a cached block out of the eviction list. Every caller has
// just established that the block is in it: a block leaves the list exactly
// when it is evicted or when a lease takes a reference to it.
func (p *Pool) lruRemove(id int) {
	b := &p.blk[id]
	if b.prev >= 0 {
		p.blk[b.prev].next = b.next
	} else {
		p.head = b.next
	}
	if b.next >= 0 {
		p.blk[b.next].prev = b.prev
	} else {
		p.tail = b.prev
	}
	b.prev, b.next = -1, -1
}
