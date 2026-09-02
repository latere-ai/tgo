// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package prefix

import "fmt"

// Lease is one sequence's hold on the blocks its KV lives in.
//
// A lease is owned by one sequence and is NOT safe for concurrent use, which
// matches a Session (007-D1). The pool behind it is, so leases held by
// different goroutines are fine — that is the case this package exists to make
// safe.
type Lease struct {
	pool  *Pool
	scope Scope
	seed  [32]byte

	ids    []int      // the sequence's tokens, prompt then generated
	blocks []int      // physical block per logical block, in order
	hashes [][32]byte // one per complete block; empty under ScopeOff

	matched   int // blocks that came from the cache
	reused    int // positions that came from the cache
	published int // leading blocks a hash entry already names

	released bool
}

// Blocks is the page table row: the physical block holding each logical block
// of the sequence, in order. Bind it to accel's AttentionOptions.Pages with
// Block set to [Pool.Block].
//
// Re-read it after every [Lease.Publish]: publishing can swap a block for the
// one that won a race to name the same prefix.
func (l *Lease) Blocks() []int {
	return append(make([]int, 0, len(l.blocks)), l.blocks...)
}

// Reused is the number of leading positions whose KV the pool already holds.
// It is a multiple of [Pool.Block], it is at most one less than the prompt
// length, and it is the prefill's first position — accel's
// AttentionOptions.BaseName.
func (l *Lease) Reused() int { return l.reused }

// Matched is the number of leading blocks taken from the cache.
func (l *Lease) Matched() int { return l.matched }

// Len is the positions the lease covers: the prompt, plus everything
// [Lease.Append] has added.
func (l *Lease) Len() int { return len(l.ids) }

// Row is the physical row that holds position t of the sequence.
//
// The engine's key and value states are Pool.Blocks()*Pool.Block() rows per
// layer, and this is the index into them. It panics on a position the lease
// does not cover, which is a caller bug rather than a state a request reaches.
func (l *Lease) Row(t int) int {
	if t < 0 || t >= len(l.ids) {
		panic(fmt.Sprintf("prefix: position %d is outside the lease's %d positions",
			t, len(l.ids)))
	}
	return l.blocks[t/l.pool.block]*l.pool.block + t%l.pool.block
}

// Grow makes sure the lease's blocks cover n more positions than its ids do.
//
// It is the half of [Lease.Append] a caller needs *before* the step that
// computes those positions: a token needs a row to be written to. It records no
// ids and chains no hashes, because what those tokens are is not settled until
// the step lands -- a step that fails is a step whose tokens the caller may
// replace, and a hash chained over a token nobody computed names a block
// holding something else.
//
// It takes every position or none: a failure part way through would leave the
// sequence's length disagreeing with the blocks that back it.
func (l *Lease) Grow(n int) error {
	if l.released {
		return ErrReleased
	}
	if n < 0 {
		return fmt.Errorf("prefix: growing by %d positions", n)
	}
	if n == 0 {
		return nil
	}
	p := l.pool
	want := (len(l.ids) + n + p.block - 1) / p.block
	if want > p.blocks {
		return fmt.Errorf("prefix: %d positions need %d blocks and the pool holds %d: %w",
			len(l.ids)+n, want, p.blocks, ErrExhausted)
	}
	if k := want - len(l.blocks); k > 0 {
		p.mu.Lock()
		fresh, err := p.alloc(k)
		p.mu.Unlock()
		if err != nil {
			return err
		}
		l.blocks = append(l.blocks, fresh...)
	}
	return nil
}

// Commit records tokens whose key/value state a step has computed, chaining the
// hash of every block they completed.
//
// It is the other half of [Lease.Append], and it runs *after* the step. The
// blocks must already cover the positions, which [Lease.Grow] is for.
func (l *Lease) Commit(ids ...int) error {
	if l.released {
		return ErrReleased
	}
	if len(ids) == 0 {
		return nil
	}
	p := l.pool
	if want := (len(l.ids) + len(ids) + p.block - 1) / p.block; want > len(l.blocks) {
		return fmt.Errorf("prefix: committing %d tokens at position %d needs %d blocks "+
			"and the lease holds %d; grow it before the step that computes them",
			len(ids), len(l.ids), want, len(l.blocks))
	}
	l.ids = append(l.ids, ids...)
	if l.scope != ScopeOff {
		// Hash the blocks the new tokens completed. A block's hash is chained
		// on its predecessor's, so this continues the chain rather than
		// starting one.
		prev := l.seed
		if n := len(l.hashes); n > 0 {
			prev = l.hashes[n-1]
		}
		for i := len(l.hashes); i < len(l.ids)/p.block; i++ {
			prev = chain(prev, l.ids[i*p.block:(i+1)*p.block])
			l.hashes = append(l.hashes, prev)
		}
	}
	return nil
}

// Append is [Lease.Grow] and [Lease.Commit] in one call, for a caller whose
// step cannot fail between them.
func (l *Lease) Append(ids ...int) error {
	if err := l.Grow(len(ids)); err != nil {
		return err
	}
	return l.Commit(ids...)
}

// Publish offers every complete block the lease holds to the cache, so that a
// later request with the same prefix reuses it. Call it once the KV for those
// positions is computed, and never before: a published block is immutable and
// another sequence may be attending to it before the call returns.
//
// It returns the page table row, which the caller must adopt. Two sequences can
// miss on one prefix concurrently, compute it twice and both publish; the pool
// keeps one block and the loser drops its own and takes the winner's, so the
// two sequences end up sharing one block rather than leaking one (016 §10.4).
// The same swap happens without any concurrency, when a full hit's cap
// (016-D10) made this lease decline a block that is already cached.
//
// Under [ScopeOff] it publishes nothing and returns the blocks unchanged.
func (l *Lease) Publish(written int) []int {
	if l.released || l.scope == ScopeOff {
		return l.Blocks()
	}
	p := l.pool
	// Only blocks every position of which a step has actually computed.
	//
	// The parameter is the whole point and the reason this is not
	// `Publish()`. A lease covers the positions a caller *may* write --
	// [Pool.Acquire] leases the whole prompt so admission is a promise, and
	// [Lease.Grow] takes blocks before the step that fills them -- so the
	// blocks it holds run ahead of the blocks it has. Publishing on the lease's
	// extent instead of on this offers another sequence a block holding
	// nothing, and it reads that as context: a chunked prefill 32 tokens into a
	// 192-token prompt published all six blocks, and the next request with the
	// same prefix reused 192 positions of which 160 were never written.
	complete := written / p.block
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := l.published; i < len(l.hashes) && i < complete; i++ {
		h := l.hashes[i]
		mine := l.blocks[i]
		if theirs, ok := p.entry[h]; ok {
			if theirs != mine {
				p.hold(theirs)
				p.drop(mine)
				l.blocks[i] = theirs
				p.stats.Adoptions++
			}
			l.published = i + 1
			continue
		}
		p.entry[h] = mine
		p.blk[mine].hash = h
		p.blk[mine].published = true
		p.stats.Publishes++
		l.published = i + 1
	}
	return l.Blocks()
}

// Release gives the blocks back.
//
// A block whose last reference goes is not freed: it is the cache, and it stays
// until the pool needs it (016-D5). A block that no hash entry names — the
// sequence's partial tail — goes straight back to the free list, because
// nothing can ever find it again.
//
// Release is idempotent.
func (l *Lease) Release() {
	if l.released {
		return
	}
	l.released = true
	p := l.pool
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, id := range l.blocks {
		p.drop(id)
	}
}
