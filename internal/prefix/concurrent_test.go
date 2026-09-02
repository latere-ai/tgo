// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package prefix

import (
	"errors"
	"sync"
	"testing"
)

func TestConcurrentMissesOnOnePrefixKeepOneBlock(t *testing.T) {
	// 016 §10.4. The default scope is the process, so every session's
	// goroutine reaches one pool while a Session is deliberately unlocked
	// (007-D1). Neither vLLM nor sglang has this problem -- both schedulers
	// are single-threaded loops -- so there is no prior art to copy.
	//
	// Several goroutines miss on one prefix, compute it, and publish. The pool
	// must keep ONE block per logical block: the loser drops its own and takes
	// the winner's, or the loser leaks a block and the winner's refcount is
	// short by one.
	const workers = 9 // distinct from every other dimension here
	p := newPool(t, Config{Block: testBlock, Blocks: 31})
	prompt := run(100, 9) // two complete blocks and a partial tail

	var start sync.WaitGroup
	start.Add(1)
	// acquired holds every worker at the point where all of them have missed
	// and none has published. That is the interleaving the pool has to survive
	// and the one a scheduler will not reproduce on demand.
	var acquired sync.WaitGroup
	acquired.Add(workers)
	var done sync.WaitGroup
	pages := make([][]int, workers)
	leases := make([]*Lease, workers)
	errs := make([]error, workers)
	for i := range workers {
		done.Go(func() {
			start.Wait()
			l, err := p.Acquire(Request{IDs: prompt})
			if err != nil {
				errs[i] = err
				acquired.Done()
				return
			}
			if l.Matched() != 0 {
				errs[i] = errors.New("matched a prefix nobody had published yet")
			}
			acquired.Done()
			acquired.Wait()
			pages[i] = l.Publish(l.Len())
			leases[i] = l
			_ = p.Stats() // a reader racing the writers
			l.Release()
		})
	}
	start.Done()
	done.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("worker %d: %v", i, err)
		}
	}
	// Every worker's page table must agree with the map, or a swap updated one
	// side and not the other. Checked here rather than in the goroutines,
	// because a t.Fatalf outside the test goroutine reports nothing.
	for _, l := range leases {
		checkPublished(t, p, l)
	}
	for i := 1; i < workers; i++ {
		for b := range 2 { // the complete blocks, which are the shareable ones
			if pages[i][b] != pages[0][b] {
				t.Fatalf("worker %d holds block %d at %d and worker 0 at %d; the "+
					"pool kept two blocks for one prefix", i, b, pages[i][b], pages[0][b])
			}
		}
	}
	s := p.Stats()
	if s.Publishes != 2 {
		t.Fatalf("Publishes = %d, want 2: one per logical block, whoever won", s.Publishes)
	}
	if want := 2 * (workers - 1); s.Adoptions != want {
		t.Fatalf("Adoptions = %d, want %d", s.Adoptions, want)
	}
	if s.InUse != 0 || s.Cached != 2 || s.Free != 31-2 {
		t.Fatalf("InUse = %d, Cached = %d, Free = %d; want 0, 2 and %d",
			s.InUse, s.Cached, s.Free, 31-2)
	}
}

func TestConcurrentTrafficLeavesEveryBlockAccountedFor(t *testing.T) {
	// The same pool under hits, misses, growth and eviction pressure at once.
	// What it asserts is the arithmetic that a lost refcount breaks: every
	// block is in use, cached or free, and once the traffic stops none is in
	// use. Run under -race, this is also the test that says the lock covers
	// lookup, allocate, insert and evict rather than three of the four.
	const workers = 11
	const rounds = 13
	p := newPool(t, Config{Block: testBlock, Blocks: 17})
	shared := run(100, 9)

	var start sync.WaitGroup
	start.Add(1)
	var done sync.WaitGroup
	for w := range workers {
		done.Go(func() {
			start.Wait()
			for r := range rounds {
				ids := shared
				if w%3 == 0 { // some traffic shares nothing
					ids = run(1000*(w+1)+r, 9)
				}
				l, err := p.Acquire(Request{IDs: ids, Salt: []string{"", "x"}[w%2]})
				if err != nil {
					if !errors.Is(err, ErrExhausted) {
						t.Errorf("worker %d: %v", w, err)
					}
					continue
				}
				l.Publish(l.Len())
				if err := l.Append(700+r, 701+r, 702+r); err != nil &&
					!errors.Is(err, ErrExhausted) {
					t.Errorf("worker %d: Append = %v", w, err)
				}
				l.Publish(l.Len())
				l.Release()
				l.Release() // idempotent, and racing nothing
			}
		})
	}
	start.Done()
	done.Wait()

	s := p.Stats()
	if s.InUse != 0 {
		t.Fatalf("InUse = %d after every lease was released, want 0", s.InUse)
	}
	if got := s.Cached + s.Free; got != 17 {
		t.Fatalf("Cached + Free = %d, want the whole pool, %d", got, 17)
	}
	if s.Acquires == 0 {
		t.Fatal("no worker acquired anything")
	}
	// Every block a hash entry names must exist, be published and be
	// unreferenced. This is 016-D5's invariant read directly off the pool.
	p.mu.Lock()
	defer p.mu.Unlock()
	if got, want := len(p.entry), s.Cached; got != want {
		t.Fatalf("%d hash entries name %d cached blocks", got, want)
	}
	for h, id := range p.entry {
		b := p.blk[id]
		if !b.published || b.hash != h {
			t.Fatalf("hash entry names block %d, which is published=%v", id, b.published)
		}
		if b.refs != 0 {
			t.Fatalf("block %d is cached with %d references", id, b.refs)
		}
	}
}
