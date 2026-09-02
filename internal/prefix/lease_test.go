// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package prefix

import (
	"errors"
	"strings"
	"testing"
)

func TestRowMapsAPositionToItsBlocksRow(t *testing.T) {
	// The engine's key and value states are Blocks*Block rows per layer, and
	// this is the arithmetic that finds a sequence's position in them. It is
	// worth a test because a block index used as a row -- or a row used as a
	// block -- reads plausibly and corrupts silently.
	//
	// A fresh pool hands out blocks 0, 1, 2 in order, which makes the page
	// table the IDENTITY and this whole test vacuous: Row(t) could return t
	// and pass. So the fixture scrambles the table first -- a lease that takes
	// and gives back the low blocks leaves them at the end of the free list,
	// and the next lease gets them in reverse.
	p := testPool(t)
	churn := acquire(t, p, Request{IDs: run(500, 3*testBlock)})
	pin := acquire(t, p, Request{IDs: run(700, 1)})
	defer pin.Release()
	churn.Release()

	l := acquire(t, p, Request{IDs: run(100, 9)})
	defer l.Release()

	blocks := l.Blocks()
	// The guard that keeps this test honest. Without it a later change to the
	// allocator could quietly restore the identity, and the assertions below
	// would go on passing while testing nothing.
	identity := true
	for i, b := range blocks {
		if b != i {
			identity = false
		}
	}
	if identity {
		t.Fatalf("the page table is %v, the identity; Row cannot be "+
			"distinguished from its argument", blocks)
	}
	if got := l.Row(0); got == 0 {
		t.Fatal("Row(0) = 0: the fixture did not move the first block")
	}

	for _, tc := range []struct{ pos, want int }{
		{0, blocks[0]*testBlock + 0},
		{3, blocks[0]*testBlock + 3},
		{4, blocks[1]*testBlock + 0},
		{7, blocks[1]*testBlock + 3},
		{8, blocks[2]*testBlock + 0},
	} {
		if got := l.Row(tc.pos); got != tc.want {
			t.Fatalf("Row(%d) = %d, want %d (page table %v)",
				tc.pos, got, tc.want, blocks)
		}
	}
}

func TestRowPanicsOutsideTheLease(t *testing.T) {
	p := testPool(t)
	l := acquire(t, p, Request{IDs: run(100, 5)})
	defer l.Release()

	for _, pos := range []int{-1, 5} {
		func() {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("Row(%d) did not panic", pos)
				}
				if !strings.Contains(r.(string), "outside the lease") {
					t.Fatalf("Row(%d) panicked with %v", pos, r)
				}
			}()
			l.Row(pos)
		}()
	}
}

func TestBlocksIsACopyTheCallerCannotCorrupt(t *testing.T) {
	p := testPool(t)
	l := acquire(t, p, Request{IDs: run(100, 9)})
	defer l.Release()

	got := l.Blocks()
	got[0] = 999
	if again := l.Blocks(); again[0] == 999 {
		t.Fatal("Blocks() handed out the lease's own slice")
	}
}

func TestAppendAllocatesBlocksAsTheSequenceGrows(t *testing.T) {
	p := testPool(t)
	l := acquire(t, p, Request{IDs: run(100, 5)}) // two blocks, one partial
	defer l.Release()
	if got := len(l.Blocks()); got != 2 {
		t.Fatalf("the prompt took %d blocks, want 2", got)
	}

	// Three more tokens fill the partial block and start no new one.
	if err := l.Append(1, 2, 3); err != nil {
		t.Fatalf("Append = %v", err)
	}
	if got := len(l.Blocks()); got != 2 {
		t.Fatalf("after filling the block the sequence holds %d blocks, want 2", got)
	}
	if got := l.Len(); got != 8 {
		t.Fatalf("Len() = %d, want 8", got)
	}
	// One more needs a block.
	if err := l.Append(4); err != nil {
		t.Fatalf("Append = %v", err)
	}
	if got := len(l.Blocks()); got != 3 {
		t.Fatalf("the ninth position took %d blocks, want 3", got)
	}
	if got := l.Row(8); got != l.Blocks()[2]*testBlock {
		t.Fatalf("Row(8) = %d, want the start of block %d", got, l.Blocks()[2])
	}
}

func TestAppendCompletesABlockWithoutAllocatingOne(t *testing.T) {
	// Filling a partial block allocates nothing and yet makes a new block
	// publishable. A length check that only counted blocks would miss the
	// second half of that, and the completed block would never be shared.
	p := testPool(t)
	l := acquire(t, p, Request{IDs: run(100, 5)})
	defer l.Release()
	l.Publish(l.Len())
	if got := p.Stats().Publishes; got != 1 {
		t.Fatalf("Publishes = %d after the prompt, want 1", got)
	}
	if err := l.Append(1, 2, 3); err != nil {
		t.Fatalf("Append = %v", err)
	}
	if got := len(l.Blocks()); got != 2 {
		t.Fatalf("filling the tail block allocated one: %d blocks, want 2", got)
	}
	l.Publish(l.Len())
	if got := p.Stats().Publishes; got != 2 {
		t.Fatalf("Publishes = %d after the block was filled, want 2", got)
	}
	checkPublished(t, p, l)
}

func TestAppendFillsTheLastBlockThePoolHas(t *testing.T) {
	// The boundary the length check owns: the sequence grows into the very
	// last block, which must be allowed, and one position further, which must
	// not.
	p := newPool(t, Config{Block: testBlock, Blocks: 3})
	l := acquire(t, p, Request{IDs: run(100, 2*testBlock)})
	defer l.Release()
	if err := l.Append(1, 2, 3, 4); err != nil {
		t.Fatalf("Append into the last block = %v", err)
	}
	if got := len(l.Blocks()); got != 3 {
		t.Fatalf("the sequence holds %d blocks, want 3", got)
	}
	if got := l.Len(); got != 3*testBlock {
		t.Fatalf("Len() = %d, want %d", got, 3*testBlock)
	}
	l.Publish(l.Len())
	if got := p.Stats().Publishes; got != 3 {
		t.Fatalf("Publishes = %d, want 3: the pool learned every block", got)
	}
	checkPublished(t, p, l)
	if err := l.Append(5); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Append past the last block = %v, want ErrExhausted", err)
	}
}

func TestAppendingNothingDoesNothing(t *testing.T) {
	p := testPool(t)
	l := acquire(t, p, Request{IDs: run(100, 5)})
	defer l.Release()
	if err := l.Append(); err != nil {
		t.Fatalf("Append() = %v", err)
	}
	if got := l.Len(); got != 5 {
		t.Fatalf("Len() = %d, want 5", got)
	}
}

func TestAGeneratedTailIsPublishedAndReusedByTheNextTurn(t *testing.T) {
	// The multi-turn win: turn n+1 re-sends everything turn n produced, so the
	// blocks the answer filled are the next prompt's prefix.
	p := testPool(t)
	prompt := run(100, 5)
	l := acquire(t, p, Request{IDs: prompt})
	l.Publish(l.Len()) // one complete block
	if err := l.Append(700, 701, 702, 703); err != nil {
		t.Fatalf("Append = %v", err)
	}
	l.Publish(l.Len()) // the block the answer completed
	l.Release()
	if got := p.Stats().Publishes; got != 2 {
		t.Fatalf("Publishes = %d, want 2", got)
	}

	next := append(append([]int{}, prompt...), 700, 701, 702, 703, 900)
	turn := acquire(t, p, Request{IDs: next})
	defer turn.Release()
	if got, want := turn.Matched(), 2; got != want {
		t.Fatalf("the next turn matched %d blocks, want %d", got, want)
	}
	if got, want := turn.Reused(), 8; got != want {
		t.Fatalf("the next turn reused %d positions, want %d", got, want)
	}
}

func TestAppendRefusesToGrowPastThePool(t *testing.T) {
	p := newPool(t, Config{Block: testBlock, Blocks: 3})
	l := acquire(t, p, Request{IDs: run(100, 3*testBlock)})
	defer l.Release()
	err := l.Append(1)
	if !errors.Is(err, ErrExhausted) {
		t.Fatalf("Append past the pool = %v, want ErrExhausted", err)
	}
	if got := l.Len(); got != 3*testBlock {
		t.Fatalf("a refused Append changed the length to %d", got)
	}
}

func TestAppendRefusesWhenNoBlockIsFree(t *testing.T) {
	// The pool is big enough for the sequence and every block belongs to
	// somebody else. That is ErrExhausted from the allocator rather than from
	// the length check, and it must leave the lease untouched.
	p := newPool(t, Config{Block: testBlock, Blocks: 3})
	mine := acquire(t, p, Request{IDs: run(100, testBlock)})
	defer mine.Release()
	theirs := acquire(t, p, Request{IDs: run(500, 2*testBlock)})
	defer theirs.Release()

	if err := mine.Append(1); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Append with no free block = %v, want ErrExhausted", err)
	}
	if got := mine.Len(); got != testBlock {
		t.Fatalf("a refused Append changed the length to %d", got)
	}
	if got := len(mine.Blocks()); got != 1 {
		t.Fatalf("a refused Append left %d blocks, want 1", got)
	}
}

func TestAReleasedLeaseRefusesToGrow(t *testing.T) {
	p := testPool(t)
	l := acquire(t, p, Request{IDs: run(100, 5)})
	l.Release()
	if err := l.Append(1); !errors.Is(err, ErrReleased) {
		t.Fatalf("Append after Release = %v, want ErrReleased", err)
	}
	if got := l.Publish(l.Len()); len(got) != 2 {
		t.Fatalf("Publish after Release = %v, want the blocks unchanged", got)
	}
	if got := p.Stats().Publishes; got != 0 {
		t.Fatalf("a released lease published %d blocks, want 0", got)
	}
}

func TestReleaseIsIdempotent(t *testing.T) {
	// A second Release that decremented again would hand a live block to the
	// next request, which is 016-D5's invariant broken from the other side.
	p := testPool(t)
	prompt := run(100, 9)
	a := acquire(t, p, Request{IDs: prompt})
	a.Publish(a.Len())
	b := acquire(t, p, Request{IDs: prompt})
	defer b.Release()

	a.Release()
	a.Release()
	if got := p.blk[b.Blocks()[0]].refs; got != 1 {
		t.Fatalf("the shared block has %d references after a double release, want 1", got)
	}
	if s := p.Stats(); s.InUse != 3 {
		t.Fatalf("InUse = %d after a double release, want 3", s.InUse)
	}
}

func TestPublishingTwiceLearnsEachBlockOnce(t *testing.T) {
	p := testPool(t)
	l := acquire(t, p, Request{IDs: run(100, 9)})
	defer l.Release()
	first := l.Publish(l.Len())
	second := l.Publish(l.Len())
	if got := p.Stats().Publishes; got != 2 {
		t.Fatalf("Publishes = %d after publishing twice, want 2", got)
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("the page table moved between publishes: %v then %v", first, second)
		}
	}
}

func TestAnUnpublishedTailGoesStraightBackToTheFreeList(t *testing.T) {
	// Nothing names a partial block, so nothing can ever find it again. Held
	// as cache it would be a block the pool cannot use and cannot hit.
	p := testPool(t)
	l := acquire(t, p, Request{IDs: run(100, 9)}) // two whole blocks and one partial
	l.Publish(l.Len())
	l.Release()
	s := p.Stats()
	if s.Cached != 2 || s.Free != testBlocks-2 {
		t.Fatalf("Cached = %d and Free = %d, want 2 and %d", s.Cached, s.Free, testBlocks-2)
	}
}
