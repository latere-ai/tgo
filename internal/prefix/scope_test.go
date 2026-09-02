// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package prefix

import "testing"

func TestSessionScopeDoesNotShareAcrossSessions(t *testing.T) {
	// 016-D7. A hit is faster than a miss and that timing is observable, so a
	// prefix shared across sessions is a membership oracle over another
	// session's prompts.
	p := newPool(t, Config{Block: testBlock, Blocks: testBlocks, Scope: ScopeSession})
	prompt := run(100, 9)
	warm(t, p, Request{IDs: prompt, Session: "alice"})

	other := acquire(t, p, Request{IDs: prompt, Session: "bob"})
	defer other.Release()
	if other.Matched() != 0 {
		t.Fatalf("another session matched %d blocks, want 0", other.Matched())
	}

	same := acquire(t, p, Request{IDs: prompt, Session: "alice"})
	defer same.Release()
	if got, want := same.Matched(), 2; got != want {
		t.Fatalf("the same session matched %d blocks, want %d -- the multi-turn "+
			"win is the reason this scope is usable", got, want)
	}
}

func TestASessionNameCannotBeConfusedWithASalt(t *testing.T) {
	// Session "a" with salt "b" and session "ab" with no salt are different
	// domains. An unprefixed concatenation makes them one, which is the leak
	// the scope exists to close arriving through the encoding.
	p := newPool(t, Config{Block: testBlock, Blocks: testBlocks, Scope: ScopeSession})
	prompt := run(100, 9)
	warm(t, p, Request{IDs: prompt, Session: "a", Salt: "b"})

	l := acquire(t, p, Request{IDs: prompt, Session: "ab"})
	defer l.Release()
	if l.Matched() != 0 {
		t.Fatalf("session %q matched %d blocks of session %q's prefix, want 0",
			"ab", l.Matched(), "a")
	}
}

func TestASaltIsolatesAnOtherwiseIdenticalPrompt(t *testing.T) {
	// The caller-supplied cache_salt of 016 §7.1: the layer that knows who the
	// caller is salts by tenant, and blocks match only within one salt.
	p := testPool(t)
	prompt := run(100, 9)
	warm(t, p, Request{IDs: prompt, Salt: "tenant-7"})

	stranger := acquire(t, p, Request{IDs: prompt})
	if stranger.Matched() != 0 {
		t.Fatalf("an unsalted request matched %d blocks of a salted prefix, want 0",
			stranger.Matched())
	}
	stranger.Release()

	elsewhere := acquire(t, p, Request{IDs: prompt, Salt: "tenant-9"})
	if elsewhere.Matched() != 0 {
		t.Fatalf("another tenant matched %d blocks, want 0", elsewhere.Matched())
	}
	elsewhere.Release()

	mine := acquire(t, p, Request{IDs: prompt, Salt: "tenant-7"})
	defer mine.Release()
	if got, want := mine.Matched(), 2; got != want {
		t.Fatalf("the same salt matched %d blocks, want %d", got, want)
	}
}

func TestScopeOffSharesNothingWithItself(t *testing.T) {
	// The cold baseline 016 §7 measures against. It still hands out blocks, so
	// the engine keeps one code path.
	p := newPool(t, Config{Block: testBlock, Blocks: testBlocks, Scope: ScopeOff})
	prompt := run(100, 9)
	warm(t, p, Request{IDs: prompt})

	l := acquire(t, p, Request{IDs: prompt})
	defer l.Release()
	if l.Matched() != 0 || l.Reused() != 0 {
		t.Fatalf("scope off matched %d blocks and reused %d positions, want 0 and 0",
			l.Matched(), l.Reused())
	}
	if got := len(l.Blocks()); got != 3 {
		t.Fatalf("scope off handed out %d blocks, want 3", got)
	}
	if s := p.Stats(); s.Publishes != 0 || s.Cached != 0 {
		t.Fatalf("scope off published %d blocks and cached %d, want 0 and 0",
			s.Publishes, s.Cached)
	}
	// Every block comes back, because none of them is the cache.
	l.Release()
	if got := p.Stats().Free; got != testBlocks {
		t.Fatalf("Free = %d after scope off released everything, want %d",
			got, testBlocks)
	}
}

func TestScopeOffStillGrowsWithTheSequence(t *testing.T) {
	p := newPool(t, Config{Block: testBlock, Blocks: testBlocks, Scope: ScopeOff})
	l := acquire(t, p, Request{IDs: run(100, 5)})
	defer l.Release()
	if err := l.Append(1, 2, 3, 4); err != nil {
		t.Fatalf("Append = %v", err)
	}
	if got := len(l.Blocks()); got != 3 {
		t.Fatalf("the grown sequence holds %d blocks, want 3", got)
	}
	if got := p.Stats().Publishes; got != 0 {
		t.Fatalf("scope off published %d blocks, want 0", got)
	}
}

func TestTwoPoolsWithTheSameScopeAgreeOnAKey(t *testing.T) {
	// The hash is a pure function of (scope, domain, salt, ids), with no
	// per-process randomness: 016-D9 keeps SHA-256 deterministic so separate
	// processes could share a cache without weakening collision resistance.
	a := seed(ScopeProcess, "", "pepper")
	b := seed(ScopeProcess, "", "pepper")
	if a != b {
		t.Fatal("the same scope, domain and salt produced two seeds")
	}
	if seed(ScopeProcess, "", "pepper") == seed(ScopeSession, "", "pepper") {
		t.Fatal("two scopes produced one seed")
	}
}
