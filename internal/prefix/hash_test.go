// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package prefix

import "testing"

func TestTheChainMakesABlocksHashDependOnEverythingBeforeIt(t *testing.T) {
	// h_i = H(h_{i-1} || ids[iB:(i+1)B]). The same block content behind a
	// different predecessor is a different block, which is what turns a
	// content key into a prefix key (016-D2).
	s := seed(ScopeProcess, "", "")
	shared := run(9000, 4)
	first := chainAll(s, append(run(100, 4), shared...), 4)
	second := chainAll(s, append(run(500, 4), shared...), 4)

	if first[0] == second[0] {
		t.Fatal("two different leading blocks hashed the same")
	}
	if first[1] == second[1] {
		t.Fatal("the same interior run hashed the same behind different " +
			"predecessors; the hash is a content key, not a prefix key")
	}
	if again := chainAll(s, append(run(100, 4), shared...), 4); again[1] != first[1] {
		t.Fatal("the same prefix hashed to two values")
	}
}

func TestOnlyCompleteBlocksAreHashed(t *testing.T) {
	// A partial block is not shareable: sharing it would mean two sequences
	// writing one block at different offsets (016-D4).
	s := seed(ScopeProcess, "", "")
	if got := len(chainAll(s, run(100, 3), 4)); got != 0 {
		t.Fatalf("a partial block produced %d hashes, want 0", got)
	}
	if got := len(chainAll(s, run(100, 9), 4)); got != 2 {
		t.Fatalf("nine positions produced %d hashes, want 2", got)
	}
}

func TestIdsAreEncodedFixedWidth(t *testing.T) {
	// A decimal or varint encoding gives [1, 23] and [12, 3] one byte stream,
	// and the pool would hand one prompt the other's KV.
	s := seed(ScopeProcess, "", "")
	if chain(s, []int{1, 23}) == chain(s, []int{12, 3}) {
		t.Fatal("two id sequences share a hash")
	}
	if chain(s, []int{0, 0}) == chain(s, []int{0}) {
		t.Fatal("a shorter id sequence hashed like a longer one")
	}
}

func TestTheSaltReachesEveryBlockThroughTheChain(t *testing.T) {
	// 016 §7.1: the salt is mixed into h_0 and the chain carries it to the end
	// of the sequence, so a salted prompt shares no block with an unsalted one.
	ids := run(100, 12)
	plain := chainAll(seed(ScopeProcess, "", ""), ids, 4)
	salted := chainAll(seed(ScopeProcess, "", "tenant-7"), ids, 4)
	for i := range plain {
		if plain[i] == salted[i] {
			t.Fatalf("block %d hashed the same with and without a salt", i)
		}
	}
}
