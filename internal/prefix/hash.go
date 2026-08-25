// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package prefix

import (
	"crypto/sha256"
	"encoding/binary"
)

// label separates this construction from any other use of SHA-256 in tgo, and
// versions it: a change to the encoding below changes the label, so blocks
// hashed under the old one can never match under the new.
const label = "tgo/prefix/v1"

// seed is h_{-1}: the chain's starting value, carrying the isolation scope, the
// scope's domain and the request's salt.
//
// Mixing them here rather than into each block is what 016 §7.1 means by "the
// chain propagates it": one hash of the seed puts the salt in h_0, and h_0 is
// in h_1, and so on to the end of the sequence. Blocks match only within one
// (scope, domain, salt).
//
// Every variable-length field is length-prefixed, so a session named "a" with
// salt "b" cannot produce the seed of a session named "ab" with no salt. That
// is the same confusion the scope exists to prevent, arriving through the
// encoding instead.
func seed(s Scope, domain, salt string) [32]byte {
	h := sha256.New()
	h.Write([]byte(label))
	var u [8]byte
	binary.LittleEndian.PutUint64(u[:], uint64(s))
	h.Write(u[:])
	for _, f := range [...]string{domain, salt} {
		binary.LittleEndian.PutUint64(u[:], uint64(len(f)))
		h.Write(u[:])
		h.Write([]byte(f))
	}
	var out [32]byte
	h.Sum(out[:0])
	return out
}

// chain is h_i = H(h_{i-1} || ids), the hash of one block.
//
// H is SHA-256 (016-D9). A collision here does not corrupt data — it hands one
// request another request's KV and the output stays fluent — so the choice is
// adversarial rather than a matter of speed. A fast non-cryptographic hash
// would need a per-process random seed, which vLLM learned by shipping a
// predictable one (vllm#12621).
//
// Ids are encoded fixed-width, so no two id sequences share a byte stream:
// a decimal or varint encoding makes [1, 23] and [12, 3] the same input.
func chain(prev [32]byte, ids []int) [32]byte {
	h := sha256.New()
	h.Write(prev[:])
	var u [8]byte
	for _, id := range ids {
		binary.LittleEndian.PutUint64(u[:], uint64(id))
		h.Write(u[:])
	}
	var out [32]byte
	h.Sum(out[:0])
	return out
}

// chainAll hashes every complete block of ids. A trailing partial block gets no
// hash: it is not shareable, because sharing it would mean two sequences
// writing one block at different offsets (016-D4).
func chainAll(s [32]byte, ids []int, block int) [][32]byte {
	n := len(ids) / block
	if n == 0 {
		return nil
	}
	out := make([][32]byte, 0, n)
	prev := s
	for i := range n {
		prev = chain(prev, ids[i*block:(i+1)*block])
		out = append(out, prev)
	}
	return out
}
