// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package grammar

import (
	"encoding/binary"
	"slices"
	"sync"
)

// dstate is one determinized state: a set of NFA states, plus the two caches
// that make a per-step mask affordable.
//
// 015-D1 is the design of this file. A grammar walk over 152k tokens per step
// is the naive implementation and it is too slow; the answer is to compute, per
// automaton state, the set of tokens that can be read from it, and to keep it,
// because states repeat heavily -- every `"` in a document with six string
// properties reaches the same state. Built on first visit rather than at
// compile time, because the eager version pays the whole vocabulary cross
// product at startup for states a request may never reach.
type dstate struct {
	// set is the epsilon-closed NFA state set, sorted. key is its encoding.
	set []int
	key string

	// acc is whether the document is complete here.
	acc bool

	// next is the byte transition cache. A present entry with a nil value
	// means the byte is dead, which is worth caching: most bytes are dead in
	// most states and recomputing that is the inner loop of the token walk.
	//
	// Guarded by Grammar.mu.
	next map[byte]*dstate

	// once guards the token cache below. After it has run, allowed and dest
	// are immutable and readable without the lock, which is what lets several
	// requests share one Grammar (015-D1's "cached across requests").
	once    sync.Once
	allowed []int
	dest    []*dstate
}

// encodeSet turns a sorted NFA state set into a map key.
//
// A varint encoding rather than fmt, because this runs once per newly reached
// state and the alternative formats a slice of ints into a string to compare it
// against other strings.
func encodeSet(set []int) string {
	buf := make([]byte, 0, len(set)*2)
	for _, s := range set {
		buf = binary.AppendUvarint(buf, uint64(s))
	}
	return string(buf)
}

// intern returns the dstate for an epsilon-closed set, creating it once.
//
// Called with mu held.
func (g *Grammar) intern(set []int) *dstate {
	key := encodeSet(set)
	if d, ok := g.states[key]; ok {
		return d
	}
	d := &dstate{
		set: set,
		key: key,
		acc: slices.Contains(set, g.accept),
	}
	g.states[key] = d
	return d
}

// stepLocked advances one byte, returning nil when the byte cannot be read.
//
// Called with mu held.
func (g *Grammar) stepLocked(d *dstate, b byte) *dstate {
	if to, ok := d.next[b]; ok {
		return to
	}
	var move []int
	for _, s := range d.set {
		for _, e := range g.n.move[s] {
			if b >= e.lo && b <= e.hi {
				move = append(move, e.to)
			}
		}
	}
	var to *dstate
	if len(move) > 0 {
		to = g.intern(g.n.closure(move))
	}
	if d.next == nil {
		d.next = make(map[byte]*dstate)
	}
	d.next[b] = to
	return to
}

// tokens fills d's admissible-token cache, once.
//
// The admissibility rule is specs/015-structured-output.md section 2 stated
// exactly: a token is admissible only if EVERY intermediate byte is. A token is
// walked byte by byte from d and kept only if the walk never dies, so `":` is
// admissible precisely where a string terminator may be followed by a colon and
// nowhere else. Checking only the first byte, or only the state the token lands
// in, would admit a token whose middle leaves the grammar.
func (g *Grammar) tokens(d *dstate) {
	d.once.Do(func() {
		g.builds.Add(1)
		// One acquisition for the whole vocabulary rather than one per byte.
		// The walk is the expensive part of a first visit and it is pure
		// lookup, so holding the lock across it costs less than the lock
		// traffic of releasing it 152k times.
		g.mu.Lock()
		defer g.mu.Unlock()

		var allowed []int
		var dest []*dstate
		for id := 0; id < g.size; id++ {
			if g.stopSet[id] {
				// A stop token is not text. It is admitted by acceptance
				// below, never by its bytes -- a tokenizer that hands back the
				// surface form of a control token would otherwise let
				// <|im_end|> be typed inside a JSON string.
				continue
			}
			bs := g.v.Bytes(id)
			if len(bs) == 0 {
				// No text: a control token, or a hole in the vocabulary. It
				// consumes no bytes, so admitting it would let the mask
				// approve a step that advances nothing.
				continue
			}
			cur := d
			for _, b := range bs {
				if cur = g.stepLocked(cur, b); cur == nil {
					break
				}
			}
			if cur != nil {
				allowed = append(allowed, id)
				dest = append(dest, cur)
			}
		}
		if d.acc {
			allowed, dest = mergeStop(allowed, dest, g.stop)
		}
		d.allowed, d.dest = allowed, dest
	})
}

// mergeStop folds the stop ids into a sorted admissible set, with a nil
// destination standing for "the document ends here".
//
// The ids stay sorted because Mask walks the mask and the admissible set
// together in one pass, and Advance binary-searches it.
func mergeStop(allowed []int, dest []*dstate, stop []int) ([]int, []*dstate) {
	outIDs := make([]int, 0, len(allowed)+len(stop))
	outDest := make([]*dstate, 0, len(allowed)+len(stop))
	i, j := 0, 0
	for i < len(allowed) || j < len(stop) {
		switch {
		case j == len(stop) || (i < len(allowed) && allowed[i] < stop[j]):
			outIDs = append(outIDs, allowed[i])
			outDest = append(outDest, dest[i])
			i++
		default:
			outIDs = append(outIDs, stop[j])
			outDest = append(outDest, nil)
			j++
		}
	}
	return outIDs, outDest
}
