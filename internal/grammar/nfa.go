// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package grammar

import "slices"

// The automaton is over BYTES, not over runes and not over characters.
//
// It has to be. specs/015-structured-output.md section 2 is the whole reason
// this package exists: the mask is over the tokenizer's vocabulary, and a BPE
// token is a byte string that need not align to a character, to a UTF-8
// sequence, or to a grammar boundary. `":` is one token and it is a string
// terminator followed by a structural character; a multi-byte rune can arrive
// split across two tokens. An automaton whose alphabet is the rune could not
// step through either. So the alphabet is the byte, and well-formed UTF-8 is
// spelled out as byte ranges in schema.go rather than assumed.

// brange is one inclusive byte range in a character class.
type brange struct{ lo, hi byte }

// edge is one byte-range transition.
type edge struct {
	lo, hi byte
	to     int
}

// nfa is a Thompson automaton: epsilon edges and byte-range edges, states
// numbered from zero.
//
// Thompson construction rather than a direct DFA build because the front end
// composes fragments -- an alternation of enum literals, a bounded repetition
// of a string character -- and composition is what epsilon edges are for. The
// determinization in dfa.go is lazy, so the nondeterminism costs nothing until
// a state is actually visited.
type nfa struct {
	eps  [][]int
	move [][]edge

	// mark is closure's visited set, kept here so a closure does not allocate
	// one per call. It is only ever touched under the grammar's lock, and
	// every entry closure sets it also clears.
	mark []bool
}

// frag is a sub-automaton with one entry and one exit. Both may be the same
// state; nothing downstream cares.
type frag struct{ in, out int }

func (n *nfa) state() int {
	n.eps = append(n.eps, nil)
	n.move = append(n.move, nil)
	n.mark = append(n.mark, false)
	return len(n.eps) - 1
}

func (n *nfa) link(from, to int) { n.eps[from] = append(n.eps[from], to) }

// closure returns the epsilon closure of seed, sorted ascending and deduped.
//
// Sorted because the sorted set IS the DFA state's identity: dfa.go keys the
// state cache on it, and two seeds that reach the same set must produce the
// same key or the cache that 015-D1 calls "the design" would hold a separate
// entry per path taken to a state.
func (n *nfa) closure(seed []int) []int {
	stack := slices.Clone(seed)
	var out []int
	for len(stack) > 0 {
		s := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if n.mark[s] {
			continue
		}
		n.mark[s] = true
		out = append(out, s)
		stack = append(stack, n.eps[s]...)
	}
	for _, s := range out {
		n.mark[s] = false
	}
	slices.Sort(out)
	return out
}

// empty matches the empty string.
func (n *nfa) empty() frag {
	s := n.state()
	return frag{s, s}
}

// lit matches the bytes of s exactly.
func (n *nfa) lit(s string) frag {
	in := n.state()
	cur := in
	for i := 0; i < len(s); i++ {
		next := n.state()
		n.move[cur] = append(n.move[cur], edge{s[i], s[i], next})
		cur = next
	}
	return frag{in, cur}
}

// class matches one byte from any of the ranges.
func (n *nfa) class(rs ...brange) frag {
	in, out := n.state(), n.state()
	for _, r := range rs {
		n.move[in] = append(n.move[in], edge{r.lo, r.hi, out})
	}
	return frag{in, out}
}

// cat concatenates. An empty list is the empty string.
func (n *nfa) cat(fs ...frag) frag {
	if len(fs) == 0 {
		return n.empty()
	}
	for i := 1; i < len(fs); i++ {
		n.link(fs[i-1].out, fs[i].in)
	}
	return frag{fs[0].in, fs[len(fs)-1].out}
}

// alt is the union. An empty list matches nothing at all, which is a language
// with no members -- the front end never builds one, but the shape is right:
// an entry state with no way out is dead, and dead is what "no member" means.
func (n *nfa) alt(fs ...frag) frag {
	in, out := n.state(), n.state()
	for _, f := range fs {
		n.link(in, f.in)
		n.link(f.out, out)
	}
	return frag{in, out}
}

// opt matches f or the empty string.
func (n *nfa) opt(f frag) frag {
	in, out := n.state(), n.state()
	n.link(in, f.in)
	n.link(f.out, out)
	n.link(in, out)
	return frag{in, out}
}

// star matches zero or more of what gen builds.
//
// gen is a generator rather than a fragment because the loop needs exactly one
// copy and the bounded repetition below needs several; taking a generator lets
// both go through the same builder without either having to know which.
func (n *nfa) star(gen func() frag) frag {
	in, out := n.state(), n.state()
	f := gen()
	n.link(in, f.in)
	n.link(f.out, in)
	n.link(in, out)
	return frag{in, out}
}

// rep matches between min and max copies, max < 0 meaning unbounded.
//
// The bounded tail is nested optionals rather than a counter, because the
// automaton has no registers: "at most three more" is three nested "one more,
// or stop". That is what makes minLength and maxItems regular and what makes
// minimum and multipleOf not -- a numeric bound is arithmetic on the value, and
// no amount of nesting spells arithmetic.
func (n *nfa) rep(gen func() frag, min, max int) frag {
	var parts []frag
	for range min {
		parts = append(parts, gen())
	}
	switch {
	case max < 0:
		parts = append(parts, n.star(gen))
	case max > min:
		// Build the optional tail inside out, so the innermost copy is the
		// last one allowed.
		tail := n.empty()
		for i := 0; i < max-min; i++ {
			tail = n.opt(n.cat(gen(), tail))
		}
		parts = append(parts, tail)
	}
	return n.cat(parts...)
}

// size is how many states the automaton holds. schema.go's maxStates is a
// bound on it, checked where the compilation recurses.
func (n *nfa) size() int { return len(n.eps) }
