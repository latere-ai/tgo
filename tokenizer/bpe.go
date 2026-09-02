// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tokenizer

// Byte-pair encoding, specs/002-tokenizer.md section 5.

// pair is one merge-table key.
type pair struct{ left, right string }

// merge repeatedly joins the adjacent symbol pair with the lowest merge rank,
// where the rank is the pair's index in the ordered merge list, and breaks a
// rank tie by taking the leftmost such pair (002-D1).
//
// The tie-break is the whole reason this is written out rather than folded into
// a scan. Ranks are unique per pair, so a tie arises exactly when the *same*
// pair occupies two overlapping positions -- "eee" with ("e","e") in it -- and
// the strict < below is what makes the leftmost one win. Writing <= here would
// take the rightmost, would still look like a correct reading of "the lowest
// rank", and would produce different ids for about one short string in thirty.
// ("Ġ","Ġ") is rank 0 in the Qwen tables, so ordinary double spaces exercise
// this path constantly.
//
// The scan is the naive O(n^2) of 002-D2: a piece from section 4's split is a
// word, and the heap that would make it O(n log n) needs a benchmark behind it.
func (t *Tokenizer) merge(syms []string) []string {
	for len(syms) > 1 {
		best, bestRank := -1, 0
		for i := 0; i+1 < len(syms); i++ {
			rank, ok := t.merges[pair{syms[i], syms[i+1]}]
			if !ok {
				continue
			}
			if best < 0 || rank < bestRank {
				best, bestRank = i, rank
			}
		}
		if best < 0 {
			break
		}
		syms[best] += syms[best+1]
		syms = append(syms[:best+1], syms[best+2:]...)
	}
	return syms
}

// symbols splits a byte-mapped piece into one symbol per code point, the
// starting state of the merge loop.
func symbols(piece string) []string {
	out := make([]string, 0, len(piece))
	for _, r := range piece {
		out = append(out, string(r))
	}
	return out
}
