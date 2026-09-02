// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package oracle

// Penalize is specs/006-sampling.md §3.1's three penalties, in float64.
//
// It is written from the spec's prose rather than from sample/stages.go, which
// is the whole point of it: the shipped implementation was checked against
// hand-computed constants written by the same reader in the same sitting, so
// the two agreed about what the rules meant rather than about what they are.
// 010 §5 calls that a reference and not a restatement.
//
// The rules, in the order §3.1 states them:
//
//   - repetition applies **once per distinct token** in the window, not once
//     per occurrence, and it divides a positive logit while it multiplies a
//     non-positive one. Dividing a negative logit by r > 1 moves it toward
//     zero and makes a penalised token more likely, which is the classic bug
//     the asymmetry exists to avoid;
//   - presence subtracts once per distinct token;
//   - frequency subtracts once per occurrence.
//
// A repetition penalty of 0 or 1 is no penalty: 1 is the identity and 0 is the
// zero value of a field a caller did not set, which would otherwise multiply
// every repeated logit to zero.
//
// window is the number of most recent history entries to count, or 0 for all of
// them.
func Penalize(logits []float64, history []int, repetition, presence, frequency float64, window int) []float64 {
	out := make([]float64, len(logits))
	copy(out, logits)

	rep := repetition != 0 && repetition != 1
	add := presence != 0 || frequency != 0
	if !rep && !add {
		return out
	}

	seen := history
	if window > 0 && len(seen) > window {
		seen = seen[len(seen)-window:]
	}
	counts := map[int]int{}
	for _, id := range seen {
		counts[id]++
	}

	for id, c := range counts {
		if id < 0 || id >= len(out) {
			continue
		}
		if rep {
			if out[id] > 0 {
				out[id] /= repetition
			} else {
				out[id] *= repetition
			}
		}
		if add {
			out[id] -= presence + frequency*float64(c)
		}
	}
	return out
}
