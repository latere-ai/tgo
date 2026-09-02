// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package sample

import (
	"math"
	"slices"
)

// dist is a post-policy distribution: the kept token ids and their softmax
// weights, unnormalized.
//
// Unnormalized, and the total carried beside them, because that is what accel's
// SampleCategorical walks -- it compares against draw*total rather than against
// the draw, so a mask that leaves the weights summing below one needs no
// renormalizing pass. Dividing here and multiplying there would be two roundings
// where the device has none.
type dist struct {
	// ids are the kept token ids, ascending. Nil means every id in the
	// vocabulary is kept, and w is then indexed by id rather than parallel.
	ids   []int
	w     []float32
	total float32
}

// walk returns the token whose cumulative weight first exceeds u*total.
//
//	token = min{ i : sum_{j<=i} w_j > u * total }
//
// The walk is sequential and in ascending id order, exactly as accel's
// SampleCategorical does it: a parallel prefix sum would form different partial
// sums and put the boundary elsewhere when two weights are equal, and accel 028
// takes reproducibility over speed there. Skipping the zeroed ids changes
// nothing, since adding zero is exact.
func (d dist) walk(u float32) int {
	target := u * d.total
	acc := float32(0)
	last := 0
	for i, w := range d.w {
		id := i
		if d.ids != nil {
			id = d.ids[i]
		}
		acc += w
		if acc > target {
			return id
		}
		last = id
	}
	// Reached only when rounding leaves the running sum at or below the
	// target after every entry. The last kept id is the answer, as in the
	// kernel.
	return last
}

// spread writes the normalized distribution into out, which must be the
// vocabulary length and zeroed.
func (d dist) spread(out []float32) {
	for i, w := range d.w {
		id := i
		if d.ids != nil {
			id = d.ids[i]
		}
		out[id] = w / d.total
	}
}

// policyDist applies specs/006-sampling.md section 3 to logits, in place, and
// returns the distribution the draw is taken against.
//
// scratch is a vocabulary-length buffer this may take over as the weight
// vector; it is never returned to the caller of the package.
func policyDist(logits []float32, history []int, p Policy, scratch []float32) dist {
	// Bias first. A logit bias is the caller's absolute statement about a
	// token, so a penalty computed on a biased logit still means what it says,
	// while biasing an already penalised one does not.
	for id, b := range p.LogitBias {
		logits[id] += b
	}

	penalize(logits, history, p)

	// Temperature zero is greedy, and greedy is a branch rather than a
	// division. It is taken here because everything above reorders candidates
	// and everything below cannot: temperature and the two truncations are
	// argmax-invariant, so running them before the branch could not change the
	// answer and running them after is free.
	if p.Temperature == 0 {
		return dist{ids: []int{argmax(logits)}, w: []float32{1}, total: 1}
	}

	// Temperature before the truncations. Top-p is a mass threshold and
	// temperature is what changes the mass, so truncating first would make the
	// same p mean a different set at every temperature.
	for i := range logits {
		logits[i] /= p.Temperature
	}

	switch {
	case p.TopK > 0:
		// Top-k before top-p: k is a hard cap on the candidate count and p
		// then trims within it. The reverse order lets p admit more than k.
		//
		// The selection is on the softmax **weights**, which is where accel
		// makes it: TopKMask runs after the softmax, so its (value, index) tie
		// rule compares exp(l - m) rather than l.
		//
		// Selecting on the logits gives the same set almost everywhere,
		// because exp is monotonic -- but not at the tail, where it is
		// many-to-one in f32. Two logits that differ as f32 can share one f32
		// weight, and there the rules disagree: on logits the larger one wins,
		// and accel, seeing two equal values, keeps the smaller id. A k = 128
		// boundary over a 152k vocabulary sits deep in that tail, so the case
		// is reachable rather than theoretical, and [006-D1] makes this path
		// the reference the device is checked against.
		//
		// It costs a whole-vocabulary exp pass that selecting on the logits
		// did not. That is the same pass the top-p branch below already takes,
		// and correctness against the reference is what it buys.
		all, _ := weightsAll(logits, scratch)
		cand := topN(all, min(p.TopK, len(all)))
		w, total := pick(all, cand)
		if p.TopP > 0 {
			// The nucleus is a fraction of its input's own total, which here
			// is the top-k set: accel's TopPMask sums what it is given.
			cand, w, total = nucleus(cand, w, total, p.TopP)
		}
		return keep(cand, w, total)

	case p.TopP > 0:
		// No top-k, so the nucleus is a fraction of the whole vocabulary's
		// mass -- but it can still only be as wide as accel's kernel walks.
		// The 128 largest candidates are therefore enough to build it.
		all, total := weightsAll(logits, scratch)
		cand := topN(all, min(TopMaxRounds, len(all)))
		w, _ := pick(all, cand)
		cand, w, total = nucleus(cand, w, total, p.TopP)
		return keep(cand, w, total)

	default:
		// No truncation stage at all: softmax over the whole vocabulary.
		w, total := weightsAll(logits, scratch)
		return dist{w: w, total: total}
	}
}

// penalize applies section 3.1 to the logits, over the window of history.
//
// The window covers prompt and generated tokens together. Penalising only what
// the model produced lets it repeat the prompt verbatim.
func penalize(logits []float32, history []int, p Policy) {
	rep := p.RepetitionPenalty != 0 && p.RepetitionPenalty != 1
	add := p.PresencePenalty != 0 || p.FrequencyPenalty != 0
	if !rep && !add {
		return
	}

	window := history
	if p.PenaltyWindow > 0 && len(window) > p.PenaltyWindow {
		window = window[len(window)-p.PenaltyWindow:]
	}
	counts := make(map[int]int, len(window))
	for _, id := range window {
		counts[id]++
	}

	for id, c := range counts {
		if rep {
			// The asymmetry is the point. Dividing a negative logit by r > 1
			// moves it toward zero, which makes a penalised token *more*
			// likely; multiplying keeps the penalty monotone on both sides.
			// Zero takes the multiply branch, so a zero logit is left alone
			// either way but the branch is the one section 3.1 names.
			if logits[id] > 0 {
				logits[id] /= p.RepetitionPenalty
			} else {
				logits[id] *= p.RepetitionPenalty
			}
		}
		if add {
			// Presence is once per token that appears, frequency once per
			// occurrence. Repetition is also once per token, not once per
			// occurrence: it is not applied c times.
			logits[id] -= p.PresencePenalty + p.FrequencyPenalty*float32(c)
		}
	}
}

// argmax returns the index of the largest logit, ties to the lowest index.
//
// The scan is upward and the comparison is strict, so an equal pair keeps the
// first -- accel's SampleArgmax tie rule. Equal logits are ordinary at
// saturation, and a rule nobody stated would let two backends return different
// tokens.
func argmax(logits []float32) int {
	best := logits[0]
	at := 0
	for i, v := range logits[1:] {
		if v > best {
			best = v
			at = i + 1
		}
	}
	return at
}

// topN returns the n largest logits as ids, ordered by (value, id)
// lexicographically descending: larger value first, and the smaller id first
// among equals.
//
// Selection rather than a sort. A vocabulary is 152k entries and n is at most
// TopMaxRounds, so a full sort would be thousands of comparisons to answer a
// question about the first few dozen -- and its tie behaviour would be the
// standard library's rather than the kernel's.
func topN(logits []float32, n int) []int {
	out := make([]int, 0, n)
	for i, v := range logits {
		if len(out) == n && !above(v, i, logits[out[n-1]], out[n-1]) {
			continue
		}
		if len(out) < n {
			out = append(out, 0)
		}
		// Shift the entries this one outranks up by one. Ids arrive
		// ascending, so an equal value stops the shift and lands after the
		// entries it ties with, which is the tie rule.
		j := len(out) - 1
		for j > 0 && above(v, i, logits[out[j-1]], out[j-1]) {
			out[j] = out[j-1]
			j--
		}
		out[j] = i
	}
	return out
}

// above reports whether (v, i) outranks (w, j) lexicographically.
func above(v float32, i int, w float32, j int) bool {
	return v > w || (v == w && i < j)
}

// weightsAll fills buf with exp(l - max) over the whole vocabulary and returns
// it with its total.
//
// The maximum is subtracted for the usual reason, and the total is summed in
// ascending id order. That order is not the kernel's -- accel reduces across
// 128 lanes and then a tree -- so the two totals differ in their last bits,
// which is specs/010-conformance.md section 5 territory rather than something
// either side can fix.
func weightsAll(logits, buf []float32) ([]float32, float32) {
	m := logits[argmax(logits)]
	total := float32(0)
	for i, v := range logits {
		e := expf(v - m)
		buf[i] = e
		total += e
	}
	return buf, total
}

// pick gathers the candidates' weights out of the whole-vocabulary vector,
// parallel to ids, with their total.
//
// It reads weights rather than recomputing exp from the logits, so the number a
// candidate is selected by is the number it is then weighted by. Recomputing
// them is where the two could disagree, and a set chosen on one quantity and
// weighted by another is the defect this file just fixed, one step further on.
func pick(all []float32, ids []int) ([]float32, float32) {
	w := make([]float32, len(ids))
	total := float32(0)
	for i, id := range ids {
		w[i] = all[id]
		total += w[i]
	}
	return w, total
}

// expf is exp in float32, the width the device computes it in.
func expf(x float32) float32 { return float32(math.Exp(float64(x))) }

// nucleus keeps the smallest prefix of the descending candidates whose weight
// reaches p of total, and returns it with its own total.
//
// The entry that *crosses* the threshold is kept, which is what makes the set
// the smallest one reaching p rather than the largest one below it (006-D6).
//
// The candidate list is at most TopMaxRounds long, so a nucleus wider than 128
// tokens is capped rather than refused: accel's TopPMask walks 128 rounds and
// stops whether or not it reached its mass, and a host reference that panicked
// where the device kept 128 would not be a reference. specs/006-sampling.md
// section 3 says such a policy "must refuse"; refusing here would abort a
// decode loop on the data rather than on the configuration, since a p of 0.95
// over a 152k vocabulary exceeds 128 candidates routinely.
func nucleus(ids []int, w []float32, total, p float32) ([]int, []float32, float32) {
	target := total * p
	acc := float32(0)
	n := 0
	for n < len(ids) && acc < target {
		acc += w[n]
		n++
	}
	return ids[:n], w[:n], acc
}

// keep turns a candidate list into a dist with its ids ascending.
//
// Ascending because the categorical walk goes in id order, and the ids arrive
// from the truncation stages in descending value order.
func keep(ids []int, w []float32, total float32) dist {
	order := make([]int, len(ids))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return ids[a] - ids[b] })

	sortedIDs := make([]int, len(ids))
	sortedW := make([]float32, len(w))
	for i, o := range order {
		sortedIDs[i] = ids[o]
		sortedW[i] = w[o]
	}
	return dist{ids: sortedIDs, w: sortedW, total: total}
}
