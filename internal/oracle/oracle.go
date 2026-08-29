// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package oracle is a host-side float64 implementation of a transformer
// forward pass, written from the mathematics rather than from tgo's graph.
//
// It is the reference every device result is checked against
// (specs/010-conformance.md §5). Its value comes from being an independent
// derivation: it is written by reading the equations in
// specs/004-model-graph.md §2, it imports nothing from tgo's nn or model
// packages, and it takes weights as []float64 and shapes as plain ints rather
// than as tgo types. Two derivations of the same mathematics agreeing is
// evidence; one derivation compared against itself is not (010-D2).
//
// Everything here is float64 and everything here is slow. Triple-nested loops
// are the point: on a disagreement with the device the oracle is presumed
// right, because it is the simpler program (010-D5). Nothing in this package
// may be rewritten for speed at the cost of being obviously correct.
//
// Shapes are carried in the arguments, not in the slices. A tensor of shape
// [a, b, c] is a flat []float64 of length a*b*c in row-major order, so the
// element at (i, j, k) is at i*b*c + j*c + k. A shape that does not match its
// slice is a programming error in the caller and panics; these are refusals in
// the sense of specs/004-model-graph.md §7, and a wrong shape that computed
// something anyway would be the accept-and-silently-wrong class that
// specs/010-conformance.md §2 exists to catch.
package oracle

import "math"

// RMSNorm normalizes each row of x over its last axis and scales by gain:
//
//	y_i = x_i / sqrt(mean(x^2) + eps) * g_i
//
// as specs/004-model-graph.md §2.2 defines it.
//
// The row count is inferred from len(x)/len(gain) rather than passed, and that
// is what makes Qwen3's per-head QK-norm a reshape rather than a second
// operator (004 §2.4): normalizing [T·H, d_h] with a d_h gain is this same
// call, while normalizing over the full H·d_h row — the bug 004 §2.4 names —
// is a different one that a caller has to write on purpose.
//
// The mean is over the row width, so eps sits inside the square root and is
// not a floor on the result.
func RMSNorm(x, gain []float64, eps float64) []float64 {
	width := len(gain)
	if width == 0 {
		panic("oracle: RMSNorm gain is empty")
	}
	if len(x)%width != 0 {
		panic("oracle: RMSNorm x length is not a multiple of the gain width")
	}
	rows := len(x) / width
	out := make([]float64, len(x))
	for r := range rows {
		row := x[r*width : (r+1)*width]
		sum := 0.0
		for _, v := range row {
			sum += v * v
		}
		scale := 1 / math.Sqrt(sum/float64(width)+eps)
		for c, v := range row {
			out[r*width+c] = v * scale * gain[c]
		}
	}
	return out
}

// MatMul multiplies x[m,k] by w[k,n] and returns [m,n], all row-major.
//
// This is specs/004-model-graph.md §2.1's projection, y = xW, with no bias:
// Qwen3 has none on its projections.
//
// The accumulation order is the naive one, first index to last. A tiled GEMM
// on a device accumulates in a different order, and the difference is the
// sqrt(K)·eps_32 term that specs/010-conformance.md §5.1 derives. In float64
// that term is ~1e-16 relative and is not what any tolerance here is set by.
func MatMul(x, w []float64, m, k, n int) []float64 {
	if m < 0 || k < 0 || n < 0 {
		panic("oracle: MatMul has a negative dimension")
	}
	if len(x) != m*k {
		panic("oracle: MatMul x length does not match m*k")
	}
	if len(w) != k*n {
		panic("oracle: MatMul w length does not match k*n")
	}
	out := make([]float64, m*n)
	for i := range m {
		for j := range n {
			sum := 0.0
			for p := range k {
				sum += x[i*k+p] * w[p*n+j]
			}
			out[i*n+j] = sum
		}
	}
	return out
}

// SiLU applies x·sigmoid(x) elementwise.
//
// Written as x/(1+exp(-x)). For very negative x the exponential overflows to
// +Inf and the quotient underflows to zero, which is the mathematical limit,
// so no branch is needed to keep it finite.
func SiLU(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v / (1 + math.Exp(-v))
	}
	return out
}

// SwiGLU returns SiLU(gate) ⊙ up, the activation of
// specs/004-model-graph.md §2.3's MLP.
//
// This is the composed SiLU-then-multiply form on purpose: 004-D5 keeps the
// composition as the reference a fused kernel is checked against, so that a
// fusion bug is a test failure rather than a quality loss nobody can see.
func SwiGLU(gate, up []float64) []float64 {
	if len(gate) != len(up) {
		panic("oracle: SwiGLU gate and up lengths differ")
	}
	out := SiLU(gate)
	for i := range out {
		out[i] *= up[i]
	}
	return out
}

// Softmax returns the normalized exponentials of x over the whole vector.
//
// The maximum is subtracted before exponentiating. That is algebraically an
// identity — exp(x_i-c)/sum(exp(x_j-c)) is independent of c — and numerically
// it is what keeps the largest exponential at exactly 1 instead of overflowing.
// specs/010-conformance.md §5.1 calls the softmax term benign for that reason.
func Softmax(x []float64) []float64 {
	if len(x) == 0 {
		panic("oracle: Softmax of an empty vector")
	}
	max := math.Inf(-1)
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	out := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		e := math.Exp(v - max)
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// GatherRows returns the rows of table named by ids, as [len(ids), width].
//
// This is the embedding lookup of specs/004-model-graph.md §3 row 3. The table
// is [vocab, width] and is already row-per-token, which is why 004 §4 marks
// the embedding as the one weight that is not transposed at load.
//
// An id outside [0, vocab) panics rather than clamping: a clamped id produces
// a plausible embedding for the wrong token, which is fluent wrong output.
func GatherRows(table []float64, vocab, width int, ids []int) []float64 {
	if vocab < 0 || width < 0 {
		panic("oracle: GatherRows has a negative dimension")
	}
	if len(table) != vocab*width {
		panic("oracle: GatherRows table length does not match vocab*width")
	}
	out := make([]float64, len(ids)*width)
	for i, id := range ids {
		if id < 0 || id >= vocab {
			panic("oracle: GatherRows id is out of range")
		}
		copy(out[i*width:(i+1)*width], table[id*width:(id+1)*width])
	}
	return out
}
