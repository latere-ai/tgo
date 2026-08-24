// Copyright 2026 The tgo Authors. All rights reserved.

package nn

import (
	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// Linear is a projection, y = xW, with x [M, K] f32 and W [K, N] f16 or int8.
//
// f32 in, f32 out, and no cast: accel accepts f32 activations against narrow
// weights directly (specs/010-conformance.md C8), so the weight stays narrow
// for memory, the activation stays wide for accuracy, and the accumulator was
// always f32. A cast here would be one dispatch per projection buying nothing.
//
// Qwen3 has no biases on its projections, so [tensor.Linear]'s fused epilogue
// is unused (004-D5).
func Linear(g *Graph, x *tensor.Tensor, w Operand) *tensor.Tensor {
	if !w.ok() {
		return g.fail("Linear", "the weight carries neither a dense plane nor a complete "+
			"quantized pair; an operand is one of the two (004-D6)")
	}
	if w.IsQuant() {
		return tensor.QuantMatMul(g.B, x, w.Quant)
	}
	return tensor.MatMul(g.B, x, w.Dense)
}

// RMSNorm normalizes each row of x by its root mean square and scales by gain.
//
// It reduces over the last axis and takes a gain of one value per feature,
// which is what makes the per-head QK-norm of [Attention] a reshape rather than
// a different operator.
func RMSNorm(g *Graph, x *tensor.Tensor, gain *tensor.Tensor) *tensor.Tensor {
	return tensor.RMSNorm(g.B, x, gain, g.Eps)
}

// SwiGLUMLP is SiLU(x*Wgate) * (x*Wup) * Wdown.
//
// [tensor.SwiGLU] is one authored kernel; the composed SiLU-then-Mul form stays
// the correctness reference, so a fusion bug is a test failure rather than a
// quality loss nobody can see (004-D5).
func SwiGLUMLP(g *Graph, x *tensor.Tensor, gate, up, down Operand) *tensor.Tensor {
	gated := Linear(g, x, gate)
	value := Linear(g, x, up)
	return Linear(g, tensor.SwiGLU(g.B, gated, value), down)
}

// rows reports x's row count and width, and whether x is a matrix at all.
func rows(x *tensor.Tensor) (m, k int, ok bool) {
	if x == nil {
		return 0, 0, false
	}
	s := x.Shape()
	if len(s) != 2 {
		return 0, 0, false
	}
	return s[0], s[1], true
}

// f32 reports whether x is an f32 activation.
func f32(x *tensor.Tensor) bool { return x != nil && x.DType() == accel.F32 }
