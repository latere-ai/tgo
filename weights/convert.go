// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"

	"github.com/latere-ai/tgo/safetensors"
)

// The f16 constants specs/001-weights.md §3 states the rule in terms of.
const (
	// maxF16 is the largest finite f16. Overflow lands here rather than on an
	// infinity (001-D2).
	maxF16 = 65504.0

	// minNormalF16 is 2^-14, f16's smallest normal. Everything below it flushes
	// to zero: a subnormal f16 is under 6e-8 and contributes nothing to a dot
	// product against activations of order 1.
	minNormalF16 = 1.0 / 16384.0

	// bitsMaxF16 and bitsQuietNaNF16 are the encodings the two special results
	// use. 0x7bff is the largest finite value, 0x7c00 the infinity it must
	// never become.
	bitsMaxF16      = 0x7bff
	bitsQuietNaNF16 = 0x7e00
)

// decodeF32 widens a raw checkpoint plane into f32 scratch.
//
// Only the three float widths a checkpoint stores weights in are accepted. An
// integer plane is refused rather than reinterpreted: a loader that read an I64
// index table as weights would produce a well-shaped tensor of nonsense, which
// is the failure mode 001-D6 exists to prevent.
func decodeF32(dt safetensors.DType, src []byte, dst []float32) error {
	want := len(dst) * dt.Size()
	if dt.Size() == 0 {
		return fmt.Errorf("weights: dtype %q is not a float plane this loader converts", string(dt))
	}
	if len(src) != want {
		return fmt.Errorf("weights: %v plane is %d bytes, but %d elements need %d",
			dt, len(src), len(dst), want)
	}
	switch dt {
	case safetensors.BF16:
		// Exact and free: bf16 is the top 16 bits of an f32, so widening is a
		// 16-bit left shift with no rounding and no table (001-D1).
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[2*i:])) << 16)
		}
	case safetensors.F16:
		for i := range dst {
			dst[i] = accel.Float16FromBits(binary.LittleEndian.Uint16(src[2*i:])).F32()
		}
	case safetensors.F32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[4*i:]))
		}
	default:
		return fmt.Errorf("weights: dtype %v is not a float plane this loader converts; "+
			"a weight must be BF16, F16 or F32", dt)
	}
	return nil
}

// transpose writes src, a [rows, cols] matrix, into dst as [cols, rows].
//
// On the host, at load, once. accel's MatMul contracts x[M,K] against w[K,N]
// while Hugging Face stores a Linear weight as [out, in], so every projection
// weight arrives with its axes the wrong way round. tensor.Transpose cannot do
// it: accel's view operators reach elementwise operators only and refuse a
// strided view into MatMul, which is the correct refusal (001-D3).
func transpose(src, dst []float32, rows, cols int) {
	for r := range rows {
		row := src[r*cols : (r+1)*cols]
		for c, v := range row {
			dst[c*rows+r] = v
		}
	}
}

// permuteHeads rewrites each head's channels from the half-split convention the
// checkpoint uses into the interleaved one accel's RoPE rotates:
//
//	y[2i] = x[i], y[2i+1] = x[i + headDim/2]
//
// accel's kernel rotates the pairs (x0,x1), (x2,x3), … while Qwen3 — and every
// HF Llama-family checkpoint — pairs (x0, x_{d/2}), (x1, x_{d/2+1}), …. Nothing
// refuses the mismatch; every shape checks and the model emits fluent text whose
// long-range coherence rots (specs/004-model-graph.md §2.5.2, 004-D9).
//
// x is the whole post-transpose plane. lastAxis is its trailing dimension, which
// after the transpose is the projection's *output* channels; segments of headDim
// within it are heads. The head count is derived here rather than passed, so the
// same call is correct for q_proj (H segments), k_proj (H_kv segments) and the
// q_norm/k_norm gains (one segment).
//
// In place, with a temporary of one head, because the permutation of a segment
// is not decomposable into swaps a reader can follow.
func permuteHeads(x []float32, lastAxis, headDim int) error {
	if headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("weights: head dim %d must be positive and even; RoPE rotates pairs", headDim)
	}
	if lastAxis%headDim != 0 {
		return fmt.Errorf("weights: output axis %d is not a multiple of head dim %d, so it does "+
			"not divide into heads", lastAxis, headDim)
	}
	half := headDim / 2
	tmp := make([]float32, headDim)
	for off := 0; off < len(x); off += headDim {
		seg := x[off : off+headDim]
		copy(tmp, seg)
		for i := range half {
			seg[2*i] = tmp[i]
			seg[2*i+1] = tmp[i+half]
		}
	}
	return nil
}

// toF16 narrows src into dst as little-endian f16 storage and reports how many
// elements saturated.
//
// dst must hold 2*len(src) bytes. It is normally a slice of device memory: the
// caller is inside Buffer.Access, so the converted plane never exists on the
// host (001-D8).
func toF16(src []float32, dst []byte) int {
	saturated := 0
	for i, v := range src {
		bits, sat := f32ToF16Bits(v)
		if sat {
			saturated++
		}
		binary.LittleEndian.PutUint16(dst[2*i:], bits)
	}
	return saturated
}

// f32ToF16Bits is specs/001-weights.md §3's rule, in the order it states it.
//
// Round to nearest with ties to even, so a weight converted here and a weight
// converted by a device Cast agree. Overflow saturates to ±65504 and is
// reported, never to ±Inf: one infinity in a weight matrix makes every output
// of that row NaN and the failure surfaces many layers later as gibberish
// rather than as an error, while a saturated weight is wrong by a bounded
// amount (001-D2). Subnormals flush to zero.
func f32ToF16Bits(x float32) (bits uint16, saturated bool) {
	b := math.Float32bits(x)
	sign := uint16(b >> 16 & 0x8000)

	// NaN before the magnitude tests, because every comparison against a NaN is
	// false and it would otherwise fall through to the rounding branch and be
	// re-encoded as a finite number.
	if b&0x7fffffff > 0x7f800000 {
		return bitsQuietNaNF16, false
	}
	abs := math.Float32frombits(b & 0x7fffffff)
	switch {
	case abs > maxF16:
		// Infinity arrives here too, which is the point: an infinite weight in
		// the file is exactly the case a saturating conversion has to catch.
		return sign | bitsMaxF16, true
	case abs < minNormalF16:
		// The sign survives. A negative zero that came back positive would flip
		// the sign of a later division.
		return sign, false
	default:
		exp := int32(b>>23&0xff) - 127
		mant := b & 0x7fffff
		// Added rather than OR-ed: roundShift returns 1024 when the mantissa
		// rounds out of ten bits and that has to carry into the exponent.
		return sign | uint16(uint32(exp+15)<<10+roundShift(mant, 13)), false
	}
}

// roundShift shifts v right by n, rounding to nearest with ties to even.
func roundShift(v, n uint32) uint32 {
	if n == 0 {
		return v
	}
	keep := v >> n
	rest := v & (1<<n - 1)
	half := uint32(1) << (n - 1)
	if rest > half || (rest == half && keep&1 == 1) {
		keep++
	}
	return keep
}

// quantizeInto writes src as int8 quants and f16 block scales.
//
// It calls quant.Int8Quantize one aligned block at a time rather than on the
// whole matrix. quant blocks from offset zero in runs of quant.Int8Block, so a
// windowed call and a whole-matrix call produce identical bytes — and the
// windowed form holds 32 floats on the host instead of a second copy of the
// tensor, which is what keeps the int8 path inside §7.1's promise that the
// converted plane never exists.
//
// quants must hold len(src) bytes and scales 2*blocks(len(src)) bytes.
func quantizeInto(src []float32, quants, scales []byte) {
	for lo := 0; lo < len(src); lo += quant.Int8Block {
		hi := min(lo+quant.Int8Block, len(src))
		q, s := quant.Int8Quantize(src[lo:hi])
		for i, v := range q {
			quants[lo+i] = byte(v)
		}
		binary.LittleEndian.PutUint16(scales[2*(lo/quant.Int8Block):], s[0].Bits())
	}
}

// blocks is how many scales a plane of n weights carries. A trailing partial
// block gets its own scale, which is what quant.Int8Quantize does.
func blocks(n int) int { return (n + quant.Int8Block - 1) / quant.Int8Block }
