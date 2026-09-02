// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 The tgo Authors. All rights reserved.

package nn_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

// parts is one attention block's arguments, ready to be spoiled one at a time.
//
// A refusal test that rebuilt the whole block would differ from the passing
// case in more than the thing it names, and a reader could not tell which
// difference produced the diagnostic.
type parts struct {
	r                                       *rig
	x                                       *tensor.Tensor
	w                                       nn.AttentionWeights
	k, v                                    *tensor.State
	posQ, posK, slots, lens, pages, extents *tensor.Tensor
	cfg                                     nn.AttentionConfig
}

func newParts(t *testing.T) *parts {
	t.Helper()
	r := newRig(t, attEps)
	r.scalarF32("rope_base", attBase)
	r.scalarF32("scale", float32(1/math.Sqrt(attHeadDim)))
	r.scalarU32("base", 0)
	p := &parts{
		r:     r,
		x:     r.input("x", accel.F32, tensor.Shape{attT, attHidden}),
		posQ:  r.input("posq", accel.U32, tensor.Shape{attT * attQHeads}),
		posK:  r.input("posk", accel.U32, tensor.Shape{attT * attKVHeads}),
		slots: r.input("slots", accel.U32, tensor.Shape{attT}),
		lens:  r.input("lengths", accel.U32, tensor.Shape{1}),
		cfg: nn.AttentionConfig{
			QHeads: attQHeads, KVHeads: attKVHeads, HeadDim: attHeadDim,
			RoPEBase: "rope_base", ScaleName: "scale", BaseName: "base", QKNorm: true,
		},
	}
	p.w = nn.AttentionWeights{
		Q:     r.g.Weight("wq", tensor.Shape{attHidden, attQHeads * attHeadDim}),
		K:     r.g.Weight("wk", tensor.Shape{attHidden, attKVHeads * attHeadDim}),
		V:     r.g.Weight("wv", tensor.Shape{attHidden, attKVHeads * attHeadDim}),
		O:     r.g.Weight("wo", tensor.Shape{attQHeads * attHeadDim, attHidden}),
		QNorm: r.g.Gain("qnorm", attHeadDim),
		KNorm: r.g.Gain("knorm", attHeadDim),
	}
	p.k = tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "k", DType: accel.F32, Shape: tensor.Shape{attCap, attKVHeads, attHeadDim},
	})
	p.v = tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "v", DType: accel.F32, Shape: tensor.Shape{attCap, attKVHeads, attHeadDim},
	})
	return p
}

func (p *parts) record() *tensor.Tensor {
	return nn.Attention(p.r.g, p.x, p.w, p.k, p.v, nn.Step{PosQ: p.posQ, PosK: p.posK, Slots: p.slots, Lengths: p.lens, Pages: p.pages, Extents: p.extents}, p.cfg)
}

// One position per row, not per token. A positions tensor built per token has
// T entries where the rotation has T*H rows, and every head after the first
// would rotate at another token's position.
func TestRoPEPositionsAreOnePerRowAndNotOnePerToken(t *testing.T) {
	p := newParts(t)
	p.posQ = p.r.input("posq_per_token", accel.U32, tensor.Shape{attT})
	p.r.refuses(p.record(), "rows and positions holds")
}

func TestAHiddenStateIsAMatrix(t *testing.T) {
	p := newParts(t)
	p.x = p.r.input("x3", accel.F32, tensor.Shape{1, attT, attHidden})
	p.r.refuses(p.record(), "the hidden state is [T, hidden]")
}

func TestAHiddenStateIsF32(t *testing.T) {
	p := newParts(t)
	p.x = p.r.input("x16", accel.F16, tensor.Shape{attT, attHidden})
	p.r.refuses(p.record(), "a hidden state is f32")
}

func TestQueryHeadsAreAMultipleOfKeyValueHeads(t *testing.T) {
	p := newParts(t)
	p.cfg.KVHeads = 3
	p.r.refuses(p.record(), "num_attention_heads", "num_key_value_heads")
}

func TestEveryHeadCountIsPositive(t *testing.T) {
	p := newParts(t)
	p.cfg.HeadDim = 0
	p.r.refuses(p.record(), "all three are positive")
}

func TestAnOddHeadDimIsRefused(t *testing.T) {
	p := newParts(t)
	p.cfg.HeadDim = 5
	p.r.refuses(p.record(), "RoPE rotates pairs")
}

func TestQKNormNeedsBothGains(t *testing.T) {
	p := newParts(t)
	p.w.KNorm = nil
	p.r.refuses(p.record(), "the q or k norm gain is missing")
}

// head_dim is not always hidden_size/num_attention_heads, and the Qwen3-0.6B
// checkpoint is exactly the case that differs (specs/004-model-graph.md
// section 5). A config that assumed it produces a projection of the wrong
// width, and the diagnostic names the field rather than two shapes.
func TestAProjectionWiderThanTheConfigIsRefused(t *testing.T) {
	p := newParts(t)
	p.cfg.HeadDim = attHeadDim * 2
	p.r.refuses(p.record(), "q_proj is 16 wide and the config asks for 32", "head_dim")
}

func TestAPoisonedHiddenStateIsSomebodyElsesDiagnostic(t *testing.T) {
	p := newParts(t)
	p.x = nil
	if got := p.record(); got != nil {
		t.Fatalf("a nil hidden state produced %v; it should propagate as poison", got)
	}
	if err := p.r.g.Err(); err != nil {
		t.Fatalf("a nil hidden state was reported here as well: %v", err)
	}
}

func TestAPoisonedProjectionIsSomebodyElsesDiagnostic(t *testing.T) {
	p := newParts(t)
	// A weight of the wrong contracted axis: MatMul refuses it, and the width
	// check downstream must not add a second diagnostic about a value that
	// does not exist.
	p.w.Q = p.r.g.Weight("wq_wrong", tensor.Shape{attHidden + 1, attQHeads * attHeadDim})
	p.r.refuses(p.record(), "contracted axes")
	if got := countDiagnostics(p.r.t, p.r.g.Err().Error(), "q_proj"); got != 0 {
		t.Fatalf("the width check spoke about a poisoned projection %d times", got)
	}
}

func countDiagnostics(t *testing.T, err, needle string) int {
	t.Helper()
	n := 0
	for i := 0; i+len(needle) <= len(err); i++ {
		if err[i:i+len(needle)] == needle {
			n++
		}
	}
	return n
}

// A page table and a block size are one binding in two values, so half of it is
// a refusal in both directions.
//
// accel refuses a table with no block, because a table addresses blocks and how
// big one is is not derivable from its shape. The other direction is the one
// accel cannot see: a block size with no table is a perfectly valid contiguous
// read, so it compiles, runs, and quietly ignores the paging the caller thought
// it had configured.
func TestAPageTableAndABlockSizeAreOneBinding(t *testing.T) {
	t.Run("a table with no block size", func(t *testing.T) {
		p := newParts(t)
		p.pages = p.r.input("pages", accel.U32, tensor.Shape{1, attCap / 4})
		p.r.refuses(p.record(), "set together or neither is")
	})
	t.Run("a block size with no table", func(t *testing.T) {
		p := newParts(t)
		p.cfg.Block = 4
		p.r.refuses(p.record(), "set together or neither is")
	})
}
