// Copyright 2026 The tgo Authors. All rights reserved.

package nn_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

// One block at a real checkpoint's shapes, which is where head_dim stops being
// hidden_size/num_attention_heads.
//
// Qwen3-0.6B has hidden_size 1024 and 16 attention heads, and head_dim 128
// rather than the 64 that ratio implies (specs/004-model-graph.md section 5).
// A block that inferred it would build every shape wrong, and the synthetic
// tests above cannot see that: they are free to choose shapes that agree.
//
// Gated on TGO_MODEL and skipped by default: specs/000-decisions.md decision 8
// keeps a 1.5 GB checkpoint out of every run. Only config.json is read -- the
// weights here are zeros, because what is under test is the graph and not the
// numbers a checkpoint would put through it.
func TestOneBlockAtACheckpointsShapes(t *testing.T) {
	dir := os.Getenv("TGO_MODEL")
	if dir == "" {
		t.Skip("set TGO_MODEL to a model directory to run this")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	var cfg struct {
		HiddenSize       int     `json:"hidden_size"`
		Heads            int     `json:"num_attention_heads"`
		KVHeads          int     `json:"num_key_value_heads"`
		HeadDim          int     `json:"head_dim"`
		IntermediateSize int     `json:"intermediate_size"`
		Eps              float32 `json:"rms_norm_eps"`
		Theta            float32 `json:"rope_theta"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("config: %v", err)
	}
	if cfg.HeadDim == cfg.HiddenSize/cfg.Heads {
		t.Logf("this checkpoint's head_dim is hidden_size/num_attention_heads (%d); "+
			"the case this test exists for is a checkpoint where it is not", cfg.HeadDim)
	}

	const tokens, capacity = 1, 8
	r := newRig(t, cfg.Eps)
	r.scalarF32("rope_base", cfg.Theta)
	r.scalarF32("scale", float32(1/math.Sqrt(float64(cfg.HeadDim))))

	x := r.input("x", accel.F32, tensor.Shape{tokens, cfg.HiddenSize})
	r.f32("x", make([]float32, tokens*cfg.HiddenSize))
	posQ := r.input("posq", accel.U32, tensor.Shape{tokens * cfg.Heads})
	posK := r.input("posk", accel.U32, tensor.Shape{tokens * cfg.KVHeads})
	slots := r.input("slots", accel.U32, tensor.Shape{tokens})
	lengths := r.input("lengths", accel.U32, tensor.Shape{1})
	r.u32("posq", make([]uint32, tokens*cfg.Heads))
	r.u32("posk", make([]uint32, tokens*cfg.KVHeads))
	r.u32("slots", make([]uint32, tokens))
	r.u32("lengths", []uint32{1})

	qWidth, kvWidth := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim
	w := nn.AttentionWeights{
		Q:     r.g.Weight("wq", tensor.Shape{cfg.HiddenSize, qWidth}),
		K:     r.g.Weight("wk", tensor.Shape{cfg.HiddenSize, kvWidth}),
		V:     r.g.Weight("wv", tensor.Shape{cfg.HiddenSize, kvWidth}),
		O:     r.g.Weight("wo", tensor.Shape{qWidth, cfg.HiddenSize}),
		QNorm: r.g.Gain("qnorm", cfg.HeadDim),
		KNorm: r.g.Gain("knorm", cfg.HeadDim),
	}
	r.f16("block.wq", make([]float32, cfg.HiddenSize*qWidth))
	r.f16("block.wk", make([]float32, cfg.HiddenSize*kvWidth))
	r.f16("block.wv", make([]float32, cfg.HiddenSize*kvWidth))
	r.f16("block.wo", make([]float32, qWidth*cfg.HiddenSize))
	r.f32("block.qnorm", make([]float32, cfg.HeadDim))
	r.f32("block.knorm", make([]float32, cfg.HeadDim))

	kc := tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "k", DType: accel.F32, Shape: tensor.Shape{capacity, cfg.KVHeads, cfg.HeadDim},
	})
	vc := tensor.NewState(r.g.B, tensor.StateDesc{
		Name: "v", DType: accel.F32, Shape: tensor.Shape{capacity, cfg.KVHeads, cfg.HeadDim},
	})
	r.f32("k", make([]float32, capacity*cfg.KVHeads*cfg.HeadDim))
	r.f32("v", make([]float32, capacity*cfg.KVHeads*cfg.HeadDim))

	h := nn.Attention(r.g, x, w, kc, vc, posQ, posK, slots, lengths, nn.AttentionConfig{
		QHeads: cfg.Heads, KVHeads: cfg.KVHeads, HeadDim: cfg.HeadDim,
		// BaseName is named and no scalar is declared for it: this is one
		// token, so the block drops it. A prefill here would need the
		// declaration, which is the asymmetry section 3.1 describes.
		RoPEBase: "rope_base", ScaleName: "scale", BaseName: "base", QKNorm: true,
	})
	gate := r.g.Weight("wgate", tensor.Shape{cfg.HiddenSize, cfg.IntermediateSize})
	up := r.g.Weight("wup", tensor.Shape{cfg.HiddenSize, cfg.IntermediateSize})
	down := r.g.Weight("wdown", tensor.Shape{cfg.IntermediateSize, cfg.HiddenSize})
	r.f16("block.wgate", make([]float32, cfg.HiddenSize*cfg.IntermediateSize))
	r.f16("block.wup", make([]float32, cfg.HiddenSize*cfg.IntermediateSize))
	r.f16("block.wdown", make([]float32, cfg.IntermediateSize*cfg.HiddenSize))

	out := nn.SwiGLUMLP(r.g, tensor.Add(r.g.B, x, h), gate, up, down)
	got, _ := r.run(out)
	if len(got) != tokens*cfg.HiddenSize {
		t.Fatalf("the block produced %d values, want %d", len(got), tokens*cfg.HiddenSize)
	}
}
