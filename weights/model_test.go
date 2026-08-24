// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package weights

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"testing"
)

// qwen3Config is the part of a Hugging Face config.json this test reads. It is
// here rather than in the package because the model owns its own configuration
// (specs/004-model-graph.md §1); weights takes declarations, not a config.
type qwen3Config struct {
	NumHiddenLayers   int `json:"num_hidden_layers"`
	HiddenSize        int `json:"hidden_size"`
	NumAttentionHeads int `json:"num_attention_heads"`
	NumKeyValueHeads  int `json:"num_key_value_heads"`
	HeadDim           int `json:"head_dim"`
	IntermediateSize  int `json:"intermediate_size"`
	VocabSize         int `json:"vocab_size"`
}

// qwen3Tensors is the declaration a Qwen3 model builder will make: which
// tensors transpose, and which carry a head dimension to permute. It is written
// out here, as caller-supplied data, because 001-D4 puts the mapping in the
// model and never in this package.
func qwen3Tensors(c qwen3Config) []Tensor {
	out := []Tensor{
		{Name: "model.embed_tokens.weight"},
		{Name: "model.norm.weight"},
		{Name: "lm_head.weight", Transpose: true},
	}
	for l := range c.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d.", l)
		out = append(out,
			Tensor{Name: p + "input_layernorm.weight"},
			Tensor{Name: p + "post_attention_layernorm.weight"},
			// q_proj, k_proj and the two QK-norm gains carry the permutation.
			// v_proj and o_proj do not: v is never rotated, and attention's q·kᵀ
			// is invariant under a permutation applied to both q and k, so
			// nothing downstream compensates (004-D9).
			Tensor{Name: p + "self_attn.q_proj.weight", Transpose: true, HeadDim: c.HeadDim},
			Tensor{Name: p + "self_attn.k_proj.weight", Transpose: true, HeadDim: c.HeadDim},
			Tensor{Name: p + "self_attn.v_proj.weight", Transpose: true},
			Tensor{Name: p + "self_attn.o_proj.weight", Transpose: true},
			Tensor{Name: p + "self_attn.q_norm.weight", HeadDim: c.HeadDim},
			Tensor{Name: p + "self_attn.k_norm.weight", HeadDim: c.HeadDim},
			Tensor{Name: p + "mlp.gate_proj.weight", Transpose: true},
			Tensor{Name: p + "mlp.up_proj.weight", Transpose: true},
			Tensor{Name: p + "mlp.down_proj.weight", Transpose: true},
		)
	}
	return out
}

// TestRealCheckpoint loads a Qwen3 dense checkpoint at f16 and asserts the
// shapes accel will see and that nothing saturated.
//
// It is gated on TGO_MODEL and skips without it. No test that runs by default
// reads a real checkpoint: the smallest Qwen3 is over a gigabyte, CI never sets
// the variable, and this is the release gate run by hand
// (specs/000-decisions.md decision 8, specs/011-sequencing.md §4).
func TestRealCheckpoint(t *testing.T) {
	dir := os.Getenv("TGO_MODEL")
	if dir == "" {
		t.Skip("TGO_MODEL is not set; this is the by-hand release gate, not a CI test")
	}
	dev := openCPU(t)
	repo := openRepoAt(t, dir)

	var cfg qwen3Config
	if err := json.Unmarshal(repo.Config(), &cfg); err != nil {
		t.Fatalf("config.json: %v", err)
	}
	// The case specs/004-model-graph.md §5 says not to infer. Qwen3-0.6B stores
	// head_dim 128 while hidden_size/num_attention_heads is 64, so a loader that
	// derived the head width would permute the wrong channels and produce a
	// model that reads fluently and loses coherence.
	if cfg.HeadDim == cfg.HiddenSize/cfg.NumAttentionHeads {
		t.Logf("head_dim %d equals hidden/heads; this checkpoint does not exercise 004 §5",
			cfg.HeadDim)
	}

	decls := qwen3Tensors(cfg)
	set, err := Load(dev, repo, decls, Options{
		Policy: F16,
		// The whole model at f16, so the saturation count below is over every
		// weight rather than over the few that stayed wide.
		Budget: 4 << 30,
		Log:    io.Discard,
	})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer set.Close()

	d, f, v, hq, hkv := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize,
		cfg.NumAttentionHeads*cfg.HeadDim, cfg.NumKeyValueHeads*cfg.HeadDim

	want := map[string][]int{
		"model.embed_tokens.weight": {v, d},
		"model.norm.weight":         {d},
		"lm_head.weight":            {d, v},
	}
	for l := range cfg.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d.", l)
		want[p+"input_layernorm.weight"] = []int{d}
		want[p+"post_attention_layernorm.weight"] = []int{d}
		want[p+"self_attn.q_proj.weight"] = []int{d, hq}
		want[p+"self_attn.k_proj.weight"] = []int{d, hkv}
		want[p+"self_attn.v_proj.weight"] = []int{d, hkv}
		want[p+"self_attn.o_proj.weight"] = []int{hq, d}
		want[p+"self_attn.q_norm.weight"] = []int{cfg.HeadDim}
		want[p+"self_attn.k_norm.weight"] = []int{cfg.HeadDim}
		want[p+"mlp.gate_proj.weight"] = []int{d, f}
		want[p+"mlp.up_proj.weight"] = []int{d, f}
		want[p+"mlp.down_proj.weight"] = []int{f, d}
	}

	if got := len(set.Names()); got != len(want) {
		t.Fatalf("loaded %d tensors, declared %d", got, len(want))
	}
	var elements int64
	for name, shape := range want {
		got, ok := set.Get(name)
		if !ok {
			t.Fatalf("%s was not loaded", name)
		}
		if fmt.Sprint(got.Shape) != fmt.Sprint(shape) {
			t.Errorf("%s shape = %v, want %v", name, got.Shape, shape)
		}
		// Zero, not "under the threshold". Trained transformer weights are
		// almost entirely within [-1, 1], so a single saturation in a bf16
		// checkpoint is a finding about the file (001-D2).
		if got.Saturated != 0 {
			t.Errorf("%s saturated %d of %d elements", name, got.Saturated, got.Elements)
		}
		elements += int64(got.Elements)
	}

	rep := set.Report()
	if rep.Saturated != 0 {
		t.Errorf("the load saturated %d elements", rep.Saturated)
	}
	if rep.Bytes != elements*2 {
		t.Errorf("Bytes = %d, want %d", rep.Bytes, elements*2)
	}
	t.Logf("%d tensors, %s parameters, %s resident at f16, mapped=%v",
		len(want), humanCount(elements), humanBytes(rep.Bytes), rep.Mapped)
}

func humanCount(n int64) string { return fmt.Sprintf("%.2fe9", float64(n)/1e9) }
