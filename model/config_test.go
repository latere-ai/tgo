// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"encoding/json"
	"strings"
	"testing"
)

// good is the config every refusal test starts from: a small Qwen3-shaped
// model with head_dim stated explicitly and not equal to hidden_size/heads,
// which is the case specs/004-model-graph.md §5 says not to infer.
//
// A map rather than a struct, because a struct cannot express "hidden_size is
// absent", "head_dim is a string", or "vocab_size is negative", which are the
// cases ParseConfig must refuse.
//
// Every dimension and every value derived from one is a distinct number, and
// that is the property to preserve when editing this fixture. The real
// checkpoint cannot discriminate a weight map that confuses two of them: for
// Qwen3-0.6B, H_kv·d_h = 8·128 = 1024 = d, so k_proj and v_proj are [d, d] and
// a map that wrote hidden_size where it meant KVWidth loads a real model with
// every shape checking. An earlier fixture reproduced that coincidence exactly
// — d = 64 with H_kv·d_h = 64, and V = f = 128 — and six wrong weight maps
// passed the table test below. The numbers here collide nowhere:
//
//	d = 80    L = 2     H = 8     H_kv = 2    d_h = 48
//	f = 176   V = 112   H·d_h = 384          H_kv·d_h = 96
//	H/H_kv = 4          d/H = 10
func good() map[string]any {
	return map[string]any{
		"architectures":           []string{Qwen3Architecture},
		"hidden_size":             80,
		"num_hidden_layers":       2,
		"num_attention_heads":     8,
		"num_key_value_heads":     2,
		"head_dim":                48,
		"intermediate_size":       176,
		"vocab_size":              112,
		"rms_norm_eps":            1e-06,
		"rope_theta":              1000000,
		"tie_word_embeddings":     true,
		"max_position_embeddings": 4096,
		"rope_scaling":            nil,
		"sliding_window":          nil,
		"use_sliding_window":      false,
	}
}

// raw marshals a config map.
func raw(t *testing.T, cfg map[string]any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	return b
}

// parse parses a config map and fails the test if it is refused.
func parse(t *testing.T, cfg map[string]any) *Config {
	t.Helper()
	c, err := ParseConfig(raw(t, cfg))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	return c
}

func TestParseConfigGood(t *testing.T) {
	c := parse(t, good())
	if c.Architecture != Qwen3Architecture {
		t.Errorf("Architecture = %q, want %q", c.Architecture, Qwen3Architecture)
	}
	if c.HiddenSize != 80 || c.NumLayers != 2 || c.NumHeads != 8 || c.NumKVHeads != 2 {
		t.Errorf("shape = d%d L%d H%d Hkv%d, want d80 L2 H8 Hkv2",
			c.HiddenSize, c.NumLayers, c.NumHeads, c.NumKVHeads)
	}
	// The whole point of the field: 80/8 is 10 and the file says 48.
	if c.HeadDim != 48 {
		t.Errorf("HeadDim = %d, want the stated 48 rather than hidden/heads", c.HeadDim)
	}
	if c.IntermediateSize != 176 || c.VocabSize != 112 || c.MaxPositionEmbeddings != 4096 {
		t.Errorf("f=%d V=%d maxpos=%d", c.IntermediateSize, c.VocabSize, c.MaxPositionEmbeddings)
	}
	if c.RMSNormEps != 1e-06 {
		t.Errorf("RMSNormEps = %v, want 1e-06", c.RMSNormEps)
	}
	if c.RoPETheta != 1e6 {
		t.Errorf("RoPETheta = %v, want 1e6", c.RoPETheta)
	}
	if !c.TieWordEmbeddings {
		t.Error("TieWordEmbeddings = false, want true")
	}
	// The three derived widths, each asserted against a literal rather than
	// against the expression that computes it, and each distinct from d: a
	// checkpoint where KVWidth happens to equal hidden_size — which Qwen3-0.6B
	// is — cannot tell the two apart.
	if got := c.KVGroup(); got != 4 {
		t.Errorf("KVGroup() = %d, want 4 = H/H_kv", got)
	}
	if got := c.QWidth(); got != 384 {
		t.Errorf("QWidth() = %d, want 384 = H*d_h", got)
	}
	if got := c.KVWidth(); got != 96 {
		t.Errorf("KVWidth() = %d, want 96 = H_kv*d_h", got)
	}
}

func TestParseConfigDefaults(t *testing.T) {
	cfg := good()
	delete(cfg, "head_dim")
	delete(cfg, "num_key_value_heads")
	delete(cfg, "tie_word_embeddings")
	c := parse(t, cfg)
	if c.HeadDim != 10 {
		t.Errorf("HeadDim = %d, want the default hidden_size/num_attention_heads = 10", c.HeadDim)
	}
	if c.NumKVHeads != c.NumHeads {
		t.Errorf("NumKVHeads = %d, want the default num_attention_heads = %d", c.NumKVHeads, c.NumHeads)
	}
	if c.TieWordEmbeddings {
		t.Error("TieWordEmbeddings = true, want the default false")
	}
}

// TestParseConfigNullsAreAbsent pins the shape every real Hugging Face config
// ships: rope_scaling and sliding_window are present keys whose value is null.
// Refusing on key presence would refuse Qwen3-0.6B itself.
func TestParseConfigNullsAreAbsent(t *testing.T) {
	cfg := good()
	cfg["rope_scaling"] = nil
	cfg["sliding_window"] = nil
	cfg["use_sliding_window"] = false
	parse(t, cfg)

	// The same config with the keys removed rather than nulled.
	cfg = good()
	delete(cfg, "rope_scaling")
	delete(cfg, "sliding_window")
	delete(cfg, "use_sliding_window")
	parse(t, cfg)
}

// TestParseConfigRefusals is specs/004-model-graph.md §7, one row per case, plus
// the missing and malformed fields of §5's required column. Each case asserts
// the message names the config field, because "invalid config" sends a reader
// to read the file rather than to the line of it that is wrong.
func TestParseConfigRefusals(t *testing.T) {
	cases := []struct {
		name  string
		mut   func(cfg map[string]any)
		names string // a substring the message must carry
	}{
		{"rope scaling", func(c map[string]any) {
			c["rope_scaling"] = map[string]any{"rope_type": "yarn", "factor": 4.0}
		}, "rope_scaling"},
		{"rope scaling legacy type", func(c map[string]any) {
			c["rope_scaling"] = map[string]any{"type": "linear", "factor": 2.0}
		}, "linear"},
		{"rope scaling unnamed", func(c map[string]any) {
			c["rope_scaling"] = map[string]any{"factor": 2.0}
		}, "unnamed"},
		{"rope scaling not an object", func(c map[string]any) {
			c["rope_scaling"] = "yarn"
		}, "rope_scaling"},
		{"sliding window enabled", func(c map[string]any) {
			c["use_sliding_window"] = true
		}, "use_sliding_window"},
		{"sliding window length without the flag", func(c map[string]any) {
			delete(c, "use_sliding_window")
			c["sliding_window"] = 4096
		}, "sliding_window"},
		{"head dim odd", func(c map[string]any) { c["head_dim"] = 33 }, "head_dim"},
		{"head dim zero", func(c map[string]any) { c["head_dim"] = 0 }, "head_dim"},
		{"head dim negative", func(c map[string]any) { c["head_dim"] = -32 }, "head_dim"},
		{"head dim absent and not divisible", func(c map[string]any) {
			delete(c, "head_dim")
			c["num_attention_heads"] = 6 // 80/6 is not an integer
		}, "head_dim"},
		{"kv heads do not divide", func(c map[string]any) {
			c["num_key_value_heads"] = 3 // 8 is not a multiple of 3
		}, "num_key_value_heads"},
		{"kv heads zero", func(c map[string]any) {
			c["num_key_value_heads"] = 0
		}, "num_key_value_heads"},
		{"architectures missing", func(c map[string]any) {
			delete(c, "architectures")
		}, "architectures[0]"},
		{"architectures empty string", func(c map[string]any) {
			c["architectures"] = []string{""}
		}, "architectures[0]"},
		{"hidden size missing", func(c map[string]any) {
			delete(c, "hidden_size")
		}, "hidden_size"},
		{"hidden size zero", func(c map[string]any) { c["hidden_size"] = 0 }, "hidden_size"},
		{"layers missing", func(c map[string]any) {
			delete(c, "num_hidden_layers")
		}, "num_hidden_layers"},
		{"heads missing", func(c map[string]any) {
			delete(c, "num_attention_heads")
		}, "num_attention_heads"},
		{"intermediate missing", func(c map[string]any) {
			delete(c, "intermediate_size")
		}, "intermediate_size"},
		{"vocab missing", func(c map[string]any) { delete(c, "vocab_size") }, "vocab_size"},
		{"vocab negative", func(c map[string]any) { c["vocab_size"] = -1 }, "vocab_size"},
		{"max positions missing", func(c map[string]any) {
			delete(c, "max_position_embeddings")
		}, "max_position_embeddings"},
		{"eps missing", func(c map[string]any) { delete(c, "rms_norm_eps") }, "rms_norm_eps"},
		{"eps zero", func(c map[string]any) { c["rms_norm_eps"] = 0 }, "rms_norm_eps"},
		{"eps negative", func(c map[string]any) { c["rms_norm_eps"] = -1e-6 }, "rms_norm_eps"},
		{"eps underflows f32", func(c map[string]any) { c["rms_norm_eps"] = 1e-300 }, "rms_norm_eps"},
		{"theta missing", func(c map[string]any) { delete(c, "rope_theta") }, "rope_theta"},
		{"theta overflows f32", func(c map[string]any) { c["rope_theta"] = 1e300 }, "rope_theta"},
		{"wrong type", func(c map[string]any) { c["hidden_size"] = "1024" }, "parse config"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := good()
			tc.mut(cfg)
			c, err := ParseConfig(raw(t, cfg))
			if err == nil {
				t.Fatalf("ParseConfig accepted %s and returned %+v", tc.name, c)
			}
			if !strings.Contains(err.Error(), tc.names) {
				t.Errorf("error %q does not name %q", err, tc.names)
			}
		})
	}
}

// TestParseConfigNotJSON covers the byte-level failure, which is separate from
// a well-formed config with a wrong field.
func TestParseConfigNotJSON(t *testing.T) {
	if _, err := ParseConfig([]byte("{")); err == nil {
		t.Fatal("ParseConfig accepted a truncated document")
	}
}
