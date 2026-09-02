// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// have builds the tensor-name-to-shape map a checkpoint header would give, from
// a weight map. Two specs naming one tensor collapse to one entry, which is
// what a tied checkpoint looks like on disk.
func have(specs []WeightSpec) map[string][]int {
	out := make(map[string][]int, len(specs))
	for _, s := range specs {
		out[s.Tensor] = append([]int(nil), s.Shape...)
	}
	return out
}

// sameSpec compares two rows of the weight map on every field, so that a field
// nothing else asserts — Kind, Layer, Alias, and the Heads that goes with
// Permute — cannot drift.
func sameSpec(a, b WeightSpec) bool {
	return a.Tensor == b.Tensor && a.Port == b.Port && a.Layer == b.Layer &&
		a.Kind == b.Kind && a.Transpose == b.Transpose && a.Permute == b.Permute &&
		a.Heads == b.Heads && a.Alias == b.Alias && sameShape(a.Shape, b.Shape)
}

// checkPorts asserts that every row of want appears in the map, whole.
func checkPorts(t *testing.T, specs, want []WeightSpec) {
	t.Helper()
	byPort := make(map[string]WeightSpec, len(specs))
	for _, s := range specs {
		if _, dup := byPort[s.Port]; dup {
			t.Fatalf("port %q is bound twice", s.Port)
		}
		byPort[s.Port] = s
	}
	for _, w := range want {
		got, ok := byPort[w.Port]
		if !ok {
			t.Errorf("port %q is not in the weight map", w.Port)
			continue
		}
		if !sameSpec(got, w) {
			t.Errorf("port %q =\n  %+v\nwant\n  %+v", w.Port, got, w)
		}
	}
}

// TestQwen3WeightsTable is specs/004-model-graph.md §4 asserted row by row, on
// the config good() documents: d=80, H=8, H_kv=2, d_h=48, f=176, V=112. No two
// of the six widths that appear in a shape are equal, so a map that wrote
// hidden_size where §4 says H·d_h, or vocab_size where it says intermediate_size,
// fails here rather than on a checkpoint whose numbers happen to coincide.
func TestQwen3WeightsTable(t *testing.T) {
	c := parse(t, good())
	checkPorts(t, qwen3Weights(c), []WeightSpec{
		{Tensor: qwen3Embed, Port: "embed", Shape: []int{112, 80}, Layer: -1, Kind: KindEmbedding},
		{Tensor: "model.layers.1.input_layernorm.weight", Port: "1.attn_norm",
			Shape: []int{80}, Layer: 1, Kind: KindGain},
		{Tensor: "model.layers.1.self_attn.q_proj.weight", Port: "1.wq",
			Shape: []int{384, 80}, Layer: 1, Kind: KindProjection,
			Transpose: true, Permute: true, Heads: 8},
		{Tensor: "model.layers.1.self_attn.k_proj.weight", Port: "1.wk",
			Shape: []int{96, 80}, Layer: 1, Kind: KindProjection,
			Transpose: true, Permute: true, Heads: 2},
		{Tensor: "model.layers.1.self_attn.v_proj.weight", Port: "1.wv",
			Shape: []int{96, 80}, Layer: 1, Kind: KindProjection, Transpose: true},
		{Tensor: "model.layers.1.self_attn.o_proj.weight", Port: "1.wo",
			Shape: []int{80, 384}, Layer: 1, Kind: KindProjection, Transpose: true},
		{Tensor: "model.layers.1.self_attn.q_norm.weight", Port: "1.qnorm",
			Shape: []int{48}, Layer: 1, Kind: KindGain, Permute: true, Heads: 1},
		{Tensor: "model.layers.1.self_attn.k_norm.weight", Port: "1.knorm",
			Shape: []int{48}, Layer: 1, Kind: KindGain, Permute: true, Heads: 1},
		{Tensor: "model.layers.1.post_attention_layernorm.weight", Port: "1.ffn_norm",
			Shape: []int{80}, Layer: 1, Kind: KindGain},
		{Tensor: "model.layers.1.mlp.gate_proj.weight", Port: "1.wgate",
			Shape: []int{176, 80}, Layer: 1, Kind: KindProjection, Transpose: true},
		{Tensor: "model.layers.1.mlp.up_proj.weight", Port: "1.wup",
			Shape: []int{176, 80}, Layer: 1, Kind: KindProjection, Transpose: true},
		{Tensor: "model.layers.1.mlp.down_proj.weight", Port: "1.wdown",
			Shape: []int{80, 176}, Layer: 1, Kind: KindProjection, Transpose: true},
		{Tensor: qwen3FinalNorm, Port: "final_norm", Shape: []int{80}, Layer: -1, Kind: KindGain},
		// The tied head, whole. §4's permute column is "no" for it: the head
		// reads the residual stream, which carries no rotated channels.
		{Tensor: qwen3Embed, Port: "lm_head", Shape: []int{112, 80}, Layer: -1,
			Kind: KindLMHead, Transpose: true, Alias: qwen3LMHead},
	})
}

// TestQwen3PermuteHeadsSplitIntoHeadDim is the invariant that makes Heads a
// contract rather than four numbers that happen to be right: the rotary
// permutation (004-D9) reorders each head's channels among themselves, so the
// axis it acts on must divide into exactly Heads groups of head_dim. A row
// whose Heads is wrong reorders across a head boundary, which every shape check
// downstream accepts and which produces fluent text with degraded long-range
// coherence.
func TestQwen3PermuteHeadsSplitIntoHeadDim(t *testing.T) {
	c := parse(t, good())
	permuted := 0
	for _, s := range qwen3Weights(c) {
		if !s.Permute {
			if s.Heads != 0 {
				t.Errorf("port %q does not permute and carries Heads = %d", s.Port, s.Heads)
			}
			continue
		}
		permuted++
		// Shape is the file's layout, [out, in] for a projection and [d_h] for
		// a gain; the permuted axis is the output one, which is Shape[0] in
		// both.
		if s.Heads <= 0 || s.Shape[0]%s.Heads != 0 || s.Shape[0]/s.Heads != c.HeadDim {
			t.Errorf("port %q: Permute splits %d channels into %d heads, which is not "+
				"%d wide", s.Port, s.Shape[0], s.Heads, c.HeadDim)
		}
	}
	// q_proj, k_proj, q_norm and k_norm, per layer, and nothing else.
	if want := 4 * c.NumLayers; permuted != want {
		t.Errorf("%d ports permute, want %d", permuted, want)
	}
}

// TestQwen3WeightsUntiedTable is the same tail row for a checkpoint that ships
// its own head. It is a separate test because Tensor and Alias are the two
// fields the tie flips, and the rest of the row must not move with them.
func TestQwen3WeightsUntiedTable(t *testing.T) {
	cfg := good()
	cfg["tie_word_embeddings"] = false
	c := parse(t, cfg)
	checkPorts(t, qwen3Weights(c), []WeightSpec{
		{Tensor: qwen3Embed, Port: "embed", Shape: []int{112, 80}, Layer: -1, Kind: KindEmbedding},
		{Tensor: qwen3LMHead, Port: "lm_head", Shape: []int{112, 80}, Layer: -1,
			Kind: KindLMHead, Transpose: true},
	})
}

// TestQwen3WeightsCount pins the arithmetic: eleven tensors per layer, plus the
// embedding and the final norm, plus a head. A row added to or dropped from
// §4's table without a test moving is a checkpoint that loads with a layer
// missing.
func TestQwen3WeightsCount(t *testing.T) {
	c := parse(t, good())
	specs := qwen3Weights(c)
	if want := 11*c.NumLayers + 3; len(specs) != want {
		t.Fatalf("len(Weights()) = %d, want %d", len(specs), want)
	}
	if len(qwen3Layer) != 11 {
		t.Fatalf("the layer table has %d rows, want 11", len(qwen3Layer))
	}
	// Every layer index appears, and the map is ordered embedding first, then
	// the layers in order, then the tail.
	if specs[0].Port != "embed" {
		t.Errorf("first spec is %q, want embed", specs[0].Port)
	}
	if specs[len(specs)-1].Port != "lm_head" {
		t.Errorf("last spec is %q, want lm_head", specs[len(specs)-1].Port)
	}
	for l := 0; l < c.NumLayers; l++ {
		want := fmt.Sprintf("%d.wq", l)
		found := false
		for _, s := range specs {
			found = found || s.Port == want
		}
		if !found {
			t.Errorf("layer %d has no wq port", l)
		}
	}
}

// TestQwen3WeightsTied is 004-D7: two planes from one file tensor, in two
// layouts, and the alias that makes the contradiction detectable.
func TestQwen3WeightsTied(t *testing.T) {
	c := parse(t, good())
	if !c.TieWordEmbeddings {
		t.Fatal("the good config is meant to be tied")
	}
	specs := qwen3Weights(c)
	var embed, head WeightSpec
	for _, s := range specs {
		switch s.Port {
		case "embed":
			embed = s
		case "lm_head":
			head = s
		}
	}
	if head.Tensor != qwen3Embed {
		t.Errorf("tied lm_head reads %q, want %q", head.Tensor, qwen3Embed)
	}
	if head.Alias != qwen3LMHead {
		t.Errorf("tied lm_head Alias = %q, want %q", head.Alias, qwen3LMHead)
	}
	// The two planes are the same source in different layouts, which is why
	// they cannot share a device buffer.
	if !head.Transpose || embed.Transpose {
		t.Errorf("tied planes transpose = head %v, embed %v; want true, false",
			head.Transpose, embed.Transpose)
	}
	if !sameShape(embed.Shape, head.Shape) {
		t.Errorf("tied planes disagree on the file shape: %v and %v", embed.Shape, head.Shape)
	}
}

func TestQwen3WeightsUntied(t *testing.T) {
	cfg := good()
	cfg["tie_word_embeddings"] = false
	c := parse(t, cfg)
	specs := qwen3Weights(c)
	head := specs[len(specs)-1]
	if head.Port != "lm_head" || head.Tensor != qwen3LMHead {
		t.Fatalf("untied head = %+v, want the %s tensor", head, qwen3LMHead)
	}
	if head.Alias != "" {
		t.Errorf("untied head Alias = %q, want empty", head.Alias)
	}
	if !head.Transpose {
		t.Error("untied head does not transpose; MatMul needs [d, V]")
	}
}

func TestCheckExact(t *testing.T) {
	for _, tie := range []bool{true, false} {
		cfg := good()
		cfg["tie_word_embeddings"] = tie
		specs := qwen3Weights(parse(t, cfg))
		if err := Check(specs, have(specs)); err != nil {
			t.Errorf("tie=%v: Check refused the checkpoint its own map describes: %v", tie, err)
		}
	}
}

// TestCheckTiedShipsLMHead is §4's contradiction: the config says the head is
// the embedding and the file ships a head of its own. Refusing beats picking
// one, because the two are different weights and nothing says which the model
// was trained with.
func TestCheckTiedShipsLMHead(t *testing.T) {
	c := parse(t, good())
	specs := qwen3Weights(c)
	h := have(specs)
	h[qwen3LMHead] = []int{c.VocabSize, c.HiddenSize}
	err := Check(specs, h)
	if err == nil {
		t.Fatal("Check accepted a tied checkpoint that also ships lm_head.weight")
	}
	if !strings.Contains(err.Error(), "tie_word_embeddings") ||
		!strings.Contains(err.Error(), qwen3LMHead) {
		t.Errorf("error %q does not name the contradiction", err)
	}
	// The sentinel is the hook a loader holding both planes uses: it may prove
	// them identical and re-run Check without the alias.
	if !errors.Is(err, ErrTiedHeadShipped) {
		t.Errorf("error %v does not wrap ErrTiedHeadShipped", err)
	}
	delete(h, qwen3LMHead)
	if err := Check(specs, h); err != nil {
		t.Errorf("Check refuses the tied checkpoint once the duplicate is gone: %v", err)
	}
}

func TestCheckMissing(t *testing.T) {
	specs := qwen3Weights(parse(t, good()))
	h := have(specs)
	delete(h, "model.layers.1.mlp.down_proj.weight")
	err := Check(specs, h)
	if err == nil {
		t.Fatal("Check accepted a checkpoint with a tensor missing")
	}
	if !strings.Contains(err.Error(), "down_proj") {
		t.Errorf("error %q does not name the missing tensor", err)
	}
}

// TestCheckExtra is §4's "a tensor the map does not mention is not silently
// ignored". Loading the intersection produces a model that runs.
func TestCheckExtra(t *testing.T) {
	specs := qwen3Weights(parse(t, good()))
	h := have(specs)
	h["model.layers.0.self_attn.rotary_emb.inv_freq"] = []int{16}
	err := Check(specs, h)
	if err == nil {
		t.Fatal("Check accepted a checkpoint with a tensor the map does not name")
	}
	if !strings.Contains(err.Error(), "inv_freq") {
		t.Errorf("error %q does not name the extra tensor", err)
	}
}

// TestCheckListsAreTruncated keeps a wrong-architecture checkpoint from
// printing every tensor it has at a caller who needs the first one.
func TestCheckListsAreTruncated(t *testing.T) {
	specs := qwen3Weights(parse(t, good()))
	h := map[string][]int{}
	err := Check(specs, h)
	if err == nil {
		t.Fatal("Check accepted an empty checkpoint")
	}
	if !strings.Contains(err.Error(), "and ") || !strings.Contains(err.Error(), "more") {
		t.Errorf("error %q is not truncated: %d tensors are missing", err, len(specs))
	}
}

// TestCheckVocabSize is §7's row. The message must name the config field, not
// the tensor: the field a caller changes is in config.json.
func TestCheckVocabSize(t *testing.T) {
	c := parse(t, good())
	specs := qwen3Weights(c)
	h := have(specs)
	h[qwen3Embed] = []int{c.VocabSize + 1, c.HiddenSize}
	err := Check(specs, h)
	if err == nil {
		t.Fatal("Check accepted an embedding table whose rows are not vocab_size")
	}
	if !strings.Contains(err.Error(), "vocab_size") {
		t.Errorf("error %q does not name vocab_size", err)
	}
}

// TestCheckShapeMismatch is the general case: a tensor of the right name and
// the wrong width, reported in the file's own layout.
func TestCheckShapeMismatch(t *testing.T) {
	c := parse(t, good())
	specs := qwen3Weights(c)
	cases := []struct {
		name  string
		shape []int
	}{
		{"model.layers.0.self_attn.q_proj.weight", []int{c.HiddenSize, c.HiddenSize}},
		{qwen3Embed, []int{c.VocabSize, c.HiddenSize + 1}},
		{qwen3FinalNorm, []int{c.HiddenSize, 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := have(specs)
			h[tc.name] = tc.shape
			err := Check(specs, h)
			if err == nil {
				t.Fatalf("Check accepted %s with shape %v", tc.name, tc.shape)
			}
			if !strings.Contains(err.Error(), tc.name) {
				t.Errorf("error %q does not name the tensor", err)
			}
		})
	}
}

func TestKindString(t *testing.T) {
	cases := map[Kind]string{
		KindGain: "gain", KindProjection: "projection",
		KindEmbedding: "embedding", KindLMHead: "lm_head", Kind(9): "Kind(9)",
	}
	for k, want := range cases {
		if got := k.String(); got != want {
			t.Errorf("Kind(%d).String() = %q, want %q", int(k), got, want)
		}
	}
}

// TestCheckVocabSizeIsOrderIndependent pins the property the dedupe in Check
// exists for: a tied checkpoint binds two ports to one tensor, and the
// vocab_size message must come from the embedding row whichever order the map
// lists them in. Built by hand rather than through qwen3Weights, because the
// order that breaks it is one qwen3Weights does not produce.
func TestCheckVocabSizeIsOrderIndependent(t *testing.T) {
	specs := []WeightSpec{
		{Tensor: qwen3Embed, Port: "lm_head", Shape: []int{128, 64},
			Layer: -1, Kind: KindLMHead, Transpose: true, Alias: qwen3LMHead},
		{Tensor: qwen3Embed, Port: "embed", Shape: []int{128, 64},
			Layer: -1, Kind: KindEmbedding},
	}
	err := Check(specs, map[string][]int{qwen3Embed: {129, 64}})
	if err == nil {
		t.Fatal("Check accepted an embedding table whose rows are not vocab_size")
	}
	if !strings.Contains(err.Error(), "vocab_size") {
		t.Errorf("error %q does not name vocab_size", err)
	}
}

// TestTiedHeadRedundancyIsAccepted is 004-D10, and the target checkpoint is why
// it exists: Qwen3-0.6B sets tie_word_embeddings, ships lm_head.weight anyway,
// and its two planes are byte-identical. The first version of this rule refused
// any tied checkpoint that shipped a head, which refused the model tgo exists
// to run. A header carries shapes and shapes cannot tell redundancy from
// contradiction, so the decision needs the bytes.
func TestTiedHeadRedundancyIsAccepted(t *testing.T) {
	raw := good()
	raw["tie_word_embeddings"] = true
	specs := qwen3Weights(parse(t, raw))

	have := map[string][]int{}
	for _, s := range specs {
		have[s.Tensor] = s.Shape
	}
	// The checkpoint ships the head as well, which is what Qwen3-0.6B does.
	var alias string
	for _, s := range specs {
		if s.Alias != "" {
			alias = s.Alias
			have[alias] = s.Shape
		}
	}
	if alias == "" {
		t.Fatal("a tied config produced no aliased head port; the fixture is wrong")
	}

	// Without a comparator the refusal stands: guessing which plane the model
	// was trained with is not a decision this package may make.
	if err := Check(specs, have); !errors.Is(err, ErrTiedHeadShipped) {
		t.Fatalf("no comparator: err = %v, want ErrTiedHeadShipped", err)
	}

	// Identical planes are redundancy, and load.
	same := func(a, b string) (bool, error) { return true, nil }
	if err := Check(specs, have, WithPlaneComparator(same)); err != nil {
		t.Fatalf("identical planes: %v, want accepted", err)
	}

	// Differing planes are the real contradiction and stay refused.
	differ := func(a, b string) (bool, error) { return false, nil }
	err := Check(specs, have, WithPlaneComparator(differ))
	if !errors.Is(err, ErrTiedHeadShipped) {
		t.Fatalf("differing planes: err = %v, want ErrTiedHeadShipped", err)
	}
	if !strings.Contains(err.Error(), "bytes differ") {
		t.Fatalf("differing planes: message %q does not say the bytes differ", err)
	}

	// A comparator that cannot answer must not be read as agreement.
	boom := func(a, b string) (bool, error) { return false, errors.New("unreadable") }
	if err := Check(specs, have, WithPlaneComparator(boom)); err == nil ||
		!strings.Contains(err.Error(), "unreadable") {
		t.Fatalf("failing comparator: err = %v, want the comparator's error", err)
	}
}
