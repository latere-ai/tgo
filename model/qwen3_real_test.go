// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"errors"
	"os"
	"sort"
	"testing"

	"github.com/latere-ai/tgo/safetensors"
)

// TestQwen3RealCheckpoint reads a checkpoint on disk.
//
// It is skipped unless TGO_MODEL names a model directory
// (specs/000-decisions.md decision 8): the smallest Qwen3 is over a gigabyte,
// and a CI that downloads one is a CI nobody runs locally. It stays in the tree
// because the synthetic configs above are written by the same hand that wrote
// the map, and this is the one test where the map meets a file it did not
// choose.
func TestQwen3RealCheckpoint(t *testing.T) {
	dir := os.Getenv("TGO_MODEL")
	if dir == "" {
		t.Skip("TGO_MODEL is not set; this test reads a real checkpoint")
	}

	b, err := Open(dir)
	if err != nil {
		t.Fatalf("Open(%s): %v", dir, err)
	}
	c := b.Config()

	// Qwen3-0.6B: d=1024 over H=16 heads, and head_dim 128 rather than the
	// 1024/16 = 64 a defaulting parser would invent. This is the assertion the
	// whole gated test exists for -- specs/004-model-graph.md §5's "the field
	// that bites" -- and every shape below follows from it.
	if c.HeadDim == c.HiddenSize/c.NumHeads {
		t.Errorf("HeadDim = %d = hidden_size/num_attention_heads (%d/%d); "+
			"this checkpoint states head_dim explicitly and it is not that value",
			c.HeadDim, c.HiddenSize, c.NumHeads)
	}
	want := Config{
		Architecture: Qwen3Architecture, HiddenSize: 1024, NumLayers: 28,
		NumHeads: 16, NumKVHeads: 8, HeadDim: 128, IntermediateSize: 3072,
		VocabSize: 151936, RMSNormEps: 1e-6, RoPETheta: 1e6,
		TieWordEmbeddings: true, MaxPositionEmbeddings: 40960,
	}
	if *c != want {
		t.Errorf("Config =\n  %+v\nwant\n  %+v", *c, want)
	}
	// The two projections whose width discriminates: H·d_h = 2048 is twice d,
	// so a map that used hidden_size for the q and o projections fails here.
	// k_proj and v_proj are [1024, 1024], which coincides with [d, d]; they are
	// asserted below through the header comparison, but they prove nothing on
	// their own.
	if c.QWidth() != 2048 || c.KVWidth() != 1024 {
		t.Errorf("QWidth=%d KVWidth=%d, want 2048 and 1024", c.QWidth(), c.KVWidth())
	}

	// The weight map against the checkpoint's own header. OpenRepo parses the
	// header and reads no tensor data, so this costs nothing on a 1.5 GB file.
	// This is the one place §4's table meets a file this repository did not
	// write.
	repo, err := safetensors.OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo(%s): %v", dir, err)
	}
	defer repo.Close()

	held := make(map[string][]int)
	for _, name := range repo.Names() {
		e, _, ok := repo.Tensor(name)
		if !ok {
			t.Fatalf("%s: named by the repo and not resolvable", name)
		}
		held[name] = e.Shape
	}

	specs := b.Weights()
	// 11 tensors per layer over 28 layers, plus the embedding and the final
	// norm, is 310 distinct names -- and 311 specs, because the tied head is a
	// second port on the embedding tensor.
	if len(specs) != 311 {
		t.Errorf("len(Weights()) = %d, want 311", len(specs))
	}
	named := make(map[string]bool, len(specs))
	for _, s := range specs {
		named[s.Tensor] = true
		shape, ok := held[s.Tensor]
		if !ok {
			t.Errorf("port %s: the checkpoint has no %s", s.Port, s.Tensor)
			continue
		}
		if !sameShape(s.Shape, shape) {
			t.Errorf("%s: the map expects %v and the checkpoint holds %v",
				s.Tensor, s.Shape, shape)
		}
	}
	if len(named) != 310 {
		t.Errorf("the map names %d distinct tensors, want 310", len(named))
	}
	var unmapped []string
	for name := range held {
		if !named[name] {
			unmapped = append(unmapped, name)
		}
	}
	sort.Strings(unmapped)

	// The discrepancy, asserted rather than worked around. config.json sets
	// tie_word_embeddings and the checkpoint ships lm_head.weight anyway: one
	// [151936, 1024] BF16 plane whose bytes are identical to the embedding's,
	// materialised by whatever wrote the file. §4 calls that combination a
	// contradiction and refuses it, so the map does not name the tensor and
	// Check refuses the checkpoint this project targets.
	//
	// The refusal is right on the evidence a header carries -- identical shapes
	// say nothing about identical weights -- and it is the reading a loader has
	// to resolve with the bytes in hand. This test pins the state of it so that
	// a change in either direction is visible.
	if !c.TieWordEmbeddings {
		t.Fatal("the checkpoint is expected to set tie_word_embeddings")
	}
	if want := []string{qwen3LMHead}; !slicesEqual(unmapped, want) {
		t.Errorf("tensors the map does not name = %v, want %v", unmapped, want)
	}
	err = Check(specs, held)
	if !errors.Is(err, ErrTiedHeadShipped) {
		t.Errorf("Check error = %v, want one wrapping ErrTiedHeadShipped", err)
	}

	// Everything except the head, which is the whole checkpoint the map claims.
	delete(held, qwen3LMHead)
	if err := Check(specs, held); err != nil {
		t.Errorf("with the duplicated head removed, Check still refuses: %v", err)
	}
}

// slicesEqual compares two string slices.
func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
