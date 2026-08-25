// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/latere-ai/tgo/model"
)

// The synthetic model every test below runs on: specs/007-engine.md §8's
// "synthetic 2-layer config", sized so no two extents collide.
//
// d=64 over H=8 heads with head_dim 16 rather than the 8 that ratio implies, so
// a shape taken from d/H is wrong everywhere; H_kv=2, so the grouped-query
// ratio is 4 and a q positions tensor used for k is the wrong length; f=128,
// which is neither d nor H·d_h.
//
// V is 640 because the tokenizer fixture's id space is 582 wide: a vocabulary
// smaller than the tokenizer can produce would gather out of range on a token
// nobody chose.
// No two of these are equal, and that is a rule rather than an accident. A
// fixture where the layer count equals the key/value head count is the identity
// for every confusion between the two, so a wrong shape reads as correct: two
// earlier waves each lost a whole class of bug to exactly that collision
// (specs/011-sequencing.md's Wave 2 and Wave 3 entries). synthKVHeads was 2,
// which is synthLayers, and is now 4 -- synthHeads still divides by it, which
// is what GQA requires.
//
// Derived extents are distinct too: H·d_h = 128 collides with synthIntermediate
// by construction, so read that one carefully if you change either.
const (
	synthHidden       = 64
	synthLayers       = 2
	synthHeads        = 8
	synthKVHeads      = 4
	synthHeadDim      = 16
	synthIntermediate = 176
	synthVocab        = 640
)

// tokenizerFixture is the tokenizer every test loads.
//
// It is tgo/tokenizer's own synthetic vocabulary, read rather than copied: it
// is 582 byte-level BPE tokens with Qwen's control tokens and the thinking
// markers, which is exactly what this package's block state machine reads, and
// a second copy in this directory would be a fixture that drifts.
const tokenizerFixture = "tokenizer/testdata/synthetic.json"

// checkpoint describes a synthetic checkpoint to write.
type checkpoint struct {
	// tie sets tie_word_embeddings.
	tie bool

	// shipHead writes an lm_head.weight beside a tied embedding, which is what
	// Qwen3-0.6B does. identicalHead says whether its bytes match the
	// embedding's, which is the difference between 004-D10's accept and its
	// refuse.
	shipHead      bool
	identicalHead bool

	// vocab overrides the vocabulary size, for the refusal tests.
	vocab int
}

// write builds a checkpoint directory and returns its path.
//
// The tensor set comes from the model's own weight map rather than from a list
// written here: a fixture that named its own tensors would pass while the map
// named others, which is the one thing this fixture must not be able to do.
func (c checkpoint) write(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	vocab := c.vocab
	if vocab == 0 {
		vocab = synthVocab
	}
	cfg := map[string]any{
		"architectures":           []string{model.Qwen3Architecture},
		"hidden_size":             synthHidden,
		"num_hidden_layers":       synthLayers,
		"num_attention_heads":     synthHeads,
		"num_key_value_heads":     synthKVHeads,
		"head_dim":                synthHeadDim,
		"intermediate_size":       synthIntermediate,
		"vocab_size":              vocab,
		"rms_norm_eps":            1e-6,
		"rope_theta":              1e6,
		"tie_word_embeddings":     c.tie,
		"max_position_embeddings": 4096,
	}
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	tok, err := os.ReadFile(tokenizerFixture)
	if err != nil {
		t.Fatalf("read %s: %v", tokenizerFixture, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), tok, 0o600); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}

	b, err := model.Open(dir)
	if err != nil {
		t.Fatalf("model.Open on the fixture's own config: %v", err)
	}
	planes := map[string][]float32{}
	order := []string{}
	for _, s := range b.Weights() {
		if _, ok := planes[s.Tensor]; ok {
			continue
		}
		n := 1
		for _, d := range s.Shape {
			n *= d
		}
		planes[s.Tensor] = plane(s.Tensor, n, s.Kind == model.KindGain)
		order = append(order, s.Tensor)
	}
	shapes := map[string][]int{}
	for _, s := range b.Weights() {
		shapes[s.Tensor] = s.Shape
	}
	if c.shipHead {
		const head = "lm_head.weight"
		src := planes["model.embed_tokens.weight"]
		p := make([]float32, len(src))
		copy(p, src)
		if !c.identicalHead {
			p[0] += 0.5
		}
		planes[head] = p
		shapes[head] = shapes["model.embed_tokens.weight"]
		order = append(order, head)
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), order, shapes, planes)
	return dir
}

// plane is one tensor's deterministic contents.
//
// Every value is a multiple of 1/32 with magnitude below 1, which bf16 and f16
// both hold exactly: the storage format then contributes nothing to any
// comparison, and what a test measures is the graph rather than the rounding
// (the same trick as model/graph_rig_test.go's f16exact).
func plane(name string, n int, gain bool) []float32 {
	out := make([]float32, n)
	seed := uint32(2166136261)
	for i := 0; i < len(name); i++ {
		seed = (seed ^ uint32(name[i])) * 16777619
	}
	for i := range out {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		v := float32(int32(seed%32)-16) / 32
		if gain {
			// A norm gain near one: a gain centred on zero would zero the
			// residual stream and every logit with it.
			v = 1 + v/4
		}
		out[i] = v
	}
	return out
}

// writeSafetensors writes one shard: the 8-byte header length, the header, and
// the planes, as bf16 — which is what a Hugging Face checkpoint holds and the
// branch of the widening a real load takes.
func writeSafetensors(t *testing.T, path string, order []string, shapes map[string][]int,
	planes map[string][]float32) {

	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, name := range order {
		p := planes[name]
		begin := len(data)
		for _, v := range p {
			data = binary.LittleEndian.AppendUint16(data, uint16(math.Float32bits(v)>>16))
		}
		header[name] = entry{DType: "BF16", Shape: shapes[name], DataOffsets: [2]int{begin, len(data)}}
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(raw)))
	out = append(out, raw...)
	out = append(out, data...)
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// openSynthetic opens the default fixture on the CPU backend.
//
// One Model, one device and one plan cache per test, which is what lets almost
// every test in this package call t.Parallel: nothing here is shared, and a
// plan-count assertion reads its own cache. Three tests do not, and each says
// why -- two swap os.Stderr and one counts the process's goroutines.
//
// Parallelism is not a preference. accel's CPU backend runs a forward pass at
// roughly half a core, and under -race the whole suite takes longer than go
// test's default ten-minute timeout on one goroutine at a time, so a serial
// suite fails the race gate on the clock rather than on a race.
func openSynthetic(t *testing.T, opts ...Option) *Model {
	t.Helper()
	dir := checkpoint{tie: true}.write(t)
	return openAt(t, dir, opts...)
}

func openAt(t *testing.T, dir string, opts ...Option) *Model {
	t.Helper()
	opts = append([]Option{WithDevice(CPU)}, opts...)
	m, err := Open(dir, opts...)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("Model.Close: %v", err)
		}
	})
	return m
}

// session opens a session and closes it with the test.
func session(t *testing.T, m *Model, opts ...SessionOption) *Session {
	t.Helper()
	s, err := m.NewSession(opts...)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Session.Close: %v", err)
		}
	})
	return s
}

// collect drains a stream into its text and its events.
func collect(t *testing.T, st *Stream) (string, []Event) {
	t.Helper()
	var text string
	var evs []Event
	for st.Next() {
		evs = append(evs, st.Event())
		text += st.Text()
	}
	return text, evs
}

// greedy is the policy every determinism test uses.
func greedy(max int) Policy { return Policy{MaxTokens: max} }

func fmtIDs(ids []int) string { return fmt.Sprint(ids) }
