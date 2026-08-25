// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/server"
)

// The end-to-end test 009-D4 asks for: a real [tgo.Model], through the real
// handler, over the real dialect codecs.
//
// It is what proves the fake engine's contract is the one the engine keeps.
// Every other test in this package scripts a token stream; this one runs the
// forward pass, which is the only way to find out that [server.Wrap] forwards
// the session options, that a [tgo.Stream] satisfies [server.Stream], and that
// the block events a real generation produces frame correctly.
//
// It needs no checkpoint and no network: the model is synthetic, written into a
// temporary directory, and it runs on accel's CPU backend.

// The synthetic model's shape, which is specs/007-engine.md §8's.
//
// No two of these are equal, and none is a multiple of another where the
// confusion would be silent: a fixture whose layer count equals its key/value
// head count is the identity for every confusion between the two, and three of
// this project's waves each lost a class of bug to exactly that
// (specs/011-sequencing.md).
const (
	synthHidden       = 64
	synthLayers       = 2
	synthHeads        = 8
	synthKVHeads      = 4
	synthHeadDim      = 16
	synthIntermediate = 176
	synthVocab        = 640
)

// synthName is the model id the synthetic server serves.
const synthName = "synthetic-qwen3"

// writeCheckpoint builds a checkpoint directory and returns its path.
//
// The tensor set comes from the model package's own weight map rather than from
// a list written here, so a fixture cannot pass while naming tensors the loader
// does not want.
func writeCheckpoint(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	cfg, err := json.Marshal(map[string]any{
		"architectures":           []string{model.Qwen3Architecture},
		"hidden_size":             synthHidden,
		"num_hidden_layers":       synthLayers,
		"num_attention_heads":     synthHeads,
		"num_key_value_heads":     synthKVHeads,
		"head_dim":                synthHeadDim,
		"intermediate_size":       synthIntermediate,
		"vocab_size":              synthVocab,
		"rms_norm_eps":            1e-6,
		"rope_theta":              1e6,
		"tie_word_embeddings":     true,
		"max_position_embeddings": 4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o600); err != nil {
		t.Fatal(err)
	}
	// tgo/tokenizer's own synthetic vocabulary, read rather than copied: it
	// carries Qwen's control tokens and the thinking markers, which is exactly
	// what the stream's block state machine reads.
	tok, err := os.ReadFile(filepath.Join("..", "tokenizer", "testdata", "synthetic.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), tok, 0o600); err != nil {
		t.Fatal(err)
	}

	b, err := model.Open(dir)
	if err != nil {
		t.Fatalf("model.Open on the fixture's own config: %v", err)
	}
	planes := map[string][]float32{}
	shapes := map[string][]int{}
	var order []string
	for _, s := range b.Weights() {
		if _, seen := planes[s.Tensor]; seen {
			continue
		}
		n := 1
		for _, d := range s.Shape {
			n *= d
		}
		planes[s.Tensor] = plane(s.Tensor, n, s.Kind == model.KindGain)
		shapes[s.Tensor] = s.Shape
		order = append(order, s.Tensor)
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), order, shapes, planes)
	return dir
}

// plane is one tensor's deterministic contents.
//
// Every value is a multiple of 1/32 with magnitude below one, which f16 holds
// exactly, so the storage format contributes nothing to what the test sees.
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
			// A norm gain near one. A gain centred on zero would zero the
			// residual stream and every logit with it.
			v = 1 + v/4
		}
		out[i] = v
	}
	return out
}

// writeSafetensors writes one shard as bf16, which is what a Hugging Face
// checkpoint holds and the branch of the widening a real load takes.
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
		begin := len(data)
		for _, v := range planes[name] {
			data = binary.LittleEndian.AppendUint16(data, uint16(math.Float32bits(v)>>16))
		}
		header[name] = entry{DType: "BF16", Shape: shapes[name], DataOffsets: [2]int{begin, len(data)}}
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(raw)))
	out = append(out, raw...)
	out = append(out, data...)
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
}

// openSynthetic opens the fixture on the CPU backend, which is the tier-1
// device every test in this tree runs on.
func openSynthetic(t *testing.T) *tgo.Model {
	t.Helper()
	// 96 rather than 64: a context equal to the hidden size is the identity for
	// every confusion between a position and a channel, which is the fixture
	// shape specs/011-sequencing.md's waves paid for.
	m, err := tgo.Open(writeCheckpoint(t), tgo.WithDevice(tgo.CPU), tgo.WithContext(96))
	if err != nil {
		t.Fatalf("tgo.Open: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("Model.Close: %v", err)
		}
	})
	return m
}

func TestASyntheticModelAnswersThroughTheRealHandler(t *testing.T) {
	m := openSynthetic(t)
	// The KV budget is divided by the model's own reservation, which is the
	// number specs/005-kv-cache.md computes and Info reports before anything
	// is allocated.
	s, err := server.New(server.Wrap(m, synthName), server.WithNotice(&strings.Builder{}),
		server.WithKVBudget(2*m.Info().CacheBytesPerSession))
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	if got := s.Concurrency(); got != 2 {
		t.Errorf("concurrency = %d, want the budget divided by one session's cache", got)
	}

	// The engine's own numbers reach the surface, rather than a copy of them
	// written here.
	if got := m.Info().VocabSize; got != synthVocab {
		t.Fatalf("the fixture's vocabulary is %d, want %d", got, synthVocab)
	}
	health := do(t, s, http.MethodGet, "/health", "")
	if !strings.Contains(health.Body.String(), synthName) {
		t.Errorf("/health does not name the model: %s", health.Body.String())
	}

	cases := []struct{ name, path, body string }{
		{"chat", "/v1/chat/completions", `{"model":"` + synthName + `","max_tokens":2,` +
			`"messages":[{"role":"user","content":"hi"}]}`},
		{"messages", "/v1/messages", `{"model":"` + synthName + `","max_tokens":2,` +
			`"messages":[{"role":"user","content":"hi"}]}`},
		{"responses", "/v1/responses", `{"model":"` + synthName + `","max_output_tokens":2,` +
			`"input":"hi"}`},
		{"completions", "/v1/completions", `{"model":"` + synthName + `","max_tokens":2,` +
			`"prompt":"hi"}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			w := do(t, s, http.MethodPost, c.path, c.body)
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", w.Code, w.Body.String())
			}
			var body map[string]any
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
				t.Fatalf("the answer is not JSON: %v: %s", err, w.Body.String())
			}
		})
		t.Run(c.name+" streaming", func(t *testing.T) {
			body := strings.TrimSuffix(c.body, "}") + `,"stream":true}`
			w := do(t, s, http.MethodPost, c.path, body)
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", w.Code, w.Body.String())
			}
			if got := w.Header().Get("Content-Type"); got != "text/event-stream" {
				t.Fatalf("Content-Type = %q", got)
			}
			if !strings.Contains(w.Body.String(), "data: ") {
				t.Fatalf("the stream carried no frames: %q", w.Body.String())
			}
		})
	}
}

// A seed makes the same request give the same answer, through the server as
// well as through the engine. It is the one property a caller can check for
// themselves, and the one the loss report would quietly break.
func TestTheSameSeedGivesTheSameAnswerThroughTheServer(t *testing.T) {
	m := openSynthetic(t)
	s, err := server.New(server.Wrap(m, synthName), server.WithNotice(&strings.Builder{}))
	if err != nil {
		t.Fatal(err)
	}
	body := `{"model":"` + synthName + `","max_tokens":3,"temperature":0.8,"seed":11,` +
		`"messages":[{"role":"user","content":"hi"}]}`
	first := answerOf(t, do(t, s, http.MethodPost, "/v1/chat/completions", body))
	second := answerOf(t, do(t, s, http.MethodPost, "/v1/chat/completions", body))
	if first != second {
		t.Errorf("the same seed gave %q and then %q", first, second)
	}
	// And an advisory field does not move it, which is §4's rule stated as a
	// property of the real engine rather than of a script.
	withUser := strings.TrimSuffix(body, "}") + `,"user":"someone"}`
	w := do(t, s, http.MethodPost, "/v1/chat/completions", withUser)
	if got := answerOf(t, w); got != first {
		t.Errorf("user changed the answer: %q vs %q", got, first)
	}
	if loss := w.Header().Get("X-Tgo-Loss"); loss != "user" {
		t.Errorf("X-Tgo-Loss = %q, want %q", loss, "user")
	}
}

// A refusal still refuses when the engine is real, which is the case a fake
// cannot rule out: the vocabulary is the model's now.
func TestARefusalHoldsAgainstTheRealVocabulary(t *testing.T) {
	m := openSynthetic(t)
	s, err := server.New(server.Wrap(m, synthName), server.WithNotice(&strings.Builder{}))
	if err != nil {
		t.Fatal(err)
	}
	body := `{"model":"` + synthName + `","max_tokens":2,"logit_bias":{"999999":-100},` +
		`"messages":[{"role":"user","content":"hi"}]}`
	w := do(t, s, http.MethodPost, "/v1/chat/completions", body)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "logit_bias") {
		t.Errorf("the refusal does not name the field: %s", w.Body.String())
	}
}

func do(t *testing.T, s *server.Server, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	return w
}

// answerOf reads the assistant's content out of an OpenAI Chat body.
func answerOf(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var body struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("%v: %s", err, w.Body.String())
	}
	if len(body.Choices) != 1 {
		t.Fatalf("choices = %d: %s", len(body.Choices), w.Body.String())
	}
	return body.Choices[0].Message.ReasoningContent + "|" + body.Choices[0].Message.Content
}
