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
	"strconv"
	"strings"
	"testing"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/server"
	"github.com/latere-ai/tgo/tokenizer"
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

// A request carrying a schema, through the real handler, over a real model,
// comes back as a document that parses and that matches the schema.
//
// This is the whole of specs/015-structured-output.md reachable from a request:
// `response_format` is mapped onto [tgo.Policy], the schema is compiled against
// the model's own vocabulary, and the mask is applied on every step. Every
// other test of the grammar runs against a vocabulary the test built; this one
// runs against a byte-level BPE, which is where the bytes a token stands for
// stop being the bytes a vocabulary file spells it with.
func TestASchemaThroughTheServerProducesADocumentThatMatchesIt(t *testing.T) {
	dir := writeCheckpoint(t)
	m, err := tgo.Open(dir, tgo.WithDevice(tgo.CPU), tgo.WithContext(512))
	if err != nil {
		t.Fatalf("tgo.Open: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("Model.Close: %v", err)
		}
	})
	s, err := server.New(server.Wrap(m, synthName), server.WithNotice(&strings.Builder{}))
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}

	const budget = 64
	body := `{"model":"` + synthName + `","max_tokens":` + strconv.Itoa(budget) +
		`,"messages":[{"role":"user","content":"describe a place"}],` +
		`"logit_bias":` + banWhitespace(t, dir) + `,` +
		`"response_format":{"type":"json_schema","json_schema":{"name":"place","strict":true,` +
		`"schema":` + placeSchema + `}}}`
	w := do(t, s, http.MethodPost, "/v1/chat/completions", body)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	answer := contentOf(t, w)
	if !json.Valid([]byte(answer)) {
		t.Fatalf("the answer is not valid JSON: %q", answer)
	}
	var got struct {
		City    string `json:"city"`
		Capital bool   `json:"capital"`
	}
	dec := json.NewDecoder(strings.NewReader(answer))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&got); err != nil {
		t.Fatalf("the answer does not match the schema: %v: %q", err, answer)
	}
	if got.City != "oslo" {
		t.Errorf(`city = %q, want "oslo", the only member of its enum: %q`, got.City, answer)
	}
	// The properties are in the schema's order, which is one of the compiler's
	// narrowings and is what a caller reading the raw body sees.
	if i, j := strings.Index(answer, `"city"`), strings.Index(answer, `"capital"`); i > j {
		t.Errorf("the properties are not in the schema's order: %q", answer)
	}
	// It ended because the document was complete, not because it ran out of
	// budget. A grammar carrying no stop ids masks every token away on the step
	// after the last brace, so this is the assertion that says the stop set
	// reached the compiler.
	if reason := finishOf(t, w); reason != "stop" {
		t.Errorf("finish_reason = %q, want %q: the completion was cut off rather than "+
			"finished: %q", reason, "stop", answer)
	}
}

// placeSchema is the fixture's schema: two required properties, of different
// types, in a closed object.
//
// Its language is finite, which is what makes the budget above a bound rather
// than a hope. An "integer" property would not be: JSON Schema spells a
// magnitude bound as "maximum", the compiler refuses that as arithmetic on the
// value, and a model whose weights are arbitrary will type digits for as long
// as it is allowed to. That is the package's documented narrowing seen from the
// test's side, and the honest answer is a schema whose language ends.
const placeSchema = `{"type":"object","properties":{` +
	`"city":{"type":"string","enum":["oslo"]},` +
	`"capital":{"type":"boolean"}},` +
	`"required":["city","capital"],"additionalProperties":false}`

// banWhitespace is a logit_bias member banning every token that carries an
// ASCII space, tab, newline or carriage return.
//
// It is a property of this fixture and not of constrained decoding. JSON admits
// whitespace before every token, so the grammar admits it too; the synthetic
// checkpoint's weights are arbitrary and it draws a space as readily as a
// brace, and a run that spends its budget on indentation says nothing about the
// mask. With whitespace banned the structural tokens are the only admissible
// ones, so the budget above is a real bound rather than a hope.
//
// The ban is a large finite number rather than negative infinity because it
// travels as JSON, which has no spelling for an infinity.
func banWhitespace(t *testing.T, dir string) string {
	t.Helper()
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("loading the fixture's tokenizer: %v", err)
	}
	bias := map[string]float64{}
	for id := 0; id < synthVocab; id++ {
		if strings.ContainsAny(string(tk.TextBytes(id)), " \t\n\r") {
			bias[strconv.Itoa(id)] = -1e30
		}
	}
	if len(bias) == 0 {
		t.Fatal("the fixture holds no whitespace token, so this ban bans nothing")
	}
	raw, err := json.Marshal(bias)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// contentOf reads the assistant's answer out of an OpenAI Chat body. Compare
// answerOf, which joins the thinking block to it with a separator that is not
// JSON.
func contentOf(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var body struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("the answer is not JSON: %v: %s", err, w.Body.String())
	}
	if len(body.Choices) != 1 {
		t.Fatalf("choices = %d, want 1: %s", len(body.Choices), w.Body.String())
	}
	return body.Choices[0].Message.Content
}

// finishOf reads finish_reason out of an OpenAI Chat body.
func finishOf(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var body struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("the answer is not JSON: %v: %s", err, w.Body.String())
	}
	if len(body.Choices) != 1 {
		t.Fatalf("choices = %d, want 1: %s", len(body.Choices), w.Body.String())
	}
	return body.Choices[0].FinishReason
}

// A schema the compiler refuses is a 400 through the real engine too, carrying
// the compiler's reason, and no session is built.
//
// Every other refusal test in this package goes through fakeEngine, which does
// its own compilation. That leaves [server.Wrap]'s one-line join to
// [github.com/latere-ai/tgo.Model.CheckSchema] untested: an engine that
// answered nil there would accept the request, allocate the session, and only
// then fail inside the generation -- after taking the KV reservation this file
// refuses in order to protect, and with the compiler's reason no longer on the
// path a caller reads.
func TestAnUncompilableSchemaIsRefusedByTheRealEngine(t *testing.T) {
	dir := writeCheckpoint(t)
	m, err := tgo.Open(dir, tgo.WithDevice(tgo.CPU), tgo.WithContext(512))
	if err != nil {
		t.Fatalf("tgo.Open: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("Model.Close: %v", err)
		}
	})
	s, err := server.New(server.Wrap(m, synthName), server.WithNotice(&strings.Builder{}))
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	// A numeric bound: the automaton counts characters, so a magnitude is
	// arithmetic on the value and the compiler says so by name.
	body := `{"model":"` + synthName + `","max_tokens":8,` +
		`"messages":[{"role":"user","content":"how many"}],` +
		`"response_format":{"type":"json_schema","json_schema":{"name":"n","strict":true,` +
		`"schema":{"type":"integer","minimum":1}}}}`
	w := do(t, s, http.MethodPost, "/v1/chat/completions", body)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d: %s", w.Code, http.StatusBadRequest, w.Body.String())
	}
	// The body is JSON, so the compiler's quotes around the keyword arrive
	// escaped.
	for _, want := range []string{"arithmetic on the value", `\"minimum\"`, "response_format"} {
		if !strings.Contains(w.Body.String(), want) {
			t.Errorf("the refusal does not carry %q: %s", want, w.Body.String())
		}
	}
}
