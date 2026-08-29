// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.design/x/accel"

	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/nn"
	"github.com/latere-ai/tgo/safetensors"
	"github.com/latere-ai/tgo/weights"
)

// TestOpenRefusals: every one names what was wrong, because a checkpoint is a
// file somebody downloaded and "could not open the model" is not a sentence
// anybody can act on.
func TestOpenRefusals(t *testing.T) {
	t.Parallel()
	good := checkpoint{tie: true}.write(t)

	t.Run("a context of zero", func(t *testing.T) {
		if _, err := Open(good, WithContext(0)); err == nil {
			t.Error("a zero context was accepted")
		}
	})
	t.Run("a directory that is not there", func(t *testing.T) {
		if _, err := Open(filepath.Join(t.TempDir(), "absent")); err == nil {
			t.Error("a missing directory was accepted")
		}
	})
	t.Run("no tokenizer", func(t *testing.T) {
		dir := checkpoint{tie: true}.write(t)
		if err := os.Remove(filepath.Join(dir, "tokenizer.json")); err != nil {
			t.Fatal(err)
		}
		_, err := Open(dir, WithDevice(CPU))
		if err == nil || !strings.Contains(err.Error(), "tokenizer") {
			t.Errorf("error = %v; it should name the tokenizer", err)
		}
	})
	t.Run("an unknown architecture", func(t *testing.T) {
		dir := checkpoint{tie: true}.write(t)
		raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
		if err != nil {
			t.Fatal(err)
		}
		bad := strings.Replace(string(raw), model.Qwen3Architecture, "LlamaForCausalLM", 1)
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(bad), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err = Open(dir, WithDevice(CPU))
		if err == nil || !strings.Contains(err.Error(), model.Qwen3Architecture) {
			t.Errorf("error = %v; an unknown architecture is refused with the known list",
				err)
		}
	})
	t.Run("a device that is not one", func(t *testing.T) {
		if _, err := Open(good, WithDevice(Device(9))); err == nil {
			t.Error("Device(9) was accepted")
		}
	})
}

// TestTiedCheckpointThatAlsoShipsAHead is 004-D10, and the reason the plane
// comparator lives in this package: shapes cannot tell the two cases apart, and
// the bytes are the engine's because the engine holds the file.
func TestTiedCheckpointThatAlsoShipsAHead(t *testing.T) {
	t.Parallel()
	t.Run("identical planes are accepted", func(t *testing.T) {
		dir := checkpoint{tie: true, shipHead: true, identicalHead: true}.write(t)
		m := openAt(t, dir)
		if m.Info().Layers != synthLayers {
			t.Errorf("Layers = %d", m.Info().Layers)
		}
	})
	t.Run("planes that differ are refused", func(t *testing.T) {
		dir := checkpoint{tie: true, shipHead: true, identicalHead: false}.write(t)
		_, err := Open(dir, WithDevice(CPU))
		if !errors.Is(err, model.ErrTiedHeadShipped) {
			t.Errorf("error = %v, want one wrapping ErrTiedHeadShipped: the config says "+
				"the head is the embedding and the weights say it is not, and picking "+
				"one is a guess", err)
		}
	})
	t.Run("an untied checkpoint carries its own head", func(t *testing.T) {
		dir := checkpoint{tie: false, shipHead: false}.write(t)
		m := openAt(t, dir)
		if _, ok := m.set.Get("lm_head"); !ok {
			t.Error("the lm_head port was not loaded")
		}
	})
}

// TestInfoReportsWhatWasResolved is what a caller who passed AutoDevice or
// AutoPrecision has no other way to learn.
func TestInfoReportsWhatWasResolved(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t, WithPrecision(F16))
	got := m.Info()
	want := Info{
		Architecture: model.Qwen3Architecture, Layers: synthLayers,
		HiddenSize: synthHidden, Heads: synthHeads, KVHeads: synthKVHeads,
		HeadDim: synthHeadDim, IntermediateSize: synthIntermediate,
		VocabSize: synthVocab, TrainedContext: 4096, Context: DefaultContext,
		Device: CPU, Precision: F16,
		WeightBytes:          got.WeightBytes,
		CacheBytesPerSession: cacheBytes(m.cfg, DefaultContext, accel.F32),
		// A dense stack caches every layer, and saying so is what lets a
		// reader divide the bytes back into a width. A hybrid's is one in four
		// (specs/023-cache-kinds.md section 8).
		CachedLayers: synthLayers,
	}
	if got != want {
		t.Errorf("Info() =\n  %+v\nwant\n  %+v", got, want)
	}
	if got.WeightBytes <= 0 {
		t.Errorf("WeightBytes = %d", got.WeightBytes)
	}
	// The gains are on the device and the loader's report does not count them,
	// because the loader did not upload them.
	if got.WeightBytes <= m.set.Report().Bytes {
		t.Errorf("WeightBytes %d does not exceed the loader's %d; the f32 norm gains are "+
			"this package's upload and are part of what the model costs",
			got.WeightBytes, m.set.Report().Bytes)
	}
}

// TestRaisingTheContextPrintsWhatItCosts is 005-D3: the user learns the number
// when they ask for it, not from an out-of-memory error.
func TestRaisingTheContextPrintsWhatItCosts(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved := os.Stderr
	os.Stderr = w
	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()

	dir := checkpoint{tie: true}.write(t)
	m, err := Open(dir, WithDevice(CPU), WithPrecision(F16), WithContext(DefaultContext*2))
	os.Stderr = saved
	_ = w.Close()
	out := <-done
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = m.Close() }()

	if !strings.Contains(out, "8192") || !strings.Contains(out, "key/value cache") {
		t.Errorf("raising the context printed %q; it should name the positions and the "+
			"cache they cost", out)
	}

	// And the default does not print, because a number nobody asked for is
	// noise on every run.
	r2, w2, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w2
	done2 := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r2)
		done2 <- string(b)
	}()
	m2, err := Open(dir, WithDevice(CPU), WithPrecision(F16))
	os.Stderr = saved
	_ = w2.Close()
	quiet := <-done2
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = m2.Close() }()
	if quiet != "" {
		t.Errorf("the default context at a named precision printed %q", quiet)
	}
}

// TestAutomaticPrecisionIsPrinted is specs/001-weights.md §5: the choice is
// printed, never silent. A caller who named a precision chose it themselves and
// gets nothing, which is the test above.
func TestAutomaticPrecisionIsPrinted(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved := os.Stderr
	os.Stderr = w
	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()

	dir := checkpoint{tie: true}.write(t)
	m, err := Open(dir, WithDevice(CPU))
	os.Stderr = saved
	_ = w.Close()
	out := <-done
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = m.Close() }()

	if !strings.Contains(out, "weights:") ||
		!strings.Contains(out, m.Info().Precision.String()) {
		t.Errorf("an automatic precision printed %q; it should name what it chose", out)
	}
}

// TestRequestRefusals: everything a caller can get wrong about a request, each
// refused before anything is submitted.
func TestRequestRefusals(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(64))
	ctx := t.Context()

	long := make([]int, 0, 80)
	for i := 0; i < 80; i++ {
		long = append(long, i%synthVocab)
	}

	t.Run("an empty prompt", func(t *testing.T) {
		if _, err := s.Complete(ctx, "", greedy(2)); err == nil {
			t.Error("an empty prompt was accepted")
		}
	})
	t.Run("a nil context", func(t *testing.T) {
		//lint:ignore SA1012 the nil is the case under test.
		if _, err := s.start(nil, []int{1}, greedy(2)); err == nil {
			t.Error("a nil context was accepted")
		}
	})
	t.Run("a prompt longer than the cache", func(t *testing.T) {
		_, err := s.start(ctx, long, greedy(2))
		if !errors.Is(err, ErrContextExhausted) {
			t.Errorf("error = %v, want ErrContextExhausted: a context is refused, never "+
				"truncated", err)
		}
	})
	t.Run("a budget the cache cannot pay", func(t *testing.T) {
		_, err := s.Complete(ctx, "short", greedy(64))
		if !errors.Is(err, ErrContextExhausted) {
			t.Errorf("error = %v, want ErrContextExhausted", err)
		}
	})
	t.Run("a closed session", func(t *testing.T) {
		s2, err := m.NewSession(WithSessionContext(64))
		if err != nil {
			t.Fatal(err)
		}
		if err := s2.Close(); err != nil {
			t.Fatal(err)
		}
		if err := s2.Close(); err != nil {
			t.Errorf("a second Close: %v", err)
		}
		if _, err := s2.Complete(ctx, "x", greedy(1)); err == nil {
			t.Error("a closed session accepted work")
		}
		if _, err := s2.Chat(ctx, nil, greedy(1)); err == nil {
			t.Error("a closed session accepted a chat")
		}
	})
	t.Run("a session with no context", func(t *testing.T) {
		if _, err := m.NewSession(WithSessionContext(0)); err == nil {
			t.Error("a zero-position session was created")
		}
	})
	t.Run("a message the template refuses", func(t *testing.T) {
		_, err := s.Chat(ctx, []chat.Message{{
			Role:   chat.User,
			Blocks: []chat.Block{{Type: chat.BlockThinking, Text: "not a user's block"}},
		}}, greedy(1))
		if err == nil {
			t.Error("a thinking block on a user turn was rendered")
		}
	})
}

// TestPolicyRefusals: each names its field, because the sampler one layer down
// panics and a caller who typed a number into a request should get an error.
func TestPolicyRefusals(t *testing.T) {
	t.Parallel()
	nan := float32(math.NaN())
	for _, c := range []struct {
		name  string
		p     Policy
		field string
	}{
		{"negative max tokens", Policy{MaxTokens: -1}, "MaxTokens"},
		{"negative temperature", Policy{Temperature: -1}, "Temperature"},
		{"NaN temperature", Policy{Temperature: nan}, "Temperature"},
		{"top-k above the kernel's rounds", Policy{TopK: 500}, "TopK"},
		{"negative top-k", Policy{TopK: -1}, "TopK"},
		{"top-p above one", Policy{TopP: 1.5}, "TopP"},
		{"NaN top-p", Policy{TopP: nan}, "TopP"},
		{"negative repetition penalty", Policy{RepetitionPenalty: -1}, "RepetitionPenalty"},
		{"NaN repetition penalty", Policy{RepetitionPenalty: nan}, "RepetitionPenalty"},
		{"negative penalty window", Policy{PenaltyWindow: -3}, "PenaltyWindow"},
		{"a bias outside the vocabulary", Policy{LogitBias: map[int]float32{9999: 1}},
			"LogitBias"},
		{"a negative bias id", Policy{LogitBias: map[int]float32{-1: 1}}, "LogitBias"},
		{"a NaN bias", Policy{LogitBias: map[int]float32{3: nan}}, "LogitBias"},
		{"a bias of positive infinity",
			Policy{LogitBias: map[int]float32{3: float32(math.Inf(1))}}, "LogitBias"},
		{"an empty stop string", Policy{Stop: []string{""}}, "Stop"},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := c.p.check(synthVocab)
			if err == nil {
				t.Fatalf("%+v was accepted", c.p)
			}
			if !strings.Contains(err.Error(), c.field) {
				t.Errorf("error %q does not name %s", err, c.field)
			}
		})
	}
	// And a ban, which is what negative infinity is for.
	if err := (Policy{Temperature: 0.7, TopK: 40, TopP: 0.9, RepetitionPenalty: 1.1,
		PenaltyWindow: 64, Stop: []string{"x"}, MaxTokens: 4,
		LogitBias: map[int]float32{1: -1, 2: float32(math.Inf(-1))},
	}).check(synthVocab); err != nil {
		t.Errorf("a whole valid policy was refused: %v", err)
	}
}

// TestStepFillRefusals covers the shapes a step cannot take.
func TestStepFillRefusals(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	var d stepData
	c := m.cfg
	d = stepData{
		ids:     make([]uint32, 8),
		posq:    make([]uint32, 8*c.NumHeads),
		posk:    make([]uint32, 8*c.NumKVHeads),
		slots:   make([]uint32, 8),
		lengths: make([]uint32, 1),
	}
	for _, c2 := range []struct {
		name           string
		rows           int
		toks           []int
		first, cap     int
		wantSubstrings string
	}{
		{"no tokens", 8, nil, 0, 64, "cannot score"},
		{"more tokens than rows", 2, []int{1, 2, 3}, 0, 64, "cannot score"},
		{"a negative position", 8, []int{1}, -1, 64, "do not fit"},
		{"past the capacity", 8, []int{1, 2}, 63, 64, "do not fit"},
		{"a token outside the vocabulary", 8, []int{synthVocab}, 0, 64, "vocabulary"},
		{"a negative token", 8, []int{-1}, 0, 64, "vocabulary"},
	} {
		t.Run(c2.name, func(t *testing.T) {
			err := d.fill(c, c2.rows, c2.toks, c2.first, cacheLayout{rows: c2.cap, limit: c2.cap})
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), c2.wantSubstrings) {
				t.Errorf("error = %q, want one mentioning %q", err, c2.wantSubstrings)
			}
		})
	}
}

// TestWidenReadsEveryFloatWidth is the gain path's decode step: the three
// widths a checkpoint holds, and a refusal for anything else.
func TestWidenReadsEveryFloatWidth(t *testing.T) {
	t.Parallel()
	want := []float32{-1.5, 0, 0.25, 2}
	cases := []struct {
		dt  safetensors.DType
		raw []byte
	}{
		{safetensors.F32, nil},
		{safetensors.F16, nil},
		{safetensors.BF16, nil},
	}
	for i := range cases {
		for _, v := range want {
			switch cases[i].dt {
			case safetensors.F32:
				cases[i].raw = binary.LittleEndian.AppendUint32(cases[i].raw,
					math.Float32bits(v))
			case safetensors.F16:
				cases[i].raw = binary.LittleEndian.AppendUint16(cases[i].raw,
					accel.ToFloat16(v).Bits())
			case safetensors.BF16:
				cases[i].raw = binary.LittleEndian.AppendUint16(cases[i].raw,
					uint16(math.Float32bits(v)>>16))
			}
		}
	}
	for _, c := range cases {
		got := make([]float32, len(want))
		if err := widen(c.dt, c.raw, got); err != nil {
			t.Fatalf("%v: %v", c.dt, err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("%v: got %v, want %v", c.dt, got, want)
				break
			}
		}
	}
	if err := widen(safetensors.I8, make([]byte, 4), make([]float32, 4)); err == nil {
		t.Error("an int plane was read as a gain")
	}
	if err := widen(safetensors.DType("F8"), nil, make([]float32, 1)); err == nil {
		t.Error("an unknown dtype was accepted")
	}
	if err := widen(safetensors.F32, make([]byte, 7), make([]float32, 2)); err == nil {
		t.Error("a short plane was accepted")
	}
}

// TestPermuteHeads is specs/004-model-graph.md §2.5.2's rewrite, which nothing
// downstream refuses and nothing downstream undoes.
func TestPermuteHeads(t *testing.T) {
	t.Parallel()
	in := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	one := append([]float32(nil), in...)
	if err := permuteHeads(one, 1); err != nil {
		t.Fatal(err)
	}
	// d_h = 8, half = 4: y[2i] = x[i], y[2i+1] = x[i+4].
	if want := []float32{0, 4, 1, 5, 2, 6, 3, 7}; !equalF32(one, want) {
		t.Errorf("one head gave %v, want %v", one, want)
	}
	two := append([]float32(nil), in...)
	if err := permuteHeads(two, 2); err != nil {
		t.Fatal(err)
	}
	if want := []float32{0, 2, 1, 3, 4, 6, 5, 7}; !equalF32(two, want) {
		t.Errorf("two heads gave %v, want %v", two, want)
	}
	if err := permuteHeads(in, 0); err == nil {
		t.Error("zero heads was accepted")
	}
	if err := permuteHeads(in, 3); err == nil {
		t.Error("a plane that does not split into three heads was accepted")
	}
	if err := permuteHeads(make([]float32, 3), 1); err == nil {
		t.Error("an odd head width was accepted; RoPE rotates pairs")
	}
}

// TestPlainValueHelpers covers the small functions a reader would otherwise
// have to trust.
func TestPlainValueHelpers(t *testing.T) {
	t.Parallel()
	for _, c := range []struct {
		n    int64
		want string
	}{
		{0, "0 B"}, {512, "512 B"}, {2048, "2.00 KiB"},
		{3 << 20, "3.00 MiB"}, {5 << 30, "5.00 GiB"},
	} {
		if got := bytesText(c.n); got != c.want {
			t.Errorf("bytesText(%d) = %q, want %q", c.n, got, c.want)
		}
	}
	for _, c := range []struct {
		d    Device
		want string
	}{{AutoDevice, "auto"}, {CPU, "cpu"}, {Metal, "metal"}, {Device(7), "Device(7)"}} {
		if got := c.d.String(); got != c.want {
			t.Errorf("Device(%d).String() = %q, want %q", c.d, got, c.want)
		}
	}
	for _, c := range []struct {
		p    Precision
		want string
	}{{AutoPrecision, "auto"}, {F16, "f16"}, {Int8, "int8"},
		{Precision(7), "Precision(7)"}} {
		if got := c.p.String(); got != c.want {
			t.Errorf("Precision(%d).String() = %q, want %q", c.p, got, c.want)
		}
	}
	for _, c := range []struct {
		k    EventKind
		want string
	}{{TextDelta, "text_delta"}, {ThinkingDelta, "thinking_delta"},
		{ToolArgsDelta, "tool_args_delta"}, {BlockStart, "block_start"},
		{BlockStop, "block_stop"}, {EventKind(9), "unknown"}} {
		if got := c.k.String(); got != c.want {
			t.Errorf("EventKind(%d).String() = %q, want %q", c.k, got, c.want)
		}
	}
	for _, c := range []struct {
		p    Precision
		want weights.Precision
	}{{F16, weights.F16}, {Int8, weights.Int8}, {AutoPrecision, weights.Auto}} {
		if got := toLoader(c.p); got != c.want {
			t.Errorf("toLoader(%v) = %v, want %v", c.p, got, c.want)
		}
		if got := fromLoader(c.want); c.p != AutoPrecision && got != c.p {
			t.Errorf("fromLoader(%v) = %v, want %v", c.want, got, c.p)
		}
	}
	if got := fromLoader(weights.Inherit); got != AutoPrecision {
		t.Errorf("fromLoader(Inherit) = %v, want auto", got)
	}
	if got := deltaKind(chat.BlockToolResult); got != TextDelta {
		t.Errorf("deltaKind(tool_result) = %v, want a text delta", got)
	}
	// The scale-plane suffix is nn's, and the two must agree or a quantized
	// weight binds its quants and not its scales.
	// The plane names are nn's and are used from there rather than respelled.
	// They were duplicated here so this package did not import nn for one
	// string; it imports nn for [nn.Form] now, so the copy lost its reason and
	// the agreement is by construction instead of by assertion.
	if nn.ScaleSuffix != ".scales" || nn.ZeroSuffix != ".zeros" {
		t.Errorf("the plane suffixes are %q and %q", nn.ScaleSuffix, nn.ZeroSuffix)
	}
}

// TestOptionsResolve checks that every option reaches the field it names.
func TestOptionsResolve(t *testing.T) {
	t.Parallel()
	o := defaults()
	if o.device != AutoDevice || o.precision != AutoPrecision || o.context != DefaultContext {
		t.Errorf("defaults() = %+v", o)
	}
	for _, fn := range []Option{WithDevice(Metal), WithPrecision(Int8), WithContext(77)} {
		fn(&o)
	}
	if o.device != Metal || o.precision != Int8 || o.context != 77 {
		t.Errorf("options = %+v", o)
	}
	so := sessionOptions{context: 1, thinking: true}
	tools := []chat.ToolSpec{{Name: "search"}}
	for _, fn := range []SessionOption{WithSessionContext(9), WithThinking(false),
		WithTools(tools...)} {
		fn(&so)
	}
	if so.context != 9 || so.thinking || len(so.tools) != 1 {
		t.Errorf("session options = %+v", so)
	}
}

// TestOpenDeviceResolvesAuto checks the branch a caller who named nothing
// takes: whichever device is there, reported as the one that was opened.
func TestOpenDeviceResolvesAuto(t *testing.T) {
	t.Parallel()
	dev, got, err := openDevice(AutoDevice)
	if err != nil {
		t.Fatalf("openDevice(auto): %v", err)
	}
	defer func() { _ = dev.Close() }()
	if got != CPU && got != Metal {
		t.Errorf("auto resolved to %v", got)
	}
	if got == CPU && dev.Info().Backend != accel.BackendCPU {
		t.Errorf("auto reported cpu and opened %v", dev.Info().Backend)
	}
	if _, _, err := openDevice(Device(42)); err == nil {
		t.Error("Device(42) opened something")
	}
}

// TestThinkingOffPreClosesTheBlock is specs/003-chat-template.md §3: thinking
// off does not omit the block, it pre-closes one, because omitting it leaves
// the model free to open its own.
func TestThinkingOffPreClosesTheBlock(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	on := session(t, m, WithSessionContext(128), WithThinking(true))
	off := session(t, m, WithSessionContext(128), WithThinking(false))
	msg := []chat.Message{{Role: chat.User,
		Blocks: []chat.Block{{Type: chat.BlockText, Text: "hello"}}}}

	lengths := map[string]int{}
	for name, s := range map[string]*Session{"on": on, "off": off} {
		st, err := s.Chat(t.Context(), msg, greedy(1))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		collect(t, st)
		lengths[name] = st.Usage().PromptTokens
	}
	if lengths["off"] <= lengths["on"] {
		t.Errorf("thinking off rendered %d prompt tokens and thinking on rendered %d; "+
			"off adds a pre-closed block rather than removing one",
			lengths["off"], lengths["on"])
	}
}

// TestMaxTokensDefaultsToTheRemainingContext is why context exhaustion is
// unreachable from inside the loop: a stream with no budget takes the one the
// cache can pay.
func TestMaxTokensDefaultsToTheRemainingContext(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(64))
	ids := m.tok.Encode("short", false)
	st, err := s.start(t.Context(), ids, Policy{})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if want := 64 - len(ids); st.maxTokens != want {
		t.Errorf("maxTokens = %d, want %d", st.maxTokens, want)
	}
	st.abandon()
}

// TestPlanRefusesAStepTheGraphCannotHold covers the recording error, which is
// separate from the compile error because model.Record refuses before it
// records anything and an empty graph compiles.
func TestPlanRefusesAStepTheGraphCannotHold(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	for _, c := range []struct{ tokens, capacity int }{
		{0, 64}, {-1, 64}, {8, 0}, {65, 64},
	} {
		if _, err := m.plan(c.tokens, c.capacity, 0, 1, accel.F32); err == nil {
			t.Errorf("plan(T=%d, C=%d) was recorded", c.tokens, c.capacity)
		}
	}
	if n := m.cache.Len(); n != 0 {
		t.Errorf("the refusals left %d plans in the cache", n)
	}
}

// TestEncodeRefusesAControlTokenTheTokenizerDoesNotHold: a renderer that emits
// a marker the vocabulary does not carry would otherwise encode it as
// characters, which is content forging a turn boundary from the other side.
func TestEncodeRefusesAControlTokenTheTokenizerDoesNotHold(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(64))
	_, err := s.m.encode(chat.Prompt{Parts: []chat.Part{
		{Text: "fine"},
		{Control: "<|not_in_this_vocabulary|>"},
	}})
	if err == nil || !strings.Contains(err.Error(), "control token") {
		t.Errorf("error = %v; it should name the control token", err)
	}
}

// TestGainPlaneRefusals covers the two ways a gain can fail to be one.
func TestGainPlaneRefusals(t *testing.T) {
	t.Parallel()
	dir := checkpoint{tie: true}.write(t)
	repo, err := safetensors.OpenRepo(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = repo.Close() }()

	if _, err := gainPlane(repo, model.WeightSpec{Tensor: "not.a.tensor"}); err == nil {
		t.Error("a tensor that is not in the checkpoint was read")
	}
	_, err = gainPlane(repo, model.WeightSpec{Tensor: "model.embed_tokens.weight"})
	if err == nil || !strings.Contains(err.Error(), "one value per feature") {
		t.Errorf("error = %v; a matrix is not a norm gain", err)
	}
	if _, err := planeDigest(repo, "not.a.tensor"); err == nil {
		t.Error("a missing plane was digested")
	}
}

// TestRunRefusesAStepThatDoesNotFit is the same refusal one level up from
// [stepData.fill], where the caller is the decode loop rather than a test.
func TestRunRefusesAStepThatDoesNotFit(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	s := session(t, m, WithSessionContext(64))
	if _, _, err := s.run(1, []int{1}, 64); err == nil {
		t.Error("a step past the capacity ran")
	}
	if _, _, err := s.run(1, []int{synthVocab + 5}, 0); err == nil {
		t.Error("a token outside the vocabulary ran")
	}
}

// TestBindBufferRefusesAViewTheBufferCannotGive guards the one place a name and
// an extent are joined by hand.
func TestBindBufferRefusesAViewTheBufferCannotGive(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	into := map[string]accel.BufferView{}
	v, _ := m.set.Get("embed")
	if err := bindBuffer(into, "embed", v.Data, accel.F16, v.Elements*4); err == nil {
		t.Error("a view longer than its buffer was bound")
	}
	if len(into) != 0 {
		t.Errorf("the failed binding left %d entries", len(into))
	}
}

// TestInt8WeightsRunTheSameLoop is 004-D6 from the engine's side: precision is
// a load-time decision and not a graph one, so the int8 path is the same loop
// with different buffers bound — quants plus a scale plane per weight, and the
// scale plane's name is the one tgo/nn declares.
func TestInt8WeightsRunTheSameLoop(t *testing.T) {
	t.Parallel()
	dir := checkpoint{tie: true}.write(t)
	m := openAt(t, dir, WithPrecision(Int8))
	if got := m.Info().Precision; got != Int8 {
		t.Fatalf("Precision = %v, want int8", got)
	}
	if m.stored("embed") != nn.FormInt8 {
		t.Errorf("the embedding was not stored as int8")
	}
	if _, ok := m.weightBind["embed"+nn.ScaleSuffix]; !ok {
		t.Errorf("the embedding's scale plane is not bound under %q", "embed"+nn.ScaleSuffix)
	}
	// The gains stay f32 whatever the policy: a gain is one value per feature,
	// and its rounding would reach every row it scales.
	if _, ok := m.set.Get("0.attn_norm"); ok {
		t.Error("a norm gain went through the loader; it is uploaded as f32 by this package")
	}
	if m.gains["0.attn_norm"].DType() != accel.F32 {
		t.Errorf("the attention norm gain is %v, want f32", m.gains["0.attn_norm"].DType())
	}

	s := session(t, m, WithSessionContext(64))
	st, err := s.Complete(t.Context(), "quantized", greedy(4))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	collect(t, st)
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if st.Usage().CompletionTokens != 4 {
		t.Errorf("usage = %+v", st.Usage())
	}
}

// TestGainsAreF32BecauseTheGraphDeclaresThemSo is the discrepancy this package
// reports, asserted so that closing it upstream shows up here.
//
// nn.Graph.Gain declares an f32 port and takes no policy;
// specs/001-weights.md §2's pipeline ends at f16 or int8, so weights.Load has
// no f32 output at all; and accel binds by exact dtype. The two shipped
// packages cannot be composed without a widening step, and this package is the
// first place they meet.
func TestGainsAreF32BecauseTheGraphDeclaresThemSo(t *testing.T) {
	t.Parallel()
	m := openSynthetic(t)
	for _, s := range m.builder.Weights() {
		if s.Kind != model.KindGain {
			continue
		}
		buf, ok := m.gains[s.Port]
		if !ok {
			t.Errorf("%s was not uploaded as a gain", s.Port)
			continue
		}
		if buf.DType() != accel.F32 {
			t.Errorf("%s is %v; the graph declares a gain f32", s.Port, buf.DType())
		}
		if _, through := m.set.Get(s.Port); through {
			t.Errorf("%s also went through the loader, which would bind an f16 view to "+
				"an f32 port", s.Port)
		}
	}
	// And the plan agrees, which is the assertion that would fail if nn ever
	// changed its mind.
	p, err := m.plan(1, 64, 0, 1, accel.F32)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	seen := 0
	for _, port := range p.Ports() {
		if !strings.HasSuffix(port.Name, "norm") && port.Name != "final_norm" {
			continue
		}
		seen++
		if port.DType != accel.F32 {
			t.Errorf("the %s port is declared %v", port.Name, port.DType)
		}
	}
	if seen == 0 {
		t.Error("the decode plan declares no norm gain ports")
	}
}

// TestCacheBytesCarriesTheWidth is specs/005-kv-cache.md §3's arithmetic on the
// library side, against the same worked example cmd/tgo's TestKVCacheArithmetic
// uses: a Qwen3-4B shape, L=36, H_kv=8, d_h=128, is 2·36·8·128 = 73728 elements
// per position — 288 KB in f32 and 144 KB in f16.
//
// The width was a `const f32 = 4` here until 2026-08-27, so under
// --prefix-cache process, where the shared pool is f16, Info.CacheBytesPerSession
// and the startup cost print both reported twice what the cache costs. The
// command line already had the width and priced it correctly, which is what
// made the two disagree without either being obviously wrong.
func TestCacheBytesCarriesTheWidth(t *testing.T) {
	c := &model.Config{NumLayers: 36, NumKVHeads: 8, HeadDim: 128}
	for _, tc := range []struct {
		dt   accel.DType
		want int64
	}{
		{accel.F32, 288 * 1024},
		{accel.F16, 144 * 1024},
	} {
		if got := cacheBytes(c, 1, tc.dt); got != tc.want {
			t.Errorf("cacheBytes at %v = %d, want %d", tc.dt, got, tc.want)
		}
	}

	// The scope is what picks the width, because it is what picks the pool: a
	// shared pool is f16 (blocks.go) and a session's own contiguous cache is
	// f32. Getting this backwards is how the number went wrong.
	for scope, want := range map[CacheScope]accel.DType{
		CacheOff:     accel.F32,
		CacheSession: accel.F32,
		CacheProcess: accel.F16,
	} {
		if got := cacheDType(scope); got != want {
			t.Errorf("cacheDType(%v) = %v, want %v", scope, got, want)
		}
	}

	// And the two halves agree: a process-scoped model reports half what an
	// unscoped one does for the same context.
	shared := cacheBytes(c, 4096, cacheDType(CacheProcess))
	own := cacheBytes(c, 4096, cacheDType(CacheOff))
	if shared*2 != own {
		t.Errorf("process scope reports %d and no scope reports %d; the pool is "+
			"half the width, so the first must be half the second", shared, own)
	}
}

// TestTheCacheIsPricedOverTheLayersThatCache is [023-D3] in the one place a
// wrong answer costs admissions: three layers in four of a hybrid write no key
// or value, and a block priced over all sixty-four is wrong by 4x in the
// direction that refuses requests a device has room for.
func TestTheCacheIsPricedOverTheLayersThatCache(t *testing.T) {
	t.Parallel()
	const context = 1024
	dense := &model.Config{NumLayers: 64, NumKVHeads: 4, HeadDim: 256}
	hybrid := &model.Config{
		NumLayers: 64, NumKVHeads: 4, HeadDim: 256,
		LayerTypes: func() model.LayerSchedule {
			s := make(model.LayerSchedule, 64)
			for i := range s {
				if i%4 == 3 {
					s[i] = model.LayerFullAttention
					continue
				}
				s[i] = model.LayerGatedDelta
			}
			return s
		}(),
	}
	d := cacheBytes(dense, context, accel.F16)
	h := cacheBytes(hybrid, context, accel.F16)
	if want := d / 4; h != want {
		t.Errorf("a hybrid's cache is %d bytes and a dense stack's is %d; one layer "+
			"in four caches, so it is a quarter (%d)", h, d, want)
	}
	// And the dense answer is unchanged, so the arithmetic did not move for
	// every model in the registry.
	if want := int64(2*64*context*4*256) * 2; d != want {
		t.Errorf("a dense stack's cache is %d bytes, want %d", d, want)
	}
}
