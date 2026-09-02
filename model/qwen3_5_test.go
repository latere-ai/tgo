// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/nn"
)

// newTestBuilder is a builder on the CPU backend, for a test that only needs a
// graph to record a refusal onto.
func newTestBuilder(t *testing.T) *tensor.Builder {
	t.Helper()
	dev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open CPU: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Errorf("device close: %v", err)
		}
	})
	rt, err := tensor.NewRuntime(dev)
	if err != nil {
		t.Fatalf("runtime: %v", err)
	}
	return rt.NewBuilder("qwen3_5")
}

// specs/024-qwen3-5-architecture.md sub-scope B, against the checkpoint's own
// config.json.
//
// The file in testdata is `Qwen/Qwen3.5-27B`'s, verbatim, 4 KiB of it. It is
// not a weight, so [000 D8](../specs/000-decisions.md) is untouched — and it is
// what makes this parser checkable against the thing it parses rather than
// against a fixture written from the same misreading. Four of §4.5's draft rows
// were wrong; a test written from the draft would have agreed with all four.

// realConfig is the 27B's config.json.
func realConfig(t *testing.T) json.RawMessage {
	t.Helper()
	b, err := os.ReadFile("testdata/qwen3_5-27b-config.json")
	if err != nil {
		t.Fatalf("reading the checkpoint's config: %v", err)
	}
	return b
}

// TestTheRegistryKnowsQwen35 is [004-D2](../specs/004-model-graph.md) working:
// an architecture the registry does not know is refused with the list, and this
// one is now on it.
func TestTheRegistryKnowsQwen35(t *testing.T) {
	names := Architectures()
	var found bool
	for _, n := range names {
		if n == Qwen35Architecture {
			found = true
		}
	}
	if !found {
		t.Fatalf("Architectures() = %v, and %q is not on it", names, Qwen35Architecture)
	}
	// The MoE sibling is a different key, so it is refused with the list
	// rather than mis-built by this entry (§2.5). They share every
	// linear-attention field and differ in the MLP, which is the shape a
	// shared key would have hidden.
	moe := strings.Replace(string(realConfig(t)), Qwen35Architecture,
		"Qwen3_5MoeForConditionalGeneration", 1)
	_, err := New([]byte(moe))
	if err == nil {
		t.Fatal("the MoE sibling was accepted by the qwen3_5 entry")
	}
	if !strings.Contains(err.Error(), "unknown architecture") {
		t.Errorf("the refusal %q does not name it as unknown", err)
	}
}

// TestTheCheckpointsConfigParses is §2, and every number is the file's.
func TestTheCheckpointsConfigParses(t *testing.T) {
	b, err := New(realConfig(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	c := b.Config()

	for _, x := range []struct {
		name      string
		got, want int
	}{
		{"layers", c.NumLayers, 64},
		{"hidden size", c.HiddenSize, 5120},
		{"heads", c.NumHeads, 24},
		{"kv heads", c.NumKVHeads, 4},
		{"head dim", c.HeadDim, 256},
		{"intermediate", c.IntermediateSize, 17408},
		{"vocabulary", c.VocabSize, 248320},
		{"trained context", c.MaxPositionEmbeddings, 262144},
	} {
		if x.got != x.want {
			t.Errorf("%s = %d, want %d", x.name, x.got, x.want)
		}
	}
	// Read from rope_parameters and not from a top-level key (§2.3).
	if c.RoPETheta != 1e7 {
		t.Errorf("rope theta = %v, want 1e7 from rope_parameters", c.RoPETheta)
	}
	if c.RMSNormEps != 1e-6 {
		t.Errorf("rms norm eps = %v", c.RMSNormEps)
	}
	// Stated at the **top** level, and the file ships lm_head.weight.
	if c.TieWordEmbeddings {
		t.Error("tie_word_embeddings is false in the file")
	}

	q := b.(*qwen35).Qwen35()
	if !q.OutputGate {
		t.Error("attn_output_gate is true in the file")
	}
	if q.RotaryFactor != 0.25 || q.RotaryDim() != 64 {
		t.Errorf("partial rotary factor %v gives a width of %d, want 0.25 and 64",
			q.RotaryFactor, q.RotaryDim())
	}
	// §2.4: the sections partition the rotary pairs, which is the whole of why
	// mRoPE reduces to ordinary RoPE for text.
	sum := 0
	for _, v := range q.MRoPESection {
		sum += v
	}
	if sum != q.RotaryDim()/2 {
		t.Errorf("mrope_section %v sums to %d, want %d pairs", q.MRoPESection, sum,
			q.RotaryDim()/2)
	}
	if q.ImageToken != 248056 || q.VideoToken != 248057 {
		t.Errorf("the multimodal token ids are %d and %d", q.ImageToken, q.VideoToken)
	}
	if q.MTPLayers != 1 {
		t.Errorf("mtp_num_hidden_layers = %d, want 1", q.MTPLayers)
	}
	// Qwen3's renderer until a qwen3_5 template is read from a checkpoint,
	// which is specs/003-chat-template.md's rather than this spec's.
	if b.Template() == nil {
		t.Error("the builder has no chat renderer")
	}
}

// TestTheCheckpointsScheduleAgreesWithItsInterval is §3.1: the file carries
// both fields and they say the same thing, so the agree branch of §3.2's rule
// runs on every load rather than being a guard against a shape nobody has.
func TestTheCheckpointsScheduleAgreesWithItsInterval(t *testing.T) {
	b, err := New(realConfig(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	s := b.Config().LayerTypes
	if len(s) != 64 {
		t.Fatalf("the schedule is %d layers, want 64", len(s))
	}
	if got, want := s.Count(LayerFullAttention), 16; got != want {
		t.Errorf("%d full-attention layers, want %d", got, want)
	}
	if got, want := s.Count(LayerGatedDelta), 48; got != want {
		t.Errorf("%d gated-delta layers, want %d", got, want)
	}
	// (ℓ+1) mod 4 == 0: the full layers are 3, 7, …, 63 and the **last** layer
	// is full. The opposite convention is equally plausible from the field name
	// and produces a model with the same shapes and different output, which is
	// why 024-D1 reads it from the file (§3.1).
	for i, k := range s {
		want := LayerGatedDelta
		if (i+1)%4 == 0 {
			want = LayerFullAttention
		}
		if k != want {
			t.Fatalf("layer %d is %v, want %v", i, k, want)
		}
	}
	if s[0] == LayerFullAttention || s[63] != LayerFullAttention {
		t.Error("layer 0 is full or layer 63 is not; the convention is the other one")
	}
}

// TestTheCheckpointsRecurrentGeometry is §2.2's folding and §4.5's correction to
// it: the state folds three value heads into one band and the output norm does
// not, so H_v is carried rather than derived.
func TestTheCheckpointsRecurrentGeometry(t *testing.T) {
	b, err := New(realConfig(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	r := b.Config().Recurrent
	if r == nil {
		t.Fatal("a hybrid parsed no recurrent geometry")
	}
	for _, x := range []struct {
		name      string
		got, want int
	}{
		{"key heads", r.Heads, 16},
		{"key dim", r.KeyDim, 128},
		{"value heads", r.ValueHeads, 48},
		// 48/16 value heads per key head, at 128 each.
		{"folded value dim", r.ValueDim, 384},
		{"taps", r.Taps, 4},
		// 2·W_k + W_v = 2·2048 + 6144, which is conv1d.weight's [10240, 1, 4]
		// and the number 023 §6's table assumed.
		{"conv width", r.ConvWidth, 10240},
		{"value width", r.ValueWidth(), 6144},
	} {
		if x.got != x.want {
			t.Errorf("%s = %d, want %d", x.name, x.got, x.want)
		}
	}
	// The output norm is one gain per value head, and the folded width would
	// scale three of them by one head's gain.
	if got, want := r.ValueWidth()/r.ValueHeads, 128; got != want {
		t.Errorf("the output norm is [%d], want [%d] — the file's is [128]", got, want)
	}
}

// TestTheCheckpointsWeightMap is §4.5, and every shape is the header's.
func TestTheCheckpointsWeightMap(t *testing.T) {
	b, err := New(realConfig(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	by := map[string][]int{}
	for _, s := range b.Weights() {
		by[s.Tensor] = s.Shape
	}
	for _, c := range []struct {
		tensor string
		shape  []int
	}{
		// Layer 0 is gated-delta, layer 3 is full attention.
		{"model.language_model.layers.0.linear_attn.in_proj_qkv.weight", []int{10240, 5120}},
		{"model.language_model.layers.0.linear_attn.in_proj_z.weight", []int{6144, 5120}},
		{"model.language_model.layers.0.linear_attn.in_proj_b.weight", []int{48, 5120}},
		{"model.language_model.layers.0.linear_attn.in_proj_a.weight", []int{48, 5120}},
		{"model.language_model.layers.0.linear_attn.conv1d.weight", []int{10240, 1, 4}},
		{"model.language_model.layers.0.linear_attn.dt_bias", []int{48}},
		{"model.language_model.layers.0.linear_attn.A_log", []int{48}},
		{"model.language_model.layers.0.linear_attn.norm.weight", []int{128}},
		{"model.language_model.layers.0.linear_attn.out_proj.weight", []int{5120, 6144}},
		// The gate is the second half of q_proj: 2·24·256.
		{"model.language_model.layers.3.self_attn.q_proj.weight", []int{12288, 5120}},
		{"model.language_model.layers.3.self_attn.k_proj.weight", []int{1024, 5120}},
		{"model.language_model.layers.3.self_attn.v_proj.weight", []int{1024, 5120}},
		{"model.language_model.layers.3.self_attn.o_proj.weight", []int{5120, 6144}},
		{"model.language_model.layers.3.self_attn.q_norm.weight", []int{256}},
		{"model.language_model.embed_tokens.weight", []int{248320, 5120}},
		{"lm_head.weight", []int{248320, 5120}},
	} {
		got, ok := by[c.tensor]
		if !ok {
			t.Errorf("the map does not name %s", c.tensor)
			continue
		}
		if len(got) != len(c.shape) {
			t.Errorf("%s is %v, want %v", c.tensor, got, c.shape)
			continue
		}
		for i := range got {
			if got[i] != c.shape[i] {
				t.Errorf("%s is %v, want %v", c.tensor, got, c.shape)
				break
			}
		}
	}
	// The two readings the draft could not choose between, asserted as the
	// negative: neither shipped.
	for _, absent := range []string{
		"model.language_model.layers.3.self_attn.gate_proj.weight",
		"model.language_model.layers.0.linear_attn.in_proj_qkvz.weight",
		"model.language_model.layers.0.linear_attn.in_proj_ba.weight",
		"model.layers.0.linear_attn.in_proj_qkv.weight",
	} {
		if _, ok := by[absent]; ok {
			t.Errorf("the map names %s, which the checkpoint does not carry", absent)
		}
	}
	// A gated-delta layer has no key/value projection, and a full one has no
	// recurrence. A map that emitted both per layer would ask for 48 layers of
	// tensors the file lacks.
	if _, ok := by["model.language_model.layers.0.self_attn.q_proj.weight"]; ok {
		t.Error("a gated-delta layer was given a q projection")
	}
	if _, ok := by["model.language_model.layers.3.linear_attn.in_proj_qkv.weight"]; ok {
		t.Error("a full-attention layer was given a recurrence")
	}
}

// TestTheTwoTowersAreNamedAndIgnored is 024-D13. Check refuses a tensor the map
// does not name, and the checkpoint holds 348 of them.
func TestTheTwoTowersAreNamedAndIgnored(t *testing.T) {
	for _, n := range []string{
		"model.visual.blocks.0.attn.qkv.weight",
		"model.visual.patch_embed.proj.weight",
		"mtp.fc.weight",
		"mtp.layers.0.self_attn.q_proj.weight",
	} {
		if !Qwen35Ignored(n) {
			t.Errorf("%s is not in the ignore set, so Check would refuse the "+
				"checkpoint over a weight this graph does not read", n)
		}
	}
	// And the set is the two towers rather than everything unrecognised: a
	// silent drop is the failure §7's general row is about.
	for _, n := range []string{
		"model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
		"lm_head.weight",
		"model.language_model.norm.weight",
	} {
		if Qwen35Ignored(n) {
			t.Errorf("%s is in the ignore set and it is a weight this graph reads", n)
		}
	}
}

// TestQwen35ForwardIsRefusedByName is [000 D1](../specs/000-decisions.md)'s
// output. tgo knows this architecture, has read its config and its weight map,
// and states the one operator it needs -- rather than not knowing it, or
// quietly building something else.
func TestQwen35ForwardIsRefusedByName(t *testing.T) {
	b, err := New(realConfig(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	g := &nn.Graph{B: newTestBuilder(t)}
	out := b.Forward(g, Inputs{})
	if out != nil {
		t.Fatal("a qwen3_5 forward pass recorded something")
	}
	err = g.Err()
	if err == nil {
		t.Fatal("the refusal left no reason on the graph")
	}
	for _, want := range []string{"accel#27", "48", "gate is one per token"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal %q does not say %q", err, want)
		}
	}
}

// synthQwen35 is §8's fixture as JSON: 4 layers at I=4, so three linear and one
// full, with no two dimensions equal where the confusion would be silent.
func synthQwen35(edit func(top, text map[string]any)) json.RawMessage {
	rope := map[string]any{
		"rope_type": "default", "rope_theta": 1e7,
		// 0.75 x 8 = 6 channels, three pairs, so the sections sum to 3.
		"partial_rotary_factor": 0.75, "mrope_interleaved": true,
		"mrope_section": []any{1, 1, 1},
	}
	text := map[string]any{
		"num_hidden_layers": 4, "full_attention_interval": 4,
		"layer_types": []any{"linear_attention", "linear_attention",
			"linear_attention", "full_attention"},
		"hidden_size": 64, "intermediate_size": 176, "num_attention_heads": 4,
		"num_key_value_heads": 2, "head_dim": 8, "attn_output_gate": true,
		"linear_conv_kernel_dim": 3, "linear_num_key_heads": 2,
		"linear_key_head_dim": 6, "linear_num_value_heads": 6,
		"linear_value_head_dim": 4, "mamba_ssm_dtype": "float32",
		"rms_norm_eps": 1e-6, "vocab_size": 128, "max_position_embeddings": 4096,
		"mlp_only_layers": []any{}, "mtp_num_hidden_layers": 1,
		"rope_parameters": rope,
	}
	top := map[string]any{
		"architectures": []any{Qwen35Architecture}, "model_type": "qwen3_5",
		"tie_word_embeddings": false, "image_token_id": 100, "video_token_id": 101,
		"vision_start_token_id": 102, "vision_end_token_id": 103,
		"text_config": text, "vision_config": map[string]any{"depth": 2},
	}
	if edit != nil {
		edit(top, text)
	}
	b, err := json.Marshal(top)
	if err != nil {
		panic(err)
	}
	return b
}

// TestQwen35RefusesWhatItCannotHonour is §7's table. Each row names the field,
// which is [004 §7](../specs/004-model-graph.md)'s precedent: refuse at parse,
// name the field, and refuse anything the graph cannot honour rather than
// approximating it.
func TestQwen35RefusesWhatItCannotHonour(t *testing.T) {
	// The fixture itself must be accepted, or every row below passes for the
	// wrong reason.
	if _, err := New(synthQwen35(nil)); err != nil {
		t.Fatalf("the fixture was refused: %v", err)
	}
	for _, c := range []struct {
		name string
		edit func(top, text map[string]any)
		want string
	}{
		{"a top-level key this graph does not implement", func(top, _ map[string]any) {
			top["quantization_config"] = map[string]any{"bits": 4}
		}, "quantization_config"},
		{"a text_config key this graph does not implement", func(_, text map[string]any) {
			text["num_experts"] = 512
		}, "num_experts"},
		{"no text_config at all", func(top, _ map[string]any) {
			delete(top, "text_config")
		}, "no text_config"},
		{"no rope_parameters", func(_, text map[string]any) {
			delete(text, "rope_parameters")
		}, "no rope_parameters"},
		{"a schedule that disagrees with the interval", func(_, text map[string]any) {
			text["layer_types"] = []any{"full_attention", "linear_attention",
				"linear_attention", "linear_attention"}
		}, "refused rather than reconciled"},
		{"neither field", func(_, text map[string]any) {
			delete(text, "layer_types")
			delete(text, "full_attention_interval")
		}, "nothing else says which"},
		{"a schedule of the wrong length", func(_, text map[string]any) {
			text["layer_types"] = []any{"linear_attention", "full_attention"}
		}, "names 2 layers and num_hidden_layers is 4"},
		{"a third layer type", func(_, text map[string]any) {
			text["layer_types"] = []any{"linear_attention", "linear_attention",
				"sliding_attention", "full_attention"}
		}, `layer_types[2] is "sliding_attention"`},
		{"heads that do not group", func(_, text map[string]any) {
			text["linear_num_value_heads"] = 5
		}, "the heads do not group"},
		{"a state precision that is not there", func(_, text map[string]any) {
			text["mamba_ssm_dtype"] = "bfloat16"
		}, "asks for a precision that is not there"},
		{"a third layer kind by another name", func(_, text map[string]any) {
			text["mlp_only_layers"] = []any{1}
		}, "mlp_only_layers names 1 layer"},
		{"a rotary scaling this graph does not implement", func(_, text map[string]any) {
			text["rope_parameters"].(map[string]any)["rope_type"] = "yarn"
		}, "implements no rotary scaling"},
		{"a rotary width that is not pairs", func(_, text map[string]any) {
			text["rope_parameters"].(map[string]any)["partial_rotary_factor"] = 0.125
		}, "rotates pairs"},
		{"sections that do not partition the pairs", func(_, text map[string]any) {
			text["rope_parameters"].(map[string]any)["mrope_section"] = []any{1, 1, 2}
		}, "the sections partition the pairs"},
		{"a field at the wrong level", func(top, text map[string]any) {
			delete(text, "hidden_size")
			top["hidden_size"] = 64
		}, "hidden_size"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := New(synthQwen35(c.edit))
			if err == nil {
				t.Fatal("a config this graph cannot honour was accepted")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("the refusal %q does not say %q", err, c.want)
			}
		})
	}
}

// TestQwen35DerivesTheScheduleFromTheIntervalAlone is §3.2's other branch: a
// checkpoint with only one of the two fields is built from it rather than
// refused, and the convention is the one 024-D1 records.
func TestQwen35DerivesTheScheduleFromTheIntervalAlone(t *testing.T) {
	b, err := New(synthQwen35(func(_, text map[string]any) {
		delete(text, "layer_types")
	}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	s := b.Config().LayerTypes
	want := LayerSchedule{LayerGatedDelta, LayerGatedDelta, LayerGatedDelta,
		LayerFullAttention}
	for i := range want {
		if s[i] != want[i] {
			t.Fatalf("the derived schedule is %v, want %v", s, want)
		}
	}
	// And the list alone, with no interval to cross-check it.
	b, err = New(synthQwen35(func(_, text map[string]any) {
		delete(text, "full_attention_interval")
	}))
	if err != nil {
		t.Fatalf("New with the list alone: %v", err)
	}
	if got := b.Config().LayerTypes.Count(LayerFullAttention); got != 1 {
		t.Errorf("the named schedule has %d full layers, want 1", got)
	}
}
