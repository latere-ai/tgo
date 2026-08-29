// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/nn"
)

// specs/024-qwen3-5-architecture.md, sub-scope B: the config, the schedule, the
// refusals, the weight map and the registry entry.
//
// What this entry does **not** do is run. The gated delta block needs a gate
// with a head axis and accel's is per token
// ([C27](../specs/010-conformance.md), accel#27), so a forward pass is refused
// by name. That is [000 D1](../specs/000-decisions.md)'s output: tgo knows this
// architecture and says what it cannot do yet, rather than not knowing it or
// quietly building something else.
//
// Every field and every tensor name below was read from `Qwen/Qwen3.5-27B` on
// 2026-08-29, over HTTP and without downloading a weight (024-D14). Four of the
// draft's readings were wrong, and both alternatives it named as "what to
// expect if these are wrong" are what shipped.

// Qwen35Architecture is the architectures[0] value a qwen3_5 checkpoint carries.
//
// The MoE sibling is `Qwen3_5MoeForConditionalGeneration`, a different key, so
// it is refused by the registry with the list of what tgo knows rather than
// mis-built by this entry (§2.5). They share every linear-attention field and
// differ in the MLP, which is the shape a shared key would have hidden.
const Qwen35Architecture = "Qwen3_5ForConditionalGeneration"

// The checkpoint tensor names outside the layer stack.
//
// A multimodal checkpoint nests the text tower under `language_model` beside
// `visual`, so nothing here is `model.` alone (§4.5).
const (
	qwen35Embed     = "model.language_model.embed_tokens.weight"
	qwen35FinalNorm = "model.language_model.norm.weight"
	qwen35LMHead    = "lm_head.weight"

	// qwen35LayerPrefix is §4's "model.layers.ℓ." for a nested text tower.
	qwen35LayerPrefix = "model.language_model.layers.%d."

	// qwen35VisionPrefix and qwen35MTPPrefix are the two towers this graph
	// does not read: 333 tensors and 15 (024-D13).
	qwen35VisionPrefix = "model.visual."
	qwen35MTPPrefix    = "mtp."
)

func init() { Register(Qwen35Architecture, newQwen35) }

// qwen35 is the qwen3_5 builder.
type qwen35 struct{ cfg *qwen35Config }

// qwen35Config is [Config] plus the fields this architecture adds.
//
// A separate type rather than a wider [Config], which is 004 §5's table shared
// by every architecture. `Builder.Config` already promises this: *"Fields a
// specific architecture adds beyond §5's table are reachable through the
// concrete builder type."*
type qwen35Config struct {
	*Config

	// Interval is `full_attention_interval`, and Types is `layer_types`
	// verbatim. Both are present in the 27B and §3.2 refuses a checkpoint that
	// carries one saying something the other does not.
	Interval int
	Types    []string

	// OutputGate is `attn_output_gate`. There is no gate tensor: it is the
	// second half of q_proj's output, whose width says so (§4.5).
	OutputGate bool

	// RotaryFactor is `rope_parameters.partial_rotary_factor`, ρ, so the
	// rotary width is ρ·d_h and not d_h.
	RotaryFactor float64

	// MRoPESection is `rope_parameters.mrope_section`. It is read and checked
	// rather than used: for text every position component is equal, so mRoPE
	// reduces to ordinary RoPE exactly, and checking that the sections
	// partition the rotary pairs is what keeps the reduction an argument
	// rather than a hope (024-D12).
	MRoPESection []int

	// Vision and MTP are what makes this checkpoint multimodal and
	// speculative. Neither is read by this graph, and both are refused rather
	// than ignored (024-D13, and 026 owns the first).
	ImageToken, VideoToken           int
	VisionStartToken, VisionEndToken int
	MTPLayers                        int
}

// qwen35Raw is the top level of a qwen3_5 config.json.
//
// Everything the graph needs is one level down, in `text_config`. ParseConfig
// reads the top level, so a parser written from 018 §1's quote reads **zero**
// for hidden_size and every other width -- and a config of zeros is not a
// refusal, it is a model with no width (§2.1).
type qwen35Raw struct {
	Architectures   []string        `json:"architectures"`
	ModelType       string          `json:"model_type"`
	TieEmbeddings   bool            `json:"tie_word_embeddings"`
	ImageToken      int             `json:"image_token_id"`
	VideoToken      int             `json:"video_token_id"`
	VisionStart     int             `json:"vision_start_token_id"`
	VisionEnd       int             `json:"vision_end_token_id"`
	Text            json.RawMessage `json:"text_config"`
	Vision          json.RawMessage `json:"vision_config"`
	TransformersVer string          `json:"transformers_version"`
}

// qwen35Text is `text_config`.
type qwen35Text struct {
	NumLayers        int         `json:"num_hidden_layers"`
	LayerTypes       []string    `json:"layer_types"`
	Interval         int         `json:"full_attention_interval"`
	HiddenSize       int         `json:"hidden_size"`
	IntermediateSize int         `json:"intermediate_size"`
	NumHeads         int         `json:"num_attention_heads"`
	NumKVHeads       int         `json:"num_key_value_heads"`
	HeadDim          int         `json:"head_dim"`
	OutputGate       bool        `json:"attn_output_gate"`
	ConvKernel       int         `json:"linear_conv_kernel_dim"`
	KeyHeads         int         `json:"linear_num_key_heads"`
	KeyHeadDim       int         `json:"linear_key_head_dim"`
	ValueHeads       int         `json:"linear_num_value_heads"`
	ValueHeadDim     int         `json:"linear_value_head_dim"`
	MambaDType       string      `json:"mamba_ssm_dtype"`
	RMSNormEps       float32     `json:"rms_norm_eps"`
	VocabSize        int         `json:"vocab_size"`
	MaxPositions     int         `json:"max_position_embeddings"`
	MLPOnlyLayers    []int       `json:"mlp_only_layers"`
	MTPLayers        int         `json:"mtp_num_hidden_layers"`
	Rope             *qwen35Rope `json:"rope_parameters"`
}

// qwen35Rope is `text_config.rope_parameters`, which carries both the rotary
// base and the partial factor -- neither where ParseConfig looks (§2.3).
type qwen35Rope struct {
	RopeType        string  `json:"rope_type"`
	RopeTheta       float32 `json:"rope_theta"`
	PartialRotary   float64 `json:"partial_rotary_factor"`
	MRoPEInterleave bool    `json:"mrope_interleaved"`
	MRoPESection    []int   `json:"mrope_section"`
}

// qwen35TopKeys and qwen35TextKeys are §7's allow lists, one per level.
//
// rawConfig ignores an unknown JSON key, which is right for a dense model whose
// extra fields are metadata and wrong here: a qwen3_5 config carries fields that
// change the arithmetic, and a field tgo silently ignores is a model tgo
// silently gets wrong. So an unknown key is a refusal that names it -- and a
// field at the *wrong level* is a named refusal rather than a zero, which §2.1
// shows the cost of.
var qwen35TopKeys = map[string]bool{
	"architectures": true, "model_type": true, "tie_word_embeddings": true,
	"image_token_id": true, "video_token_id": true,
	"vision_start_token_id": true, "vision_end_token_id": true,
	"text_config": true, "vision_config": true, "transformers_version": true,
	"dtype": true, "torch_dtype": true,
}

var qwen35TextKeys = map[string]bool{
	"num_hidden_layers": true, "layer_types": true, "full_attention_interval": true,
	"hidden_size": true, "intermediate_size": true, "num_attention_heads": true,
	"num_key_value_heads": true, "head_dim": true, "attn_output_gate": true,
	"linear_conv_kernel_dim": true, "linear_num_key_heads": true,
	"linear_key_head_dim": true, "linear_num_value_heads": true,
	"linear_value_head_dim": true, "mamba_ssm_dtype": true, "rms_norm_eps": true,
	"vocab_size": true, "max_position_embeddings": true, "mlp_only_layers": true,
	"mtp_num_hidden_layers": true, "mtp_use_dedicated_embeddings": true,
	"rope_parameters": true,
	// Metadata the graph reads and checks, or does not read at all.
	"model_type": true, "hidden_act": true, "dtype": true, "torch_dtype": true,
	"attention_bias": true, "attention_dropout": true, "initializer_range": true,
	"use_cache": true, "eos_token_id": true, "bos_token_id": true,
	"pad_token_id": true, "tie_word_embeddings": true,
}

// newQwen35 is the registry constructor.
func newQwen35(raw json.RawMessage) (Builder, error) {
	c, err := parseQwen35Config(raw)
	if err != nil {
		return nil, err
	}
	return &qwen35{cfg: c}, nil
}

// Config is 004 §5's table, filled from `text_config`.
func (m *qwen35) Config() *Config { return m.cfg.Config }

// Qwen35 is the architecture's own fields, which [Builder.Config] does not
// carry.
func (m *qwen35) Qwen35() *qwen35Config { return m.cfg }

// Template is Qwen3's renderer until a qwen3_5 template is read from a
// checkpoint, which is specs/003-chat-template.md's.
func (m *qwen35) Template() chat.Renderer { return chat.Qwen3() }

// Weights is §4.5's map with the layer templating expanded.
func (m *qwen35) Weights() []WeightSpec { return qwen35Weights(m.cfg) }

// parseQwen35Config reads both levels, refuses what the graph cannot honour,
// and fills 004 §5's table plus this architecture's own fields.
func parseQwen35Config(raw json.RawMessage) (*qwen35Config, error) {
	if err := onlyKeys(raw, qwen35TopKeys, "the config"); err != nil {
		return nil, err
	}
	var top qwen35Raw
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("model: reading a %s config: %w", Qwen35Architecture, err)
	}
	if len(top.Text) == 0 {
		return nil, fmt.Errorf("model: the config has no text_config; every field " +
			"this graph reads is under it, and a config parsed at the top level " +
			"would be zero-width rather than refused " +
			"(specs/024-qwen3-5-architecture.md §2.1)")
	}
	if err := onlyKeys(top.Text, qwen35TextKeys, "text_config"); err != nil {
		return nil, err
	}
	var t qwen35Text
	if err := json.Unmarshal(top.Text, &t); err != nil {
		return nil, fmt.Errorf("model: reading a %s text_config: %w",
			Qwen35Architecture, err)
	}
	if t.Rope == nil {
		return nil, fmt.Errorf("model: text_config has no rope_parameters, which is " +
			"where this architecture states rope_theta and partial_rotary_factor " +
			"(specs/024-qwen3-5-architecture.md §2.3)")
	}

	cfg := &Config{
		Architecture: Qwen35Architecture, HiddenSize: t.HiddenSize,
		NumLayers: t.NumLayers, NumHeads: t.NumHeads, NumKVHeads: t.NumKVHeads,
		HeadDim: t.HeadDim, IntermediateSize: t.IntermediateSize,
		VocabSize: t.VocabSize, RMSNormEps: t.RMSNormEps,
		RoPETheta: t.Rope.RopeTheta, TieWordEmbeddings: top.TieEmbeddings,
		MaxPositionEmbeddings: t.MaxPositions,
	}
	q := &qwen35Config{
		Config: cfg, Interval: t.Interval, Types: t.LayerTypes,
		OutputGate: t.OutputGate, RotaryFactor: t.Rope.PartialRotary,
		MRoPESection: t.Rope.MRoPESection, ImageToken: top.ImageToken,
		VideoToken: top.VideoToken, VisionStartToken: top.VisionStart,
		VisionEndToken: top.VisionEnd, MTPLayers: t.MTPLayers,
	}

	if err := q.check(&t); err != nil {
		return nil, err
	}
	sched, err := qwen35Schedule(t.NumLayers, t.LayerTypes, t.Interval)
	if err != nil {
		return nil, err
	}
	cfg.LayerTypes = sched
	cfg.Recurrent = &Recurrent{
		Heads: t.KeyHeads, KeyDim: t.KeyHeadDim,
		// §2.2's folding: several value heads share a key head and are
		// disjoint row bands of one state, so the state's value width is the
		// whole band. ValueHeads carries H_v because the output norm is one
		// gain per value head and the folded width would broadcast it wrong.
		ValueDim:   t.ValueHeads / t.KeyHeads * t.ValueHeadDim,
		ValueHeads: t.ValueHeads,
		Taps:       t.ConvKernel,
		ConvWidth:  2*t.KeyHeads*t.KeyHeadDim + t.ValueHeads*t.ValueHeadDim,
	}
	if err := cfg.Recurrent.check(); err != nil {
		return nil, err
	}
	return q, nil
}

// check is §7's table, minus the two the schedule owns.
func (q *qwen35Config) check(t *qwen35Text) error {
	c := q.Config
	switch {
	case c.NumLayers < 1 || c.HiddenSize < 1 || c.NumHeads < 1 || c.VocabSize < 1:
		return fmt.Errorf("model: text_config is %d layers, %d hidden, %d heads and "+
			"a vocabulary of %d; a field read as zero is a field at the wrong level",
			c.NumLayers, c.HiddenSize, c.NumHeads, c.VocabSize)
	case t.ValueHeads%t.KeyHeads != 0:
		return fmt.Errorf("model: linear_num_value_heads is %d and "+
			"linear_num_key_heads is %d; the heads do not group, and value heads "+
			"sharing a key head are disjoint row bands of one state "+
			"(specs/024-qwen3-5-architecture.md §4.3)", t.ValueHeads, t.KeyHeads)
	case t.MambaDType != "" && t.MambaDType != "float32":
		return fmt.Errorf("model: mamba_ssm_dtype is %q and accel's recurrent state "+
			"is f32; a config asking for anything else asks for a precision that is "+
			"not there", t.MambaDType)
	case len(t.MLPOnlyLayers) != 0:
		return fmt.Errorf("model: mlp_only_layers names %d layer(s) and this graph "+
			"builds two kinds; a third is a model it does not implement",
			len(t.MLPOnlyLayers))
	case t.Rope.RopeType != "" && t.Rope.RopeType != "default":
		return fmt.Errorf("model: rope_parameters.rope_type is %q and this graph "+
			"implements no rotary scaling; a scaling it ignored would cap the model "+
			"at a context it was not trained for (specs/004-model-graph.md §7)",
			t.Rope.RopeType)
	}
	// ρ·d_h is the rotary width and RoPE rotates pairs, so it is an even
	// integer or it is not a width.
	w := q.RotaryFactor * float64(c.HeadDim)
	rot := int(w)
	if q.RotaryFactor <= 0 || q.RotaryFactor > 1 || float64(rot) != w || rot%2 != 0 {
		return fmt.Errorf("model: partial_rotary_factor %v over a head dim of %d is a "+
			"rotary width of %v; it is an even whole number of channels, because RoPE "+
			"rotates pairs", q.RotaryFactor, c.HeadDim, w)
	}
	// The sections partition the rotary *pairs*, which is what makes the
	// text-only reduction exact rather than approximate (024-D12).
	if n := len(t.Rope.MRoPESection); n > 0 {
		sum := 0
		for _, v := range t.Rope.MRoPESection {
			sum += v
		}
		if sum != rot/2 {
			return fmt.Errorf("model: mrope_section %v sums to %d and the rotary width "+
				"is %d channels, which is %d pairs; the sections partition the pairs, "+
				"and sections that do not are a rotary this graph cannot reduce to the "+
				"text-only case (specs/024-qwen3-5-architecture.md §2.4)",
				t.Rope.MRoPESection, sum, rot, rot/2)
		}
	}
	return nil
}

// RotaryDim is ρ·d_h, the width RoPE rotates.
func (q *qwen35Config) RotaryDim() int { return int(q.RotaryFactor * float64(q.HeadDim)) }

// qwen35Schedule is §3.2's rule.
//
// A checkpoint that says one thing in two places is **refused, not
// reconciled**. [000 D6](../specs/000-decisions.md) is the authority, and a
// schedule guessed from the field a reader happened to prefer is the same
// failure one level in: both schedules build, both run, and only one of them is
// the model.
func qwen35Schedule(layers int, types []string, interval int) (LayerSchedule, error) {
	if len(types) == 0 && interval == 0 {
		return nil, fmt.Errorf("model: neither layer_types nor "+
			"full_attention_interval is present, and nothing else says which of the "+
			"%d layers are which", layers)
	}
	var named LayerSchedule
	if len(types) > 0 {
		if len(types) != layers {
			return nil, fmt.Errorf("model: layer_types names %d layers and "+
				"num_hidden_layers is %d; a list that does not cover the stack cannot "+
				"be read as a repeating pattern either, because a period-%d pattern is "+
				"one full layer in %d (specs/024-qwen3-5-architecture.md §3.1)",
				len(types), layers, len(types), len(types))
		}
		named = make(LayerSchedule, layers)
		for i, s := range types {
			switch s {
			case "full_attention":
				named[i] = LayerFullAttention
			case "linear_attention":
				named[i] = LayerGatedDelta
			default:
				return nil, fmt.Errorf("model: layer_types[%d] is %q, and this graph "+
					"builds full_attention and linear_attention; a third layer type is "+
					"a model it does not implement", i, s)
			}
		}
	}
	if interval == 0 {
		return named, nil
	}
	if interval < 0 {
		return nil, fmt.Errorf("model: full_attention_interval is %d; it is how many "+
			"layers apart the full-attention layers sit", interval)
	}
	// ℓ is full attention iff (ℓ+1) mod I == 0, so the full layers are
	// I-1, 2I-1, … and the last layer is full. The opposite convention --
	// ℓ mod I == 0 -- is equally plausible from the field name alone and
	// produces a model with the same shapes, the same parameter count and
	// different output, which is why it is read from the file rather than
	// chosen (024-D1, confirmed in §3.1).
	derived := make(LayerSchedule, layers)
	for i := range derived {
		derived[i] = LayerGatedDelta
		if (i+1)%interval == 0 {
			derived[i] = LayerFullAttention
		}
	}
	if named == nil {
		return derived, nil
	}
	for i := range derived {
		if named[i] != derived[i] {
			return nil, fmt.Errorf("model: layer_types says layer %d is %v and "+
				"full_attention_interval %d makes it %v; a checkpoint that says one "+
				"thing in two places is refused rather than reconciled, because both "+
				"schedules build and only one is the model "+
				"(specs/024-qwen3-5-architecture.md §3.2)",
				i, named[i], interval, derived[i])
		}
	}
	return named, nil
}

// onlyKeys refuses a JSON object carrying a key that is not on the list.
func onlyKeys(raw json.RawMessage, allow map[string]bool, what string) error {
	var got map[string]json.RawMessage
	if err := json.Unmarshal(raw, &got); err != nil {
		return fmt.Errorf("model: reading %s: %w", what, err)
	}
	var extra []string
	for k := range got {
		if !allow[k] {
			extra = append(extra, k)
		}
	}
	if len(extra) == 0 {
		return nil
	}
	sort.Strings(extra)
	return fmt.Errorf("model: %s carries %s, which this graph does not implement; a "+
		"qwen3_5 config states fields that change the arithmetic and a field tgo "+
		"ignored would be a model tgo silently got wrong "+
		"(specs/024-qwen3-5-architecture.md §7)", what, strings.Join(extra, ", "))
}

// Forward is refused, by name.
//
// specs/024-qwen3-5-architecture.md §11: sub-scope C -- `nn.GatedDelta` and the
// graph -- is **gated on an accel answer that does not exist**. The gated delta
// recurrence gives each value head its own decay and accel's gate is per token
// ([C27](../specs/010-conformance.md), accel#27), so 48 gates in 48 have no
// operand to arrive on.
//
// The two workarounds are both worse than a refusal, and
// [000 D1](../specs/000-decisions.md) names why: a mean over the 48 gates runs,
// produces plausible numbers, and is a different model; and one dispatch per
// head is 48 per layer over 48 layers, which is the private device code this
// project exists not to write.
//
// So this refuses and says what it is waiting for. That is the output 000 D1
// asks for: tgo knows this architecture, has read its config and its weight
// map, and states the one operator it needs -- rather than not knowing it, or
// quietly building something else.
func (m *qwen35) Forward(g *nn.Graph, in Inputs) *tensor.Tensor {
	if g == nil || g.B == nil {
		return nil
	}
	c := m.cfg.Config
	poison(g, fmt.Errorf("model: %s is known and cannot be run yet: its %d "+
		"gated-delta layers give each of %d value heads its own decay, and accel's "+
		"gate is one per token (C27, accel#27). The config, the layer schedule and "+
		"the weight map are built and tested; the forward pass waits on a "+
		"[tokens, heads] gate (specs/024-qwen3-5-architecture.md §4.4)",
		Qwen35Architecture, c.LayerTypes.Count(LayerGatedDelta),
		c.Recurrent.ValueHeads))
	_ = in
	return nil
}
