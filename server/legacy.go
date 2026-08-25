// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"latere.ai/x/pkg/llmdialect/ir"
)

// The legacy completions surface, which llmdialect does not carry.
//
// llmdialect translates the three chat dialects; /v1/completions is not one of
// them, and it is the one route where no template runs: the body's `prompt`
// reaches the model as the bytes the caller wrote (specs/007-engine.md's
// Session.Complete). So tgo owns this codec end to end -- decode, encode and
// SSE -- and owns it as an llmdialect.Frontend, so it travels the same pipeline
// as the other three and its refusals and its loss report come out of the same
// two tables rather than a fourth copy of them.

// dialectLegacy names this surface. It is not one of ir's four, and it is a
// value of the same type so that everything keyed by dialect keeps working.
const dialectLegacy ir.Dialect = "openai-completions"

// legacyKeys are the members this decoder understands. Anything else lands in
// the loss report, which is where the sampling knobs land too: they are
// subtracted back out by [lossReport], exactly as they are on the other three
// surfaces (§4.1).
//
// logprobs is deliberately absent. This encoder answers with logprobs:null on
// every choice, so the member is accepted and not acted on -- which is the
// definition of a loss, and is what the other three surfaces report for it.
var legacyKeys = map[string]bool{
	"model": true, "prompt": true, "max_tokens": true, "temperature": true,
	"top_p": true, "stop": true, "stream": true, "stream_options": true,
	"n": true, "user": true, "suffix": true, "echo": true, "best_of": true,
}

// legacyFrontend is the caller-side codec for /v1/completions.
type legacyFrontend struct{}

func newLegacyFrontend() *legacyFrontend { return &legacyFrontend{} }

// Name returns the dialect name.
func (*legacyFrontend) Name() ir.Dialect { return dialectLegacy }

// DecodeRequest parses a legacy completions body into the IR.
//
// The prompt becomes a single user message so that everything the shared
// pipeline does -- the refusals, the loss report, the policy mapping -- reads
// one shape. [legacyFrontend.finish] then moves it back out to
// [request.prompt], because this route renders no template.
func (*legacyFrontend) DecodeRequest(body []byte) (*ir.Request, error) {
	top, err := topLevel(body)
	if err != nil {
		return nil, err
	}
	req := &ir.Request{}
	for k := range top {
		if !legacyKeys[k] {
			req.Loss.Add(ir.LossRequestFieldOf(k))
		}
	}
	var wire struct {
		Model       string          `json:"model"`
		MaxTokens   *int64          `json:"max_tokens"`
		Temperature *float64        `json:"temperature"`
		TopP        *float64        `json:"top_p"`
		Stop        json.RawMessage `json:"stop"`
		Stream      bool            `json:"stream"`
		User        string          `json:"user"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, fmt.Errorf("malformed completions request: %w", err)
	}
	if wire.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	prompt, err := legacyPrompt(top)
	if err != nil {
		return nil, err
	}
	stop, err := legacyStop(wire.Stop)
	if err != nil {
		return nil, err
	}
	req.Model = wire.Model
	req.MaxTokens = wire.MaxTokens
	req.Temperature = wire.Temperature
	req.TopP = wire.TopP
	req.StopSequences = stop
	req.Stream = wire.Stream
	req.UserID = wire.User
	req.Messages = []ir.Message{{
		Role:   ir.RoleUser,
		Blocks: []ir.Block{{Type: ir.BlockText, Text: prompt}},
	}}
	return req, nil
}

// legacyPrompt reads the prompt, which the wire format allows to be a string
// or an array.
//
// An array of more than one prompt is n>1 wearing another name, and an array of
// token ids is a prompt this server cannot render back to the text
// Session.Complete takes. Both are refused rather than approximated.
func legacyPrompt(top map[string]json.RawMessage) (string, error) {
	raw, ok := top["prompt"]
	if !ok || isNull(raw) {
		return "", fmt.Errorf("prompt is required")
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return "", fmt.Errorf("prompt must be a string or an array of strings: token id " +
			"prompts are not supported")
	}
	if len(many) != 1 {
		return "", fmt.Errorf("prompt must hold exactly one string, got %d: more than one "+
			"completion per request needs batching (specs/008-scheduler.md)", len(many))
	}
	return many[0], nil
}

// legacyStop reads stop, which is a string or an array of them.
func legacyStop(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || isNull(raw) {
		return nil, nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, fmt.Errorf("stop must be a string or an array of strings")
	}
	return many, nil
}

// finish moves the prompt out of the message list and refuses the members that
// would change the answer.
func (*legacyFrontend) finish(top map[string]json.RawMessage, out *request) *apiError {
	if raw, ok := top["suffix"]; ok && !isNull(raw) {
		return refusal("suffix", "tgo: suffix is not supported: filling in the middle needs "+
			"the prompt built around the completion, and this route sends the prompt verbatim")
	}
	if raw, ok := top["echo"]; ok && !isNull(raw) {
		var echo bool
		if json.Unmarshal(raw, &echo) == nil && echo {
			return refusal("echo", "tgo: echo is not supported: the answer would carry the "+
				"prompt back, which this server does not re-tokenize")
		}
	}
	if raw, ok := top["best_of"]; ok && !isNull(raw) {
		var n int
		if json.Unmarshal(raw, &n) == nil && n > 1 {
			return refusal("best_of", "tgo: best_of=%d is not supported: sampling several "+
				"completions and picking one needs batching (specs/008-scheduler.md)", n)
		}
	}
	// One message, one text block: it is what DecodeRequest built.
	if len(out.msgs) == 1 && len(out.msgs[0].Blocks) == 1 {
		out.prompt = out.msgs[0].Blocks[0].Text
	}
	if out.prompt == "" {
		return badRequest("tgo: the prompt is empty, and a completion of nothing is nothing")
	}
	out.msgs = nil
	return nil
}

// EncodeResponse renders an IR response as a legacy completions body.
//
// Every block's text is concatenated, thinking included: this route ran no
// template, so what the model produced is the completion the caller asked for
// and there is no turn structure to separate it from.
func (*legacyFrontend) EncodeResponse(resp *ir.Response) ([]byte, error) {
	var text strings.Builder
	for _, b := range resp.Blocks {
		text.WriteString(b.Text)
	}
	return json.Marshal(map[string]any{
		"id": resp.ID, "object": "text_completion", "created": 0, "model": resp.Model,
		"choices": []map[string]any{{
			"text": text.String(), "index": 0, "logprobs": nil,
			"finish_reason": legacyFinish(resp.StopReason),
		}},
		"usage": legacyUsage(resp.Usage),
	})
}

// legacyFinish maps the IR stop vocabulary onto finish_reason.
func legacyFinish(stop ir.StopReason) string {
	if stop == ir.StopMaxTokens {
		return "length"
	}
	return "stop"
}

func legacyUsage(u ir.Usage) map[string]any {
	return map[string]any{
		"prompt_tokens":     u.InputTokens,
		"completion_tokens": u.OutputTokens,
		"total_tokens":      u.InputTokens + u.OutputTokens,
	}
}

// NewEventEncoder returns an encoder writing text_completion chunks to w.
func (*legacyFrontend) NewEventEncoder(w io.Writer) ir.EventEncoder {
	return &legacyEventEncoder{w: w}
}

// legacyEventEncoder writes the legacy chunk stream: one text_completion
// object per delta, a terminal chunk carrying finish_reason and usage, and
// [DONE].
type legacyEventEncoder struct {
	w     io.Writer
	id    string
	model string
}

// Encode writes the SSE frame for one IR event.
func (e *legacyEventEncoder) Encode(ev ir.Event) error {
	switch ev.Type {
	case ir.EventMessageStart:
		e.id = ev.ID
		e.model = ev.Model
		return nil
	case ir.EventTextDelta, ir.EventThinkingDelta, ir.EventArgsDelta:
		return e.chunk(ev.Delta, nil, nil)
	case ir.EventBlockStart, ir.EventBlockStop, ir.EventSignatureDelta:
		// The legacy format has no block structure: a completion is one run of
		// text, which is the whole difference between this route and the
		// other three.
		return nil
	case ir.EventMessageDelta:
		finish := legacyFinish(ev.StopReason)
		var usage map[string]any
		if ev.Usage != nil {
			usage = legacyUsage(*ev.Usage)
		}
		return e.chunk("", &finish, usage)
	case ir.EventMessageStop:
		_, err := fmt.Fprint(e.w, "data: [DONE]\n\n")
		return err
	default:
		return fmt.Errorf("tgo: unknown event type %q", ev.Type)
	}
}

func (e *legacyEventEncoder) chunk(text string, finish *string, usage map[string]any) error {
	choice := map[string]any{"text": text, "index": 0, "logprobs": nil, "finish_reason": nil}
	if finish != nil {
		choice["finish_reason"] = *finish
	}
	body := map[string]any{
		"id": e.id, "object": "text_completion", "created": 0, "model": e.model,
		"choices": []map[string]any{choice},
	}
	if usage != nil {
		body["usage"] = usage
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(e.w, "data: %s\n\n", raw)
	return err
}
