// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"encoding/json"
	"math"
	"strings"

	"github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/chat"
	"latere.ai/x/pkg/llmdialect/ir"
)

// This file is the whole of the mapping between llmdialect's vocabulary and
// tgo's: an ir.Request becomes messages, a session shape and a
// [github.com/latere-ai/tgo.Policy], and nothing else in the tree imports ir
// for that purpose (009-D10). One file, so the two vocabularies meet once.

// request is one decoded request, checked and mapped.
type request struct {
	dialect ir.Dialect

	// msgs is the conversation, with the system turn first where there is one.
	msgs []chat.Message

	// prompt is the raw text /v1/completions generates from, and is set only
	// on that route, where no template runs.
	prompt string

	spec   SessionSpec
	policy tgo.Policy

	// loss is what X-Tgo-Loss carries, already corrected both ways (loss.go).
	loss []string

	stream bool
}

// adapt maps a decoded ir.Request onto tgo's types.
//
// Everything that can refuse does so here, before a session is allocated: a
// request that will not run must not first take a KV reservation from one that
// would.
func adapt(d ir.Dialect, req *ir.Request, ex extras, raw map[string]bool, eng Engine) (*request, *apiError) {
	msgs, aerr := mapMessages(req)
	if aerr != nil {
		return nil, aerr
	}
	pol, aerr := mapPolicy(req, ex, eng.VocabSize())
	if aerr != nil {
		return nil, aerr
	}
	if aerr := mapSchema(d, req, eng, &pol); aerr != nil {
		return nil, aerr
	}
	return &request{
		dialect: d,
		msgs:    msgs,
		spec: SessionSpec{
			Tools:    mapTools(req.Tools),
			Thinking: !ex.thinkingOff,
			Key:      ex.cacheSalt,
			Recorder: nil, // set by the handler, which owns the metrics
		},
		policy: pol,
		loss:   lossReport(d, req, raw),
		stream: req.Stream,
	}, nil
}

// mapSchema carries a requested JSON schema onto the policy, refusing one the
// grammar compiler will not compile.
//
// The compilation happens here, before a session exists, for the reason the
// whole file is arranged this way: a request that will not run must not first
// take a KV reservation from one that would. It is not wasted work either --
// the model keeps what it compiled, so the request that follows finds it
// (015-D1).
//
// The refusal carries the compiler's own words. 015-D4 makes the reason the
// point: "a numeric bound is arithmetic on the value" tells a caller to move
// the bound into their own validation, while "unsupported" sends them to guess
// which of their keywords was the problem.
func mapSchema(d ir.Dialect, req *ir.Request, eng Engine, pol *tgo.Policy) *apiError {
	if req.Schema == nil {
		return nil
	}
	f := schemaField(d)
	if len(req.Schema.Schema) == 0 {
		return refusal(f, "tgo: %s asks for a JSON schema and carries none", f)
	}
	// Refused here as well as in tgo.Policy, because this is the layer that can
	// answer it: Policy.check runs inside Session.start, after this file has
	// already allocated the session, and its error would arrive as a 500 with
	// no field on it. A stop string cuts the completion where it matched, so it
	// would end a constrained request on half a document (015-D9).
	if len(pol.Stop) > 0 {
		return refusal(f, "tgo: %s asks for a JSON schema and the request also carries a "+
			"stop sequence %q; a stop string cuts the completion where it matched, so it "+
			"would end the document early and it would not parse. Send one or the other",
			f, pol.Stop)
	}
	if err := eng.CheckSchema(req.Schema.Schema); err != nil {
		return refusal(f, "tgo: %s cannot be compiled to a token mask: %v", f, err)
	}
	pol.Schema = req.Schema.Schema
	return nil
}

// schemaField is what one dialect calls the member that asked for a schema.
// A refusal names what the caller sent rather than what the IR called it.
func schemaField(d ir.Dialect) string {
	switch d {
	case ir.DialectAnthropicMessages:
		return "output_format"
	case ir.DialectOpenAIResponses:
		return "text.format"
	default:
		return "response_format"
	}
}

// mapMessages turns the IR conversation into chat turns.
//
// Two shapes differ and both matter. The IR carries the system prompt beside
// the messages while the Qwen3 template reads it from position zero, and the IR
// puts tool results in a user turn while tgo gives them their own Tool role --
// which is what lets the renderer wrap each in <tool_response> without
// inspecting text (specs/003-chat-template.md 003-D8).
func mapMessages(req *ir.Request) ([]chat.Message, *apiError) {
	var out []chat.Message
	if sys := textOf(req.System); sys != "" {
		out = append(out, chat.Message{
			Role:   chat.System,
			Blocks: []chat.Block{{Type: chat.BlockText, Text: sys}},
		})
	}
	for _, m := range req.Messages {
		role := chat.User
		if m.Role == ir.RoleAssistant {
			role = chat.Assistant
		}
		turns, aerr := mapTurn(role, m.Blocks)
		if aerr != nil {
			return nil, aerr
		}
		out = append(out, turns...)
	}
	if len(out) == 0 || out[len(out)-1].Role == chat.System {
		return nil, badRequest("tgo: the request carries no message to answer")
	}
	return out, nil
}

// mapTurn splits one IR message into the turns tgo renders.
//
// A run of tool results becomes Tool turns and the rest stays on the original
// role, in the order the blocks arrived: a caller who put a result between two
// sentences gets it back in that place rather than sorted.
func mapTurn(role chat.Role, blocks []ir.Block) ([]chat.Message, *apiError) {
	var out []chat.Message
	var pending []chat.Block
	flush := func() {
		if len(pending) > 0 {
			out = append(out, chat.Message{Role: role, Blocks: pending})
			pending = nil
		}
	}
	for _, b := range blocks {
		switch b.Type {
		case ir.BlockText:
			pending = append(pending, chat.Block{Type: chat.BlockText, Text: b.Text})
		case ir.BlockThinking:
			if role != chat.Assistant {
				// The renderer refuses it too, and refusing here names the
				// dialect member rather than the internal turn index.
				return nil, badRequest("tgo: a thinking block belongs to an assistant turn")
			}
			pending = append(pending, chat.Block{Type: chat.BlockThinking, Text: b.Text})
		case ir.BlockRedactedThinking:
			// Opaque, and a prior turn's thinking is stripped from the prompt
			// anyway (specs/003-chat-template.md §3), so nothing is lost by
			// dropping it that would not be dropped by rendering it.
		case ir.BlockToolUse:
			if b.ToolUse == nil {
				return nil, badRequest("tgo: a tool_use block carries no call")
			}
			pending = append(pending, chat.Block{Type: chat.BlockToolUse, ToolUse: &chat.ToolUse{
				ID: b.ToolUse.ID, Name: b.ToolUse.Name, Args: b.ToolUse.Args,
			}})
		case ir.BlockToolResult:
			if b.ToolResult == nil {
				return nil, badRequest("tgo: a tool_result block carries no result")
			}
			text, aerr := textOfChecked(b.ToolResult.Blocks)
			if aerr != nil {
				return nil, aerr
			}
			flush()
			out = append(out, chat.Message{Role: chat.Tool, Blocks: []chat.Block{{
				Type: chat.BlockToolResult,
				ToolResult: &chat.ToolResult{
					ToolUseID: b.ToolResult.ToolUseID,
					Text:      text,
					IsError:   b.ToolResult.IsError,
				},
			}}})
		case ir.BlockImage:
			return nil, imageRefusal()
		default:
			return nil, refusal(string(b.Type), "tgo: a %q content block has no place in a "+
				"text-only model's prompt", b.Type)
		}
	}
	flush()
	return out, nil
}

// imageRefusal is §4's rule applied to a picture: dropping it would answer a
// question the caller did not ask, so it is refused rather than recorded.
// specs/004-model-graph.md 004-D2 makes a vision model additive, and that is
// when this stops being a refusal.
func imageRefusal() *apiError {
	return refusal("image", "tgo: image content is not supported: this model is text-only, "+
		"and dropping the image would answer a different question")
}

// textOf joins the text of a block list, which is how a multi-part system
// prompt reaches a template that has one slot for it.
func textOf(blocks []ir.Block) string {
	var parts []string
	for _, b := range blocks {
		if b.Type == ir.BlockText && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n\n")
}

// textOfChecked is textOf, refusing an image rather than dropping it.
func textOfChecked(blocks []ir.Block) (string, *apiError) {
	for _, b := range blocks {
		if b.Type == ir.BlockImage {
			return "", imageRefusal()
		}
	}
	return textOf(blocks), nil
}

// mapTools carries tool declarations through to the template.
func mapTools(tools []ir.Tool) []chat.ToolSpec {
	if len(tools) == 0 {
		return nil
	}
	out := make([]chat.ToolSpec, 0, len(tools))
	for _, t := range tools {
		out = append(out, chat.ToolSpec{
			Name: t.Name, Description: t.Description, InputSchema: json.RawMessage(t.InputSchema),
		})
	}
	return out
}

// mapPolicy builds one request's sampling configuration.
//
// The IR half and the raw-body half meet here: temperature, top_p, top_k,
// max_tokens and the stop strings come from ir.Request, and the seed, the
// penalties and the bias come from [extras], because the IR has no room for
// them (§4.1).
func mapPolicy(req *ir.Request, ex extras, vocab int) (tgo.Policy, *apiError) {
	var p tgo.Policy
	if req.Temperature != nil {
		p.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		p.TopP = float32(*req.TopP)
	}
	switch {
	case req.TopK != nil:
		p.TopK = int(*req.TopK)
	case ex.topK != nil:
		// Only the Anthropic dialect has a top_k field, so on the OpenAI
		// surfaces the IR never sees one and the raw body is the only source.
		p.TopK = *ex.topK
	}
	if req.MaxTokens != nil {
		if *req.MaxTokens <= 0 {
			return p, badRequest("tgo: max_tokens must be positive, got %d", *req.MaxTokens)
		}
		if *req.MaxTokens > int64(math.MaxInt32) {
			return p, badRequest("tgo: max_tokens is out of range: %d", *req.MaxTokens)
		}
		p.MaxTokens = int(*req.MaxTokens)
	}
	p.Stop = req.StopSequences
	if ex.seed != nil {
		p.Seed = *ex.seed
	}
	if ex.presencePenalty != nil {
		p.PresencePenalty = *ex.presencePenalty
	}
	if ex.frequencyPenalty != nil {
		p.FrequencyPenalty = *ex.frequencyPenalty
	}
	if ex.repetitionPenalty != nil {
		p.RepetitionPenalty = *ex.repetitionPenalty
	}
	if ex.penaltyWindow != nil {
		if *ex.penaltyWindow < 0 {
			return p, badRequest("tgo: penalty_window must not be negative, got %d", *ex.penaltyWindow)
		}
		p.PenaltyWindow = *ex.penaltyWindow
	}
	if len(ex.logitBias) > 0 {
		for id := range ex.logitBias {
			if id < 0 || id >= vocab {
				return p, refusal("logit_bias", "tgo: logit_bias names token %d, which is "+
					"outside this model's vocabulary of %d: a bias that lands on no token "+
					"changes the answer without saying so", id, vocab)
			}
		}
		p.LogitBias = ex.logitBias
	}
	return p, nil
}
