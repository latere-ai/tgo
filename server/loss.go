// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"slices"
	"strings"

	"latere.ai/x/pkg/llmdialect/ir"
)

// The loss report is two corrections applied to llmdialect's, in that order.
//
// llmdialect accumulates the fields *its IR* cannot carry. That is the right
// list for a gateway and the wrong one for tgo in both directions:
//
//   - ir.Request has no seed, logit_bias or penalties, so every frontend files
//     each as an unknown top-level field -- and tgo implements all of them
//     (specs/007-engine.md §1's Policy). Emitting the list verbatim would
//     report as unhonoured exactly the knobs that were honoured (009-D12).
//     honoured is subtracted.
//
//   - ir.Request *does* carry things tgo drops: a caller identity, a
//     prompt-cache breakpoint, a thinking budget, a tool_choice tgo cannot
//     force without constrained decoding (009-D6). llmdialect files none of
//     those, because its IR represents them fine. dropped is added.
//
// Both tables live here so a new sampling knob or a new accepted-and-ignored
// field is one edit, and a Policy field missing from honoured fails a test
// rather than silently starting to report itself as lost.

// honoured maps a [github.com/latere-ai/tgo.Policy] field to the wire fields
// that set it, across all four dialects.
//
// TestEveryPolicyFieldIsHonoured reflects over Policy and fails on a field
// this map does not name, and TestHonouredFieldsAreParsed fails on a wire name
// this map claims that [parseExtras] and [applyRequest] do not actually read.
// The two together are the invariant: a name is here if and only if a request
// carrying it changes what the sampler does.
var honoured = map[string][]string{
	"Temperature":       {"temperature"},
	"TopK":              {"top_k"},
	"TopP":              {"top_p"},
	"RepetitionPenalty": {"repetition_penalty"},
	"PresencePenalty":   {"presence_penalty"},
	"FrequencyPenalty":  {"frequency_penalty"},
	"PenaltyWindow":     {"penalty_window"},
	"LogitBias":         {"logit_bias"},
	"Seed":              {"seed"},
	"MaxTokens":         {"max_tokens", "max_completion_tokens", "max_output_tokens"},
	"Stop":              {"stop", "stop_sequences"},
	"Schema":            {"response_format", "output_format", "text"},
	// Both reached by one member, and it is not top_logprobs. specs/030-logprobs.md
	// §4: the only route that can encode a logprob is /v1/completions, whose
	// member is `logprobs` and whose value *is* the count of alternatives.
	// `top_logprobs` belongs to /v1/chat/completions, which llmdialect's ir
	// cannot express, so it reaches no output and stays a loss everywhere.
	"LogProbs":    {"logprobs"},
	"TopLogProbs": {"logprobs"},
}

// honouredSession are the wire names that configure the *session* rather than
// the sampler, so they are honoured without being
// [github.com/latere-ai/tgo.Policy] fields and cannot go in [honoured], whose
// invariant is that its keys are exactly Policy's.
//
// One entry. cache_salt bounds which pooled session a request may be routed to
// (specs/019-session-affinity.md 019-D3) and reaches no Policy field at all, so
// a caller who sends it and is told it was dropped would think their cache is
// shared with everybody when it is isolated. It is honoured on every route,
// because [parseExtras] reads it from the raw body and does not know which
// dialect sent it.
var honouredSession = map[string]bool{"cache_salt": true}

// honouredWire is honoured flattened: every wire name some dialect applies.
var honouredWire = func() map[string]bool {
	m := map[string]bool{}
	for _, names := range honoured {
		for _, n := range names {
			m[n] = true
		}
	}
	return m
}()

// honouredEverywhere are the names that reach Policy on any route.
//
// The penalties, the seed, the bias and the window are read from the raw body
// by [parseExtras], which does not know which dialect sent them; temperature,
// top_p and top_k reach ir.Request or [extras] under one spelling on all four
// surfaces.
var honouredEverywhere = []string{
	"temperature", "top_p", "top_k", "seed", "logit_bias",
	"presence_penalty", "frequency_penalty", "repetition_penalty", "penalty_window",
}

// honouredHere are the names one dialect spells its own way.
//
// The subtraction has to be per dialect or it is the very bug 009-D12 names,
// from the other side: max_output_tokens sent to /v1/chat/completions sets
// nothing, because that surface's member is max_tokens, and subtracting the
// name anyway would report a bound this server never applied as honoured. The
// completion then runs unbounded and no channel says so, which is worse than
// reporting an honoured knob as lost.
// The schema is the sharpest case in the table: each dialect spells it
// differently -- response_format, output_format, text.format -- and all three
// names reach the same [github.com/latere-ai/tgo.Policy] field, so subtracting
// the union would report a schema as enforced on the three routes that never
// saw one.
var honouredHere = map[ir.Dialect][]string{
	ir.DialectOpenAIChat:        {"max_tokens", "max_completion_tokens", "stop", "response_format"},
	ir.DialectAnthropicMessages: {"max_tokens", "stop_sequences", "output_format"},
	ir.DialectOpenAIResponses:   {"max_output_tokens", "text"},
	// specs/030-logprobs.md §4: one route serves logprobs and it is this one,
	// because it is the only Frontend tgo wrote. llmdialect's ir carries no
	// logprobs shape at all, so the three dialects it encodes cannot express
	// one whatever tgo computes -- 030-D5 reports that rather than reaching
	// past the codec to append a member to a body it did not write.
	//
	// top_logprobs is not here: this surface has no such member, and its
	// logprobs is itself the count.
	dialectLegacy: {"max_tokens", "stop", "logprobs"},
}

// honouredOn is what the subtraction reads: for one dialect, the wire names a
// request carrying them actually applies.
var honouredOn = func() map[ir.Dialect]map[string]bool {
	out := make(map[ir.Dialect]map[string]bool, len(honouredHere))
	for d, own := range honouredHere {
		m := make(map[string]bool, len(own)+len(honouredEverywhere))
		for _, n := range honouredEverywhere {
			m[n] = true
		}
		for _, n := range own {
			m[n] = true
		}
		out[d] = m
	}
	return out
}()

// Loss fields this package adds, for things llmdialect represents and tgo does
// not act on.
const (
	// lossUser is a caller-supplied end-user identifier: OpenAI's `user`,
	// Anthropic's `metadata.user_id`. tgo has no per-user anything.
	lossUser ir.LossField = "user"

	// lossMetadata is a request-scoped bag the Responses dialect accepts and
	// discards; it never reaches ir.Request at all.
	lossMetadata ir.LossField = "metadata"

	// lossToolChoice is a forced or restricted tool choice. tgo renders tools
	// into the prompt and cannot force a call without constrained decoding
	// (009-D6, specs/015-structured-output.md).
	lossToolChoice ir.LossField = "tool_choice"

	// lossParallelToolCalls is parallel_tool_calls:false, which is the same
	// promise from the other side.
	lossParallelToolCalls ir.LossField = "parallel_tool_calls"
)

// lossReport is what X-Tgo-Loss carries: llmdialect's list, corrected.
//
// Order is llmdialect's insertion order first and this package's additions
// after, deduplicated, so the header is stable for a given request. The
// subtraction is the caller's dialect's, because a name is honoured only on the
// surfaces that define it.
func lossReport(d ir.Dialect, req *ir.Request, raw map[string]bool) []string {
	honours := honouredOn[d]
	var out ir.Loss
	for _, f := range req.Loss.Fields() {
		if honours[string(f)] || honouredSession[string(f)] {
			continue
		}
		out.Add(f)
	}
	for _, f := range dropped(req, raw) {
		out.Add(f)
	}
	return out.Strings()
}

// dropped is what tgo accepts, runs, and does nothing with.
//
// Every entry is advisory by §4's test: a request with it and a request without
// it produce the same tokens. A field that changed the tokens would be in
// refuse.go instead.
func dropped(req *ir.Request, raw map[string]bool) []ir.LossField {
	var out []ir.LossField
	if req.UserID != "" {
		out = append(out, lossUser)
	}
	// metadata is reported whenever it was sent, and separately from the
	// identity inside it. The Responses frontend lists metadata among the keys
	// it understands and then has no field for it; the Anthropic one reads
	// metadata.user_id and drops the rest. Either way the bag itself is gone,
	// and a caller who put a trace id in it should learn that here rather than
	// from its absence in a log.
	if raw["metadata"] {
		out = append(out, lossMetadata)
	}
	if blocksCacheHint(req.System) || messagesCacheHint(req.Messages) {
		out = append(out, ir.LossCacheControl)
	}
	if req.Reasoning != nil {
		if req.Reasoning.BudgetTokens > 0 {
			// The budget is advisory: tgo does not stop the model mid-thought
			// (§4.1's thinking row).
			out = append(out, ir.LossThinkingBudget)
		}
		if req.Reasoning.Effort != "" {
			// The template's thinking flag is a boolean, so a tier is a tier
			// tgo cannot spend.
			out = append(out, ir.LossReasoningEffort)
		}
	}
	if tc := req.ToolChoice; tc != nil {
		if tc.Mode != "" && tc.Mode != ir.ToolChoiceAuto {
			out = append(out, lossToolChoice)
		}
		if tc.DisableParallel {
			out = append(out, lossParallelToolCalls)
		}
	}
	return out
}

// blocksCacheHint reports whether any block asks for a prompt-cache
// breakpoint. tgo has no prompt cache until specs/016-prefix-cache.md.
func blocksCacheHint(blocks []ir.Block) bool {
	return slices.ContainsFunc(blocks, func(b ir.Block) bool { return b.CacheHint })
}

func messagesCacheHint(msgs []ir.Message) bool {
	return slices.ContainsFunc(msgs, func(m ir.Message) bool { return blocksCacheHint(m.Blocks) })
}

// header renders a loss list for X-Tgo-Loss.
//
// Comma-separated, because a loss list is a set of field names and an HTTP
// header holding a set is a comma-separated one (RFC 9110 §5.6.1).
func header(fields []string) string { return strings.Join(fields, ", ") }
