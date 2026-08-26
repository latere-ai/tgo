// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"fmt"
	"math"

	"github.com/latere-ai/tgo/sample"
)

// Policy is one request's sampling configuration, plus the two limits that
// belong to a sequence rather than to a row of logits.
//
// The zero value is greedy and unbounded except by the session's context:
// argmax, no bias, no penalties, no truncation.
//
// specs/007-engine.md §1. The first nine fields are
// [github.com/latere-ai/tgo/sample.Policy]'s, restated so a caller does not
// import the sampler to name a temperature; MaxTokens, Stop and Seed are the
// engine's, because a package that sees one row of logits knows nothing about
// a sequence (006-D4).
type Policy struct {
	// Temperature divides the logits. Zero means greedy, exactly: a separate
	// branch taken after the penalties, not a division.
	Temperature float32

	// TopK keeps the k largest candidates. Zero means the stage is absent.
	TopK int

	// TopP keeps the smallest set of candidates whose mass reaches p. Zero
	// means the stage is absent, not "keep one token".
	TopP float32

	// RepetitionPenalty is divisive and sign-asymmetric. One and zero both
	// mean no penalty.
	RepetitionPenalty float32

	// PresencePenalty is subtracted once from any token in the window.
	PresencePenalty float32

	// FrequencyPenalty is subtracted once per occurrence in the window.
	FrequencyPenalty float32

	// PenaltyWindow is how many tokens back the penalties read. Zero means the
	// whole context, prompt and generated tokens together.
	PenaltyWindow int

	// LogitBias is an absolute statement about a token, applied before
	// everything else. Negative infinity bans one.
	LogitBias map[int]float32

	// Seed fixes the draw stream. The same seed and the same policy give the
	// same completion on one device, including a policy that changes
	// mid-request (006-D2).
	Seed uint64

	// MaxTokens bounds the completion. Zero means "as many as the session's
	// remaining context holds", which is what makes context exhaustion a
	// refusal a caller asked for rather than one that arrives unannounced
	// (§7).
	MaxTokens int

	// Schema constrains the completion to a JSON document matching this JSON
	// Schema, by masking away at every step the tokens that cannot continue
	// one (specs/015-structured-output.md §1). The output parses by
	// construction, so a retry loop around a model that emits JSON "most of
	// the time" is unnecessary.
	//
	// The schema is compiled against the model's vocabulary, and a schema that
	// cannot be compiled is refused by naming the construct rather than
	// approximated: a keyword silently ignored produces a document that
	// validates against a schema the caller did not write (015-D4).
	// [Model.CheckSchema] performs that compilation on its own, which is what
	// lets a server answer an uncompilable schema before it allocates
	// anything, and it keeps the result so the request that follows pays
	// nothing.
	//
	// Three narrowings of JSON Schema are deliberate and are the compiler's:
	// an object's properties are emitted in the order the schema declares
	// them, an object is closed, and "integer" admits the plain spelling.
	// Each shrinks the admitted language, so a document produced here still
	// validates against the schema. What is not narrowed is a number's
	// magnitude, because JSON Schema spells that as "minimum", which is
	// refused -- so a caller who needs one checks it after decoding.
	//
	// It is refused together with Stop: a stop string cuts the completion
	// where it matched, which is half a document. MaxTokens is not refused,
	// because a budget that runs out is reported -- CompletionTokens reaches
	// it, and a server renders that as a length finish -- while a stop string
	// that fires reports an ordinary end.
	Schema []byte

	// Stop ends the completion when one of these strings appears in the
	// decoded text, and the text is cut before it.
	//
	// On text and not on token ids: a stop string need not align to a token
	// boundary, so it belongs with the detokenizer and its hold-back buffer
	// (006-D4, specs/002-tokenizer.md 002-D8). While a stop string is set the
	// stream holds back the longest suffix of its output that could still
	// begin one.
	//
	// Set together with Schema it is refused, not applied: the two end a
	// completion by different rules, and the one that wins would cut a
	// document Schema promised would parse.
	Stop []string
}

// sampling projects the policy onto what the sampler reads.
func (p Policy) sampling() sample.Policy {
	return sample.Policy{
		Temperature:       p.Temperature,
		TopK:              p.TopK,
		TopP:              p.TopP,
		RepetitionPenalty: p.RepetitionPenalty,
		PresencePenalty:   p.PresencePenalty,
		FrequencyPenalty:  p.FrequencyPenalty,
		PenaltyWindow:     p.PenaltyWindow,
		LogitBias:         p.LogitBias,
	}
}

// check refuses a policy the engine cannot honour, naming the field.
//
// The sampler panics on a policy the device could not reproduce, which is the
// right contract one layer down and the wrong one here: a caller who typed a
// number into a request should get an error, not a stack trace.
func (p Policy) check(vocab int) error {
	if p.MaxTokens < 0 {
		return fmt.Errorf("tgo: MaxTokens is %d; it is zero for unbounded or a positive "+
			"count", p.MaxTokens)
	}
	if p.Temperature < 0 || p.Temperature != p.Temperature {
		return fmt.Errorf("tgo: Temperature is %v; it is zero for greedy or positive",
			p.Temperature)
	}
	if p.TopK < 0 || p.TopK > sample.TopMaxRounds {
		return fmt.Errorf("tgo: TopK is %d; it is zero for no truncation or 1..%d, which is "+
			"what accel's kernel can reproduce", p.TopK, sample.TopMaxRounds)
	}
	if p.TopP < 0 || p.TopP > 1 || p.TopP != p.TopP {
		return fmt.Errorf("tgo: TopP is %v; it is zero for no truncation or lies in (0, 1]",
			p.TopP)
	}
	if p.RepetitionPenalty < 0 || p.RepetitionPenalty != p.RepetitionPenalty {
		return fmt.Errorf("tgo: RepetitionPenalty is %v; it is not negative",
			p.RepetitionPenalty)
	}
	if p.PenaltyWindow < 0 {
		return fmt.Errorf("tgo: PenaltyWindow is %d; it is zero for the whole context or a "+
			"positive count", p.PenaltyWindow)
	}
	for id, bias := range p.LogitBias {
		if id < 0 || id >= vocab {
			return fmt.Errorf("tgo: LogitBias names token %d and the vocabulary holds %d",
				id, vocab)
		}
		// Negative infinity bans a token, which is the point. NaN and positive
		// infinity are not statements about a token: one poisons every
		// comparison and the other makes the argmax unconditional whatever the
		// model computed, and the sampler refuses both.
		if bias != bias || math.IsInf(float64(bias), 1) {
			return fmt.Errorf("tgo: LogitBias for token %d is %v; a bias is finite or "+
				"negative infinity, which bans the token", id, bias)
		}
	}
	for _, s := range p.Stop {
		if s == "" {
			return fmt.Errorf("tgo: Stop holds an empty string, which every completion " +
				"contains before its first token")
		}
	}
	// A stop string cuts the text at the point it matched and ends the stream
	// with no error, so a stop that fires inside a document leaves a caller
	// holding half of one and reading it as a completed answer. The two
	// stopping rules are also different in kind: the grammar ends a generation
	// where the document is complete, and Stop ends it where a substring
	// appeared. Refused rather than silently ignored, because a stop string
	// dropped without a word is the same request answered differently
	// (015-D9).
	if len(p.Schema) > 0 && len(p.Stop) > 0 {
		return fmt.Errorf("tgo: Schema and Stop are set together; a stop string cuts the "+
			"completion where it matched, so it would end a constrained request on half a "+
			"document, and Schema promises one that parses. Drop Stop, or drop Schema: "+
			"%q", p.Stop)
	}
	return nil
}

// Usage is what one request consumed.
type Usage struct {
	// PromptTokens is the rendered prompt's length in tokens, padding
	// excluded: a bucket's pad rows are arithmetic the caller did not ask for
	// and did not receive.
	PromptTokens int

	// CompletionTokens is how many tokens the model produced, structural
	// markers included and the stopping token excluded.
	CompletionTokens int

	// CachedPromptTokens is how many of PromptTokens were served from
	// key/value state the session already held, and so were not prefilled
	// again. It is zero unless [WithPrefixCache] is on, and it is always at
	// least one short of PromptTokens: the cache holds key/value state and not
	// logits, so the last prompt position is scored on every request
	// (016-D10).
	//
	// It is reported rather than merely enjoyed because a cache that silently
	// stops working reads as "the framework got slower", which is the one
	// symptom nobody files (specs/016-prefix-cache.md §1).
	CachedPromptTokens int
}
