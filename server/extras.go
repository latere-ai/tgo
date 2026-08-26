// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// extras are the request fields tgo honours that llmdialect's IR does not
// carry, read from the raw body beside DecodeRequest (009-D12, §4.1).
//
// Every field here is also a name in [honoured], and the two are checked
// against each other by a test: a knob parsed but not subtracted reports itself
// as lost, and a knob subtracted but not parsed reports itself as honoured
// while being dropped. The second is the worse of the two and the harder to
// see.
type extras struct {
	seed              *uint64
	logitBias         map[int]float32
	presencePenalty   *float32
	frequencyPenalty  *float32
	repetitionPenalty *float32
	penaltyWindow     *int
	topK              *int

	// cacheSalt is 016 §7.1's caller-supplied salt, which
	// specs/019-session-affinity.md 019-D3 uses as the affinity key: a request
	// carrying one may be routed only to a pooled session whose last request
	// carried the same one, and a request carrying none only to a session that
	// had none. It is not a sampling knob and reaches no
	// [github.com/latere-ai/tgo.Policy] field, which is why it is subtracted
	// from the loss report by [honouredSession] rather than by [honoured].
	cacheSalt string

	// thinkingOff is the one signal that does not survive into ir.Request:
	// Anthropic's thinking:{"type":"disabled"} and OpenAI's
	// reasoning_effort:"none" both decode to no ir.Reasoning at all, which is
	// indistinguishable from a request that said nothing. tgo's default is on,
	// so the difference is visible in the prompt.
	thinkingOff bool
}

// topLevel splits a request body into its top-level members.
//
// The body is parsed twice on purpose: once by the dialect's frontend into the
// neutral IR, and once here for the fields that IR has no room for. A single
// parse would mean either reimplementing the dialect or dropping the knobs.
func topLevel(body []byte) (map[string]json.RawMessage, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, fmt.Errorf("the request body is not a JSON object: %w", err)
	}
	return top, nil
}

// keys is the set of member names present, which is what the loss report reads
// for a field a frontend accepts and discards.
func keys(top map[string]json.RawMessage) map[string]bool {
	out := make(map[string]bool, len(top))
	for k := range top {
		out[k] = true
	}
	return out
}

// parseExtras reads the fields ir.Request has no room for.
//
// A malformed value is an error naming the field rather than a silent zero: a
// seed that did not parse is a request whose caller believes it is reproducible
// and is not.
func parseExtras(top map[string]json.RawMessage) (extras, error) {
	var ex extras
	var err error
	if ex.seed, err = jsonUint(top, "seed"); err != nil {
		return ex, err
	}
	if ex.presencePenalty, err = jsonFloat(top, "presence_penalty"); err != nil {
		return ex, err
	}
	if ex.frequencyPenalty, err = jsonFloat(top, "frequency_penalty"); err != nil {
		return ex, err
	}
	if ex.repetitionPenalty, err = jsonFloat(top, "repetition_penalty"); err != nil {
		return ex, err
	}
	if ex.penaltyWindow, err = jsonInt(top, "penalty_window"); err != nil {
		return ex, err
	}
	if ex.topK, err = jsonInt(top, "top_k"); err != nil {
		return ex, err
	}
	if ex.logitBias, err = parseLogitBias(top); err != nil {
		return ex, err
	}
	if ex.cacheSalt, err = jsonString(top, "cache_salt"); err != nil {
		return ex, err
	}
	ex.thinkingOff = thinkingOff(top)
	return ex, nil
}

// jsonString reads an optional string member.
//
// A member of the wrong type is an error naming the field rather than a silent
// empty string: a cache_salt that did not parse is a request whose caller
// believes their cache is isolated from everybody else's and is not.
func jsonString(top map[string]json.RawMessage, name string) (string, error) {
	raw, ok := top[name]
	if !ok || isNull(raw) {
		return "", nil
	}
	var v string
	if err := json.Unmarshal(raw, &v); err != nil {
		return "", fmt.Errorf("%s must be a string: %w", name, err)
	}
	return v, nil
}

// parseLogitBias reads the {token id: bias} map.
//
// JSON object keys are strings, so the ids arrive as decimal text; a key that
// is not an integer is an error rather than a skipped entry, because a bias
// that lands on no token is a request the caller believes was applied.
func parseLogitBias(top map[string]json.RawMessage) (map[int]float32, error) {
	raw, ok := top["logit_bias"]
	if !ok || isNull(raw) {
		return nil, nil
	}
	var wire map[string]float64
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("logit_bias must be an object of token id to bias: %w", err)
	}
	if len(wire) == 0 {
		return nil, nil
	}
	out := make(map[int]float32, len(wire))
	for k, v := range wire {
		id, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("logit_bias key %q is not a token id", k)
		}
		out[id] = float32(v)
	}
	return out, nil
}

// thinkingOff reports whether the request asked the model not to think.
func thinkingOff(top map[string]json.RawMessage) bool {
	if raw, ok := top["thinking"]; ok && !isNull(raw) {
		var t struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(raw, &t) == nil && t.Type == "disabled" {
			return true
		}
	}
	if raw, ok := top["reasoning_effort"]; ok && !isNull(raw) {
		var e string
		if json.Unmarshal(raw, &e) == nil && e == "none" {
			return true
		}
	}
	if raw, ok := top["reasoning"]; ok && !isNull(raw) {
		var r struct {
			Effort string `json:"effort"`
		}
		if json.Unmarshal(raw, &r) == nil && r.Effort == "none" {
			return true
		}
	}
	return false
}

// isNull reports whether a member was sent as JSON null, which every dialect
// treats as absent.
func isNull(raw []byte) bool { return string(raw) == "null" }

func jsonUint(top map[string]json.RawMessage, name string) (*uint64, error) {
	raw, ok := top[name]
	if !ok || isNull(raw) {
		return nil, nil
	}
	var v uint64
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, fmt.Errorf("%s must be a non-negative integer: %w", name, err)
	}
	return &v, nil
}

func jsonInt(top map[string]json.RawMessage, name string) (*int, error) {
	raw, ok := top[name]
	if !ok || isNull(raw) {
		return nil, nil
	}
	var v int
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, fmt.Errorf("%s must be an integer: %w", name, err)
	}
	return &v, nil
}

func jsonFloat(top map[string]json.RawMessage, name string) (*float32, error) {
	raw, ok := top[name]
	if !ok || isNull(raw) {
		return nil, nil
	}
	var v float64
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, fmt.Errorf("%s must be a number: %w", name, err)
	}
	f := float32(v)
	return &f, nil
}
