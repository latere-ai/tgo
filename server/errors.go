// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"encoding/json"
	"fmt"
	"net/http"

	"latere.ai/x/pkg/llmdialect/ir"
)

// llmdialect's Frontend has four methods and none of them encodes an error, and
// package ir defines no error type. A refusal, a 404 and a 429 therefore have
// no body shape to borrow, and a device failure mid-generation could only reach
// the client as an abrupt close -- which a client cannot tell from a dropped
// connection. So this package owns a small per-dialect error encoder, beside
// the frontend rather than inside it (009-D13).
//
// The dialects genuinely differ, which is why this is a table and not one
// shape: Anthropic sends a named `error` event mid-stream and a
// {"type":"error",...} body; OpenAI sends an error chunk and then closes.

// errKind is the dialect-neutral class of a failure. Each dialect names it in
// its own vocabulary.
type errKind int

const (
	// errInvalidRequest is a request this server will not run: a refused
	// field, a malformed body, a context that does not fit.
	errInvalidRequest errKind = iota

	// errNotFound is a request for a model this server does not serve.
	errNotFound

	// errOverloaded is a full queue. It carries Retry-After.
	errOverloaded

	// errInternal is a failure on tgo's side, including a device failure
	// mid-generation (specs/007-engine.md §7).
	errInternal

	// errClientGone is the client hanging up. It has no body: there is nobody
	// left to read one. It exists as a class so the admission path can report
	// a cancelled wait as something other than a failure.
	errClientGone
)

// name is what one dialect calls an error class.
func (k errKind) name(d ir.Dialect) string {
	anthropic := d == ir.DialectAnthropicMessages
	switch k {
	case errNotFound:
		return "not_found_error"
	case errOverloaded:
		if anthropic {
			return "overloaded_error"
		}
		return "rate_limit_error"
	case errInternal:
		if anthropic {
			return "api_error"
		}
		return "server_error"
	case errClientGone:
		return "request_canceled"
	default:
		return "invalid_request_error"
	}
}

// status is the HTTP status a class answers with.
func (k errKind) status() int {
	switch k {
	case errNotFound:
		return http.StatusNotFound
	case errOverloaded:
		return http.StatusTooManyRequests
	case errInternal:
		return http.StatusInternalServerError
	case errClientGone:
		// 499 is nginx's, not RFC 9110's, and it is what every log pipeline
		// already reads for a client that hung up. Nothing is written with it.
		return 499
	default:
		return http.StatusBadRequest
	}
}

// apiError is one failure, before it is dressed in a dialect.
type apiError struct {
	kind errKind

	// field names the request member at fault, and is empty when none is.
	// Every §4 refusal sets it: "refuse, naming the field" is the whole rule,
	// and a refusal that does not say which knob it refused sends the caller
	// to bisect their own request.
	field string

	msg string

	// reason is the tgo_sessions_rejected_total label. It is on the error
	// rather than at the call site so that a new refusal cannot be added
	// without deciding what it is counted as.
	reason string

	// retryAfter is the Retry-After header's value, in seconds, and is set
	// only on errOverloaded: a 429 that does not say when to come back turns
	// a load problem into a retry storm (009-D3).
	retryAfter string
}

func (e *apiError) Error() string { return e.msg }

// refusal is a §4 refusal: a field whose absence would change the answer.
func refusal(field, format string, args ...any) *apiError {
	return &apiError{kind: errInvalidRequest, field: field, reason: "refused_field",
		msg: fmt.Sprintf(format, args...)}
}

// badRequest is a request that is wrong rather than unsupported.
func badRequest(format string, args ...any) *apiError {
	return &apiError{kind: errInvalidRequest, reason: "bad_request",
		msg: fmt.Sprintf(format, args...)}
}

// body is the pre-stream error body in one dialect.
func (e *apiError) body(d ir.Dialect) []byte {
	name := e.kind.name(d)
	var v any
	if d == ir.DialectAnthropicMessages {
		v = map[string]any{
			"type":  "error",
			"error": map[string]any{"type": name, "message": e.msg},
		}
	} else {
		inner := map[string]any{"message": e.msg, "type": name, "code": nil}
		inner["param"] = nullableField(e.field)
		v = map[string]any{"error": inner}
	}
	raw, err := json.Marshal(v)
	if err != nil {
		// The map holds strings and nils; this cannot fail, and a panic here
		// would be a worse answer than a plain body.
		return []byte(`{"error":{"message":"tgo: the error could not be encoded"}}`)
	}
	return raw
}

// nullableField renders an empty field name as JSON null, which is what
// OpenAI's `param` is when no member is at fault.
func nullableField(field string) any {
	if field == "" {
		return nil
	}
	return field
}

// writeError answers a request that never started streaming.
func writeError(w http.ResponseWriter, d ir.Dialect, e *apiError) {
	if e.kind == errClientGone {
		// No body -- nobody is reading one -- but the status, because a bare
		// return lets the runtime synthesize 200 with an empty body, which a
		// proxy reads as a completion that produced nothing
		// (specs/022-batched-serving.md §10's 499 rule).
		w.WriteHeader(e.kind.status())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if e.retryAfter != "" {
		w.Header().Set("Retry-After", e.retryAfter)
	}
	w.WriteHeader(e.kind.status())
	_, _ = w.Write(e.body(d))
}

// writeStreamError reports a failure after the response has begun.
//
// The status is already sent, so the only channel left is the stream itself.
// Anthropic names the frame `error`; OpenAI sends a chunk carrying an error
// member and then the terminator, because a chat client's parser reads chunks
// and would ignore a frame it has no case for.
func writeStreamError(w http.ResponseWriter, d ir.Dialect, e *apiError) {
	switch d {
	case ir.DialectAnthropicMessages:
		writeFrame(w, "error", e.body(d))
	case ir.DialectOpenAIResponses:
		raw, err := json.Marshal(map[string]any{
			"type": "error", "code": e.kind.name(d), "message": e.msg,
			"param": nullableField(e.field),
		})
		if err != nil {
			// As in body: the map holds strings and nils, so this cannot
			// fail, and a frame the client can parse beats a panic.
			raw = []byte(`{"type":"error","message":"tgo: the error could not be encoded"}`)
		}
		writeFrame(w, "error", raw)
	default:
		writeFrame(w, "", e.body(d))
		writeFrame(w, "", []byte("[DONE]"))
	}
}

// writeFrame writes one SSE frame. An empty name writes a data-only frame,
// which is what the OpenAI dialects use throughout.
func writeFrame(w http.ResponseWriter, name string, data []byte) {
	if name != "" {
		_, _ = fmt.Fprintf(w, "event: %s\n", name)
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
}
