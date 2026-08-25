// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package server serves one tgo model over four request routes: three wire
// dialects carried by llmdialect, and the legacy completions codec tgo owns.
//
//	POST /v1/chat/completions   OpenAI Chat Completions
//	POST /v1/messages           Anthropic Messages
//	POST /v1/responses          OpenAI Responses
//	POST /v1/completions        OpenAI legacy, raw text, no template
//	GET  /v1/models, /health, /metrics
//
// The three chat dialects reach one neutral request through
// [latere.ai/x/pkg/llmdialect]'s Frontend half: a Frontend decodes what the
// client sent into an ir.Request and re-encodes the answer in the dialect the
// client speaks. Translation is hub-and-spoke, so three surfaces cost one
// adapter rather than three parsers (specs/009-server.md 009-D9). llmdialect's
// Backend half encodes a request *to* an upstream provider and this package
// never uses it: tgo is the upstream.
//
// # The boundary
//
// A handler turns a request into an [Engine] call and a token stream into SSE
// frames. It holds no model state, and nothing under
// [github.com/latere-ai/tgo] knows this package exists (009-D1, 009-D10):
// a caller embedding tgo as a library inherits neither the IR types nor the
// dialect layer.
//
// # Refuse what changes the answer, record what does not
//
// A field whose absence changes what the model computes is refused by name --
// n > 1, a structured-output schema, an image, a logit_bias id outside the
// vocabulary. A field that is advisory runs and is reported: every one lands in
// the X-Tgo-Loss response header and in the tgo_request_loss_total counter
// (009-D2). llmdialect's own loss report is the input to that list and not the
// output of it: ir.Request carries no seed, logit_bias or penalties, so its
// frontends report each as unrepresentable while tgo implements every one, and
// the subtraction takes them back out, per dialect, because a name is honoured
// only on the surfaces that define it (009-D12). See loss.go for the tables.
//
// # Concurrency
//
// One [Session] per in-flight request and an admission semaphore sized by KV
// memory, because each session's cache is a fixed reservation. Over the limit
// requests queue; the queue is bounded, and a full one answers 429 with
// Retry-After rather than growing (009-D3). Without batching, concurrent
// requests do not go faster -- they interleave, and total throughput is what
// one sequence gets.
//
// # Not in scope
//
// Authentication, rate limiting, multi-model routing and a model management
// API (009-D5). The server binds to loopback unless [WithPublicBind] says
// otherwise, and says out loud that it has no authentication when it does
// (009-D8).
package server
