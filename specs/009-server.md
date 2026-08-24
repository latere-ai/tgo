---
title: "The server: an OpenAI-compatible surface that does not reach into the engine"
status: drafted
layer: api
depends_on:
  - 000-decisions.md
  - 003-chat-template.md
  - 007-engine.md
---

# The server

## 1. What compatibility means here

`POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, with
OpenAI's request and response JSON, streaming over server-sent events. That
surface is what every client library already speaks, and it is the reason not to
invent one.

Compatibility is a **serialisation decision**. The handler renders a request
into a `Session` call and a stream of tokens into SSE frames, and it holds no
model state of its own. Nothing in [007](007-engine.md) knows the server exists.

**Rejected:** an engine API shaped like OpenAI's schema. It looks like it saves
a layer. It means every future scheduler and cache decision is negotiated
against a JSON schema owned by someone else.

## 2. Where tgo differs, and says so

| field | tgo |
| --- | --- |
| `n > 1` | refused; it needs batching, which is [008](008-scheduler.md) |
| `logprobs` | supported, host-computed from the same logits |
| `tools` / `function_call` | v0 renders them into the prompt via the model's template and returns the model's text; no forced grammar |
| `logit_bias` | supported, applied before penalties |
| `seed` | supported and honoured as a stream seed, per [006 §3](006-sampling.md) |
| `response_format: json_schema` | refused in v0; see §4 |

An unsupported field is **refused with a message naming it**, never ignored. A
server that ignores `n=4` and returns one choice has produced a wrong answer to
a well-formed request.

## 3. Concurrency

One `Engine`, one `Session` per in-flight request, and a semaphore sized to the
KV memory available — because [005 §2](005-kv-cache.md) makes each session's
cache a fixed reservation. Requests over the limit queue; the queue has a bound
and a timeout, and a full queue returns 429 rather than growing.

Without batching, concurrent requests interleave at submission granularity and
throughput does not improve. The server does not pretend otherwise: `/metrics`
reports queue depth and per-request wait, so the cost of gap 1 in
[008](008-scheduler.md) is visible as a number.

## 4. Structured output is specified and not built

sglang's contribution here is constrained decoding: a grammar or JSON schema
compiled to a per-step token mask, so invalid tokens have probability zero and
the output parses by construction.

The mask is $O(V)$ per step and must be applied to the logits. On the host that
is a vector multiply — cheap. On the device it is a bound tensor and one
elementwise op, which accel can express today. What does not exist is the
compiler from a schema to a mask, which is a real piece of work: a pushdown
automaton over the tokenizer's vocabulary, with a per-state token set.

It is [015](015-structured-output.md), written and unbuilt, and it is the
highest-value thing after batching.

## 5. Not in scope

Authentication, rate limiting per key, multi-model routing, and a model
management API. tgo serves one model, and everything above belongs to whatever
runs in front of it. Saying so is the decision.

## 6. Tests

Handler tests run against a **fake engine** — a scripted token stream — so they
need no device and no weights, and they cover: SSE framing including the
terminal `[DONE]`, a client disconnect cancelling generation, every §2 refusal,
the queue bound and its 429, and malformed JSON.

One end-to-end test runs a real tiny synthetic model through the real handler.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 009-D1 | OpenAI JSON at the edge only | an OpenAI-shaped engine API | engine decisions stay free of an external schema |
| 009-D2 | refuse unsupported fields by name | ignore them | a well-formed request never gets a quietly wrong answer |
| 009-D3 | session concurrency bounded by KV memory, 429 over it | unbounded goroutines | the reservation from 005 is enforced, not hoped |
| 009-D4 | handlers tested against a fake engine | only end-to-end | the HTTP surface is fully covered without a device |
| 009-D5 | no auth, no routing, one model | a management API | the boundary is stated rather than discovered |
