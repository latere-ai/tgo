---
title: "Structured output: a schema compiled to a per-step token mask"
status: drafted
layer: engine
depends_on:
  - 000-decisions.md
  - 006-sampling.md
  - 009-server.md
---

# Structured output

Not blocked on accel. It is real work, and [011 §3](011-sequencing.md) places it
after batching.

## 1. The idea

Prompting a model to emit JSON gives JSON most of the time. Constrained
decoding gives it every time: at each step, the tokens that cannot continue a
valid document are given probability zero, so the output parses **by
construction** and a retry loop is unnecessary.

Formally, with $G$ a grammar and $y_{<t}$ the tokens so far:

$$p'(v \mid y_{<t}) \propto p(v \mid y_{<t}) \cdot \mathbb{1}\!\left[y_{<t} v \text{ is a viable prefix of } G\right]$$

## 2. The hard part

The mask is over the **tokenizer's vocabulary**, not over characters. A single
BPE token spans several characters and may cross a grammar boundary — `":` is
one token, and it is a string terminator followed by a structural character. So
the automaton advances by a token's worth of characters, and a token is
admissible only if every intermediate character is.

Computing that per step over 152k tokens is too slow to do naively. The standard
answer, from outlines and xgrammar, is to precompute per automaton state the set
of admissible tokens, cached across requests because states repeat heavily. The
cache is the design, and building it lazily on first visit is what keeps
startup from paying for the whole vocabulary cross product.

## 3. Where the mask is applied

Before penalties, as an additive $-\infty$ on the logits — so it composes with
[006 §2](006-sampling.md)'s order without a special case, and a masked token
cannot be resurrected by a penalty or a temperature.

Host in v0, alongside the rest of [006](006-sampling.md)'s host stages. It is a
bound tensor and one elementwise multiply on the device, which accel can express
today, so this does not become a conformance entry.

## 4. Scope

JSON Schema first — it is what `response_format` sends and what most callers
want. A general EBNF surface is the same machinery with a different front end
and comes second. Regex is a special case of the EBNF path.

**Refuse a schema that cannot be compiled**, naming the construct. Unsupported
keywords silently ignored produce a document that validates against a schema the
caller did not write.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 015-D1 | mask over vocabulary via a lazily-built per-state token set | per-step grammar walk over 152k tokens | startup stays cheap; the cache is the design |
| 015-D2 | applied as additive $-\infty$ before penalties | multiply after truncation | composes with 006's order with no special case |
| 015-D3 | JSON Schema first, EBNF second | EBNF only | matches `response_format`; regex falls out of EBNF |
| 015-D4 | refuse uncompilable schemas by construct | ignore unknown keywords | the caller's schema is the one enforced |
