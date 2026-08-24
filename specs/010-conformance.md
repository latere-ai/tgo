---
title: "Conformance: what tgo proves about accel, and the register of what accel cannot do"
status: drafted
layer: all
depends_on:
  - 000-decisions.md
---

# Conformance

This is the primary output of the project. [000 D1](000-decisions.md) makes tgo
accel's validating consumer; this spec is the machinery and the register.

## 1. Two directions

**Downward — does accel do what it says?** A real model exercises accel's
operators at real shapes with real weights, which no unit test in accel does.
Where tgo's host reference and accel's device result disagree, one of them is
wrong, and the finding goes to accel with the reproducer.

**Upward — what does accel not have?** Every place tgo cannot express something
gets a named test that skips with the reason and the owning accel spec number.
The suite prints them as a table. The table is the deliverable.

## 2. The register

Each entry: what tgo wanted, which accel spec owns it, what tgo does instead,
and what that costs. Kept in sync with the skipping tests, one per row.

Every row is filed upstream. The issue number is part of the row, so a reader
can follow the argument rather than take the cost on trust.

| # | what tgo cannot do | accel spec | filed | workaround | cost |
| --- | --- | --- | --- | --- | --- |
| C1 | batched paged attention from `tensor` | 010, 030 | [#1](https://github.com/golang-design/accel/issues/1) | one sequence per submission | no batching at all |
| C2 | RoPE at per-row positions | 010, 025 | [#2](https://github.com/golang-design/accel/issues/2) | scalar offset, single sequence | blocks C1 even once C1 exists |
| C3 | independent per-slot sampling draws | 028, 039 | [#3](https://github.com/golang-design/accel/issues/3) | host sampling, one sequence | blocks C1 |
| C4 | a paged KV cache (`pagetable` is internal) | 030 | [#1](https://github.com/golang-design/accel/issues/1) | one contiguous state per session | capacity reserved for the longest sequence; [005 §2](005-kv-cache.md) |
| C5 | an f16 KV cache (`Attention` requires f32) | 007, 010 | [#4](https://github.com/golang-design/accel/issues/4) | f32 | 2× the cache; 1.21 GB at 4k context for Qwen3-4B |
| C6 | penalties and temperature on device | 039 | [#6](https://github.com/golang-design/accel/issues/6) | host, before submission | a 608 KB logits readback per token |
| C7 | bf16 arithmetic (storage exists, no operator reads it) | 002, 010 | [#5](https://github.com/golang-design/accel/issues/5) | convert to f16 at load | the one inexact conversion in the pipeline; [001 §3](001-weights.md) |
| C8 | an f32 GEMM (`MatMul` requires f16 operands) | 010 | [#5](https://github.com/golang-design/accel/issues/5) | cast before every projection | 7 casts per layer, 252 dispatches for a 36-layer model |
| C9 | a strided view into `MatMul` | 025 | — | host-side transpose at load | [001 §4](001-weights.md); correct, and forecloses runtime reshaping |
| C10 | importing host memory as a device buffer | 001 | [#7](https://github.com/golang-design/accel/issues/7) | copy through a view | every weight copied twice on unified memory |

C9 is not filed: accel's refusal there is the *correct* one — a silently copied
strided view would hide a real cost — and the host-side transpose is the right
answer, not a workaround. It stays in the table because it constrains what tgo
can do at graph time, which is what this table is for.

**A row leaves this table only when its test stops skipping.** Nothing is
removed because it was worked around, and nothing is removed because an issue
was closed.

## 3. Numbers tgo reports back

Measured, not asserted, and re-measured each release:

- **CPU/Metal divergence.** Same prompt, greedy, both backends: the first token
  index at which they differ and the logit gap there. A bound nobody measured is
  not a bound.
- **Readback share of a decode step.** How much of the step is the $V$-wide
  logits transfer. This is the argument for C6 in one number.
- **Quantization error against the bound.** One layer's output at int8 against
  f16, checked against `quant.Int8ErrorBound` over the real blocks — not a
  tuned tolerance, and not synthetic weights.
- **Plan compile time** per bucket, and cache hit rate over a session.
- **Transient memory** from `Plan.Memory()`, against the hand-computed working
  set. accel's aliasing either helps by the amount it claims or it does not.

## 4. How the suite runs

Three tiers, mirroring accel's own:

| tier | needs | runs |
| --- | --- | --- |
| 1 | nothing | every push. Synthetic configs, CPU backend. |
| 2 | a Metal device | every push on macOS. `TGO_REQUIRE_METAL=1` turns a missing device into a failure, not a skip. |
| 3 | real weights | by hand, before a release. `TGO_MODEL=/path`. Never in CI. |

Tier 3's result is recorded in [011 §4](011-sequencing.md) with the date, the
checkpoint, and the numbers from §3.

## 5. The parity oracle

A pure-Go, host-side, f64 implementation of the whole forward pass. Slow,
obviously correct, no device. Every device result is checked against it.

**This is not a duplicate implementation to keep in sync.** It is written from
the model's mathematics — the equations in [004](004-model-graph.md) — rather
than from tgo's graph code, which is what makes disagreement meaningful. If both
were written from the same source, agreement would prove nothing.

Tolerances are stated per operator and derived: f32 accumulation over $K$ terms
carries a relative error of order $\sqrt{K}\,\varepsilon_{32}$, and the
quantized path adds `Int8ErrorBound`. A tolerance that had to be raised to make
a test pass is a finding, not a fix.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 010-D1 | one skipping test per register row | a prose list | the table cannot go stale silently |
| 010-D2 | the oracle is written from the mathematics, not from tgo's graph | share code with the builder | agreement is evidence |
| 010-D3 | tolerances are derived; a raised tolerance is a finding | tune until green | a numerics regression cannot be absorbed |
| 010-D4 | tier 3 never runs in CI | a nightly with a download | CI stays under a minute and offline |
