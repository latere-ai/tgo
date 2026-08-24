---
title: "Conformance: the register of what accel cannot do, and the oracle that proves what it can"
status: drafted
layer: all
depends_on:
  - 000-decisions.md
---

# Conformance

This is the primary output of the project. [000 D1](000-decisions.md) makes tgo
accel's validating consumer; this spec is the machinery, the register, and the
evidence.

## 1. Two directions

**Downward — does accel do what it says?** A real model exercises accel's
operators at real shapes with real weights, which no unit test inside accel
does. A 2560-wide RMSNorm over a tensor whose values span six orders of
magnitude is not the same test as a 64-wide one over seeded noise. Where tgo's
host oracle and accel's device result disagree, one of them is wrong, and the
finding goes upstream with a reproducer.

**Upward — what does accel not have?** Every place tgo cannot express something
gets a named test that skips with the reason and the owning accel spec. The
suite prints them as a table. **The table is the deliverable**, and §2 is it.

## 2. The register

| # | what tgo cannot do | accel spec | filed | state | workaround, and what it costs |
| --- | --- | --- | --- | --- | --- |
| C1 | batched attention from `tensor` | 010, 030, **043** | [#1](https://github.com/golang-design/accel/issues/1) | **designed** | one sequence per submission; throughput is $1/B$ of the hardware ([008 §1](008-scheduler.md)) |
| C2 | RoPE at per-row positions | 025, **043** | [#2](https://github.com/golang-design/accel/issues/2) | **landing** | *was* a scalar offset; accel's tree now has `RoPE(…, positions *Tensor)` |
| C3 | independent per-slot sampling draws | 028, 039, **043** | [#3](https://github.com/golang-design/accel/issues/3) | **designed** | host sampling, one sequence |
| C4 | a paged KV cache (`pagetable` is internal) | 030, **043** | [#1](https://github.com/golang-design/accel/issues/1) | **designed** | contiguous per session; **322×** the memory for ten short chats ([005 §3](005-kv-cache.md)) |
| C5 | an f16 KV cache (`Attention` requires f32) | 007, 010, **043 §5** | [#4](https://github.com/golang-design/accel/issues/4) | **designed** | f32; 2× the cache, 1.21 GB at 4k for Qwen3-4B |
| C6 | penalties and temperature on device | 039 | [#6](https://github.com/golang-design/accel/issues/6) | open | host; a 608 KB logits readback per token |
| C7 | bf16 arithmetic (storage exists, no operator reads it) | 002, 010 | [#5](https://github.com/golang-design/accel/issues/5) | open | convert to f16 at load; the one inexact step in the pipeline ([001 §3](001-weights.md)) |
| C8 | an f32 GEMM (`MatMul` requires f16 operands) | 010, **043 §5** | [#5](https://github.com/golang-design/accel/issues/5) | **designed** | cast before every projection; 7 per layer, 252 dispatches per forward pass |
| C9 | a strided view into `MatMul` | 025 | — | won't fix, correctly | host-side transpose at load ([001 §4](001-weights.md)) |
| C10 | importing host memory as a device buffer | 001 | [#7](https://github.com/golang-design/accel/issues/7) | open | copy through a view; every weight copied twice on unified memory |
| **C11** | **a KV cache longer than 128 positions** | 007, 010 | [#8](https://github.com/golang-design/accel/issues/8) | **open — blocking** | none. `Attention` refuses $C > 128$; see [005 §2.3](005-kv-cache.md) |
| C12 | binding a `LayerState` view to `Attention` | 007, 030 | [#9](https://github.com/golang-design/accel/issues/9) | open | one state per layer: 72 states, ports and bindings for a 36-layer model |

**States.** `landing` — the signature has changed in accel's tree.
`designed` — accel
[043](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md)
specifies it and the code does not do it yet. `open` — filed, not yet designed.
`won't fix, correctly` — see below.

C9 is not filed and should not be. accel refusing a strided view into `MatMul`
is the **correct** refusal: silently copying one would hide a real cost behind an
operator that looks free. The host-side transpose is the right answer, not a
workaround. It stays in the table because it constrains what tgo can do at graph
time, which is what this table is for.

**C11 is different in kind from every other row.** The rest are costs — tgo runs
more slowly, or uses more memory, or emits more dispatches. C11 has no
workaround and no degraded mode: `Attention` refuses a cache large enough to
hold a system prompt, so there is no context length at which a model is
servable. Everything downstream of [011 M6](011-sequencing.md) is gated on it,
including every number in §3, because a measurement taken at 128 positions
describes nothing.

It is listed last because it was found last. It is first in priority.

**A row leaves this table only when its test stops skipping.** Not when an issue
closes, not when a spec is written, and never because it was worked around.

### 2.1 What the register is worth so far

Nine reports so far. Seven produced one upstream design decision — accel 043's *a scalar is a
value every row shares; a value that differs per row is a tensor* — which
removes surface rather than adding it, and which was reached by five of the
rows above being **the same mistake seen five times**. That is the argument for
a validating consumer: no single one of C1–C5 looks like a design decision from
inside accel, and together they are one.

C11 is the second argument, and a different one. It is not a subtle design
tension — it is a hard refusal with an honest error message, sitting in a
library whose own tests all pass, because no test inside accel asks for a cache
longer than a workgroup. **A gap can be fully documented, correctly refused, and
still invisible, until something tries to do the real job.** That is what a
validating consumer is for, and it is worth more than the five rows that shared
one cause.

## 3. Numbers tgo reports back

Measured, not asserted, and re-measured each release. Each of these is a
question accel cannot answer about itself.

| measurement | why accel cannot self-report it | what it decides |
| --- | --- | --- |
| **CPU/Metal divergence** — greedy, same prompt: the first differing token index and the logit gap there | needs a real model long enough to accumulate reduction-order differences | whether "same result on both backends" is a claim tgo can make |
| **readback share of a decode step** | needs a $V = 151936$ vocabulary | the size of C6 in one number |
| **quantization error against `Int8ErrorBound`** on real blocks | needs trained weights; synthetic ones have no outliers, and the bound is driven by the largest weight in a block | whether int8 is usable, and where |
| **plan compile time per bucket**, and cache hit rate over a session | needs a real graph of ~500 nodes | whether [007-D2](007-engine.md)'s bucket set is right |
| **transient bytes** from `Plan.Memory()` vs. the hand-computed working set | needs a graph with real lifetime structure | whether accel's aliasing helps by the amount it claims |

The third deserves a note. `quant.Int8Quantize` scales a block of 32 by its
largest magnitude, so the error a weight suffers is proportional to the largest
weight *in its block*, not to its own. Synthetic weights drawn from one
distribution have no outliers and therefore flatter the scheme. Trained
transformer weights have outlier channels — this is well documented and is the
whole reason mixed-precision schemes exist — so a bound measured on real blocks
is a different number from one measured on noise, and it is the only one worth
reporting.

## 4. How the suite runs

| tier | needs | when | on failure |
| --- | --- | --- | --- |
| 1 | nothing | every push | red |
| 2 | a Metal device | every push on macOS, `TGO_REQUIRE_METAL=1` | red — a missing device is a **failure**, not a skip |
| 3 | real weights, `TGO_MODEL=/path` | by hand, before a release | blocks the release |

Tier 2's environment variable is the mechanism accel uses and the reason is the
same: a job that promises a backend and skips when it finds no device is a job
that rots green.

Tier 3 is never in CI. [000 D8](000-decisions.md) — the smallest Qwen3 is over a
gigabyte, and a CI that downloads one is a CI nobody runs locally. Its result is
recorded in [011 §4](011-sequencing.md) with the date, the checkpoint, and §3's
numbers.

## 5. The parity oracle

A pure-Go, host-side, **float64** implementation of the whole forward pass.
Slow, obviously correct, no device, no accel. Every device result is checked
against it.

**It is not a duplicate implementation to keep in sync.** It is written from the
model's mathematics — the equations in [004](004-model-graph.md) — rather than
from tgo's graph code. That is the entire point: if both were written from the
same source, agreement would prove only that the source was copied correctly.
Two independent derivations of the same mathematics agreeing is evidence; one
derivation compared against itself is not.

The practical rules that keep it independent:

- it imports nothing from tgo's `nn` or `model` packages;
- it takes weights as `[]float64` and shapes as integers, not as tgo types;
- it is written by reading the spec, not the code;
- when it disagrees with the device, **the oracle is presumed right** until
  shown otherwise, because it is the simpler program.

### 5.1 Tolerances are derived, not tuned

f32 accumulation over $K$ terms carries a relative error on the order of

$$\varepsilon_{\text{acc}} \sim \sqrt{K}\,\varepsilon_{32}, \qquad \varepsilon_{32} = 2^{-24} \approx 6\times10^{-8}$$

for a well-conditioned sum with random signs — pairwise or blocked summation, as
a tiled GEMM does, is closer to $\sqrt{\log K}$, so $\sqrt{K}$ is a safe upper
bound. For $K = 2560$ that is about $3\times10^{-6}$ relative.

Each stage adds its own term:

| stage | added error |
| --- | --- |
| f32 GEMM over $K$ | $\sqrt{K}\,\varepsilon_{32}$ |
| f16 operand storage | $\varepsilon_{16} = 2^{-11} \approx 4.9\times10^{-4}$, relative, per operand |
| int8 weights | `quant.Int8ErrorBound`, driven by the per-block maximum |
| softmax | benign; the max-subtraction makes it stable |

The f16 operand term dominates the f32 accumulation term by three orders of
magnitude, which is worth stating plainly: **the tolerance on a matmul is set by
the storage format, not by the accumulator.** That is also why C8 matters less
for accuracy than it does for bandwidth.

**A tolerance that had to be raised to make a test pass is a finding, not a
fix.** The rule is enforced socially and by a comment on every tolerance
constant naming which term above produced it. A constant with no derivation is
a bug report waiting to be written.

## 6. Reporting

The suite emits the §2 register and the §3 numbers as a generated Markdown
document, so the table in this spec is produced from the tests rather than
maintained beside them. A hand-maintained register drifts within one milestone;
that is the same failure this project exists to catch in accel.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 010-D1 | one skipping test per register row | a prose list | the table cannot go stale silently |
| 010-D2 | the oracle is written from the mathematics, not from tgo's graph | share code with the builder | agreement becomes evidence rather than tautology |
| 010-D3 | tolerances are derived and commented with their term; a raised tolerance is a finding | tune until green | a numerics regression cannot be absorbed |
| 010-D4 | tier 3 never runs in CI | a nightly with a download | CI stays offline and under a minute |
| 010-D5 | the oracle is float64 and presumed right on disagreement | float32, matching the device | it is the simpler program; matching the device would import the device's bugs |
| 010-D6 | the register is generated from the tests | maintained by hand beside them | it is the exact drift tgo exists to catch upstream |
