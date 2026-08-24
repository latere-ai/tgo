---
title: "Sequencing: the order the work is done in, and what actually landed"
status: living
layer: all
depends_on:
  - 000-decisions.md
---

# Sequencing

The other specs say what. This one says **when**, and afterwards, **what really
happened**. It is the only file here that is edited after its subject ships.

## 1. The rule that fixes the order

Nothing that can be wrong silently is built on top of something unverified. So
the order is: the pieces with a checkable ground truth first (a tokenizer has
fixed vectors; a template has a golden string), then numerics against an oracle,
then the loop, then the surfaces.

Writing the server early is the tempting mistake. A `framework` request invites
an API surface, and an API over logits that are subtly wrong is a working demo
of a broken model.

## 2. Milestones

| M | scope | done means |
| --- | --- | --- |
| M0 | module, CI, spec tree, docs | CI green on an empty build; every gate from accel's `ci.yml` present |
| M1 | tokenizer ([002](002-tokenizer.md)) | fixed vectors pass; fuzz clean; streaming equals batch |
| M2 | chat template ([003](003-chat-template.md)) | goldens match; the injection case is structural |
| M3 | safetensors + conversion ([001](001-weights.md)) | every §6 refusal is a test; bf16→f16 saturation counted |
| M4 | `nn` blocks + oracle ([004](004-model-graph.md), [010 §5](010-conformance.md)) | each block matches the f64 oracle within a derived tolerance, both backends |
| M5 | Qwen3 forward pass | a synthetic 2-layer config produces logits matching the oracle |
| M6 | KV cache + decode loop ([005](005-kv-cache.md), [007](007-engine.md)) | prefill-then-decode equals token-by-token; padded prefill leaves the cache clean. **Capped at 128 positions until [010 C11](010-conformance.md) closes** |
| M7 | sampling ([006](006-sampling.md)) | order tests pass; stream reproducibility holds across a policy change |
| M8 | CLI | `tgo run` generates from a local checkpoint |
| M9 | server ([009](009-server.md)) | handler suite against the fake engine; one golden per dialect; one real end-to-end |
| M10 | conformance report ([010](010-conformance.md)) | the register table is generated from the tests; §3 numbers measured |
| M11 | real weights | a Qwen3 dense checkpoint is coherent at f16 and int8, on both backends |

M11 is the gate in [000](000-decisions.md)'s "what v0 is". Everything before it
runs on synthetic configs.

### 2.1 What is gated upstream, and what is not

```mermaid
flowchart LR
  subgraph free["unblocked — 90% of the coverage gate lives here"]
    M1["M1 tokenizer"] --> M2["M2 templates"] --> M3["M3 loader"]
    M3 --> M4["M4 nn + oracle"] --> M5["M5 forward pass"]
  end
  M5 --> M6["M6 decode loop<br/>≤128 positions"]
  M6 --> M7["M7 sampling"] --> M8["M8 CLI"] --> M9["M9 server"]
  M9 --> M10["M10 report"] --> M11["M11 real weights"]
  C11["accel#8<br/>128-position cap"] -.blocks.-> M11
  C11 -.caps.-> M6
```

M1–M5 are entirely unblocked and are where the work goes now. They are also
where [000 D8](000-decisions.md) puts the device-free packages, so they carry
almost all of the coverage gate: a tree that reaches M5 with the gate green is a
tree whose remaining risk is concentrated in the parts a device decides.

M6 through M10 are *buildable* at 128 positions — the loop, the sampler, the CLI
and the server all work; they just cannot be pointed at a real conversation. So
they are not blocked, they are **unmeasurable**: every number in
[010 §3](010-conformance.md) taken at 128 positions describes nothing. M11 is
blocked outright.

## 3. What is deliberately not in v0

| | why |
| --- | --- |
| continuous batching | [008](008-scheduler.md); blocked on three accel gaps |
| paged KV, prefix reuse | [010](010-conformance.md) C4; the pool is unexported in accel |
| structured output | [015](015-structured-output.md); real work, and it is after batching |
| GGUF | [012](012-gguf.md); accel has no super-block kernel |
| MoE, hybrid/linear attention | Qwen3.5-class architectures; [004](004-model-graph.md)'s registry makes them additive |
| LoRA, speculative decoding, multi-device | not blocked, not v0 |

## 4. Outcomes

Appended as each milestone lands: what shipped, what deviated from the spec and
why, and for M11 the checkpoint, the date, and the
[010 §3](010-conformance.md) numbers.

### 2026-08-24 — M0 — **done**

Module, CI, and the spec tree. CI mirrors accel's gates — build, vet, test,
race, cgo-free, cross-compile, gofmt — with two additions: a **per-package**
coverage floor rather than a repository average, and `speclint`, which checks
frontmatter shape, that dependency edges resolve, that the tree is acyclic, that
a `blocked` spec names what blocks it, that every spec carries a decision
record, and that the index cannot go stale.

**Deviation from plan:** none in scope, but the tree was written twice. It was
first written against accel's signatures as they stood, and then reconciled
against accel
[043](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md),
which landed mid-drafting in answer to seven reports tgo filed. Five register
rows changed state in one commit.

**Findings, which are the actual output of M0:** nine issues on accel. Seven
before the reconciliation, two after — and the two found last are the ones that
matter most. [accel#8](https://github.com/golang-design/accel/issues/8) caps the
KV cache at 128 positions, which is the only finding so far with no workaround.
[accel#9](https://github.com/golang-design/accel/issues/9) refuses a
`LayerState` view, which corrected a decision this tree had already recorded
([005-D1](005-kv-cache.md)).

That last point is worth keeping. **A spec written against a library's
documentation was wrong about that library within a day.** Reading the
signatures is not the same as reading the refusals, and the refusals are where
the design actually lives.

**Two gates found their own bugs before any product code existed**, which is the
argument for building them first:

- The coverage gate reported *"every package at or above 90%"* over a tree whose
  only measured package was the one exempt from the check. It now fails when no
  non-exempt package was measured, and `speclint` moved from a test-only package
  into real code so the floor has something to stand on — 97.9%, measured.
- `speclint`'s negative tests immediately found a bug in its own citation regex,
  which captured the digits without the `C` and would have reported every valid
  register citation as dangling.
- CI's Windows row found that a CRLF checkout made **every spec in the tree**
  fail to parse. Nothing local would have.

CI is green on both workflows: `ci` across Linux, macOS and Windows with the
race detector, the fuzz seed corpus, the per-package coverage floor, the
cgo-free grep, ten cross-compile targets, gofmt and `speclint`; and `ci-metal`
on Apple silicon.

**M1 is next and is unblocked.**
