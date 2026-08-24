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
| M6 | KV cache + decode loop ([005](005-kv-cache.md), [007](007-engine.md)) | prefill-then-decode equals token-by-token; padded prefill leaves the cache clean |
| M7 | sampling ([006](006-sampling.md)) | order tests pass; stream reproducibility holds across a policy change |
| M8 | CLI | `tgo run` generates from a local checkpoint |
| M9 | server ([009](009-server.md)) | handler suite against the fake engine; one real end-to-end |
| M10 | conformance report ([010](010-conformance.md)) | the register table is generated from the tests; §3 numbers measured |
| M11 | real weights | a Qwen3 dense checkpoint is coherent at f16 and int8, on both backends |

M11 is the gate in [000](000-decisions.md)'s "what v0 is". Everything before it
runs on synthetic configs.

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

*(No entries yet. M0 in progress.)*
