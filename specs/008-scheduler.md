---
title: "Continuous batching: the design, and the three accel gaps that block it"
status: blocked
layer: engine
depends_on:
  - 000-decisions.md
  - 005-kv-cache.md
  - 007-engine.md
blocked_on:
  - "accel specs/040-batch-scheduler.md"
  - "accel specs/010-kernel-corpus.md"
  - "accel specs/039-sampling-policy.md"
---

# Continuous batching

**Status: blocked, deliberately.** This spec exists so that the parts which are
cheap today are not made expensive by accident, and so the gaps have a written
argument attached to them. No code in this repository implements it.

## 1. Why it is the thing worth having

A decode step is memory-bound: it reads every weight to produce one token. Two
sequences decoding together read the weights **once** and produce two tokens.
Throughput scales close to linearly in batch size until the arithmetic becomes
the limit, which for a 4B model at int8 is far above the batch sizes a
single-device server sees.

Static batching wastes it back: a batch that starts together finishes together,
so a 20-token answer occupies a slot until the 800-token answer beside it is
done. Continuous batching — vLLM's contribution — admits a new sequence into a
slot the moment one leaves.

## 2. The design, stated for when it is buildable

**Slots.** A fixed $[0, \text{maxBatch})$. A slot owns a row of the q tensor, a
row of the output, a page-table row, and an entry in `lengths`. Membership
changes are *contents* — rebind, no recompile. The batch **size** is structure:
it is a leading dimension on every port, so changing it is a different plan and
costs a drain. **The design never changes size.** Idle slots are parked on a
zero-length sequence and cost one wasted row.

**Admission.** A request is admitted when a slot is free *and* the KV pool can
hold its prompt plus a reserve for the answer. Admitting without the reserve is
how a server deadlocks: every slot full, none able to grow.

**Eviction.** Recompute rather than swap. A preempted sequence drops its blocks
and re-prefills on readmission, which costs $O(T)$ once against a swap's
$O(T)$ transfer both ways plus the host memory.

**Chunked prefill.** A long prompt is split so a prefill never starves the
decode steps sharing its step. Without it, one 8k prompt stalls every other
sequence for the length of its forward pass.

## 3. The three gaps, and why each is silent

Every one of these produces output that is well-shaped, finite, and plausible.
That is what makes them worth writing down rather than discovering.

### Gap 1 — batched paged attention is not at the tensor layer

`tensor.Attention` selects the decode and prefill kernels only. The batched and
paged kernels exist under accel's `internal/testkernels` and nothing in
`tensor/` references them. There is no exported operator that binds `pages` and
`lengths`.

**Owner:** accel 010 (registry), 030 (paging).
**Consequence if ignored:** there is no batched attention to call. This one is
not silent; it is simply absent. It is listed first because the other two only
matter once it exists.

### Gap 2 — `RoPE` conflates the row index with the position

`RoPE` computes $\text{pos} = r + \texttt{Offset}$ with `Offset` a scalar. In a
batched decode $r$ is the **slot index**, so slot 0 rotates at `Offset`, slot 1
at `Offset+1`, and exactly one member is rotated at its own cache length.

**Owner:** accel 010, 025.
**What it needs:** positions as a tensor binding, one entry per row.
**Consequence if ignored:** every member but one attends with wrong positional
phase. Output stays fluent and loses long-range coherence.

### Gap 3 — one `Draw` shared across the batch

`SampleDims.Draw` is a scalar in the uniform block, and `SampleCategorical` is
`workgroup=1` writing `out[0]`. Widening the scalar keeps accel 028's
reproducibility and destroys independence.

**Owner:** accel 028, 039.
**What it needs:** one row and one independent draw per slot.
**Consequence if ignored:** two sequences with similar distributions emit the
same token. Every existing test still passes.

### And a fourth, from this repository

**The block pool is unexported.** `tensor/internal/pagetable` is internal
because no exported operator takes a page table. Without it there is no paged
cache, so §2's slots have nothing to address through, and the cache stays
contiguous with the cost in [005 §2](005-kv-cache.md).

**Owner:** accel 030.

## 4. What tgo does now

Holds one named, skipping test per gap in [010](010-conformance.md), each
naming the accel spec that owns it. When a gap closes, its test stops skipping
and this spec moves from `blocked` to `drafted`.

## 5. What is cheap now and would be expensive later

Recorded so the shape is not foreclosed:

- The KV cache is addressed through a **`Session`** rather than by a global
  offset, so a page table can replace the addressing without touching callers.
- The RoPE offset is a **named scalar** on the plan, so it becomes a bound
  tensor without changing the builder's structure.
- The sampler consumes **one draw per step per sequence** already
  ([006-D2](006-sampling.md)), so widening to per-slot draws is a shape change,
  not a semantic one.
- `Generate` streams over a **channel**, so a scheduler can drive many of them
  without the API changing.

## 6. Prefix reuse depends on paging, not on policy

sglang's RadixAttention keys a trie on token prefixes and shares the KV blocks
beneath. Sharing blocks needs blocks. Over [005](005-kv-cache.md)'s contiguous
per-session cache the only way to share a prefix is to copy it, which costs most
of what the sharing saves. So prefix reuse is downstream of gap 4, and it is
recorded here rather than planned as a tgo feature.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 008-D1 | fixed batch size, membership is contents | resize the batch per step | membership changes are free; size changes cost a drain |
| 008-D2 | recompute on preemption | swap KV to host | no host KV mirror; $O(T)$ once |
| 008-D3 | admission reserves answer space | admit on free slot alone | a full pool cannot deadlock |
| 008-D4 | keep the four shape hooks in §5 live in v0 | build them when the gaps close | v0 costs nothing for them; the retrofit stays local |
