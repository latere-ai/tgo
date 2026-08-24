---
title: "Continuous batching: slots, admission, and the upstream change it now waits on"
status: blocked
layer: engine
depends_on:
  - 000-decisions.md
  - 005-kv-cache.md
  - 007-engine.md
blocked_on:
  - "accel specs/043-per-row-values.md (designed; implementation in flight)"
---

# Continuous batching

**Status: blocked, and the reason changed on 2026-08-24.**

It was blocked because accel had no design for per-row values. tgo filed four
reports; accel
[043](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md)
answered all four with one decision, and `RoPE` has already changed. So the
design is no longer the blocker — **the code is**, and this spec now waits on
043's implementation rather than on its absence.

Nothing in this repository implements batching. This spec exists so that the
parts which are cheap today are not made expensive by accident, and so that when
043 lands the work is a binding change rather than a redesign.

## 1. Why it is the thing worth having

A decode step is memory-bound: it reads **every weight** to produce **one
token**. For a 4B model at int8 that is 4 GB of traffic per token. Two sequences
decoding together read those 4 GB **once** and produce two tokens.

Let $W$ be the weight bytes, $A$ the per-sequence activation and cache traffic,
and $\beta$ the achievable bandwidth. One step at batch size $B$ costs

$$t(B) \approx \frac{W + B\cdot A}{\beta}, \qquad \text{throughput} = \frac{B}{t(B)} = \frac{B\beta}{W + BA}$$

Since $A \ll W$ at realistic context lengths, throughput is close to linear in
$B$ until $BA \sim W$. For the numbers above — $W = 4$ GB, $A$ on the order of
tens of MB — that crossover is far above the batch sizes a single-device server
sees. **Batching is not an optimisation at the margin; it is most of the
hardware.**

Static batching gives it back: a batch that starts together finishes together,
so a 20-token answer holds its slot until the 800-token answer beside it is
done. Continuous batching — vLLM's contribution — admits a new sequence into a
slot the moment one leaves.

## 2. Slots: membership is contents, batch size is structure

A **slot** is an index in $[0, B_{\max})$. It owns a row of `q`, a row of `out`,
a row of `positions`, an entry in `lengths`, and a page-table row. The KV blocks
are addressed *through* the page table, so they never move — which is what makes
a slot swap free.

Against accel 003's four kinds of variation:

| change | which variation | cost |
| --- | --- | --- |
| a sequence leaves, another takes its slot | **contents**: `lengths`, the page-table row, `positions`, the scatter id | free; no rebind, no recompile |
| a sequence grows by a block | **contents**: one page-table entry | free |
| the batch gains or loses a *member count* | none of the four — batch is a leading dimension on every port | a different plan, and a drain |

**The design never changes $B_{\max}$.** Idle slots are parked on a zero-length
sequence and cost one wasted row of arithmetic. accel 040 records the trap
precisely: `BatchedDims.Batch` is declared and never read, so a scheduler that
"shrinks the batch" by writing it runs the dropped slot anyway — and that slot's
`out` row still holds what the previous step left there, so the model emits a
**repeat of the last token** rather than obvious garbage.

## 3. Admission

A request is admitted when a slot is free **and** the block pool can hold its
prompt plus a reserve for the answer.

$$\text{blocks needed} = \left\lceil \frac{T_\text{prompt} + R}{B_\text{block}} \right\rceil$$

with $R$ the reserve. Admitting on a free slot alone is how a server deadlocks:
every slot occupied, the pool empty, and no sequence able to grow — so nothing
finishes and nothing can be evicted into progress.

$R$ is a policy number, not a derived one. Setting it to the request's
`max_tokens` is safe and admits too little; setting it to zero maximises
occupancy and deadlocks. The design takes $R$ as configuration with a documented
default and **reports rejected admissions**, because a server that quietly
admits fewer requests than it could is indistinguishable from a slow one.

## 4. Eviction: recompute, not swap

A preempted sequence drops its blocks and re-prefills on readmission.

| | cost |
| --- | --- |
| recompute | one prefill of $T$ tokens, on readmission |
| swap to host | $T \cdot 288$ KB out and back, plus host memory for every swapped sequence |

Recompute wins because prefill is compute-bound and parallel over $T$, while a
swap is two serial transfers over a bus, and because it needs no host mirror of
the cache. vLLM supports both and recommends recompute for exactly this reason.

**Victim choice is last-arrived-first-evicted**, which bounds the worst-case
latency of the sequences already in flight rather than spreading the damage.

## 5. Chunked prefill

A prefill and a decode step cannot share a dispatch — different shapes, different
kernels. So a long prompt either runs alone, stalling every decoding sequence
for the length of its forward pass, or is split.

Splitting a 8192-token prompt into chunks of $c$ means the decode steps beside
it wait at most one chunk's forward pass. The chunk size trades prefill
efficiency (larger $c$ is a better GEMM) against decode latency (smaller $c$ is
a shorter stall), and the useful range is 512–2048.

Chunked prefill needs a prefill whose first query token is not position 0, so
its causal mask hides the right thing. accel's `BaseName` scalar does that
today; 043 turns it into `AttentionOptions.Positions`. Either way the mechanism
exists — chunked prefill is the one part of this spec that is **not** blocked,
and it is still not built, because a scheduler with one sequence has nothing to
stall.

## 6. Prefix reuse depends on paging, not on policy

sglang's RadixAttention keys a trie on token prefixes and shares the KV blocks
beneath a common prefix. Two requests with the same 900-token system prompt
prefill it once.

**Sharing blocks needs blocks.** Over [005 §2.1](005-kv-cache.md)'s contiguous
per-session cache the only way to share a prefix is to copy it, which costs most
of what the sharing saves. So prefix reuse is downstream of 043's `Pages`
binding, and recording that is the point: it is otherwise natural to plan prefix
caching as a tgo feature and discover late that it is an accel one.

The refcounting it needs — a block freed only when the last sequence referencing
it leaves — is tgo's, and it is the reason the trie is not simply a cache with
an LRU.

## 7. What is cheap now and would be expensive later

The four hooks tgo keeps live in v0, at a cost of nothing:

| hook | v0 shape | why |
| --- | --- | --- |
| the cache is addressed through a `Session` | one session, one cache | a page table replaces the addressing without touching callers |
| positions are a **bound tensor**, not a scalar | a one-row tensor | already true as of accel's current tree; widening is a shape change |
| one draw consumed per step per sequence | one sequence | [006-D2](006-sampling.md); per-slot draws become a shape change, not a semantic one |
| `Generate` streams over a channel | one stream | a scheduler drives many without the API moving |

## 8. What tgo does now

Holds one named, skipping test per [010](010-conformance.md) register row, each
naming the accel spec that owns it. When 043's implementation lands, those tests
stop skipping and this spec moves from `blocked` to `drafted`.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 008-D1 | fixed $B_{\max}$; membership is contents | resize the batch per step | membership changes are free; a size change costs a drain. accel 040's `Batch` trap makes shrinking actively wrong |
| 008-D2 | recompute on preemption | swap KV to host | no host mirror; prefill is parallel, a swap is two serial transfers |
| 008-D3 | admission reserves answer space | admit on a free slot alone | a full pool cannot deadlock; rejections are reported |
| 008-D4 | keep §7's four hooks live in v0 | build them when 043 lands | v0 pays nothing; the retrofit stays a binding change |
| 008-D5 | last-arrived-first-evicted | round-robin or longest-running | bounds the latency of sequences already in flight rather than spreading the damage |
| 008-D6 | blocked on 043's *implementation*, not its design | build against the designed signatures now | accel 043 is drafted and partly in flight; building against unlanded signatures is building against nothing |
