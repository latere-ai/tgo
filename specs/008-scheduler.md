---
title: "Continuous batching: slots, admission, and the upstream change it now waits on"
status: drafted
layer: engine
depends_on:
  - 000-decisions.md
  - 005-kv-cache.md
  - 007-engine.md
---

# Continuous batching

**Unblocked on 2026-08-24.** This spec carried `status: blocked` from the day it
was written. accel closed the last gap — `Attention` takes a batch axis — and a
value test confirms it: two sequences of lengths 96 and 32 batched together match
two single-sequence runs to `0.00e+00`, on *the batched paged decode kernel*.

Nothing in this spec is implemented. What changed is that it can be.

**The shape accel settled on**, which §2's slots must match:

```go
q       [batch, qSeq, qHeads, headDim]   // qSeq is 1 for a decode
Lengths [batch]                          // per sequence
Pages   [batch, maxPages]                // one row of block ids per sequence
```

A batch **requires** `Pages`: members have different lengths, so a contiguous
cache cannot address them. That is not a restriction to work around — it is the
reason paging exists, and it means [016](016-prefix-cache.md)'s block pool is a
prerequisite for this spec rather than a neighbour of it.

**One shape is still per-dispatch:** `qSeq`. A batched step takes one token per
sequence, so a *batched prefill* is not expressible and a prefill cannot share a
dispatch with a decode. That is [C16](010-conformance.md), and §5 is where it
costs something.

## 1. Why it is the thing worth having

A decode step is memory-bound: it reads **every weight** to produce **one
token**. For a 4B model at int8 that is 4 GB of traffic per token. Two sequences
decoding together read those 4 GB **once** and produce two tokens.

Let $W$ be the weight bytes, $A$ the per-sequence activation and cache traffic,
and $\beta$ the achievable bandwidth. One step at batch size $B$ costs

$$t(B) \approx \frac{W + B\cdot A}{\beta}, \qquad \text{throughput} = \frac{B}{t(B)} = \frac{B\beta}{W + BA}$$

Throughput is close to linear in $B$ until $BA \sim W$, and the ceiling is

$$\frac{t(1)}{t(\infty)} = \frac{W}{A} + 1$$

**$A$ is not small, and this is where an earlier draft was wrong.** Reading the
cache dominates it, and [005 §3](005-kv-cache.md) gives 288 KiB per position, so
$A \approx L \cdot 288\text{ KiB}$ for context length $L$:

| context $L$ | $A$ | crossover $B$ | ceiling |
| --- | --- | --- | --- |
| 1024 | 295 MB | ≈ 13 | ~15× |
| 2048 | 590 MB | ≈ 7 | ~7× |
| 4096 | 1.21 GB | ≈ 3.4 | ~4.3× |

So the crossover is **inside** the range a single-device server sees, not far
above it, and **the ceiling falls as context grows.** Batching is still most of
the hardware at short contexts and much less at long ones.

That is a second argument for [C5](010-conformance.md), the f16 cache: halving
$A$ doubles the batch size worth reaching. It is also an argument for
[016](016-prefix-cache.md), which reduces the prefill work that batching does not
help with at all.

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

## 6. Prefix reuse is no longer downstream of this spec

sglang's RadixAttention keys a trie on token prefixes and shares the KV blocks
beneath a common prefix. This section previously said prefix reuse was
downstream of gap 4 — the unreachable page table — and recorded that as the
point, since it is otherwise natural to plan prefix caching as a tgo feature and
discover late that it is an accel one.

**That dependency is discharged.** `AttentionOptions.Pages` landed on
2026-08-24, so prefix reuse is buildable and it is [016](016-prefix-cache.md).

Two things the move clarified, both of which belong here rather than there:

- **It does not depend on batching.** A single-sequence server benefits from
  every request after the first, and a multi-turn conversation benefits from
  every turn after the first. So [016](016-prefix-cache.md) is not blocked on
  this spec, and is the larger win of the two for an agent workload.
- **What it would want from a scheduler** is the one thing [016-D3](016-prefix-cache.md)
  gave up in choosing a hash map over a trie: *which waiting requests share a
  prefix with this one*, so admission can group them. If this spec ever wants
  that query, it is the argument for the trie, and 016 says so.

## 7. Four hooks that are cheap now and expensive later

tgo keeps these live in v0 at a cost of nothing, so that when [C1](010-conformance.md)
closes the retrofit stays local:

| hook | v0 shape | why it matters later |
| --- | --- | --- |
| the cache is addressed through a **`Session`** | one session, one cache | a page table replaces the addressing without touching callers |
| positions are a **bound tensor** | a one-row tensor | already true; widening to B rows is a shape change, not a rewrite |
| **one draw consumed per step per sequence** | one sequence | [006-D2](006-sampling.md); per-slot draws become a shape change, not a semantic one |
| `Generate` streams over a **channel** internally | one stream | a scheduler drives many without the public API moving ([007-D6](007-engine.md)) |

The first two were free because accel 043 moved per-row values onto tensors
before tgo wrote any code. The third and fourth are decisions tgo made for
reasons that stand on their own, and which happen to survive batching.

## 8. What tgo does now

Holds one named, skipping test per [010](010-conformance.md) register row, each
naming the accel spec that owns it. When [C1](010-conformance.md) closes, those
tests stop skipping and this spec moves from `blocked` to `drafted`.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 008-D1 | fixed $B_{\max}$; membership is contents | resize the batch per step | membership changes are free; a size change costs a drain. accel 040's `Batch` trap makes shrinking actively wrong |
| 008-D2 | recompute on preemption | swap KV to host | no host mirror; prefill is parallel, a swap is two serial transfers |
| 008-D3 | admission reserves answer space | admit on a free slot alone | a full pool cannot deadlock; rejections are reported |
| 008-D4 | keep §7's four hooks live in v0 | build them when 043 lands | v0 pays nothing; the retrofit stays a binding change |
| 008-D5 | last-arrived-first-evicted | round-robin or longest-running | bounds the latency of sequences already in flight rather than spreading the damage |
| 008-D6 | blocked on 043's *implementation*, not its design | build against the designed signatures now | accel 043 is drafted and partly in flight; building against unlanded signatures is building against nothing |
