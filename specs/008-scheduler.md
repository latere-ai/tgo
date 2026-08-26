---
title: "Continuous batching: slots, admission, and the ragged step that makes a mixed dispatch possible"
status: drafted
layer: engine
depends_on:
  - 000-decisions.md
  - 005-kv-cache.md
  - 007-engine.md
---

# Continuous batching

**Fully unblocked upstream on 2026-08-26.** This spec carried `status: blocked`
from the day it was written. Two gaps closed in sequence: `Attention` took a
batch axis ([C1](010-conformance.md)), and then a *ragged* one
([C16](010-conformance.md)).

Nothing in this spec is implemented. What changed is that all of it can be.

**Two shapes accel offers, and this spec builds on the second.**

```go
// rectangular: every sequence contributes the same number of tokens
q       [batch, qSeq, qHeads, headDim]
Lengths [batch]
Pages   [batch, maxPages]

// ragged: each sequence contributes its own number, and they are end to end
q            [sum(QueryExtents), qHeads, headDim]
QueryExtents [batch]                       // tokens contributed this step
Lengths      [batch]                       // cached positions after this step
Pages        [batch, maxPages]
```

The ragged form is the one a scheduler wants, because the rectangular form
makes every member of a batch contribute the same count and **a scheduler's
whole job is that they do not**. With `QueryExtents = [512, 1, 1, 1]` a
512-token prefill chunk and three decode steps are one dispatch. A sequence
contributing zero is an ordinary member, which is what makes admitting into an
idle slot free.

A token's position follows from the two per-sequence numbers rather than from a
base: with $L$ cached positions after the step and $n$ contributed, this step's
tokens occupy the last $n$, so token $i$ sits at $L - n + i$. `BaseName` is
therefore refused on a ragged step — there is no single base for a step whose
sequences start in different places.

Both forms **require** `Pages`: members have different lengths, so a contiguous
cache would pad every sequence to the longest. That is not a restriction to
work around — it is the reason paging exists, and it means
[016](016-prefix-cache.md)'s block pool is a **prerequisite** for this spec
rather than a neighbour of it. §9 is that dependency stated as work.

**One cost the ragged step carries**, and it is [C22](010-conformance.md): the
ragged kernel reads an **f32** cache. [C5](010-conformance.md) closed on the
argument that an f16 cache halves the largest allocation a serving process has,
and §1's arithmetic makes both the batch size worth reaching and the throughput
ceiling proportional to $1/A$ — so batching a prefill currently costs half the
batch it was reaching for. Filed as
[accel#23](https://github.com/golang-design/accel/issues/23). It narrows the
win; it blocks nothing.

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

## 5. Chunked prefill, and what the ragged step changed about it

An earlier draft of this section said a prefill and a decode cannot share a
dispatch, so a long prompt either runs alone — stalling every decoding sequence
for a whole forward pass — or is split into chunks that bound the stall. It
concluded that chunking **bounds latency and recovers no throughput**.

**The second half is now wrong**, and the reason is worth keeping. The claim was
true of the rectangular batch, where one `qSeq` applies to every member: a
512-token chunk and a 1-token decode are different counts, so they were
different dispatches whatever the scheduler wanted. The ragged step removes the
premise — `QueryExtents = [c, 1, 1, 1]` is one dispatch — so a chunk does not
stall the decodes beside it at all. **It rides with them.**

$$t_{\text{mixed}} \approx \frac{W + (c + B_d)\,A'}{\beta}$$

with $B_d$ decoding sequences and $A'$ the per-token activation and cache
traffic. The weights $W$ are read once for the chunk *and* the decodes
together, where two dispatches read them twice. So chunking now recovers exactly
what batching recovers, and the chunk size trades a different pair against each
other:

| $c$ | prefill efficiency | decode latency | weight traffic |
| --- | --- | --- | --- |
| small | worse GEMM | shorter step | more steps, so $W$ is read more often |
| large | better GEMM | longer step | fewer steps |

The useful range is still 512–2048, and the reason has moved from "how long may
a decode wait" to "how much of the step is the chunk". A chunk far larger than
$B_d$ makes the step a prefill with passengers, which is fine; a chunk far
smaller wastes the weight read.

**A chunk's first query token is not at position 0**, so its causal mask must
hide the right thing. A ragged step derives that per token from `Lengths` and
`QueryExtents` rather than from `BaseName`, so the mechanism is not a scalar
that has to be right — it is the same two numbers admission already maintains.

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

## 8. What this spec is waiting on, and it is tgo

Nothing upstream. The order of work is:

1. **The page-table port.** [004 §3](004-model-graph.md)'s port table has no
   page table and `nn.Attention` binds no `Pages` or `Block`, so nothing in tgo
   can pass one however capable the kernels are. `internal/prefix` is the
   bookkeeping and has no importers. Until a `Session`'s cache is addressed
   through a table, a slot cannot be swapped and a batch cannot be formed.
   [016 §9](016-prefix-cache.md) is the same statement from the other side.
2. **Then slots and admission**, which are §2 and §3 and are pure policy over
   the port above.
3. **Then the ragged step**, which is a shape change on a graph that already
   pages.

Each is a wave. None of them is blocked.

## 9. What tgo does now

Holds one named, skipping test per [010](010-conformance.md) register row, each
naming the accel spec that owns it. [C1](010-conformance.md) and
[C16](010-conformance.md) are both closed, so this spec's own rows have stopped
skipping and what remains open beside it is [C22](010-conformance.md), the f32
cache the ragged kernel reads.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 008-D1 | fixed $B_{\max}$; membership is contents | resize the batch per step | membership changes are free; a size change costs a drain. accel 040's `Batch` trap makes shrinking actively wrong |
| 008-D2 | recompute on preemption | swap KV to host | no host mirror; prefill is parallel, a swap is two serial transfers |
| 008-D3 | admission reserves answer space | admit on a free slot alone | a full pool cannot deadlock; rejections are reported |
| 008-D4 | keep §7's four hooks live in v0 | build them when 043 lands | v0 pays nothing; the retrofit stays a binding change |
| 008-D5 | last-arrived-first-evicted | round-robin or longest-running | bounds the latency of sequences already in flight rather than spreading the damage |
| 008-D6 | blocked on 043's *implementation*, not its design | build against the designed signatures now | accel 043 is drafted and partly in flight; building against unlanded signatures is building against nothing. **Discharged 2026-08-26**: 043, 046 and the ragged kernel all landed, and [C16](010-conformance.md) closed on a value probe rather than on a commit |
| 008-D7 | the **ragged** form, not the rectangular one | batch decodes rectangularly and run prefills alone | a rectangular batch gives every member the same token count, which is the one thing a scheduler exists to avoid. It also makes §5's chunk a separate dispatch, so the weights are read twice |
| 008-D8 | the page-table port is a wave of its own, before slots | build slots and the port together | a slot swap is free only because the blocks never move, so the addressing has to be right before a swap means anything. Building both at once means a batching bug and an addressing bug are indistinguishable |
| 008-D9 | admission is a queue in front of the slots, and [019](019-session-affinity.md)'s `Pool` becomes it | keep `Pool` beside the scheduler as a second owner of sessions | two owners of "which conversation is where" is two answers to it. `Pool.route` already picks the session holding the longest matching prefix, which is what admission wants; a scheduler that ignored it would prefill what the pool had already computed |
