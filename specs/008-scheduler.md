---
title: "Continuous batching: slots, admission, and the ragged step that makes a mixed dispatch possible"
status: complete
layer: engine
depends_on:
  - 000-decisions.md
  - 005-kv-cache.md
  - 007-engine.md
  - 016-prefix-cache.md
---

# Continuous batching

**Written blocked, unblocked on 2026-08-26, shipped on 2026-08-27.** This spec
carried `status: blocked` from the day it was written. Two gaps closed in
sequence: `Attention` took a batch axis ([C1](010-conformance.md)), and then a
*ragged* one ([C16](010-conformance.md)). §8 records what shipped the day
after.

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
rather than a neighbour of it. It is a `depends_on` edge, the pool work landed
there rather than here, and `NewBatch` refuses outright without one.

**One cost the ragged step carried, and no longer does.** The first ragged
kernel read an **f32** cache, which gave back the halving
[C5](010-conformance.md) closed on: §1's arithmetic makes both the batch size
worth reaching and the throughput ceiling proportional to $1/A$, so batching a
prefill cost half the batch it was reaching for. Reported as
[C22](010-conformance.md) and filed as
[accel#23](https://github.com/golang-design/accel/issues/23);
`AttentionRaggedF16` landed against it, so the ragged step takes f16 and
batching keeps the halving.

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
cache dominates it. [005 §3](005-kv-cache.md) gives 288 KiB per position in f32
and 144 KiB in f16, and the pool tgo allocates is f16 (`blocks.go:88`), so
$A \approx L \cdot 144\text{ KiB}$ for context length $L$:

| context $L$ | $A$ | crossover $B$ | ceiling |
| --- | --- | --- | --- |
| 1024 | 147 MB | ≈ 27 | ~28× |
| 2048 | 295 MB | ≈ 13 | ~15× |
| 4096 | 590 MB | ≈ 6.8 | ~7.8× |

So the crossover is **inside** the range a single-device server sees at long
contexts and above it at short ones, and **the ceiling falls as context grows.**
Batching is most of the hardware at short contexts and much less at long ones.

Every row here is twice what it was before [C5](010-conformance.md) and
[C22](010-conformance.md) closed: halving $A$ doubled the batch size worth
reaching, and the table above is the f16 one. [016](016-prefix-cache.md) is the
other half, because it reduces the prefill work that batching does not help with
at all.

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
sequence, which costs one entry of `lengths`, one entry of `extents` and one
page-table row, and no arithmetic: a member contributing zero tokens emits no
query row. accel 040 records the trap
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
occupancy and deadlocks. The design first took $R$ as configuration with a
documented default. **That reversed in the build**: `SchedulerOptions.Reserve`
has no default, and a zero or negative reserve is refused with the argument in
the error text, because a default reserve is this section's deadlock wearing a
number a deployment never chose. Admission also **reports rejected admissions**,
because a server that quietly admits fewer requests than it could is
indistinguishable from a slow one.

## 4. Eviction: recompute, not swap

A preempted sequence drops its blocks and re-prefills on readmission.

| | cost |
| --- | --- |
| recompute | one prefill of $T$ tokens, on readmission |
| swap to host | $T \cdot 144$ KB out and back, plus host memory for every swapped sequence |

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

| hook | v0 shape | what happened |
| --- | --- | --- |
| the ports are **named tensors** | one session binds them | held. `Batch` is a second binder over the same names, which is what kept the retrofit local. The cache addressing was *not* reused: `Batch` allocates its own ports and `batchSlot` is explicitly not a `Session` |
| positions are a **bound tensor** | a one-row tensor | held. Widening to $B$ rows was a shape change, not a rewrite |
| **one draw consumed per step per sequence** | one sequence | held ([006-D2](006-sampling.md)); per-slot draws are a shape change, not a semantic one |
| **one step is one call** on the streaming surface | one stream | held in substance, not in shape. There is no `Generate` and no internal channel — [007-D6](007-engine.md) rejects channels — but one `Stream.Next` is one model step, so a scheduler drives many streams without the public API moving |

The first two were free because accel 043 moved per-row values onto tensors
before tgo wrote any code. The third and fourth are decisions tgo made for
reasons that stand on their own. Two of the four predicted the retrofit's
*shape* and were wrong about it: the page table did not slide under `Session`,
and the stream is an iterator. What made the retrofit local was the naming, not
either prediction.

## 8. Outcome

Built between 2026-08-26 and 2026-08-27, in the order 008-D8 sets: the
page-table port, then the shared block pool, then the batched graph, then
admission, then the policy over it. The mechanism is `Batch`, the policy is
`Scheduler` over `nextStep` and `victim`, and 22 tests cover them. The policy is
a pure function over integers and is tested without a device: `nextStep` and
`victim` decide from slot state alone, so every case is microseconds and the
cases can be the ones that matter rather than the ones a forward-pass fixture
can afford. What the device tests carry is the mix and the values, and the
values are asserted against real sessions.

**What shipped**, section by section:

| section | what landed | where |
| --- | --- | --- |
| 2 | `Batch`: one set of ports for the whole dispatch, `PortExtents` and `PortLast` beside the widened `Lengths` and `Pages`. Verified by a batch of three producing what three single steps produce, bit for bit, on a page table that is deliberately not the identity | `batch.go:45`, `batch.go:114`, `model/batch.go:66` |
| 2 | a step's row total is bucketed to a compiled plan per rung, with one `tensor.Bindings` cached per rung, so a moving row count is not a new compile every step | `batch.go:52`, `batch.go:65` |
| 3 | `prefix.Request.Reserve`, so $\lceil (T+R)/B \rceil$ blocks are taken together or not at all, and `Admit` distinguishes "no slot" from "the pool cannot hold this" | `batch.go:233`, `scheduler.go:92` |
| 4 | last-arrived-first, preferring a slot mid-prefill at equal arrival — 008-D2 makes eviction a recompute, and recomputing a prefill nobody has read costs no latency a caller was promised. `Finish` is the ordinary release beside it | `schedule.go:128`, `scheduler.go:149`, `scheduler.go:128` |
| 5 | `nextStep` mixes a chunk with the decodes beside it in one dispatch, prefills placed first when rows are scarce. Measured on the device: 8 prompt tokens and 2 decodes, one step | `schedule.go:76`, `schedule.go:21` |
| 6 | block sharing between slots through the shared pool, which is [016](016-prefix-cache.md)'s and which a batch collects rather than owns | `batch.go:233` |
| 7 | two hooks held as designed and two held in substance; see §7's table | `batch.go:45`, `stream.go:76` |

**What diverged** from the design, and why the code is right:

- §3 said $R$ is configuration with a documented default. There is no default:
  `NewScheduler` refuses a zero reserve and says why in the error. A default
  reserve is §3's deadlock wearing a number no deployment chose, and a refusal
  is the only report a caller cannot miss (`scheduler.go:57`).
- §7 predicted a page table sliding under `Session` and a `Generate` streaming
  over a channel. Neither exists: `Batch` owns its own ports and `batchSlot` is
  not a `Session`, and [007-D6](007-engine.md) rejects channels. The retrofit
  stayed local for a different reason — the ports were already named tensors, so
  a batch is a second binder over the same names (`batch.go:45`).
- The intro's f32 ragged kernel was a real cost for one day.
  [C22](010-conformance.md) closed, so the paragraph now records the close
  rather than the cost, and §1's table is the f16 one.

**Three defects, none of which a value test on its own would have found.**

- **A batched step padded rows that belonged to no sequence.** A single
  sequence's bucketed prefill pads with rows whose slot is the cache capacity
  and lets `ScatterRows` drop them; in a batch those rows index one past the
  offsets array — another sequence's cache on a GPU, read back fluently. That
  is [C23](010-conformance.md), filed as
  [accel#24](https://github.com/golang-design/accel/issues/24). tgo argued for
  "a row past the last extent contributes nothing" over clamping it into the
  last sequence, accel took that shape, and the interim fix that charged the
  padding to a real sequence's extent came back out: pad rows carry the cache
  capacity as their slot and move no member's extent or length
  (`model/batch.go:140`).
- **`Admit` leased the prompt and `Step` leased it again**, so the lease chained
  its block hashes over `prompt+prompt` — hashes naming blocks that hold only
  `prompt`. The logits of the step that caused it are correct, so nothing
  downstream reports it; the lease's own length is what the test asserts.
- **A returned logits slice outlived what it described.** The first draft
  aliased one shared readback, and the test written minutes later broke the rule
  and compared stale numbers, which read as a batching bug. A lifetime nobody
  can keep is not a lifetime. The rule now is one buffer per slot, valid until
  that slot steps again (`batch.go:86`).

**Not built.** Nothing in this spec's scope. The three items §9 carried moved to
specs that own them: [020](020-device-sampling.md) took sampling on the batched
path, with the measurement that chooses between the host readback and
`tensor.Sample`; [021](021-admission-queue.md) took the queue in front of
admission that 008-D9 hands to [019](019-session-affinity.md)'s `Pool`; and
[022](022-batched-serving.md) took the server change, replacing the session pool
with a scheduler engine.

**Description debt**, which is a spec that is thinner than the code rather than
work nobody did. Five behaviours are built and tested and no section states
them: `Produced.Sampleable` and `Feed`'s refusal of a token fed before the
prompt is scored (`scheduler.go:188`, `scheduler.go:239`), which §5 owns; that a
slot left out of a step is not evicted (`schedule.go:76`), which is §5's
boundary against §4; `Scheduler.Finish` as the ordinary release distinct from
`Evict` (`scheduler.go:128`), which §2 owns; the logits lifetime contract
(`batch.go:86`); and that a step reads back only the rows it produced, as one
span, because at $V = 151936$ a full readback charges every step for every idle
slot (`batch.go:300`).

## 9. What this spec handed on

This section listed three unbuilt items. Each now has a spec that owns it, and
this section stops describing the work.

**Sampling on the batched path** was that `Scheduler.Step` returns logits and
the caller feeds the token, so the readback is $B \times V$ floats per step
while [C3/C6](010-conformance.md)'s `tensor.Sample` sits unused.
[020](020-device-sampling.md) took it, including the measurement
[006-D1](006-sampling.md) makes the deciding one.

**A queue in front of admission** was that `Admit` refuses rather than waits.
[021](021-admission-queue.md) took it, and it is 008-D9's handover to
[019](019-session-affinity.md)'s `Pool`.

**The server does not use it** was that `tgo serve` pools sessions where a
scheduler over a batch would replace them.
[022](022-batched-serving.md) took it, and it waits on the two above.

## 10. What tgo does now

Holds one named, skipping test per [010](010-conformance.md) register row, each
naming the accel spec that owns it. [C1](010-conformance.md),
[C16](010-conformance.md), [C22](010-conformance.md) and
[C23](010-conformance.md) are closed, so this spec's own rows have stopped
skipping and **nothing in the register is open against this spec**.

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
| 008-D9 | admission is a queue in front of the slots, and [019](019-session-affinity.md)'s `Pool` becomes it | keep `Pool` beside the scheduler as a second owner of sessions | two owners of "which conversation is where" is two answers to it. **The stated reason was wrong and the conclusion holds.** It read "`Pool.route` already picks the session holding the longest matching prefix, which is what admission wants" — but `Session.reusable` returns 0 unless the scope is `CacheSession`, and a `Batch` requires `CacheProcess`, so in every configuration a scheduler can run in `route` scores zero for every entry and falls through to the coldest. Nothing is lost: reuse moved one layer down to the block hashes. What the scheduler inherits is the **waiting**, and `route` is what does not come with it. **The queue itself is now [021](021-admission-queue.md)'s**, which owns the design this decision states |
