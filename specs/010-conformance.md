---
title: "Conformance: the register of what accel cannot do, and the oracle that proves what it can"
status: implemented
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
| C1 | a **batched** decode | 040 | [#12](https://github.com/golang-design/accel/issues/12) | **closed** | none needed. Verified: two sequences of lengths 96 and 32 batched match two single runs to `0.00e+00` |
| C2 | RoPE at per-row positions | 025, 043 | [#2](https://github.com/golang-design/accel/issues/2) | **closed** | none needed |
| C3 | sampling of any kind at the `tensor` layer | 028, 039 | [#6](https://github.com/golang-design/accel/issues/6) | **closed** | none needed. `tensor.Sample` composes the whole policy on the device — penalties, temperature, softmax, top-k, top-p and the categorical walk — and returns a token id |
| C4 | a paged KV **decode** | 030, 043 | [#1](https://github.com/golang-design/accel/issues/1) | **closed** | none needed |
| C5 | an f16 KV cache that can be **written**, or paged | 007, 010 | [#13](https://github.com/golang-design/accel/issues/13) | **closed** | none needed; `ScatterRows`, prefill and paged decode all take f16. **Halves the cache** |
| C6 | penalties and temperature on device | 039 | [#6](https://github.com/golang-design/accel/issues/6) | **closed** | none needed. The policy runs on the device, so a step can return a token id rather than reading back 608 KB of logits |
| C7 | a **bf16 GEMM** | 002, 010 | — | won't fix, correctly | convert on the host at load, which is the right answer and not a workaround. [001 §3](001-weights.md): bf16 is the top half of an f32, so widening is a shift — exact, free, and done once. A bf16 GEMM would let tgo keep bf16 *on the device*, which costs the same two bytes as f16 and buys nothing. Filed inside [#14](https://github.com/golang-design/accel/issues/14) and answered with the mixed GEMM that closed [C8](#2-the-register); re-audited 2026-08-27 and reclassified rather than re-filed, because a capability tgo would not use is not a gap |
| C8 | f32 activations against f16 or int8 weights | 010 | [#14](https://github.com/golang-design/accel/issues/14) | **closed** | none needed. **The cast chain is gone**: 1013 selections → 760 on the Qwen3 graph |
| C9 | a strided view into `MatMul` | 025 | — | won't fix, correctly | host-side transpose at load ([001 §4](001-weights.md)) |
| C10 | avoiding a host copy of every converted weight | 001 | [#7](https://github.com/golang-design/accel/issues/7) | **closed** | none needed; `Buffer.Access` |
| C11 | a KV cache longer than 128 positions | 007, 010, 044 | [#8](https://github.com/golang-design/accel/issues/8) | **closed** | none needed |
| C12 | binding a `LayerState` view | 007, 030 | [#9](https://github.com/golang-design/accel/issues/9) | **closed** | none needed. 2 states, not 72 |
| C13 | a paged **prefill** | 010, 030 | [#10](https://github.com/golang-design/accel/issues/10) | **closed** | none needed; verified by reversing the page table |
| C14 | an f16 `GatherRows` | 010 | [#11](https://github.com/golang-design/accel/issues/11) | **closed** | none needed |
| C15 | a quantized matrix-vector kernel at $M=1$ | 010 | [#11](https://github.com/golang-design/accel/issues/11) | **closed** | none needed |
| C16 | a **batched prefill**, or prefill and decode in one dispatch | 040, 046 | [#16](https://github.com/golang-design/accel/issues/16) | **closed** | none needed. `AttentionOptions.QueryExtents` makes `q` flat — `[sum(extents), qHeads, headDim]` — so a step is a segmented extent rather than a rectangle and a 512-token chunk shares a dispatch with three decodes. Verified: a mixed step is bit-identical to the steps it batches, and re-splitting the same tokens moves the output |
| C17 | GGUF's K-quant super-blocks | 010 | [#15](https://github.com/golang-design/accel/issues/15), not planned | **open, not scheduled** | read safetensors and quantize at load ([012](012-gguf.md)) |
| C18 | `Contiguous` on Metal | 010, 021 | [#19](https://github.com/golang-design/accel/issues/19) | **closed** | none needed. It was the only kernel in the corpus with no MSL artifact, so every graph that slices — which [004 §3.2](004-model-graph.md) requires — was refused at compile. Fixed upstream the day it was filed |
| C19 | a CPU backend that dispatches in parallel | 006 | [#20](https://github.com/golang-design/accel/issues/20) | **closed** | none needed. The worker pool landed: 19.5x per prompt token on a real model, and device is 99.98% of a step, so nothing measurable remains between dispatches. The residual gap to Metal is kernel throughput rather than a missing capability |
| C21 | **4-bit weights** | 027, 048, 010 | [#22](https://github.com/golang-design/accel/issues/22) | **closed** | none needed. `quant.Int4Quantize` and `Int4MatMul` landed against this report, verified twice — against a reconstruction reference, and against the weights the checkpoint held within `quant.Int4ErrorBound`. tgo stores them since 2026-08-27, so a 27B checkpoint resolves to **13.4 GiB** rather than 26.7 ([001 §5.1](001-weights.md)). The embedding table is capped at int8, because it is gathered and there is no int4 gather |
| C22 | a **ragged step over an f16 cache** | 046, 010 | [#23](https://github.com/golang-design/accel/issues/23) | **closed** | none needed. `AttentionRaggedF16` landed against this report, so batching keeps [C5](#2-the-register)'s halving instead of giving it back. Per-sequence traffic $A$ stays halved, and [008 §1](008-scheduler.md) makes both the batch size worth reaching and the throughput ceiling proportional to $1/A$ |
| C23 | a **ragged step that tolerates a query row belonging to no sequence** | 046, 010 | [#24](https://github.com/golang-design/accel/issues/24) | **closed** | none needed. A row past the last extent is padding and reaches nothing, which is the shape this report argued for over clamping it into the last sequence — clamping would have turned an out-of-bounds read into a wrong answer. A batched step pads `q` to its plan shape freely |
| C24 | a **paged prefill over an f16 cache** | 010, 030 | [#25](https://github.com/golang-design/accel/issues/25) | **closed** | none needed. `AttentionPrefillPagedF16` landed against this report **the same day**, and the shared block pool is f16: twice the blocks, twice the prefixes worth keeping, and by [008 §1](008-scheduler.md) twice the batch size worth reaching. It was [C5](#2-the-register)'s pattern a third time — each of "`ScatterRows`, prefill and paged decode all take f16" is true and the combination was not, because the width and the paging selected separately. accel fixed the *pair* |
| C25 | declare a **reshaped** result as a graph output | 007, 025 | [#26](https://github.com/golang-design/accel/issues/26) | **closed** | none needed, and this row is why §2's rule is about values. It was the **accept-and-silently-wrong** class: correct shape, no refusal, all zeros, and `Contiguous` in front did not help. It cost an hour inside a recurrence that was correct, and it would have cost a wrong answer in production rather than a red build |
| C26 | a **depthwise causal convolution** over rows the graph computed | 025 | — | won't fix, correctly | **none, and the refusal was the right one.** The composition [018 §4.1](018-hybrid-models.md) records needs a left-padded input and `tensor` joins no two tensors along an axis, so the padded input cannot be built — but a `Concat` was never the ask: a convolution running a token at a time needs the K−1 inputs of the *previous step*, not zeros, and a decode step has no earlier rows in its own tensor at all. `nn.DepthwiseCausalConv` runs over a rolling `[K−1+T, C]` state built from `ScatterRows`, `GatherRows` and `Slice`, all of which exist. It costs ~3K+5 dispatches and K−1 copies of a `[T, C]` tensor per layer, which over 48 layers is one kernel to *want* and none to be blocked on. The row said `[slots, K−1+T, C]` until 2026-08-27: the block slices axis 0 as the time axis, so what is built holds one sequence, and the slot axis is [023](023-cache-kinds.md)'s work |
| C20 | a decode step whose submit cost is amortised | 021 | [#21](https://github.com/golang-design/accel/issues/21) | **closed** | none needed, and this row closes on a **measurement** rather than a probe because that is what it asked for. Submit went from 15.61% of a decode step to **3.34%**, throughput +43%, and p99 fell 84% — device is 94.62% of a step, which is the shape a decode step should have ([017 §4.1](017-benchmarks.md), Qwen3-0.6B f16 on Metal, 2026-08-25). The row's cost cell quoted the *before* number for two days after the spec it cites recorded the after |
| C27 | a gated delta layer whose **decay is per head** | 047, 043 | [#27](https://github.com/golang-design/accel/issues/27) | **open** | **48 dispatches per layer where one would do, or the wrong model.** `LinearOptions.Alpha` and `.Beta` are one f32 per token, while the state is `[slots, heads, valueDim, keyDim]` and the recurrence runs independently per head — every term carries a head index except the two gates. The published gated-delta formulation produces them per value head, and the target config reads `linear_num_value_heads: 48`. accel **refuses** the `[tokens, heads]` shape rather than reading the first heads-worth of it, which is the good answer and is why this row is a gap and not a defect: nothing computes a wrong decay quietly. The consumer's two workarounds are one call per head with a sliced `[T]` gate (3072 dispatches for a 64-layer model where 64 would do), or dropping the per-head decay, which is a different model. Closing it is one rank check: `[tokens]` keeps meaning every head shares a token's gate, so nothing existing moves. The shape was unconfirmed when the row was filed, because the checkpoint is a 50 GiB download. It is confirmed now without one: ollama's public `qwen3_5` loader permutes `in_proj_ba` through a permutation of length `2*valueHeads` and names the native layout as beta then alpha per key head (`x/models/qwen3_5/gdn_projections.go:64`, ollama/ollama `bd3f22e2`), and its forward pass hands the whole width to the gated delta kernel rather than reducing it (`qwen3_5.go:1091,1116`). Two independent readings of the architecture now say the gate has a head axis |

**accel moves under this table fast.** Within a day of the first filing on
2026-08-24, four rows closed: `RoPE` took a positions tensor, `Attention`
accepted an f16 cache and a page table, and `Buffer.Access` removed the host
copy. The table carries no snapshot date of its own, because it is
`Register()`'s output ([010-D10](#decision-record)) and the commit that holds it
is the date.

### How a row's state is decided

**By a probe that asserts a value, not by reading what an operator refuses.**

The rule used to be "record the graph and read the refusal", and **that rule is
what produced C13's false green.** A paged prefill compiled, so the probe
recorded it as working, and [016 §9](016-prefix-cache.md) said cross-request
prefix sharing "is expressible today". It is not. A refusal-based probe is blind
to the accept-and-silently-wrong class, which is the class that matters most.

**The rule now:** a probe binds real buffers, asserts the **output** against the
host oracle of [§5](#5-the-parity-oracle), records `Plan.Selections()`, and —
where an option is optional — **varies it and checks the output moves.** An
option that changes nothing is either honoured and irrelevant, or ignored.

An operator that accepts and computes the wrong thing is not a new register
state. It is [§1](#1-two-directions)'s **downward** direction — accel not doing
what it says — and it goes to the oracle, not to a fifth column.

**The four verdicts below are the 2026-08-24 re-derivation, kept because the
lesson stands.** All four rows have since closed or been reclassified, and the
generated table above is the current state. What they show is that reading
commits had got each of them wrong:

- **C8 looked closed and is not.** `MatMul` gained f32 operands, but it requires
  the two operands to *share* a dtype. A transformer's activations are f32 and
  its weights are f16 or int8, because a 4B model is 16 GB in f32. So f32
  operands remove the casts only for a model that stores f32 weights, which is
  the one configuration nobody runs. The row is narrowed to what would actually
  close it: a **mixed** GEMM.
- **C1 looked closed and is half closed.** Paging landed; batching did not.
  `q`'s rank is the phase, so a batch axis is read as a prefill and refused for
  a missing `BaseName`.
- **C5 closed, then reopened by using it.** `Attention` reads an f16 cache. No
  graph can *write* one: `ScatterRows` writes f32, and prefill over f16 is
  refused. accel's own test populates the cache from the host, which is a
  legitimate way to test the read path and exactly what hides the write path — a
  model has to compute KV on the device and write it from inside the graph.
- **C10 closed by a different answer than the one asked for.** The request was a
  buffer *over* caller memory; accel pointed the problem the other way with
  `Buffer.Access`, which needs no lifetime promise. Better than the ask, and
  invisible from the issue title.

This is [010-D1](#decision-record) in miniature, and it is why
[010-D6](#decision-record) generates the table, which
[010-D10](#decision-record) delivered on 2026-08-25.

**States.** Three, and not four (`internal/conformance/register.go:20`).
`closed` — accel's exported surface does the thing, verified by the probe.
`open` — it does not, and the row cites the issue or the named upstream artifact
that records the gap. `won't fix, correctly` — see below. A fourth state for a
gap accel has *designed* and not built would record accel's intent rather than
its behaviour, which is the reading [010-D7](#decision-record) removed.

C9 is not filed and should not be. accel refusing a strided view into `MatMul`
is the **correct** refusal: silently copying one would hide a real cost behind an
operator that looks free. The host-side transpose is the right answer, not a
workaround. It stays in the table because it constrains what tgo can do at graph
time, which is what this table is for.

**C11 closed on 2026-08-24.** It was the tree's headline blocker — a cache
capped at 128 positions, shorter than a system prompt. accel
[044](https://github.com/golang-design/accel/blob/main/specs/044-unbounded-context.md)
shipped the tiling loop and a 4096-position cache is verified working. Nothing
in tgo is blocked on cache size any more.

**C13 replaced it as the blocking row on 2026-08-24, and it was worse in
kind.** Every other row in this table is a *refusal*: tgo asks for something and
is told no. C13 was an **acceptance**. `Attention` took `Pages` on a prefill,
dropped it, read the cache contiguously, and returned a fluent wrong answer —
measured at a worst absolute difference of 0.74 between an identity and a
reversed page table, with `Selections()` naming the contiguous kernel both
times. That is the row [010-D7](#decision-record) was written for.

**C13 has since closed**, verified by reversing the page table and watching the
output move ([§2.2.1](#221-the-earlier-audit-kept-because-the-lesson-stands)),
and [016](016-prefix-cache.md)'s prefix cache is built in `internal/prefix`.
Cross-request prefix sharing is expressible and nothing is blocked on this
row.

**A row leaves this table only when its test stops skipping.** Not when an issue
closes, not when a spec is written, and never because it was worked around.

### 2.2.0 The re-audit when the tracker went to zero

**On 2026-08-27 every issue tgo had filed was closed, and four rows were not.**
§2.3 rule 1 says an open row cites an *open* issue, so that state is either a
register behind its evidence or a tracker ahead of its capabilities, and
[§2.2.1](#221-the-earlier-audit-kept-because-the-lesson-stands) is the reason it
cannot be assumed to be the second. Each of the four was probed.

| row | verdict | what settled it |
| --- | --- | --- |
| [C21](#2-the-register) 4-bit weights | **closed** | `Int4MatMul` computes to within a reconstruction reference at a transformer's shape. What remained was `weights.Precision` naming only f16 and int8, which was tgo's own work and shipped the same day |
| [C20](#2-the-register) submit cost | **closed** | a measurement, not a probe: 15.61% → 3.34%, +43% throughput, p99 −84% ([017 §4.1](017-benchmarks.md)) |
| [C7](#2-the-register) bf16 GEMM | **won't fix, correctly** | [001 §3](001-weights.md) widens bf16 to f32 with a shift, exactly and once. A bf16 GEMM would let tgo hold bf16 on the device, which costs what f16 costs and buys nothing |
| [C17](#2-the-register) K-quant super-blocks | **stays open** | `quant` registers one level of scale; a Q4_K super-block is two levels over eight sub-blocks with a minimum each. Nothing reads one |

**Two of the four moved for reasons the tracker could not have told anyone.**

- **C21 did not close where its issue closed.** The issue said "no 4-bit
  representation" and accel shipped one; the *row* is about a 27B model fitting
  a 24 GiB device, and that needs a loader storing int4. The row closes because
  what is left is tgo's, and the register is the register of accel's gaps —
  which is the [C8](#2-the-register) lesson read the other way round.
- **C7 was never a gap.** It sat open for a week as "narrowed to what would
  actually close it: a bf16 GEMM", and re-reading [001 §3](001-weights.md)
  says tgo would not use one. Re-filing it under §2.3 rule 2 would have asked
  accel for a kernel no consumer wants. It is [C9](#2-the-register)'s kind of
  row and is recorded as one.

**And one cell was stale against the spec it cites.** C20's cost quoted submit
at 15.61% while [017 §4.1](017-benchmarks.md) had recorded 3.34% two days
earlier. [010-D6](#decision-record)'s generator guarantees the *table* matches
`Register()`; it cannot guarantee that `Register()`'s prose matches the
measurement it names, and nothing else was checking.

### 2.2 The 2026-08-25 audit: sixteen rows closed, four open, and what that took

Re-audited on 2026-08-25 against accel HEAD `05ff997` by recording graphs and
reading `Selections()`, never by reading a commit message.

**Open on that date: C7** (a bf16 GEMM), **C16** (a batched step takes one
token per sequence, so a prefill cannot batch), **C17** (GGUF, not scheduled),
and **C20** (submit is 15.6% of a decode step). C9 is a refusal that should
stay. Everything else was closed.

**Three of those four have moved since.** C16 closed on the segmented extent
`AttentionOptions.QueryExtents` gives a step, C20 closed on a measurement, and
C7 was reclassified as a correct refusal by
[§2.2.0](#220-the-re-audit-when-the-tracker-went-to-zero)'s re-audit. **Open
today: C17 alone**, out of twenty-six rows; the correct refusals are C7, C9 and
C26.

**The two that closed most recently are the ones tgo had carried longest.**
`tensor.Sample` now composes the entire policy on the device — penalties,
temperature, softmax, top-k, top-p and the categorical walk, eight kernels — and
returns a token id. That closes **C3** (no sampling operator at all) and **C6**
(the 608 KB logits readback per token) together, and it is the row
[017 §4.1](017-benchmarks.md) measured at 1.59% of a decode step: real, and an
order of magnitude smaller than the submit cost beside it.

> **accel's composition order is not the one this tree specified**, and accel's
> argument is better. [006 §3](006-sampling.md) truncated before the softmax,
> as vLLM does; accel truncates after, because f32 rounding can make two
> distinct logits equal probabilities, so a top-$k$ over logits keeps a
> different boundary entry than the cumulative walk later sees. 006 was
> corrected rather than the divergence recorded, since
> [006-D1](006-sampling.md) makes tgo the *reference* for the device path and a
> reference that composes differently is not one.

Three earlier closures changed what tgo builds:

- **row C1** — continuous batching is expressible. [008](008-scheduler.md) was
  `blocked` from the day it was written and is not any more.
- **row C8** — **the cast chain is gone.** f32 activations now multiply f16 and
  int8 weights directly: 1013 kernel selections on the Qwen3-4B graph became 760.
- **row C5** — an f16 cache can be written *and* paged, so
  [005 §3](005-kv-cache.md)'s f16 column is the one tgo builds against.

### 2.2.1 The earlier audit, kept because the lesson stands



On 2026-08-24 accel closed ten of the eleven issues tgo filed. A re-audit of
every row at HEAD `cb82904`, asserting values and reading `Selections()` rather
than checking that a graph compiles:

| closed upstream **and** verified closed | closed upstream, **still open here** |
| --- | --- |
| C2, C4, C10, C11, C12, C13, C14, C15 | C1, C5, C7, C8 |

**Eight of those are real and two of them change this project's shape.** C12
collapses the KV cache from 72 states to 2. C13 was the blocking row — a paged
prefill now honours its page table, verified by reversing the table and watching
the output move — which unblocks [016](016-prefix-cache.md) entirely.

**Four are not, and each closed against a report that named a symptom.** C1's
issue was titled for the *block pool*, which landed; the batch axis it also
asked for did not. C5's asked for an f16 *cache* and got the read path. C7 and
C8 shared one issue about `MatMul` being f16-only, which was answered with an
f32 GEMM — and a transformer's operands are never the same dtype, so the casts
remain.

The lesson is not that accel closed things carelessly. **Each fix matched its
issue's title.** It is that a title is a summary and a register row is a
capability, and only the second one is testable.

### 2.3 Commenting on a closed issue is not reporting

There was a second failure here, and it was tgo's.

When C5 and C8 turned out to be unfixed, tgo **commented on the closed issues**
and left the register's `filed` column pointing at them. A comment on a closed
thread creates no work item and appears in nobody's queue. For a week the
register read as though six rows were tracked upstream while exactly one open
issue existed.

Worse, [012](012-gguf.md) carried `status: blocked` naming an accel *spec* and
no issue at all — a blocker that existed only in this repository.

**The rule, from now on:**

1. an open register row cites an **open** issue;
2. when accel closes an issue whose capability is still absent, tgo **files a
   new one** rather than commenting, and says in it why it is a re-file;
3. a spec with `status: blocked` names a **durable upstream record** in
   `blocked_on` — an issue, or the named thing upstream that records the gap —
   not a bare file path, because a file path is a belief about whose problem it
   is rather than something anyone can act on.

**Rule 3 was narrower when first written**, and accel corrected it. It demanded
an open *issue*; accel closed [#15](https://github.com/golang-design/accel/issues/15)
as not planned and recorded the gap as a `quant_matmul_superblock` row in its
kernel corpus, carrying the layout, the formula and both workarounds. That is a
**better** record than an issue with no plan, because the corpus is what someone
adding a kernel reads and an issue is what someone opening the tracker reads.
tgo accepted the closure and widened the rule. See
[012 §3](012-gguf.md).

`speclint` enforces (3) over the spec text, which is the one a linter can see
there. `conformance.Validate` (`internal/conformance/register.go:451`) enforces
the half of (1) a program can see over the register itself — an open row cites
an issue or a named upstream artifact, and a correct refusal cites neither — and
`TestTheRegisterObeysItsOwnRules` runs it on every test run. Whether the
citation is still *open*, and (2), are enforced by the re-audits in
[§2.2.0](#220-the-re-audit-when-the-tracker-went-to-zero) and
[§2.2](#22-the-2026-08-25-audit-sixteen-rows-closed-four-open-and-what-that-took),
which is where the gap surfaced.

> **And the generator stopped being aspirational on 2026-08-25.** 010-D6 was
> written when the register was maintained by hand, and `internal/conformance`
> now emits §2's table from `Register()` with a test that fails when the two
> disagree. It caught its author's own hand-edit on the first attempt: two new
> rows in the wrong position, which a reader comparing the two would not have
> flagged. The table above is generated output — **edit `Register()`, not this
> section.**
>
> **That rule stopped being hypothetical on 2026-08-24.** accel closed
> [#4](https://github.com/golang-design/accel/issues/4) and
> [#5](https://github.com/golang-design/accel/issues/5), and a probe against the
> same HEAD shows both capabilities still absent: an f16 cache cannot be written
> (`ScatterRows` writes f32, prefill over f16 is refused, paged+f16 is refused),
> and mixed-precision `MatMul` is still refused for both f16 and int8 weights.
>
> Neither closure was wrong from inside accel — each shipped what its issue
> asked for by name. C5's issue asked for an f16 *cache* and got the read path;
> C8's asked for an *f32 GEMM* and got one. **The reports named the symptom and
> the fix matched the name rather than the cost**, which is a failure of the
> reporting, not the fixing. C8's is squarely tgo's fault and is now refiled as
> a mixed GEMM.
>
> The general lesson is why this table exists: an issue tracker records what was
> asked, and only a test records what a consumer can do.

### 2.1 What the register is worth so far

Twenty-one distinct accel issues so far, cited by the register's twenty-six
rows. They produced one upstream design decision — accel 043's *a scalar is a
value every row shares; a value that differs per row is a tensor* — which
removes surface rather than adding it, and which was reached by five of the
rows above being **the same mistake seen five times**. That is the argument for
a validating consumer: no single one of C1–C5 looks like a design decision from
inside accel, and together they are one.

The second argument is C8, and it is about the difference between a report being
**accepted** and a cost being **removed**. accel took the argument, relaxed the
refusal, and the 252 casts are still there — because the report named the
symptom (*f16-only*) rather than the shape (*mixed precision*). A consumer that
stops measuring once a fix lands reports a win that did not happen. The
follow-up is on the same issue, with the probe output in it.

C11 is the third, and a different kind again. It is not a subtle design
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

## 3.1 Performance against vLLM, and which axes are winnable

tgo's goal is to be **faster than vLLM**, and this section says on what and how
it is measured, because an ambition with no measurement is a slogan.

**The axes tgo should win, and why:**

| axis | why tgo can win |
| --- | --- |
| **host overhead per decode token** — scheduling, sampling, detokenizing, deciding what runs next | none of it is matrix multiplication, all of it runs every token, and it is compiled Go against a Python interpreter with a global lock |
| **time to first token, cold** | tgo builds a plan in milliseconds; a Python stack loads for tens of seconds |
| **resident footprint of the runtime itself** | one static binary against an interpreter and its dependency tree |
| **hardware vLLM serves poorly** | CPU, Metal, and whatever accel adds |

**The axis tgo will lose for a long time, stated plainly:** raw GEMM and
attention throughput on NVIDIA. vLLM's kernels are years of hand-tuned CUDA —
FlashAttention, CUTLASS-derived quantized GEMMs — and accel's are portable
kernels written in a Go subset. **That gap is accel's to close, not tgo's**, and
[000 D1](000-decisions.md) means tgo's contribution to closing it is
measurement: a kernel that is slower than it should be is a report, exactly like
a kernel that is missing.

**The measurements**, run against vLLM on the same model, hardware and prompts:

- **decode tokens per second**, single sequence and at batch, which is the
  headline;
- **host time per token** — the step minus the device time — which is the axis
  above and the one tgo expects to win first;
- **time to first token**, cold and warm;
- **resident memory** at the same context and batch;
- **the readback share** ([C6](#2-the-register)), because it is host overhead
  tgo currently cannot remove.

**Losses are published.** A framework that reports only the benchmarks it wins
is not reporting. The register already commits tgo to naming what it cannot do;
the same rule covers what it does slowly.

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
fix.**

**Enforced by the type, not socially.** `internal/conformance` has no way to
write a tolerance down: a comparison takes a `Terms`, and every constructor is
one row of the table above or one ceiling from accel's own numerics spec. A
caller composes the stages its computation actually has —
`AccumF32(k).And(StoreF16(2)).And(Magnitude(scale))` — and the bound falls out.
Widening one means **adding a term**, which is an argument somebody has to make
out loud, and `Terms.Explain()` prints what produced the number.

The zero `Terms` is exact equality, which is a legitimate claim about an
operation that only moves bytes.

## 6. Reporting

The suite emits the §2 register and the §3 numbers as a generated Markdown
document, so the table in this spec is produced from the tests rather than
maintained beside them. A hand-maintained register drifts within one milestone;
that is the same failure this project exists to catch in accel.

## Outcome

The register and the oracle are built and running. `internal/conformance` is
the machinery this spec designs — the tier rule every parity test runs under,
the derived tolerance every comparison is judged by, the probe rig, the register
and its generator — and `internal/oracle` is the float64 reference it judges
against. They landed across eighteen commits between 2026-08-24 and 2026-08-27,
from Wave 2's first oracle to Wave 11's gated delta probe. The register holds
twenty-six rows: twenty-two closed, one open (C17), and three correct refusals
(C7, C9, C26). §2's table is generated output and a test refuses a document
whose table has drifted from `Register()` or has anything spliced inside it.
§3's five numbers have types, renderers and tests, and none of them has been
measured.

**What shipped**, section by section:

| section | what landed | where |
| --- | --- | --- |
| 1 | both directions: a device result compared against the oracle, and every place tgo cannot express something as a register row | `internal/conformance/conformance.go:4`, `internal/conformance/parity_test.go:52` |
| 2 | `Register()` and `Document()`, and the drift test that pins §2's table to them line by line | `internal/conformance/register.go:112`, `:429`, `internal/conformance/register_test.go:127` |
| 2 (010-D1) | one skipping subtest per open row, generated from the register, each naming the row, the capability, the owning accel spec and what the workaround costs | `internal/conformance/register_test.go:177` |
| 2, how a row's state is decided (010-D7) | `Rig.Parity`: bind real buffers, compare the output against the float64 reference under a `Terms` budget, read `Plan.Selections()`, and vary the optional binding to require the output to move | `internal/conformance/rig.go:222`, `internal/conformance/ragged_test.go:343` |
| 2.2.0 | the re-audited rows are standing probes rather than a one-time check; six run on every test run | `internal/conformance/reaudit_test.go:36` |
| 2.3 | `Validate`: an open row cites something upstream, a correct refusal cites no issue, every row names an accel spec and fills both prose cells. Rule 3 is `speclint`'s | `internal/conformance/register.go:451`, `internal/speclint/speclint.go:231` |
| 3 | `Measurements` and its five types, each with a `Value()` renderer and a JSON form, and a nil pointer meaning *not measured* rather than zero | `internal/conformance/measure.go:31` |
| 4 (010-D4) | `decide` over the three tiers, every branch tested as data including the two unreachable on one machine, `TGO_REQUIRE_METAL` in CI, and `ModelPath` for tier 3 | `internal/conformance/tier.go:74`, `internal/conformance/tier_test.go:19`, `.github/workflows/ci-metal.yml:19` |
| 5 (010-D2, 010-D5) | `internal/oracle`, float64 throughout, depending on `math` and nothing of tgo's; the whole forward pass composed from it | `internal/oracle/oracle.go:27`, `model/graph_rig_test.go:332` |
| 5.1 (010-D3) | `Terms` with nine constructors, `And` composing them, `Explain()` printing the derivation, and no way to write a tolerance down as a number | `internal/conformance/tolerance.go:62`, `internal/conformance/tolerance_test.go:20` |
| 6 | `Publish` emits the register and the numbers as one Markdown document; a verbose run logs it and `TGO_EMIT_TABLE` writes it to a file | `internal/conformance/measure.go:402`, `internal/conformance/emit_test.go:11` |

**What diverged** from the design, and why the code is right:

- §2's generated block has an extent this spec never stated: the header to the
  next blank line, and nothing but table rows inside it. The obvious reading —
  stop at the first line that is not a row — let three runs of `go test` output
  sit between the last row and the paragraph under it with every gate green
  (`internal/conformance/register_test.go:44`, `:83`). A generated block's
  extent is decided by the document, not by how far the generated part reaches.
- §5's "the whole forward pass" is true across two packages: the primitives in
  `internal/oracle`, the composition in `model/graph_rig_test.go:332`. Keeping
  the composition out of the package is what lets any package import the oracle
  without pulling a model in behind it.
- §2 says a row leaves the register when its test stops skipping, which
  describes open rows only. Closed rows keep probes too
  (`internal/conformance/reaudit_test.go`), because twenty-two of the
  twenty-six rows are closed and a closure nobody re-runs is a claim about an
  accel HEAD that has moved.
- §3.1's measurements moved to [017 §3](017-benchmarks.md), which owns the
  comparison table. The honesty rule stayed here and `tgo record` honours it: a
  record with no vLLM row names the missing row rather than omitting it
  (`cmd/tgo/record.go:138`).
- §2.3 attributed rules 1 and 2 to the manual re-audit alone. `Validate` turns
  the half of rule 1 a program can see into a lint, which is cheaper than an
  audit and runs more often.

**Not built.** §3's five measurements are named and none of them has been run:
CPU/Metal divergence, the readback share of a decode step, quantization error
against `Int8ErrorBound` on real blocks, plan compile time per bucket with the
session cache hit rate, and transient bytes from `Plan.Memory()` against the
hand-computed working set. Every `Measurements` outside `measure_test.go`'s
fixtures is the empty struct, so every document `Publish` emits today prints
five "not measured" lines. Taking them needs three things that do not exist
yet: a tier-3 checkpoint under `TGO_MODEL`, a Metal device in the same loop as
a CPU run for the divergence number, and a §4 in [011](011-sequencing.md) to
record the dated result in, which [§4](#4-how-the-suite-runs) already points
at. The quantization
number additionally needs `measure_test.go:133`'s `outlierWeights` doc comment
corrected: it claims to be what §3 asks for and it is synthetic, which is the
one thing §3 says the number cannot be measured on. §5.1's table is four rows
short of the whitelist its own escape clause points at — f32 roundings
(`RoundF32`), the softmax weight (`SoftmaxWeight`), int4 (`QuantInt4`), and
accel's ULP and absolute primitive ceilings (`PrimitiveULP`, `PrimitiveAbs`) —
and the softmax row wants splitting, because "benign" is true of the
max-subtraction against overflow and false of the weight, which carries the
score's accumulated error scaled by the exponential. The gated delta oracle
(`internal/conformance/linear_test.go:78`) needs a home: either move it into
`internal/oracle` under §5's four rules, or amend §5 to permit a reference
beside the probe it judges and say what keeps it independent. `Rig` and the
generated block's extent are the machinery a contributor writing the next probe
reads, and no section describes either. And seven call sites in four packages
read `TGO_MODEL` with `os.Getenv` directly (`cmd/tgo/engine_test.go:28`,
`nn/checkpoint_test.go:31`, `weights/model_test.go:68`,
`model/qwen3_real_test.go:24`) instead of `conformance.ModelPath`, which
`e2e_test.go:55` uses, so §4's tier rule — a path that cannot be read is a
failure, not a skip — is bypassed wherever the package is not imported.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 010-D1 | one skipping test per register row | a prose list | the table cannot go stale silently |
| 010-D2 | the oracle is written from the mathematics, not from tgo's graph | share code with the builder | agreement becomes evidence rather than tautology |
| 010-D3 | tolerances are derived and commented with their term; a raised tolerance is a finding | tune until green | a numerics regression cannot be absorbed |
| 010-D4 | tier 3 never runs in CI | a nightly with a download | CI stays offline and under a minute |
| 010-D5 | the oracle is float64 and presumed right on disagreement | float32, matching the device | it is the simpler program; matching the device would import the device's bugs |
| 010-D6 | the register is generated from the tests **at M10** | maintained by hand forever | it is the exact drift tgo exists to catch upstream. **Amended 2026-08-24:** generation needs tests, so until M10 `speclint` stands in — it checks the rows are numbered without gaps and that nothing in the tree cites a row that does not exist. A decision nothing enforces, in the spec about decisions nothing enforces, was the wrong thing to leave standing. **Superseded 2026-08-25 by [010-D10](#decision-record):** `internal/conformance` emits the table from `Register()` and a drift test fails when the two disagree, so the interim ended at Wave 4 rather than at M10. `speclint` keeps the numbering and citation checks, which read the spec text and are not what the generator does |
| 010-D7 | a probe asserts a value against the oracle and varies optional bindings | record the graph and read the refusal | the refusal-based rule was blind to C13 and reported a false green in its own spec |
| 010-D8 | an open row cites an open issue; a blocked spec names a durable upstream record, issue **or** named artifact; a closed issue with an absent capability is **re-filed**, not commented on | comment on the closed thread; demand an open issue for every blocker | a comment creates no work item, and the register read as tracked while one issue was open ([§2.3](#23-commenting-on-a-closed-issue-is-not-reporting)) |
| 010-D9 | performance against vLLM is a measured table per axis, losses included | a headline throughput claim | tgo will lose raw NVIDIA kernel throughput for a long time and should win host overhead first; one number hides both ([§3.1](#31-performance-against-vllm-and-which-axes-are-winnable)) |
| 010-D10 | the generator is the source of truth; §2's table is its output | edit the table and reconcile the code later | **Demonstrated 2026-08-25.** A hand-edit adding two rows to §2 was caught by the drift test within one CI run, including a row-order difference nobody would have noticed by eye. Adding a row means editing `Register()`, which is one place rather than two |
