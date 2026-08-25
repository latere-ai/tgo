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
| C1 | a **batched** decode | 040 | [#12](https://github.com/golang-design/accel/issues/12) | **closed** | none needed. Verified: two sequences of lengths 96 and 32 batched match two single runs to `0.00e+00` |
| C2 | RoPE at per-row positions | 025, 043 | [#2](https://github.com/golang-design/accel/issues/2) | **closed** | none needed |
| C3 | sampling of any kind at the `tensor` layer | 028, 039 | [#6](https://github.com/golang-design/accel/issues/6) | **open** | host sampling. The per-row kernels exist in the corpus; `tensor` exports no sampling operator |
| C4 | a paged KV **decode** | 030, 043 | [#1](https://github.com/golang-design/accel/issues/1) | **closed** | none needed |
| C5 | an f16 KV cache that can be **written**, or paged | 007, 010 | [#13](https://github.com/golang-design/accel/issues/13) | **closed** | none needed; `ScatterRows`, prefill and paged decode all take f16. **Halves the cache** |
| C6 | penalties and temperature on device | 039 | [#6](https://github.com/golang-design/accel/issues/6) | **open** | host, before submission; a 608 KB logits readback per token |
| C7 | a **bf16 GEMM** | 002, 010 | [#14](https://github.com/golang-design/accel/issues/14) | **open, narrowed** | convert on the host at load. `Cast` now widens bf16 to f32, so only the GEMM is missing, and a weight would not want a per-step cast anyway |
| C8 | f32 activations against f16 or int8 weights | 010 | [#14](https://github.com/golang-design/accel/issues/14) | **closed** | none needed. **The cast chain is gone**: 1013 selections → 760 on the Qwen3 graph |
| C9 | a strided view into `MatMul` | 025 | — | won't fix, correctly | host-side transpose at load ([001 §4](001-weights.md)) |
| C10 | avoiding a host copy of every converted weight | 001 | [#7](https://github.com/golang-design/accel/issues/7) | **closed** | none needed; `Buffer.Access` |
| C11 | a KV cache longer than 128 positions | 007, 010, 044 | [#8](https://github.com/golang-design/accel/issues/8) | **closed** | none needed |
| C12 | binding a `LayerState` view | 007, 030 | [#9](https://github.com/golang-design/accel/issues/9) | **closed** | none needed. 2 states, not 72 |
| C13 | a paged **prefill** | 010, 030 | [#10](https://github.com/golang-design/accel/issues/10) | **closed** | none needed; verified by reversing the page table |
| C14 | an f16 `GatherRows` | 010 | [#11](https://github.com/golang-design/accel/issues/11) | **closed** | none needed |
| C15 | a quantized matrix-vector kernel at $M=1$ | 010 | [#11](https://github.com/golang-design/accel/issues/11) | **closed** | none needed |
| C16 | a **batched prefill**, or prefill and decode in one dispatch | 040 | [#16](https://github.com/golang-design/accel/issues/16) | **open** | prefills run alone. Chunked prefill bounds latency and recovers no throughput ([008 §5](008-scheduler.md)) |
| C18 | `Contiguous` on Metal | 010, 021 | [#19](https://github.com/golang-design/accel/issues/19) | **closed** | none needed. It was the only kernel in the corpus with no MSL artifact, so every graph that slices — which [004 §3.2](004-model-graph.md) requires — was refused at compile. Fixed upstream the day it was filed |
| C19 | a CPU backend that dispatches in parallel | 006 | [#20](https://github.com/golang-design/accel/issues/20) | **open** | none. Workgroups run serially, so a 4-token prefill of 0.6B takes 3560s; a user without a GPU meets this |
| C17 | GGUF's K-quant super-blocks | 010 | [#15](https://github.com/golang-design/accel/issues/15), not planned | **open, not scheduled** | read safetensors and quantize at load ([012](012-gguf.md)) |

**This table is a dated snapshot and accel is moving under it fast.** Within a
day of filing, four rows closed: `RoPE` took a positions tensor, `Attention`
accepted an f16 cache and a page table, and `Buffer.Access` removed the host
copy. As of **2026-08-24**.

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

The re-derivation also changed three verdicts that reading commits had got
wrong:

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
[010-D6](#decision-record) generates the table at M10.

**States.** `closed` — accel's exported surface does the thing, verified by the
probe. `open` — it does not. `won't fix, correctly` — see below.
`designed` — accel
[043](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md)
specifies it and the code does not do it yet. `open` — filed, not yet designed.
`won't fix, correctly` — see below.

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

**C13 replaces it as the blocking row, and it is worse in kind.** Every other
row in this table is a *refusal*: tgo asks for something and is told no. C13 is
an **acceptance**. `Attention` takes `Pages` on a prefill, drops it, reads the
cache contiguously, and returns a fluent wrong answer — measured at a worst
absolute difference of 0.74 between an identity and a reversed page table, with
`Selections()` naming the contiguous kernel both times.

Since a paged decode is only useful over blocks a paged prefill wrote, this
makes cross-request prefix sharing inexpressible, so it blocks
[016](016-prefix-cache.md) and [011 M10b](011-sequencing.md).

**A row leaves this table only when its test stops skipping.** Not when an issue
closes, not when a spec is written, and never because it was worked around.

### 2.2 Thirteen rows closed, three open, and what that took

As of accel HEAD `ee659f6`, re-audited by asserting values and reading
`Selections()`:

**Closed: C1, C2, C4, C5, C8, C10, C11, C12, C13, C14, C15** — and C9 is a
refusal that should stay. **Open: C3 and C6** (no sampling operator at the
tensor layer, one issue), **C7** narrowed to a bf16 GEMM alone, **C16** (a
batched step takes one token per sequence), and **C17** (GGUF, not scheduled).

Three of those closures change what tgo builds:

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

`speclint` enforces (3), which is the one a linter can see. (1) and (2) are
enforced by the re-audit in §2.2, which is where the gap surfaced.

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

Nine reports so far. Seven produced one upstream design decision — accel 043's *a scalar is a
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

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 010-D1 | one skipping test per register row | a prose list | the table cannot go stale silently |
| 010-D2 | the oracle is written from the mathematics, not from tgo's graph | share code with the builder | agreement becomes evidence rather than tautology |
| 010-D3 | tolerances are derived and commented with their term; a raised tolerance is a finding | tune until green | a numerics regression cannot be absorbed |
| 010-D4 | tier 3 never runs in CI | a nightly with a download | CI stays offline and under a minute |
| 010-D5 | the oracle is float64 and presumed right on disagreement | float32, matching the device | it is the simpler program; matching the device would import the device's bugs |
| 010-D9 | performance against vLLM is a measured table per axis, losses included | a headline throughput claim | tgo will lose raw NVIDIA kernel throughput for a long time and should win host overhead first; one number hides both ([§3.1](#31-performance-against-vllm-and-which-axes-are-winnable)) |
| 010-D8 | an open row cites an open issue; a blocked spec names a durable upstream record, issue **or** named artifact; a closed issue with an absent capability is **re-filed**, not commented on | comment on the closed thread; demand an open issue for every blocker | a comment creates no work item, and the register read as tracked while one issue was open ([§2.3](#23-commenting-on-a-closed-issue-is-not-reporting)) |
| 010-D7 | a probe asserts a value against the oracle and varies optional bindings | record the graph and read the refusal | the refusal-based rule was blind to C13 and reported a false green in its own spec |
| 010-D6 | the register is generated from the tests **at M10** | maintained by hand forever | it is the exact drift tgo exists to catch upstream. **Amended 2026-08-24:** generation needs tests, so until M10 `speclint` stands in — it checks the rows are numbered without gaps and that nothing in the tree cites a row that does not exist. A decision nothing enforces, in the spec about decisions nothing enforces, was the wrong thing to leave standing |
