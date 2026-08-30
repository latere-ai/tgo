---
title: "Benchmarks: measuring where a token's time goes, and comparing honestly"
status: complete
layer: all
depends_on:
  - 000-decisions.md
  - 007-engine.md
  - 010-conformance.md
---

# Benchmarks

[000 §11](000-decisions.md) makes *faster than vLLM* the goal and
[010 §3.1](010-conformance.md) names the axes. This is the instrument.

## 1. One number is not a measurement

"Tokens per second" hides the thing tgo is betting on. A decode step is

$$t_\text{step} = t_\text{host} + t_\text{submit} + t_\text{device} + t_\text{readback}$$

and tgo's claim is that a compiled language wins $t_\text{host}$ while accel
decides $t_\text{device}$. A single throughput number cannot tell those apart,
so it cannot tell whether a regression is tgo's fault or accel's — which is the
question this project exists to answer.

**So every measurement is a breakdown**, and the breakdown is the deliverable.

## 2. What is instrumented

```go
package bench

// Step is one decode or prefill step, in nanoseconds.
type Step struct {
    Phase    Phase // Prefill or Decode
    Tokens   int   // tokens this step consumed
    Batch    int

    Host     time.Duration // sampling, detokenizing, bookkeeping
    Submit   time.Duration // building bindings and handing the plan to the queue
    Device   time.Duration // fence wait
    Readback time.Duration // logits to host
}

// Recorder collects steps with negligible overhead and no allocation on the
// hot path. Disabled by default; a nil Recorder costs one branch.
type Recorder struct{ /* ... */ }

func (r *Recorder) Step(s Step)
func (r *Recorder) Report() Report

// Report is the aggregate, with percentiles rather than means.
type Report struct {
    Prefill, Decode PhaseStats
    TTFT            Quantiles // time to first token
    Steps           int
}

type PhaseStats struct {
    TokensPerSecond float64
    Host, Submit, Device, Readback Quantiles // p50, p90, p99
    ShareOfStep     map[string]float64        // the breakdown, summing to 1
}
```

**Percentiles, not means.** A mean hides the tail, and the tail is what a user
experiences as a stall. p99 of a decode step is the number that decides whether
a stream feels smooth.

**Off by default and allocation-free when on.** An instrument that changes what
it measures is not an instrument. §6 has the test that pins it.

## 3. What is measured, and against what

| measurement | why |
| --- | --- |
| **decode tokens/second**, batch 1 and at batch | the headline, and the only one comparable across frameworks. Only batch 1 is measured today: `stream.go:314` is the one writer of a `bench.Step` and it sets `Batch: 1`, while `batch.go` and `scheduler.go` record nothing. Instrumenting the batched path is [027](027-batched-benchmarks.md) |
| **host share of a step** | [000 §11](000-decisions.md)'s bet, stated as a fraction |
| **time to first token**, cold and warm | cold includes model load and plan compile; warm is prefill only. They are different products |
| **readback share** | [010 C6](010-conformance.md) in one number, measured in production too ([009 §6](009-server.md)) |
| **resident memory** at a stated context and batch | the axis where a static binary should win outright |
| **plan compile time** per bucket, and cache hit rate | whether [007-D2](007-engine.md)'s bucket set is right |

Against **vLLM and sglang**, same model, same hardware, same prompts, same
sampling policy. Different frameworks default to different policies, and
comparing greedy against top-p is comparing nothing.

## 4. The rules that make a comparison honest

An inference benchmark is easy to make say what you want. These are binding.

1. **Publish losses.** Every row, including the ones tgo is behind on. A
   framework that reports only its wins is not reporting
   ([010-D9](010-conformance.md)).
2. **State the hardware, the model, the precision and the policy** with every
   number. A tokens-per-second figure without them is decoration.
3. **Warm up, then measure.** The first steps include plan compilation and page
   faults. Report cold separately rather than averaging it in.
4. **Same prompts, and say what they are.** Prompt length distribution decides
   the prefill/decode ratio, which decides everything. Fixed-length synthetic
   prompts flatter batching; real traces do not.
5. **No cherry-picked batch size.** Report a curve, since the shape is the
   finding — [008 §1](008-scheduler.md) shows the ceiling *falls* as context
   grows, which a single batch size would hide.
6. **The comparison is reproducible from the repository**: one command, the
   harness in tree, and the record carrying the prompt recipe
   (`cmd/tgo/record.go:60-66`). No prompt set is checked in: `tgo bench` builds
   its prompt at run time from one repeated word (`cmd/tgo/bench.go:121`), so
   rule 4's real traces are still absent and the recipe is what a reader
   reproduces from.

## 4.1 What the first real runs measured

Qwen3-0.6B at f16, 2026-08-25, on an 8-core Apple machine.

**On Metal**, 64 prompt tokens and 32 decode steps after 4 warm-up steps, before
and after accel's [#21](https://github.com/golang-design/accel/issues/21) fix:

| decode | tokens/s | p50 | p90 | p99 | host | submit | device | readback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 12.57 | 61.6ms | 78.9ms | 474.7ms | 0.93% | **15.61%** | 81.88% | 1.59% |
| **after** | **17.97** | 54.2ms | 61.5ms | **76.5ms** | 0.64% | **3.34%** | **94.62%** | 1.39% |

Prefill before the fix: 379.03 tokens/s, 168.9ms, 97.89% device.

**+43% throughput from one upstream change, and the p99 fell 84%** — from
474.7ms to 76.5ms, so the tail collapsed by more than the median moved. A
per-call reflection frame rebuild produces occasional very slow calls rather
than a uniform tax, so removing it removed the variance. [017-D2](#decision-record)
chose percentiles for a different reason; this is the first time they paid for
themselves.

Device is now **94.62%** of a decode step, which is the shape a decode step
should have: the framework spends its time in kernels rather than around them.

Cold start is 27.6s (model load and every first compile); warm time to first
token is 169ms.

**Three findings the breakdown produced that a throughput number could not:**

1. **Submit is 15.61% of a decode step and 1.12% of a prefill.** Per-dispatch
   cost is roughly fixed, so it is amortised over a 64-token prefill and paid in
   full by a one-token decode — which is the shape [008 §1](008-scheduler.md)
   argues batching fixes, visible in a measurement rather than in an argument.
   **Filed as [accel#21](https://github.com/golang-design/accel/issues/21) and
   since fixed**, which is the row above. accel attributed it further than this
   instrument could: not a per-*step* submission cost as tgo assumed, but a
   per-*node* one — the cost of calling Objective-C from Go through reflection,
   about five message sends per dispatch over a ~790-node graph. The lesson kept
   in [017-D1](#decision-record)'s terms: the breakdown says *how much* and
   never *where*, so the useful report is the number and the shape, not a theory
   about the cause. Filed with the ratio,
   because a ~790-node graph resubmitted per token is a per-dispatch cost meeting
   a step that does little arithmetic per node — and because
   [000 §11](000-decisions.md) says the parts that are not matrix multiplication
   are where tgo should win, so measuring itself losing 15% of a step to
   plumbing is the finding, not a footnote.
2. **Readback is 1.59%, not the dominant term.** [C6](010-conformance.md) — the
   608 KB of logits per token — is real and is an order of magnitude smaller
   than the submit overhead beside it. The register row stands; its priority
   does not.
3. **p99 is 7.7× p50** (474ms against 61.6ms). Percentiles were
   [017-D2](#decision-record)'s reason for existing, and a mean would have hidden
   this entirely.

**On the CPU backend**, before and after accel's
[#20](https://github.com/golang-design/accel/issues/20) worker pool:

| | before | after | |
| --- | ---: | ---: | --- |
| prefill, per prompt token | 476.2s | **24.4s** | **19.5×** |
| decode step, p50 | — | 108.8s | — |
| device share | 99.99% | 99.98% | — |

**19.5× against the 7.5× accel measured**, and the difference is the workload
rather than either benchmark being wrong: theirs was one elementwise kernel over
a flat buffer, where a pool's fixed cost is a real fraction of the work, and a
transformer prefill is ~790 nodes of GEMM-shaped work that amortises it.

**Device is 99.98% of a decode step**, so accel's question — whether the time is
inside dispatches or between them — is answered: inside. The serial node walk
they flagged as deliberate is not what tgo waits on, and there is nothing
measurable between dispatches to recover.

Against Metal on the same machine: **CPU decode is ~2000× slower per step**, and
CPU prefill ~9000× slower per token. The pool moved this from unusable in any
sense to unusable for inference and fine for correctness — which matters,
because [010 §4](010-conformance.md)'s tier 1 runs on the CPU backend and just
got 19.5× cheaper.

> Metal produced nothing at all until 2026-08-25: `Contiguous` was the only
> kernel in accel's corpus with no MSL artifact, and
> [004 §3.2](004-model-graph.md) requires it before the LM head. Filed as
> [accel#19](https://github.com/golang-design/accel/issues/19), fixed upstream
> the same day, and these are the first numbers after it.

**A comparison against vLLM is still not worth running**, and §4 rule 1 is why.
12.57 tokens/s on a 0.6B model against years of hand-tuned CUDA would report a
fact about kernel maturity dressed as a fact about tgo. The row waits for
[011 M13](011-sequencing.md), and what would make it meaningful first is the
submit overhead in finding 1 — the one axis [000 §11](000-decisions.md) says tgo
should win.

## 5. Where the numbers go

`tgo bench` writes a Markdown table and a JSON record. The JSON is what a
regression check reads: a build that loses more than a stated fraction on any
axis fails, the same way the coverage gate works.

The Markdown goes into [011 §3](011-sequencing.md)'s milestone log, at each
wave that changes the decode path. Only Wave 4 carries a table
(`011-sequencing.md:696`), so performance has a current value and not yet a
history.

## 6. Tests

| test | what it catches |
| --- | --- |
| a disabled `Recorder` allocates zero and costs one branch | §2's claim, by `testing.AllocsPerRun` |
| an enabled `Recorder` allocates zero on the hot path | the instrument perturbing the measurement |
| the breakdown sums to the step, within a tolerance | arithmetic |
| percentiles against a known distribution | the aggregation |
| a synthetic model produces a full report end to end | the harness, with no real weights |

## Outcome

Package `bench` is tgo's decode instrument. It records the four terms of every
step, aggregates them as percentiles, and `tgo bench` renders the aggregate as a
Markdown table and a versioned JSON record. The package landed in Wave 1
(2026-08-24); `WithRecorder` and the `tgo bench` command landed in Wave 4
(2026-08-25), which is when the instrument first measured the real Qwen3-0.6B
checkpoint. §4.1 above is that record and stays there rather than moving here,
because [010](010-conformance.md), [011](011-sequencing.md),
[020](020-device-sampling.md), [026](026-image-tokens.md),
[028](028-performance-gate.md) and `batch.go:402` all cite it by section number.

**What shipped**, section by section:

| section | what landed | where |
| --- | --- | --- |
| 1 | the four terms, with host as the residual rather than a fifth clock | `bench/bench.go:57`, `session.go:599-604`, `stream.go:305-317` |
| 2 | `Recorder`, `Report`, `PhaseStats`, `Quantiles`: off in the zero value, allocation-free when on | `bench/bench.go:88`, `:106`, `:123`; `bench/report.go:25-101` |
| 3 | four of six rows: throughput, host share, cold and warm TTFT, resident memory, and the readback share in production | `cmd/tgo/record.go:83-89`, `:109-117`; `server/metrics.go:124-136`, `server/generate.go:335` |
| 4 | rules 2, 3 and 4 in the harness: the conditions beside every number, warm up then `Reset`, the prompt recipe in the record | `cmd/tgo/record.go:71-89`, `cmd/tgo/bench.go:205-216`, `:113-133` |
| 4.1 | the first runs on Metal and on the CPU backend, and accel [#19](https://github.com/golang-design/accel/issues/19), [#20](https://github.com/golang-design/accel/issues/20) and [#21](https://github.com/golang-design/accel/issues/21) filed from them | §4.1 above; `011-sequencing.md:686-770` |
| 5 | the Markdown table on stdout, and the `tgo.bench/1` record on `--json`, byte-stable | `cmd/tgo/bench.go:150-164`, `cmd/tgo/record.go:22`, `:260` |
| 6 | all five rows, and the breakdown twice: through the real engine on synthetic weights, and through `cmdBench` against a fake one | `bench/bench_test.go:44`, `:81`, `:134`, `:206`; `engine_test.go:824`; `cmd/tgo/bench_test.go:178` |
| DR | D1, D2, D3, D4 and D7 built and tested | `bench/report.go:41-45`, `:184`; `bench/bench.go:106-119` with `engine_test.go:880`; `cmd/tgo/record.go:71-89`; `options.go:161` |

**What diverged** from the design, and why the code is right:

- §2's listing is a subset of what shipped. `Quantiles` carries `N`,
  `PhaseStats` carries `Steps` and `Tokens`, `Report` carries `Dropped`, and
  `Recorder` gained `Enabled`, `Reset` and `TTFT` (`bench/bench.go:119-160`,
  `bench/report.go:25-70`). Each answers a question the listing leaves
  unanswerable: how many samples are behind a quantile, and whether the report
  covers the whole run or a prefix of it.
- §4 rule 1 is enforced as *name every unmeasured axis*, not as *publish every
  row*. The record carries one named note per axis this build cannot measure
  (`cmd/tgo/record.go:131`, `:138`, `:156`, `:173`) and
  `cmd/tgo/record_test.go:392` fails if a note is dropped. A rule about rows
  nobody has measured cannot be checked, and a rule about naming the gap can.
- 017-D3 says off by default, and `server/generate.go:50` gives every request a
  `Recorder`. The default is still off in the library (`engine_test.go:880`);
  the server is one caller that opts in, because [009-D7](009-server.md) needs
  the readback share from production traffic and the cost is one recorder per
  request.
- The step p50/p90/p99 in the Markdown table, §4.1's own figures included, is
  the sum of four independently sorted quantiles (`cmd/tgo/record.go:380`
  `stepAt`), not a percentile of measured step times. `PhaseStats` reports the
  four terms separately by 017-D1, so a modeled step is the only step figure
  derivable from it. §4.1's p99-against-p50 ratio is a ratio of two such steps
  and stands.

**Not built.** Nothing that 017 owns. One item left this paragraph on
2026-08-28: the two stale strings claiming the engine exports no recorder.

Owned elsewhere, four items, each with the spec that owns it.

- **017-D5's batch curve**: [027](027-batched-benchmarks.md). It needs a
  `bench.Step` from `batch.go` and `scheduler.go` carrying the real slot count;
  today `stream.go` is the only writer of one and it hardcodes `Batch: 1`.

  The recorder window defect that sat beside it here is **fixed** (2026-08-27).
  `bench.Recorder` kept the *first* `capacity` steps while
  `server/generate.go` said a longer completion "reports quantiles over its most
  recent steps", so past `recorderCapacity` the server published a completion's
  warm-up as what the device was doing now. It is a ring now, per 027-D5, which
  keeps the newest rather than refusing to publish a truncated report — a long
  request is the one whose current behaviour a reader wants, and refusing would
  report nothing at all in that case.
- **Two stale strings in `cmd/tgo`**, and they are [027](027-batched-benchmarks.md)'s:
  `singleBatchNote` (`cmd/tgo/record.go:131`) and the `--batch` refusal
  (`cmd/tgo/bench.go:88`) both say "specs/008-scheduler.md is drafted and
  unbuilt" while 008 is `complete`, and `cmd/tgo/bench_test.go:40` asserts that
  sentence verbatim, which is why nothing caught it. 027 removes the refusal, so
  the text goes with it.

  **The other pair is corrected** (2026-08-28). `noBreakdownNote` and
  `cmd/tgo/main.go`'s discrepancy list both said specs/007-engine.md §1 exports
  no way to set or read a `bench.Recorder`, which was true when
  [017-D7](#decision-record) was filed and stopped being true in Wave 4 when
  `WithRecorder` shipped. The note now names the one thing that can still
  produce an empty breakdown — a session opened without a recorder — and
  `TestAnUnmeasuredAxisNamesAGapThatIsStillOpen` fails on the old text. The
  test beside it checked only that a note cited a spec, which a note naming a
  closed gap satisfies.
- **017-D6's regression gate**: [028](028-performance-gate.md). The record is
  written and shaped for a checker that does not exist (`cmd/tgo/record.go:205`),
  there is no benchmark counterpart to the coverage gate, and neither
  `.github/workflows/ci.yml` nor `ci-metal.yml` runs `tgo bench`.
- **Plan compile time per bucket, and plan-cache hit rate**: §3's last row waits
  on [007 §1](007-engine.md) exporting plan-cache statistics. The cache is
  unexported and a `Model` reports nothing about it, so a miss reaches this
  process only folded into the cold time to first token, where the model load
  cannot be separated from it (`cmd/tgo/record.go:173` `noPlanStatsNote`).
- **The vLLM and sglang comparison**: gated by [011 M13](011-sequencing.md), and
  §4.1 argues the row is not worth running before the submit overhead is
  addressed (`cmd/tgo/record.go:138` `comparisonNote`).

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 017-D1 | every measurement is a host/submit/device/readback breakdown | one throughput number | a regression can be attributed to tgo or to accel, which is the question the project exists to answer |
| 017-D2 | percentiles, never means | mean step time | the tail is what a user feels as a stall |
| 017-D3 | off by default, allocation-free when on | always on | an instrument that changes what it measures is not one. **Amended 2026-08-28**: `bench.Recorder` now takes a mutex on each write and each read. The original rule was "one recorder per stepping goroutine, so the hot path is free of a lock", which assumed the stepping goroutine and the reader are the same one. [022](022-batched-serving.md) makes them structurally different — one driver goroutine steps the whole batch and writes every request's recorder, and the request's own goroutine reads it — so no discipline fixes it. The allocation claim is unchanged: storage is still fixed at construction |
| 017-D4 | losses are published, with hardware, model, precision and policy | headline wins | a benchmark without its conditions is decoration |
| 017-D5 | report a batch-size curve, not a point | the best batch size | [008 §1](008-scheduler.md)'s ceiling falls as context grows, and a point hides it. **Unbuilt as of 2026-08-27**, and no longer for the reason the code gives: 008 is `complete`, so the curve is buildable. [027](027-batched-benchmarks.md) owns the batched instrument and the stale refusal text the Outcome names |
| 017-D6 | JSON record gates regressions like the coverage gate | eyeball the table | a number nobody enforces drifts. The record is written and shaped for a checker — versioned schema, byte-stable encoding, a named note per unmeasured axis — and nothing reads it. [028](028-performance-gate.md) owns the checker, the way [027](027-batched-benchmarks.md) owns D5's instrument |
| 017-D7 | the engine takes a caller's `Recorder` through a session option | keep the recorder internal | the engine recorded all four terms and exported no way to set or read one, so `tgo bench` could only print the breakdown as missing — which is the single number 017-D1 says cannot attribute a regression. `WithRecorder` shipped in Wave 4 (`options.go:161`) and every `tgo bench` run now carries the four terms; the sentence survives verbatim in `cmd/tgo/record.go:156` and `cmd/tgo/main.go:53`, where it is now false |
