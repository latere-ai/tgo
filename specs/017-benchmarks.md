---
title: "Benchmarks: measuring where a token's time goes, and comparing honestly"
status: drafted
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
| **decode tokens/second**, batch 1 and at batch | the headline, and the only one comparable across frameworks |
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
   harness in tree, the prompts checked in.

## 4.1 What the first real run measured

Qwen3-0.6B at f16 on accel's CPU backend, 2026-08-25:

| batch | tokens/s | p50 step | host | submit | device | readback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | 3560.8s | 0.00% | 0.01% | **99.99%** | 0.00% |

**The breakdown did its job on the first run.** 99.99% device says the cost is
accel's kernels and not tgo's loop, which is a conclusion a throughput number
cannot support — and the readback share [C6](010-conformance.md) predicted
would dominate is 0.00%, because the kernels are so much slower that nothing
else is visible. Both are findings, and neither would exist without §1's split.

The two causes are upstream and filed:
[accel#19](https://github.com/golang-design/accel/issues/19) (no Metal path for
`Contiguous`, so this is CPU) and
[accel#20](https://github.com/golang-design/accel/issues/20) (the CPU backend
dispatches serially).

**A comparison against vLLM is not worth running yet**, and §4 rule 1 is why:
publishing a loss of this magnitude against a framework whose kernels are years
of hand-tuned CUDA would report a fact about accel's backend maturity dressed as
a fact about tgo. The row waits for a backend that can finish a token.

## 5. Where the numbers go

`tgo bench` writes a Markdown table and a JSON record. The JSON is what a
regression check reads: a build that loses more than a stated fraction on any
axis fails, the same way the coverage gate works.

The Markdown goes into [011 §5](011-sequencing.md) at each milestone, so
performance has a history rather than a current value.

## 6. Tests

| test | what it catches |
| --- | --- |
| a disabled `Recorder` allocates zero and costs one branch | §2's claim, by `testing.AllocsPerRun` |
| an enabled `Recorder` allocates zero on the hot path | the instrument perturbing the measurement |
| the breakdown sums to the step, within a tolerance | arithmetic |
| percentiles against a known distribution | the aggregation |
| a synthetic model produces a full report end to end | the harness, with no real weights |

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 017-D1 | every measurement is a host/submit/device/readback breakdown | one throughput number | a regression can be attributed to tgo or to accel, which is the question the project exists to answer |
| 017-D2 | percentiles, never means | mean step time | the tail is what a user feels as a stall |
| 017-D3 | off by default, allocation-free when on | always on | an instrument that changes what it measures is not one |
| 017-D4 | losses are published, with hardware, model, precision and policy | headline wins | a benchmark without its conditions is decoration |
| 017-D5 | report a batch-size curve, not a point | the best batch size | [008 §1](008-scheduler.md)'s ceiling falls as context grows, and a point hides it |
| 017-D6 | JSON record gates regressions like the coverage gate | eyeball the table | a number nobody enforces drifts. **Unbuilt as of 2026-08-25**: the record is written and shaped for a checker (versioned schema, byte-stable encoding, a named note per unmeasured axis) and nothing reads it yet |
| 017-D7 | the engine takes a caller's `Recorder` through a session option | keep the recorder internal | the engine recorded all four terms and exported no way to set or read one, so `tgo bench` could only print the breakdown as missing — which is the single number 017-D1 says cannot attribute a regression |
