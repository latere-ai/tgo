---
title: "Benchmarking the batched path: the throughput curve 017-D5 designed and nothing measures"
status: drafted
layer: engine
depends_on:
  - "000-decisions.md"
  - "008-scheduler.md"
  - "017-benchmarks.md"
---

# Benchmarking the batched path

[017](017-benchmarks.md) built the instrument and measured one sequence.
[008](008-scheduler.md) built the batch. Nothing connects them.

`bench` ships and is at 100% statement coverage, but every observation it holds
comes from one `Session`: `stream.go:306-318` is the only writer of a
`bench.Step` in the tree, and it writes `Batch: 1` as a literal. `batch.go` and
`scheduler.go` emit nothing. `tgo bench --batch 8` is refused at
`cmd/tgo/bench.go:86-91`.

So [017-D5](017-benchmarks.md) — report a curve, because
[008 §1](008-scheduler.md) shows the throughput ceiling *falls* as context grows
and a point hides that shape — designs a measurement that has never been taken.
This spec takes it.

It also fixes one live defect the same audit found. `server/generate.go:33-36`
says a completion longer than `recorderCapacity` "reports quantiles over its
most recent steps". `bench.Recorder.Step` (`bench/bench.go:123-135`) keeps the
**first** `capacity` observations and counts the rest in `Dropped`, and
`Server.report` (`server/generate.go:335-344`) publishes `d.*.P50` without
consulting `Dropped`. A 2000-step completion publishes the device's behaviour
from two thousand steps ago as its current one.

## 1. What is already true

| fact | where |
| --- | --- |
| `Step` carries `Batch int`, and no writer ever sets it above 1 | `bench/bench.go:57-66`, `stream.go:315` |
| the four terms are measured around one submission | `session.go:599-701` (`timings`), `stream.go:306-318` |
| a recorder reaches the decode loop through a session option | `options.go:144-162` (`WithRecorder`), `session.go:107-146` |
| `Scheduler.Step` knows the mix it ran | `scheduler.go:190-233` (`StepResult.PrefillTokens`, `Decodes`) |
| `Batch.Step` owns the submit, the wait and the readback and times none of them | `batch.go:300-436` (submit at `:390`, readback at `:413`) |
| the record is an axis with one point and a note saying why | `cmd/tgo/record.go:119-135`, `cmd/tgo/bench.go:237-244` |

`nextStep` (`schedule.go:76-114`) places prefill chunks first, then decodes into
whatever rows are left. A decoding slot that does not fit **is skipped and waits
a step** (`schedule.go:103`). That single line is why §2 needs two latency
definitions rather than one.

## 2. What a step means when it produces several tokens

017 §3 names three quantities: time to first token, inter-token latency, and
tokens per second. Under batching each splits in two, and the split is not a
presentational choice.

Let a step $k$ carry $d_k$ decoding slots and $c_k$ prompt tokens, and take
$t_k$ as its modeled time, the sum of 017 §1's four terms.

**Aggregate throughput** is what the device delivered:

$$\text{tokens/s}_\text{agg} = \frac{\sum_k d_k}{\sum_k t_k}$$

**Per-request inter-token latency** is what one caller waits between two of its
own tokens. Slot $s$ produced a token at step $k$ and the next at step $k'$, so

$$\text{ITL}_s = \sum_{j=k+1}^{k'} t_j$$

and $k' > k+1$ whenever `nextStep` skipped that slot. The two are different
measurements of the same steps, and the batch size moves them in opposite
directions: aggregate throughput rises with $B$ until the ceiling
[008 §1](008-scheduler.md) computes, while every request's ITL rises
monotonically because $t_k$ grows with the rows in the step.

**Time to first token** under batching is per-request only. There is no
aggregate form of it, and the existing stream — `Recorder.TTFT`
(`bench/bench.go:143-152`) — already records it per request.

**The curve plots aggregate decode throughput against batch size.** That is
017-D5's shape and [008 §1](008-scheduler.md)'s claim. Per-request ITL p50 and
p99 ride in the same table, because a throughput curve with no latency column is
the flattering half of a measurement and 017 §4 rule 1 binds this file.

## 3. Where the instrument goes

Three places can hold it.

| site | what it can see | what it costs |
| --- | --- | --- |
| a wrapper around `Scheduler.Step` | one wall-clock duration per step | nothing, and it is the single number [017-D1](017-benchmarks.md) says cannot attribute a regression |
| `Batch.Step` | submit, device, readback | it is the mechanism and does not know prefill from decode: `Batch` has no `schedState`, so it cannot classify the step (008-D8) |
| `Scheduler.Step` | the mix, from `schedPlan`, and the phase | it must obtain the three device-side durations from `Batch`, which is one unexported seam |

So the instrument is split the way the work is: `Batch` measures, `Scheduler`
classifies and records. `Batch` grows an unexported
`step(work []Work) ([][]float32, timings, error)` holding the present body, and
the exported `Batch.Step` becomes a two-line wrapper that discards the timings.
`timings` already exists in this package (`session.go:599-605`) and stays
unexported, so no public shape changes and no caller of `Batch.Step` recompiles.

`Scheduler` gains a recorder through `SchedulerOptions`, matching how a session
gets one (`options.go:148-162`): the recorder is the **caller's**, its window and
its lifetime are the caller's, and a nil one costs the branch
`Recorder.Enabled` already documents. That is what makes the batched path
measurable by a user of the library and not only by `cmd/tgo`.

**The submit region counts the lock wait**, as `session.go:621-623` already
does. `Scheduler.Step` and `Batch.Step` both take a mutex, and the model lock at
`batch.go:366` is contended by every other slot's step. Excluding it would make
the batched submit number incomparable with the single-session one, and
comparability along the axis is the entire purpose of the curve.

## 4. Prefill riding along with decodes

`nextStep` puts a 512-token chunk and the decodes beside it in one dispatch —
that is [C16](010-conformance.md), and [008 §5](008-scheduler.md) is why. The
consequence for this instrument is that some steps are neither prefill nor
decode, and `bench.Phase` has only those two values.

Folding a mixed step into `Decode` charges the chunk's device time to the decode
distribution while its $c_k$ prompt tokens are not in the numerator, so the
headline decode throughput falls whenever a prefill is in flight and no reader
can tell that from a slower device. Folding it into `Prefill` — which is what
today's code would do by default, since `report.go:83-89` sends everything that
is not `Decode` to prefill — deletes those decodes from the decode statistics
entirely.

**So `bench` gains a third phase, `Mixed`**, for a step with $c_k > 0$ and
$d_k > 0$, and `Report` gains a `Mixed PhaseStats` beside the two it has. The
curve plots decode-only steps, and reports mixed steps as their own row: a curve
whose prefill share varies with the batch size measures the harness's arrival
pattern rather than the engine.

Three call sites classify by phase today and each is a silent misattribution if
it is not changed with the enum:

- `report.go:83-89` — the `if s.Phase == Decode { ... } else { prefill }`
  becomes an explicit three-way `switch` with a named default. The comment at
  `bench/bench.go:31-33` calls `Prefill` "the phase whose statistics matter
  least", which is the reason the zero value is safe and also the reason a
  fall-through default is not.
- `cmd/tgo/record.go:242-251` — `hasBreakdown` iterates `{Prefill, Decode}`, so
  a run made only of mixed steps renders `noBreakdownNote` and reports itself as
  unmeasured.
- `cmd/tgo/record.go:313-319` — `renderMarkdown` writes two phase sections and
  needs a third.

## 5. How the harness drives a batch

`tgo bench --batch N` admits $N$ synthetic conversations into one
`Scheduler`, each with the same `--prompt-tokens` prompt
(`cmd/tgo/bench.go:107-133` builds it), **all admitted before the first step**,
each generating `--tokens` tokens. The run ends when every slot has finished.
One row per requested batch size; `--batch 1,2,4,8` gives the curve four points,
and `--batch 1` keeps the present behaviour.

**What this measures**: the steady-state ceiling of a saturated batch of equal
work. That is exactly the quantity [008 §1](008-scheduler.md) predicts, so it is
the quantity that can falsify the prediction.

**What it does not show**, stated because 017 §4 rule 4 says fixed-length
synthetic prompts flatter batching and real traces do not:

1. queueing. Every request is admitted at once, so no request waits for a slot,
   and the ITL distribution has no admission delay in it.
2. the prefill/decode mix a server sees. All $N$ prefills run in the first few
   steps and every step after them is decode-only, so the `Mixed` row is a
   handful of steps at the head of the run rather than a steady fraction.
3. a length distribution. Equal prompts finish together, so the batch never
   drains one slot at a time — which is the regime where the ceiling is
   *between* two points of this curve.

An arrival generator with a length distribution would show all three. It is not
in this spec (§9), and the record says so in a field rather than by omission
(027-D7).

## 6. The recorder window

The defect is stated at the top. The two candidate fixes are not symmetric.

**Refusing to publish quantiles when `Dropped > 0`** makes `Server.report`
silent for exactly the long completions whose step time is most interesting, and
`tgo_decode_step_seconds` would stop moving the moment a request got long. A
metric that switches itself off under load is worse than one with a stated
window.

**A ring** makes `server/generate.go:33-36`'s existing promise true, changes
nothing for `cmd/tgo/bench` — which sizes the recorder above its window at
`cmd/tgo/bench.go:186` and never drops — and costs one modulo on the hot path.
The ring is the fix.

`Recorder.steps` and `Recorder.ttfts` keep their fixed length and gain a write
index; `Step` writes at `n % len(steps)` and increments `Dropped` once the
window has wrapped. `Report` reads the whole slice and does not care about
order: `phaseStats` sums (`bench/report.go:106`) and `quantiles` sorts an independent copy (`bench/report.go:159`).
Allocation-free stays structural, which is [017-D3](017-benchmarks.md), and
`TestEnabledRecorderAllocatesZero` (`bench/bench_test.go:81`) is what keeps it
honest.

**`Dropped` keeps its name and changes its meaning**, so four documents flip
from prefix to suffix and all four must move in the same change:

- `bench/report.go:63-67` — "the report describes a prefix of the run" becomes
  the most recent `capacity` observations, and `Dropped` counts what the window
  overwrote.
- `bench/bench.go:80-86` — same sentence in the `Recorder` doc.
- `bench/bench.go:102-106` — `NewRecorder`'s warm-up advice. Under a ring a
  recorder that fills during warm-up is harmless, because `Reset` clears it.
- `cmd/tgo/record.go:342-347` — the Markdown warning says the percentiles
  "describe the start of the run". They now describe its end, over a stated
  window, which is a qualification rather than an invalidation.

`Server.report` needs no change once the ring is in. That is the test of the
fix: the root cause is in `bench`, and patching the server would leave every
other reader of a full recorder wrong.

## 7. What the JSON record gains

`recordSchema` (`cmd/tgo/record.go:17-22`) goes from `tgo.bench/1` to
`tgo.bench/2`. The fields added:

| field | why |
| --- | --- |
| `batches[].slots` | the scheduler's slot count, which is the batch that was *asked for*; `batch` stays the number of sequences that ran |
| `batches[].mix` | `prefill_tokens`, `decodes`, `mixed_steps`, `decode_only_steps` — 008 §5's mix, reported rather than inferred, from `StepResult` |
| `batches[].inter_token` | per-request ITL as `bench.Quantiles`, §2's second measurement |
| `batches[].report.mixed` | §4's third phase |
| `batch_axis.arrival` | §5's three limitations, as a note (027-D7) |

**There is no reader of the record in `cmd/tgo/record.go`.** `encodeRecord`
(`:260-266`) only writes. Every reader today is a test unmarshalling into
`benchRecord` — `cmd/tgo/bench_test.go:201`, `:288`, `cmd/tgo/record_test.go:155`
and `:200` — and 028's regression gate is the future one. Neither of those two
breaks on the bump: `:155` decodes into a struct of named fields and `:200`
compares `m["schema"]` against the `recordSchema` constant rather than a
literal. What makes the version
bump safe is therefore an invariant rather than a compatibility shim:
**`/2` is additive over `/1`.** No `/1` field is renamed, retyped, or given a
new meaning. A reader that knows `/1` decodes a `/2` document and finds every
field it knows, which is what lets 017-D6's gate reject an unknown schema string
without also having to reject a superset of one it understands.

The invariant is pinned by a checked-in `/1` document, not by review: §8's
golden-fixture test decodes it into the current `benchRecord` and asserts every
`/1` field is populated. A field that moves fails that test.

## 8. Tests

| test | what it asserts |
| --- | --- |
| `TestRecorderKeepsTheMostRecentSteps` | record `capacity + k` steps with strictly increasing `Device`; the report's quantiles come from the last `capacity` and none of the first `k` values appears. This is the defect in §6, and it fails against today's code |
| `TestFullRecorderCountsDrops` (**amended**, `bench/bench_test.go:108`) | `Dropped` still counts every overwritten observation. The content assertion at `:121-125` — `Host.P99 == 2µs`, "the recorder keeps the prefix" — inverts to the newest window |
| `TestEnabledRecorderAllocatesZero` (**confirmed**, `bench/bench_test.go:81`) | `AllocsPerRun` is still 0 with the modulo in the write path |
| `TestReportJSONStable` (**amended**, `bench/bench_test.go:405`) | the marshalled report carries `mixed` beside `prefill` and `decode`, in declaration order. A field added to `Report` is not additive for a test that asserts bytes |
| `TestEmptyReportMarshals` (**amended**, `bench/bench_test.go:454`) | the third phase's `share_of_step` is the four named keys and not `null`, for the same reason the first two are |
| `TestServerPublishesTheRecentWindow` | drive `Server.report` with a recorder holding more than `recorderCapacity` steps whose device time changes partway; the published `tgo_decode_step_seconds` reflects the recent steps. Fails today because the oldest steps are the ones kept |
| `TestSchedulerRecordsAStepPerDispatch` | one `Scheduler.Step` emits exactly one `bench.Step`, with `Batch` equal to the slots that contributed and the four terms all non-zero |
| `TestAMixedStepIsNeitherPhase` | a step carrying a chunk and two decodes lands in `Report.Mixed` and in neither `Prefill` nor `Decode`; §4's misattribution, caught at the classifier |
| `TestPhaseClassificationIsExhaustive` | an unknown `Phase` value is not silently counted as prefill (`bench/report.go:83-89`) |
| `TestBatchedHarnessProducesOneRowPerBatchSize` | `--batch 1,2,4` yields `batch_axis.points` of length 3 and three `batches` entries, each with its own report. No point is copied from another |
| `TestInterTokenLatencyCountsSkippedSteps` | a slot that `nextStep` left out of a step (`schedule.go:103`) has that step's time in its ITL. Decided from integers, no device |
| `TestParseBenchAcceptsABatch` (**replaces a row**, `cmd/tgo/bench_test.go:39`) | `--batch 8` parses. The refusal row leaves `TestParseBenchRefusals`, and a new refusal covers a batch the pool cannot hold |
| `TestRecordSchemaIsAdditive` | the checked-in `tgo.bench/1` document decodes into `benchRecord` with every `/1` field populated (§7) |
| `TestNoNoteClaimsAnUnbuiltScheduler` | no string in `cmd/tgo` says 008 is drafted or unbuilt. §9 is why this is a test and not a checklist |

## 9. Four strings that describe a tgo that no longer exists

008 shipped on 2026-08-27 (`008 §8`) and 017-D7 exported `WithRecorder`
(`options.go:161`). Three of the four notes below still say neither happened,
and they are printed in the deliverable.

| string | where | what it wrongly claims |
| --- | --- | --- |
| `singleBatchNote` | `cmd/tgo/record.go:130-135` | "tgo does not batch yet. specs/008-scheduler.md is drafted and unbuilt, so exactly one sequence is ever in flight" |
| the `--batch` refusal | `cmd/tgo/bench.go:82-91` | "tgo runs one sequence at a time: specs/008-scheduler.md is drafted and unbuilt, so there is no batched path to measure" |
| `noBreakdownNote` | `cmd/tgo/record.go:155-162` | the engine's "public surface (§1) exports no way to set and no way to read" a recorder |
| `noPlanStatsNote` | `cmd/tgo/record.go:173-177` | plan compile time per bucket and cache hit rate are absent. **Still true**: `Model.cache` is unexported (`plan.go:86-103`) and nothing reports statistics about it |

The same stale claim appears in two more places and a rewrite that misses them
leaves the package documentation contradicting the record it documents:
`cmd/tgo/engine.go:79-90` and `cmd/tgo/main.go:48-56`.

`cmd/tgo/bench_test.go:39` pins the refusal:

```go
{"a batch tgo cannot run", []string{"--batch", "8", "d"}, "008-scheduler.md is drafted and unbuilt"},
```

That row is deleted rather than reworded, because under §5 `--batch 8` is
accepted. What replaces it is an acceptance assertion plus a refusal that is
still true — a batch whose slots the block pool cannot reserve.

`singleBatchNote` is **replaced, not deleted**. `cmd/tgo/record.go:182-186`
argues that a checker unable to tell "this axis was not measured" from "this
axis does not exist" stops gating the axis the day it disappears, and that
argument survives the scheduler landing. The new text is §5's three
limitations.

## 10. What this spec does not own

- **A baseline, and the CI regression gate over it.** 017-D6 is unbuilt and
  spec 028 is where it lands. This spec produces the
  record; nothing here fails a build.
- **An arrival generator.** §5.2 names what a Poisson arrival over a prompt
  length distribution would show. It needs a checked-in trace and a seeded load
  generator, and it is a second measurement rather than a bigger version of this
  one.
- **Plan compile statistics.** 017 §3 asks for them,
  `noPlanStatsNote` correctly says they are absent, and exporting them is
  [007](007-engine.md)'s surface to change.
- **Sampling on the batched path.** [008 §9](008-scheduler.md) makes it a
  decision with a measurement attached. The harness samples on the host, which
  is what `Scheduler.Feed` expects, and the host share it produces is one input
  to that decision rather than the decision.
- **A comparison against vLLM or sglang.** `comparisonNote`
  (`cmd/tgo/record.go:137-141`) stands unchanged.

## 11. Scope

One person, one pass, in this order: the ring and its tests (`bench`, self
contained, fails first); the `Mixed` phase and the three classifiers; the
`Batch.step`/`Scheduler` seam; the harness and the flag; the four strings. The
device-side work is one unexported method and one options field, and everything
in §2 and §5 is decided from integers, which is what
[008 §8](008-scheduler.md) found made the scheduler testable at all.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 027-D1 | a `bench.Step` is one dispatch, with `Batch` holding the slots that contributed; per-request latency comes from the harness's own per-slot clock | one `Step` per token per slot | the four terms are properties of a submission, and splitting a step's device time across its slots would report a number nobody measured. The curve plots aggregate throughput and carries per-request ITL beside it (§2) |
| 027-D2 | `Batch` measures the three device-side terms through an unexported `step`; `Scheduler` classifies the phase and writes the `bench.Step` | instrumenting `Batch.Step` directly, or wrapping `Scheduler.Step` from outside | `Batch` is mechanism and cannot tell prefill from decode (008-D8); a wrapper sees one wall-clock number, which is what 017-D1 exists to refuse. The exported surface does not change and a library user turns it on with `SchedulerOptions.Recorder` |
| 027-D3 | the harness admits N equal synthetic conversations at once and names the three things the curve does not show | a seeded arrival generator over a prompt length distribution | 017 §4 rule 4 binds this: a saturated batch of equal work is the regime 008 §1 predicts, and it is not the regime a server sees. Saying so in the record is the difference between a limitation and a flattering measurement |
| 027-D4 | a step that carries prompt tokens and decodes is a third phase, `Mixed`, and the curve plots decode-only steps | folding mixed steps into `Decode`, or into `Prefill` as today's classifier would | the first charges the chunk's time to decode throughput without its tokens; the second deletes those decodes from the decode statistics. Three call sites change with the enum (§4) |
| 027-D5 | `Recorder` becomes a ring | refusing to publish quantiles while `Dropped > 0` | refusing makes the server's step metric go silent for exactly the long completions that matter. A ring makes `server/generate.go:33-36`'s promise true, costs one modulo, and keeps 017-D3's allocation-free claim structural. `Dropped` keeps its name and inverts its meaning, so four documents move together |
| 027-D6 | the schema string becomes `tgo.bench/2` and `/2` is additive over `/1` | keeping `/1` and adding fields | a version nothing increments is decoration, and 017-D6 makes the string the first thing a gate reads. Additive-only is what lets a `/1` reader decode a `/2` document, and a checked-in `/1` fixture is what pins it rather than review |
| 027-D7 | an axis that is now measurable keeps its note field, with new text | deleting `singleBatchNote` once the scheduler ships | `cmd/tgo/record.go:182-186`'s argument does not expire: a record that cannot distinguish "not measured" from "does not exist" stops gating the axis silently. The note stops saying tgo cannot batch and starts saying what the arrival pattern hides |
