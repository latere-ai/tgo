---
title: "The decisions everything else is built on"
status: normative
layer: all
depends_on: []
---

# Decisions

This file is normative. A spec in this directory that contradicts it is wrong,
and the fix is to change this file first, with the reasoning, or to change the
spec.

Each decision states what was chosen, what was rejected, and the constraint that
decided it. A decision with no rejected alternative is not a decision; it is a
description.

---

## 1. tgo is a client of accel, not a layer that routes around it

Every operation that touches a device goes through
[`golang.design/x/accel`](https://pkg.go.dev/golang.design/x/accel). tgo
contains **no kernels, no backend code, and no device-conditional numerics.**

**Rejected:** a small private kernel package for the operations accel does not
have yet. It would work, and it would destroy the reason tgo exists.

tgo is accel's validating consumer. A framework that quietly writes its own
kernel each time accel is missing one stops reporting anything: accel's gaps
become tgo's private assets, and nobody learns which of accel's abstractions
survive contact with a real model. So when accel cannot express something, the
sequence is fixed:

1. tgo writes the test that fails, or the operation it cannot record.
2. The gap is filed against the owning accel spec, by number.
3. tgo's [conformance suite](010-conformance.md) keeps a named, skipping test
   that says which accel spec owns the gap.
4. tgo waits.

`specs/010-conformance.md` is the register of what tgo currently cannot do and
why. It is the primary output of this project, alongside the running model.

> A consequence worth stating plainly: tgo will sometimes be slower than it
> could be, and the correct response is to make accel faster.

## 2. cgo-free, all the way down

`CGO_ENABLED=0` builds and runs everything, on every supported GOOS. Inherited
from accel decision 2 and re-gated here, because a dependency added to tgo can
break it independently.

**Rejected:** a cgo fast path for tokenization or GGUF dequantization. Both are
pure Go at acceptable cost, and one cgo dependency ends cross-compilation for
the whole binary.

This forbids, permanently: `sentencepiece`, `tokenizers` (the Rust one),
`llama.cpp`, BLAS, and every ONNX runtime.

## 3. Safetensors is the weight format for v0. GGUF is specified and deferred

**Rejected:** GGUF first, which is what ollama and llama.cpp read, and which has
the larger ecosystem of pre-quantized checkpoints.

The constraint is accel's quantized representation.
[accel 027](https://github.com/golang-design/accel/blob/main/specs/027-quantization.md)
registers exactly one quantized weight: int8 quants with one fp16 scale per
block of `quant.Int8Block` weights, consumed by `tensor.QuantMatMul` and
`tensor.QuantGatherRows`. GGUF's K-quants (`Q4_K_M`, `Q5_K_S`, and the rest)
carry a super-block structure with two levels of scale and a min, and **accel
has no kernel that reads them.** Reading GGUF first therefore means one of:

- dequantizing K-quants to f16 on load, which throws away the only reason to
  read GGUF, and needs more memory than the safetensors path; or
- writing K-quant kernels, which decision 1 forbids.

Safetensors carries bf16 or f16 planes that convert to what accel consumes with
no information invented. [001](001-weights.md) owns the conversion. GGUF gets
[012](012-gguf.md), written and unbuilt, and it becomes cheap the moment accel's
corpus registers a super-block kernel.

> **Amended 2026-08-27.** accel registers two quantized weights now, not one.
> int4 landed as three planes — `u32` codes holding eight weights each, an f16
> scale and an f16 zero per `quant.Int4Group` of 128 — consumed by
> `tensor.Int4MatMul` (accel
> [048](https://github.com/golang-design/accel/blob/main/specs/048-int4.md),
> [010 C21](010-conformance.md)), and tgo stores them. Neither registered form
> reads a K-quant super-block, so the decision and its two consequences are
> unchanged and [012](012-gguf.md) waits on the same missing kernel.

## 4. The numeric plane is fixed by accel's operator signatures, not chosen

accel's tensor operators accept specific dtypes. They are not preferences; a
mismatch is refused at graph-build time. The whole model graph is shaped by this
table, so it is recorded here rather than rediscovered per model.

| accel operator | operands | result |
| --- | --- | --- |
| `GatherRows` | table `f32` or `f16`, ids `u32` | `f32` |
| `QuantGatherRows` | table `i8` + scales `f16`, ids `u32` | `f32` |
| a gather over int4 | **not registered** | — |
| `RMSNorm` | `f32`, gain `f32` | `f32` |
| `Softmax`, `RoPE` | `f32` | `f32` |
| `MatMul`, `Linear` | `f32` or `f16` activations × `f16`/`f32` weights | `f32` |
| `QuantMatMul` | `f32` or `f16` activations, weight `i8`+`f16` | `f32` |
| `Int4MatMul` | `f32` or `f16` activations, weight `u32` codes + `f16` scales + `f16` zeros | `f32` |
| `Attention` | q `f32`, k/v state `f32` or `f16` | `f32` |
| `Add`, `Mul`, `SiLU`, `SwiGLU` | `f32` or `f16` | same |
| `ScatterRows` | state `f32` or `f16`, rows the **same** dtype, ids `u32` | the state |
| `Slice` | any dtype, unit step only; the result is a view | same |
| `Contiguous` | any elementwise dtype, at bounded rank | same |
| `Cast` | `f32`↔`f16`, and `bf16`→`f32` | the target |

Two consequences the model builder cannot avoid:

**Activations are f32 and narrow only at a matmul.** Every projection is
`Cast(x, F16)` then `MatMul`, and the result is f32 again. This is not a
pessimisation to remove later — accumulating a 2560-long dot product in f16
loses accuracy badly, and accel's f32 accumulation is the correct default.

**There is no bf16 arithmetic path.** accel stores bf16 (`accel.BF16`) but no
tensor operator reads it. Qwen3 ships bf16. The loader therefore converts, and
[001 §3](001-weights.md) owns the rounding and the overflow rule.

> **Amended twice, and closed.** The table above once read "both f16" for
> `MatMul`, which forced a `Cast` before every projection — 4 per layer, because
> a transformer's activations are f32 and its weights are f16 or int8. tgo
> reported it as an f32 GEMM request, which accel shipped and which did **not**
> remove the casts, because the rule was that the two operands *share* a dtype.
> Refiled naming the shape rather than the symptom — a **mixed** GEMM — and accel
> relaxed it. Measured: 1013 kernel selections on the Qwen3-4B graph became 760.
>
> bf16 is narrower than it was too. `Cast` now widens bf16 to f32, so only the
> bf16 GEMM is missing ([010 C7](010-conformance.md)) — and a weight would not
> want a per-step cast anyway, so the host conversion in
> [001 §3](001-weights.md) stays.
>
> The lesson is [010 §2.1](010-conformance.md)'s: a report being accepted is not
> a cost being removed.

> **Amended 2026-08-27, and the table is corrected above rather than left to
> contradict the amendment beside it.** Three things changed. `Cast` reads
> `bf16`→`f32`, which the paragraph above already said and the row denied;
> narrowing *from* bf16 is refused by name rather than approximated, because
> bf16 carries f32's eight-bit exponent and a value can be outside f16's range
> entirely. `Int4MatMul` is new, and it takes **three** planes rather than two,
> because at four bits fifteen levels have to be spent where the weights are and
> not symmetrically about zero, so a group carries a zero point as well as a
> scale ([010 C21](010-conformance.md)).
>
> **There is no int4 gather**, which is why the table now records the absence.
> `QuantGatherRows` reads an int8 table and nothing reads a packed one, so the
> embedding table is capped at int8 however narrow the policy — decision 5 owns
> the rule and this table is the reason for it.
>
> And the f16 KV path runs through `ScatterRows`, `Slice` and `Contiguous`,
> which the table omitted while the section claimed the whole graph is shaped by
> it. That omission cost twice: [010 C18](010-conformance.md) blocked every
> graph that slices, and [010 C25](010-conformance.md) was an
> accept-and-silently-wrong result on a reshaped output. Both sit on operators a
> model builder would have come here to look for.

## 5. Weights are f16, int8 or int4, chosen by size, and the choice is arithmetic

A model must fit. With $P$ parameters and $b$ bytes per stored weight the
resident weight footprint is

$$M_\text{weights} \approx P \cdot b + \frac{P}{B}\cdot 2$$

where the second term is the fp16 scales, present only when $b = 1$, and
$B = \texttt{quant.Int8Block}$.

| model | $P$ | f16 (`b=2`) | int8 (`b=1`) |
| --- | --- | --- | --- |
| Qwen3-0.6B | 0.6e9 | 1.2 GB | 0.6 GB |
| Qwen3-1.7B | 1.7e9 | 3.4 GB | 1.7 GB |
| Qwen3-4B | 4.0e9 | 8.0 GB | 4.0 GB |
| 27B class | 27e9 | 54 GB | 27 GB |

Above roughly 8 GB of weights, f16 stops being an option on the machines tgo
targets, so int8 is not an optimisation there — it is the only way the model
loads. Both paths are first class and both are tested. The default is chosen at
load time from available device memory, and it is always overridable, because a
silently-quantized model is a silently-different model.

`quant.Int8ErrorBound` gives the per-block bound. [001 §5](001-weights.md)
requires it to be measured on the real weight blocks, not on synthetic ones.

> **Amended 2026-08-27.** The section read "f16 or int8" and "both paths", and
> three widths ship. int4 landed in tgo on 2026-08-27 against accel
> [048](https://github.com/golang-design/accel/blob/main/specs/048-int4.md)
> ([010 C21](010-conformance.md)). The two-width prose above is the position it
> replaced and is kept; the heading is the one thing corrected in place, because
> a heading that names two widths is what a reader searches on.
>
> **The ladder is f16 → int8 → int4**, and the rule is still the widest form
> that fits. int4 is not a preference. accel's own tests show it beating int8 on
> a group of weights clustered away from zero and losing on one centred on it,
> so it is not uniformly better, and a budget rule that reached for it early
> would quietly degrade a model that fits at int8: **`auto` never prefers int4
> to int8**, and `TestAutoNeverPrefersInt4ToInt8` is the gate.
>
> **The footprint gains a third case.** int4 packs eight 4-bit codes per `u32`
> word and carries an f16 scale *and* an f16 zero per group of 128, so
>
> $$M_\text{weights} \approx P\left(\tfrac{1}{2} + \tfrac{4}{128}\right) = 0.53125\,P$$
>
> which is exactly half of int8's $1 + \tfrac{2}{32} = 1.0625$: the group
> doubles as the payload halves, so both terms halve rather than one of them. A
> 27B-class model resolves to 13.4 GiB rather than 26.7, which is what decides
> whether it fits hardware people own. [001 §5.1](001-weights.md) carries the
> per-form arithmetic, and 001's "`auto` picks by decision 5" now points at a
> rule that covers all three.
>
> **Embeddings pin to int8 whatever the policy says.** The embedding table is
> gathered a row at a time rather than contracted against, and accel registers
> no int4 gather (decision 4), so the loader caps that one tensor at int8 under
> an int4 policy. It is a numeric-plane constraint, not an accuracy judgement,
> and a tied checkpoint's LM head still packs because that is a `MatMul`.

## 6. A model is a config plus a graph builder, resolved by name

`config.json` carries `architectures: ["Qwen3ForCausalLM"]`. tgo resolves that
string in a registry to a builder function, which records the forward pass with
[`nn`](004-model-graph.md) blocks over `tensor.Builder`.

**Rejected:** one hardcoded Qwen3 forward pass, generalised when a second model
arrives. It is genuinely cheaper today. It stops being cheaper the moment the
engine, the KV cache and the server sit on top of it, because the refactor then
crosses every one of them.

**Also rejected:** a serialised graph format (ONNX-shaped). It buys portability
tgo does not need and costs a second type system.

Adding a model is one file and one `init`. Nothing else changes. A model that
the registry does not know is refused at load with the list of what it does
know, never guessed at.

> **Amended 2026-08-27.** "One file and one `init`" holds inside the class of
> model the `nn` blocks already cover, and it was falsified for a new class:
> `nn.LinearAttention` and `nn.DepthwiseCausalConv` both had to be built before
> any hybrid architecture could register. So the rule is one file and one `init`
> **when the blocks exist**; a new layer class adds `nn` blocks first, and
> [018](018-hybrid-models.md) is the worked example of what that costs.

## 7. v0 serves one sequence at a time, and continuous batching is blocked upstream

**Rejected:** shipping a vLLM-shaped continuous batching scheduler in v0.

[accel 040](https://github.com/golang-design/accel/blob/main/specs/040-batch-scheduler.md)
names three gaps that make a *correct* batched decode inexpressible in accel
today, and all three fail silently — the output stays well-shaped, finite, and
plausible:

| gap | what breaks | owning accel spec |
| --- | --- | --- |
| batched paged attention is not exposed at the tensor layer | `tensor.Attention` selects only the decode and prefill kernels | 010, 030 |
| `RoPE` takes a scalar `Offset`, so the row index is part of the position | in a batched decode the row index is the *slot*, so only one member rotates at its own cache length | 010, 025 |
| `SampleCategorical` shares one `Draw` across the batch | two sequences with similar distributions emit the same token | 028, 039 |

A scheduler written over kernels that cannot express per-slot positions is a
scheduler whose correctness cannot be tested. [008](008-scheduler.md) states the
design, [010](010-conformance.md) holds a named failing test per gap, and the
code waits.

**Paged KV is not used in v0 either, and not by choice.** accel's block pool
lives at `tensor/internal/pagetable` and is unexported, because no exported
`tensor` operator accepts a page table — accel 030 says so in the package
comment. So the cache tgo can build is one contiguous `tensor.State` per layer,
sized for the longest sequence it will ever serve. That is a fourth gap, it is
the one that costs the most memory, and [005](005-kv-cache.md) states what it
costs.

> **Amended 2026-08-24.** All three gaps above, plus the fourth, are one
> decision. accel
> [043](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md)
> names it — *a scalar is a value every row of a dispatch shares; a value that
> differs per row is a tensor* — and moves positions, cache lengths, prefill
> bases, sampling draws and the page table onto tensor operands. `RoPE` has
> already changed. The rest is designed and unbuilt.
>
> **Amended again, later the same day.** 043 landed. `RoPE` takes positions,
> `Attention` takes `Lengths`, `Pages` and an f16 cache. tgo can now build a
> paged, narrow KV cache, which is exactly what this decision said it could not.
>
> **v0 is still not batched, for a different reason.** `q`'s rank is the *phase*
> — rank 3 means a prefill — so a batched decode has no axis to live on
> ([010 C1](010-conformance.md)). And the paged cache is **inert**: the
> the *prefill* path drops the page table it accepts, so blocks a paged decode
> would read were never written by a paged prefill ([010 C13](010-conformance.md),
> [accel#10](https://github.com/golang-design/accel/issues/10)).
>
> The paragraphs above are kept, not replaced. They are why 043 exists.

> **Amended 2026-08-27, and this decision is now about the server rather than
> about accel.** Every gap it rested on has closed.
> [010 C1](010-conformance.md) closed: two sequences of lengths 96 and 32
> batched match two single runs to `0.00e+00`, so a batched decode has an axis.
> [C4](010-conformance.md) (paged decode) and [C13](010-conformance.md) (paged
> prefill, verified by reversing the page table) closed, so the paged cache is
> not inert — a paged prefill writes the blocks a paged decode reads.
> [C16](010-conformance.md) made the step ragged through
> `AttentionOptions.QueryExtents`, so a prefill and a decode share one dispatch,
> and [C24](010-conformance.md) kept that over an f16 cache. The code is not
> waiting either: `Model.NewBatch`, `Model.NewScheduler` and
> `Scheduler.Admit/Step/Feed/Evict/Finish` are built, exported and tested, and
> [008](008-scheduler.md) reads `status: implemented`.
>
> **What is still true is narrower, and it is tgo's.** The *served* path does
> not batch: nothing under `server/` or `cmd/` calls `NewScheduler` or
> `NewBatch`, and `server/admit.go` says so where the queue is — concurrent
> requests interleave at submission granularity and the total is what one
> sequence gets. That is server wiring, not an accel gap, so it is not decision
> 1's file-register-skip-wait sequence and no register row is open on it.
> [022](022-batched-serving.md) is the spec that closes it.

## 8. No test downloads weights

The smallest Qwen3 is over a gigabyte. CI runs against **synthetic
configurations** — 2 layers, hidden size 64, vocab 640 — built by a fixture
helper, with weights generated from a fixed seed. Every layer of the stack is
covered this way, including the full forward pass and a decode loop.

Real weights run behind `TGO_MODEL`, pointing at a local directory. That job is
not part of CI and never will be. It is the release gate, run by hand, and
[011 §4](011-sequencing.md) is the release-gate record that holds its result
each time.

The corollary shapes the package layout: the pure logic — tokenizer, safetensors
parsing, chat templates, sampling policy, scheduler admission — lives in
packages with **no device and no network**, so the coverage gate is reachable
there. Device-dependent code lives in thin packages, and what they exclude is
printed rather than hidden.

> **Amended 2026-08-27.** The fixture's vocabulary reads 640 above and read 128.
> It was widened because the tokenizer fixture's id space is 582 wide, and a
> vocabulary smaller than the tokenizer can produce gathers out of range on a
> token nobody chose. Layers and hidden size are unchanged. Every other fixture
> extent is deliberately distinct from every other, so that a shape taken from
> the wrong one reads as wrong rather than as correct. The section also cited a
> §4 of [011](011-sequencing.md) that did not exist when it was written. That
> section is the release-gate record — one entry per `TGO_MODEL` run — and the
> citation above points at it.

## 9. Sampling is reproducible as a stream, not as a token

accel 028 makes the random draw an input, so one token is reproducible. That is
not the same promise as *the same prompt and seed give the same completion*,
which is what a user checks. tgo owns the stream: a seed produces a
deterministic sequence of draws, and the whole completion is reproducible.

**Greedy decoding is bit-exact across runs on one device.** It is not promised
bit-exact across the CPU backend and Metal, and [010](010-conformance.md)
measures the divergence rather than asserting a bound nobody verified.

## 10. The public surface is small and the engine is not the API

tgo exports: a `Model`, a `Session`, a `Generate` call that streams tokens, and
a sampling policy. The plan cache, the KV block pool, the graph builders and the
scheduler are internal, because every one of them is a place where accel's
shape will move under us.

The HTTP surface speaks three wire dialects — OpenAI Chat Completions, Anthropic
Messages, and OpenAI Responses — through one neutral request shape, and says so
where a dialect asks for something tgo does not do. **Compatibility is a
serialisation decision, not an architectural one, and it does not reach into the
engine.** That is what makes three cost one adapter rather than three parsers;
[009 §2](009-server.md) has the boundary.

> **Amended 2026-08-27.** Two claims above are false against what package `tgo`
> exports. There is no `Generate`: the call is `Session.Chat` or
> `Session.Complete`, and it returns a `*Stream`. And the scheduler is not
> internal. The paragraph is kept because the principle in it is the one still
> being applied; the list is replaced by this one.
>
> The surface is `Model` — `Open`, `NewSession`, `NewPool`, `NewBatch`,
> `NewScheduler`, `CheckSchema`, `Info`, `Close` — and `Session`, `Stream`,
> `Pool`, `Lease`, `Batch`, `Scheduler`, with `Policy`, `SchedulerOptions`,
> `PoolRequest`, `Work`, `StepResult`, `Produced`, `Usage`, `Event`,
> `Precision`, `Device`, `CacheScope`, the `Option` and `SessionOption`
> constructors, and the named errors. The plan cache, the KV block pool and the
> graph builders are still unexported, for the reason above.
>
> **Decided: `Scheduler` is public API, and `Batch` under it.** A deployment
> chooses `Chunk` and `Reserve`, and admission and eviction are where a serving
> policy lives, so they belong to the caller and not to tgo. A scheduler
> reachable only through tgo's own HTTP server would make tgo the sole consumer
> of [008](008-scheduler.md), which is the shape decision 1 objects to one level
> up. **Rejected: moving it back behind the engine**, exposed only as a server
> flag. That keeps the surface smaller and it makes the batching layer
> unreachable by anything that could report on it, so a defect in it would
> surface as slow serving rather than as a report. The consequence is a support
> commitment, stated rather than discovered: `Batch` and `Scheduler` are
> compatibility surface now, and `Model.NewBatch` requires
> `WithPrefixCache(CacheProcess, ...)`, because sequences of different lengths
> stepping together need a shared block pool rather than a cache padded to the
> longest.
>
> The dialect boundary holds and its shape moved. There are four wire formats,
> not three — `/v1/completions` is the fourth. Parsing and serialisation for
> three of them live in an external module, `latere.ai/x/pkg/llmdialect`, tgo
> owns one mapping file (`server/adapt.go`) and codecs the fourth itself
> (`server/legacy.go`), and all four travel the same neutral request shape. No
> dialect reaches the engine, which is the thing this decision defends. The
> module is a dependency, so decision 2 re-gates it: it is pure Go and
> `CGO_ENABLED=0` builds it.

---

## 11. Faster than vLLM is the goal, and it is measured per axis

tgo is not a convenience trade. **The goal is to be faster than vLLM**, and
[010 §3.1](010-conformance.md) says on which axes and how it is measured.

**Rejected: "fast enough, and easier to deploy."** It sounds humble and it
decides the architecture badly — it licenses a slow scheduler, a slow sampler
and a slow detokenizer, because each is individually small against a matrix
multiplication. They are not small: none of them is arithmetic, all of them run
on every token, and together they are the part of a serving stack a compiled
language should win outright.

**Also rejected: claiming it before measuring it.** Today vLLM is faster,
because tgo does not run. What is stated here is a target with a table attached
and a commitment to publish the rows tgo loses.

The axis tgo will lose longest is raw GEMM and attention throughput on NVIDIA,
against years of hand-tuned CUDA. That is accel's to close, and decision 1 makes
tgo's contribution to it the same as everywhere else: a kernel slower than it
should be is a report, exactly like a kernel that is missing.

> **Amended 2026-08-27.** The premise "tgo does not run" is no longer true. tgo
> runs and serves ([011](011-sequencing.md), Waves 4 and 5), and a decode step's
> host, submit, device and readback shares are measured per axis. What has never
> been measured is a single row against vLLM, on any axis.
> [017 §4](017-benchmarks.md) rule 1 gives the reason it is not worth running
> yet: 12.57 tokens/s on a 0.6B model against years of hand-tuned CUDA would
> report a fact about kernel maturity dressed as a fact about tgo. The row waits
> for [011](011-sequencing.md) M13, and the axis that would make it meaningful
> first is submit overhead — the one this section says tgo should win. The
> target, the axes and the commitment to publish the
> rows tgo loses are unchanged. The claim stays unmade until there is a row.

## 12. Sampling runs on the host, and that is a decision rather than unfinished work

The sampler reads a row of logits back from the device and composes penalties,
temperature, top-k, top-p, the schema mask and the categorical draw in Go
(`sample`, [006](006-sampling.md)).

**Rejected: sampling on the device.** accel registers it. `tensor.Sample`
composes the same policy in one dispatch and returns a token id, and both rows
that asked for it closed against tgo's own reports
([010 C3](010-conformance.md), [C6](010-conformance.md)). So this is a
capability tgo has and does not use, which is the case decision 1 does not
otherwise cover.

The constraint is decision 8's corollary and decision 9's promise. A host
sampler lives in a package with no device and no checkpoint, so composition
order, the mask in front of it and the seeded stream are all reachable by the
coverage gate. And one implementation decides the token on the CPU backend and
on Metal, so a divergence between them is arithmetic in the model and never
policy in the sampler, which is what makes [010](010-conformance.md)'s
divergence measurement mean anything.

**The cost is reported, not absorbed.** A decode step carries 608 KB of logits
back for four bytes of output, and `internal/conformance/measure.go` keeps
measuring that share, because "how much of a decode step is the readback" is the
question tgo exists to answer for accel. The consequence is that this decision
is deliberate today and not permanent:
[020](020-device-sampling.md) states what moving the policy onto the device
costs, and what it has to keep is this section's testability rather than the
readback.

## 13. Decision 1 needs the CI gate decision 2 has

A prohibition nobody greps is a preference. Decision 2 is enforced by a job that
rejects `import "C"` by grep rather than by build; decision 1 — no kernels, no
backend code, no device-conditional numerics — is enforced by review alone.

**Rejected: leaving it to review.** Decision 2's job greps rather than builds
because a file can violate the rule behind a build tag the platform does not
select. The same is true here: a `//go:build darwin` file carrying
Metal-conditional numerics passes every test that runs on Linux, and it is the
exact shape of the violation this decision cares most about.

The gate is a lint, not yet built, that fails when device-conditional numerics
or backend code
appear outside the one named exemption, `internal/oracle` — a host-side f32
reference implementation of RoPE, attention and the rest, used to *judge* device
results and never to produce them ([010 §5](010-conformance.md)). Naming the
exemption here is the point: an unnamed exemption is how the prohibition erodes.
It is written down before it is built, so that what it checks is decided by this
file and not by whatever the first version of the script happened to catch.

**Composition is not a workaround, and the rule says where it stops.**
`nn.DepthwiseCausalConv` builds a causal convolution out of `ScatterRows`,
`GatherRows` and `Slice` over a rolling state, at roughly $3K+5$ dispatches per
layer, because accel registers no such kernel and answered the report won't-fix
([010 C26](010-conformance.md)). That is a fourth option beside decision 1's
four steps, it is in use, and it is allowed under one rule: a composition of
**registered** operators is a model, and its dispatch cost is the report. A
private kernel is what decision 1 forbids, and the lint is what tells the two
apart.

## What v0 is

One model family (Qwen3 dense), from safetensors, on the CPU backend and Metal,
one sequence at a time, through a CLI and a server speaking the OpenAI,
Anthropic and Responses APIs, with a conformance suite that reports what accel
cannot yet do.

**Done means:** a Qwen3 dense checkpoint produces coherent text at both f16 and
int8, on both backends, with the tokenizer round-tripping and the chat template
matching the reference byte for byte.

> **v0 was gated upstream and is not any more, as of 2026-08-24.**
> `tensor.Attention` refused a KV cache longer than **128 positions**, shorter
> than a system prompt, which made the goal above unreachable for reasons
> entirely in accel. accel
> [044](https://github.com/golang-design/accel/blob/main/specs/044-unbounded-context.md)
> shipped the tiling loop; a 4096-position cache is verified working, and
> nothing in tgo is blocked on cache size.
>
> **What is still blocked is narrower.** A paged *prefill* silently ignores its
> page table ([010 C13](010-conformance.md)), so cross-request prefix sharing is
> not expressible — that blocks [016](016-prefix-cache.md), not v0. Batched
> decode ([C1](010-conformance.md)) still has no axis to live on, which is
> decision 7. Neither stops a single sequence being served end to end.
>
> This paragraph is kept in its amended form rather than deleted, because the
> register's whole claim is that a row leaves only when its test says so, and
> the same discipline applies to the prose.

> **Amended 2026-08-27. Nothing in v0 is gated upstream any more.** Both gaps
> the paragraph above names as still blocking have closed.
> [010 C13](010-conformance.md) closed, verified by reversing the page table,
> and [C24](010-conformance.md) added the paged prefill over an f16 cache, so
> cross-request prefix sharing is expressible and shipped —
> `WithPrefixCache(CacheProcess, ...)`, `internal/prefix` and a shared block
> pool proved over the wire — which makes [016](016-prefix-cache.md) and
> [019](019-session-affinity.md) built rather than blocked.
> [C1](010-conformance.md) closed and a batched decode is exercised by
> `batch_test.go`.
>
> **What is left is tgo's own wiring.** The served path still runs one sequence
> at a time because nothing under `server/` drives the scheduler, which is
> decision 7's amendment and [022](022-batched-serving.md)'s subject. Scope is
> otherwise as stated: one Qwen3 dense family, safetensors, the CPU backend and
> Metal, a CLI and a server.

