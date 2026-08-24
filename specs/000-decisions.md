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

## 4. The numeric plane is fixed by accel's operator signatures, not chosen

accel's tensor operators accept specific dtypes. They are not preferences; a
mismatch is refused at graph-build time. The whole model graph is shaped by this
table, so it is recorded here rather than rediscovered per model.

| accel operator | operands | result |
| --- | --- | --- |
| `GatherRows` | table `f32`, ids `u32` | `f32` |
| `QuantGatherRows` | table `i8` + scales `f16`, ids `u32` | `f32` |
| `RMSNorm` | `f32`, gain `f32` | `f32` |
| `Softmax`, `RoPE` | `f32` | `f32` |
| `MatMul`, `Linear` | **both `f16`** | `f32` |
| `QuantMatMul` | activations `f16`, weight `i8`+`f16` | `f32` |
| `Attention` | q `f32`, k/v state `f32` | `f32` |
| `Add`, `Mul`, `SiLU`, `SwiGLU` | `f32` or `f16` | same |
| `Cast` | `f32`↔`f16` only | the target |

Two consequences the model builder cannot avoid:

**Activations are f32 and narrow only at a matmul.** Every projection is
`Cast(x, F16)` then `MatMul`, and the result is f32 again. This is not a
pessimisation to remove later — accumulating a 2560-long dot product in f16
loses accuracy badly, and accel's f32 accumulation is the correct default.

**There is no bf16 arithmetic path.** accel stores bf16 (`accel.BF16`) but no
tensor operator reads it. Qwen3 ships bf16. The loader therefore converts, and
[001 §3](001-weights.md) owns the rounding and the overflow rule.

> **Amended 2026-08-24, then corrected the same day.** accel 043 §5 accepted the
> argument in [accel#5](https://github.com/golang-design/accel/issues/5) and
> `MatMul` now takes f32 operands. **The cast chain survived it**, and the table
> above still holds.
>
> `MatMul` requires the two operands to *share* a dtype — one kernel reads both.
> A transformer's activations are f32 and its weights are f16 or int8, because
> decision 5 makes f32 weights the choice not to load the model. So f32 operands
> help only a model storing f32 weights, and `QuantMatMul` still requires f16
> activations either way.
>
> What would close it is a **mixed** GEMM: f32 activations against f16 or int8
> weights, accumulating f32 — the shape every inference stack uses, and the one
> the original report should have named instead of "f16-only". Refiled on the
> same issue. bf16 stays open as [010 C7](010-conformance.md), and is worse than
> recorded: `Cast` cannot widen bf16 either, so the conversion is entirely the
> host's.
>
> The lesson is [010 §2.1](010-conformance.md)'s: a report being accepted is not
> a cost being removed.

## 5. Weights are f16 or int8, chosen by size, and the choice is arithmetic

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

## 8. No test downloads weights

The smallest Qwen3 is over a gigabyte. CI runs against **synthetic
configurations** — 2 layers, hidden size 64, vocab 128 — built by a fixture
helper, with weights generated from a fixed seed. Every layer of the stack is
covered this way, including the full forward pass and a decode loop.

Real weights run behind `TGO_MODEL`, pointing at a local directory. That job is
not part of CI and never will be. It is the release gate, run by hand, and
[011 §4](011-sequencing.md) records its result each time.

The corollary shapes the package layout: the pure logic — tokenizer, safetensors
parsing, chat templates, sampling policy, scheduler admission — lives in
packages with **no device and no network**, so the coverage gate is reachable
there. Device-dependent code lives in thin packages, and what they exclude is
printed rather than hidden.

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

---

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
