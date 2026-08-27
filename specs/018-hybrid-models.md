---
title: "Hybrid attention: what Qwen3.8-27B is, and the operator it now has"
status: drafted
layer: graph
depends_on:
  - 000-decisions.md
  - 004-model-graph.md
  - 005-kv-cache.md
---

# Hybrid attention

One of tgo's two named target models is not a dense transformer. This spec says
what it actually is, what it needs, and why [004-D2](004-model-graph.md)'s
"a new architecture is additive" does not hold for it.

## 1. What Qwen3.8-27B is

Read from its `config.json`, not from its name:

```json
"architectures": ["Qwen3_5ForConditionalGeneration"],
"model_type": "qwen3_5",
"layer_types": ["linear_attention", "full_attention"],
"full_attention_interval": 4,
"num_hidden_layers": 64,
"hidden_size": 5120,  "head_dim": 256,
"num_attention_heads": 24, "num_key_value_heads": 4,
"partial_rotary_factor": 0.25,
"attn_output_gate": true,
"linear_conv_kernel_dim": 4,
"linear_num_key_heads": 16,  "linear_key_head_dim": 128,
"linear_num_value_heads": 48, "linear_value_head_dim": 128,
"mamba_ssm_dtype": "float32",
"max_position_embeddings": 262144,
"vocab_size": 248320,
"image_token_id": 248056, "video_token_id": 248057
```

Four things follow, in descending order of how much they cost tgo:

1. **Three of every four layers are linear attention.** `full_attention_interval:
   4` over 64 layers means 16 softmax-attention layers and **48 gated-delta
   layers**. The operator tgo has spent its design on covers a quarter of the
   model.
2. **It is multimodal.** Vision and video token ids, and a
   `ForConditionalGeneration` head. Text-only inference is a coherent subset,
   but the tokenizer and template must not choke on the image tokens.
3. **262144 context.** Which the linear layers make affordable — their state does
   not grow with sequence length — and which the 16 full-attention layers still
   pay for.
4. `partial_rotary_factor: 0.25` and `attn_output_gate` — both **already
   expressible** (§4).

## 2. The layer accel now has

**Unblocked on 2026-08-26.** This spec carried `status: blocked` naming
[accel#17](https://github.com/golang-design/accel/issues/17) from the day it was
written. `tensor.LinearAttention` landed, and a value probe confirms it rather
than a commit message doing so:

```
selection: LinearAttention -- the gated delta scan: 3 sequences of 2 heads,
           each walking its own tokens with a [4, 3] state
4 tokens over 3 sequences, matched against a float64 reference to within a
budget derived from the scan length; halving every alpha moves the output
```

Three things about the shape accel settled on, each of which decides something
below:

- **The state is an ordinary `tensor.State`** of shape
  `[slots, heads, valueDim, keyDim]`. Its leading axis is the *sequence slot*
  rather than a position, which is what §2's second bullet said could not be
  expressed. So a hybrid model holds two States of different shapes in one
  graph and needs nothing else — §3's conclusion stands and its premise moved.
- **The tokens arrive as the same segmented extent
  [008](008-scheduler.md) batches with**, `QueryExtents`. A sequence
  contributing one token is a decode, one contributing $T$ is a prefill, and a
  step with both is mixed — so the linear layers batch the same way the softmax
  ones do, and one scheduler drives both.
- **The kernel is the sequential scan and not the chunked form.** accel records
  the chunked parallel version as deliberately unbuilt, because it reassociates
  the recurrence and therefore needs its own numeric bound derived against this
  one. §2's "expressing it as $T$ dependent graph nodes would be unusably slow"
  is answered — it is one node — but the *arithmetic* is still three passes over
  `keyDim*valueDim` per token, so a 262K prefill is a scan of 262K steps. This
  is a performance row for [010](010-conformance.md) to measure and report, not
  an expressibility gap.

**One thing to verify against the reference implementation before trusting a
generated token**, and it is the kind of thing a spec should name rather than
discover: where $\alpha_t$ sits. accel's kernel computes

$$u = S k, \qquad S \leftarrow \alpha S + \beta\,k\,(v-u)^\top$$

which is the expansion of the equation below. Qwen3.5's published form places
$\alpha$ outside the whole bracket, $S_{t-1}\alpha_t(I - \beta_t k k^\top)$,
and the two differ in whether the correction term is decayed. Both are "the
gated delta rule" in the literature. tgo's oracle was written from accel's
equation and agreed with accel's kernel — which proves the kernel implements the
equation it documents and proves nothing about which equation Qwen3.8 uses.
[010 §5](010-conformance.md)'s rule applies: the checkpoint decides, and the
check belongs in the tier-3 run against real weights.

## 2.1 Why it is a kernel and not a composition

A gated delta layer carries a **matrix-valued recurrent state** per head,
`[key_head_dim, value_head_dim]` = 128×128 here, and updates it per token:

$$S_t = S_{t-1}\big(\alpha_t I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,
\qquad o_t = S_t\, q_t$$

with $\alpha_t$ a forget gate and $\beta_t$ a write strength, both produced from
the input. There is also a short depthwise causal convolution over the input
(`linear_conv_kernel_dim: 4`) before the recurrence.

- **The recurrence is sequential in $t$.** A prefill of $T$ tokens is a scan, not
  $T$ independent rows. The efficient form is chunked — matmuls within a block,
  state carried between blocks — and that chunking *is* the kernel. Expressing
  it as $T$ dependent graph nodes would be correct and unusably slow.
- **The state is not a KV cache and cannot pretend to be one.** It does not grow
  with sequence length, which is the whole appeal at 262K. But it also has no
  per-position structure: `State` and `ScatterRows` address a row per position,
  and a recurrent state has one value per *sequence*.

## 3. The consequence for the cache, which is the interesting part

A hybrid model's cache is **two different things at once**:

| layers | state | grows with $T$? | pageable? |
| --- | --- | --- | --- |
| 16 full attention | `[C, H_kv, d_h]` KV | yes | yes — [005](005-kv-cache.md) as written |
| 48 linear attention | `[H, d_k, d_v]` per sequence | **no** | no — there are no positions to page |

So [005](005-kv-cache.md)'s design covers a quarter of the model and
[016](016-prefix-cache.md)'s prefix cache covers the same quarter. **Prefix
reuse over a recurrent layer means restoring a state snapshot, not sharing
blocks** — which is precisely the design ollama uses
([016 §10.1](016-prefix-cache.md)) and which tgo rejected for the dense case
because a block pool is better *when positions exist*.

That is the finding worth recording now: **ollama's snapshot-and-restore design,
which looked like a workload difference, is the shape a hybrid model forces.**
tgo would need both, chosen per layer.

## 4. What already works, checked rather than assumed

The full-attention quarter needs nothing new:

- `partial_rotary_factor: 0.25` → `RoPE(b, x, 64, base, positions)`. `rotaryDim`
  was always a parameter rather than the full width, so rotating 64 of 256
  records today — verified by probe.
- `attn_output_gate: true` → an elementwise multiply of the attention output by
  a projected gate. `tensor.Mul` composes.
- `head_dim: 256` with $d/H = 213$ — a case [004 §5](004-model-graph.md) already
  refuses to infer, and reads from the config instead.

## 4.1 The depthwise convolution composes over a *port*, and not over a projection

**This section was right about the arithmetic and wrong about where it applies**,
and the correction is the useful part.

`linear_conv_kernel_dim: 4` needs no kernel. Built against accel and run:

```
COMPILES: 14 selections
max |device - reference| = 6.217e-08
```

as $K$ shifted windows over a **left-padded** input, each scaled by its tap row
and summed — `Contiguous(Slice(x, 0, K-1-i, T+K-1-i))` times a broadcast
`[1, C]` weight, accumulated. The left pad makes causality **structural**: no
operator needs to know the convolution is causal.

**Cost, stated honestly:** 14 dispatches for what one kernel would do, plus
$K-1$ packing copies of a `[T, C]` tensor per layer. Across 48 linear layers
that is real. So this is one less kernel to be *blocked on*, not one less kernel
to *want*.

### 4.1.1 The correction: the probe padded an input port, and a graph cannot

`Slice(x, 0, K-1-i, T+K-1-i)` reads up to row $T+K-2$, so `x` is already
$[T+K-1, C]$ when the slice is taken. The probe supplied that as an **input
port** — a buffer the caller filled, with $K-1$ zero rows at the front.

A real layer convolves a **projection**, which the graph computes. There is
nothing to pad it with: `tensor` has no operator that joins two tensors along an
axis, so the padded input cannot be built ([C26](010-conformance.md)).

**A `Concat` is the wrong thing to ask for**, which is why this is recorded
rather than filed. A convolution that runs a token at a time needs the $K-1$
inputs of the *previous step*, not zeros — a decode step has no earlier rows in
its own tensor at all. So the layer wants a **rolling state**: a
`[slots, K-1+T_max, C]` `State` that each step scatters its rows into and slices
$K$ windows out of. `ScatterRows` and `Slice` both exist, and
[§6](#6-what-would-have-to-be-built-here-once-it-is-unblocked)'s "per-layer
cache kind" is that state.

So the convolution is not blocked and is not built: it needs the cache row of §6
rather than an operator from accel.

## 4.2 What a recurrent state needs, which is not what a cache needs

accel's answer to [#17](https://github.com/golang-design/accel/issues/17)
identified something this spec had only gestured at: `State` conflates a
per-**position** cache with a per-**sequence** recurrent state, and that is a
type distinction the library has no name for. accel recorded that if the
operator is built, **the state distinction is made first**.

What tgo needs from a recurrent state, in order:

| operation | analogue today |
| --- | --- |
| **carry** across submissions — a decode step is one submission per token, so the state must survive from $t$ to $t+1$ | `State` does this |
| **snapshot** a sequence's state somewhere it can be kept | **none** |
| **restore** a snapshot into a slot | **none** |
| index at a position | **never** — there is nothing at a position |

Snapshot and restore are **copy-shaped**; everything a KV cache does is
**address-shaped**. That is the cleanest statement of why they are two types.

It also settles [016 §10.1](016-prefix-cache.md)'s open question: prefix reuse
over a recurrent layer *is* ollama's snapshot-and-restore, and tgo would need
both mechanisms, chosen per layer — 16 paged KV layers beside 48 snapshotted
recurrent ones **in one forward pass**, not two models.

## 5. What tgo does now

Nothing. This spec is `blocked` on
[accel#17](https://github.com/golang-design/accel/issues/17), and unlike every
other blocker in this tree it is not a relaxation of a refusal — it is an
operator that has never existed.

**The honest position on the target list:** Qwen3-4B is buildable now and is what
[011 §2.2](011-sequencing.md)'s waves build. Qwen3.8-27B is a stated target and
is not reachable until accel answers #17, and tgo says so publicly rather than
implying both are equally near.

If accel's answer is that a recurrent operator is out of scope, that is
legitimate and this spec becomes `deferred` with the reason — the same shape
[012](012-gguf.md) took when accel declined GGUF.

## 6. What would have to be built here, once it is unblocked

| | |
| --- | --- |
| ~~`nn.LinearAttention`~~ | **built 2026-08-27**, over accel's scan, verified against a float64 reference derived from §2's equation. The depthwise convolution is *not* in it — see [§4.1.1](#411-the-correction-the-probe-padded-an-input-port-and-a-graph-cannot) |
| a per-layer cache kind | [005](005-kv-cache.md) gains a state per layer *type*, not one shape for all |
| snapshot/restore for recurrent state | [016](016-prefix-cache.md) gains ollama's shape for the layers that need it |
| the `qwen3_5` registry entry | config parsing, the weight map, the layer-type schedule |
| image-token tolerance | the text path must not break on a multimodal tokenizer |

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 018-D1 | Qwen3.8-27B is out of v0, and said so publicly | list it as a target and hope | 48 of 64 layers are inexpressible; implying otherwise is the overclaiming this tree exists to avoid |
| 018-D2 | file the operator as a question, not a proposal | propose a kernel shape | tgo does not know accel's kernel constraints; #17 asked whether it is in scope first. **Outcome:** accel answered *in scope, recorded, not scheduled*, and found a deeper problem tgo could not see — that `State` conflates two types ([§4.2](#42-what-a-recurrent-state-needs-which-is-not-what-a-cache-needs)) |
| 018-D5 | check what composes before asking for a kernel | ask for the convolution too | the depthwise causal convolution composes today ([§4.1](#41-the-depthwise-convolution-composes-and-was-checked)), so the ask is one kernel rather than two |
| 018-D3 | a hybrid cache is per layer *type*, not one shape | force the recurrent state into `State`'s per-position model | a recurrent state has one value per sequence and no positions to index |
| 018-D4 | record that ollama's snapshot design is forced here | keep treating it as a workload difference | [016 §10.1](016-prefix-cache.md) read it as a choice about concurrency; for a recurrent layer it is the only option |
