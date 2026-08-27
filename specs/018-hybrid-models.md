---
title: "Hybrid attention: what Qwen3.8-27B is, and the operator it now has"
status: implemented
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
4. `partial_rotary_factor: 0.25` and `attn_output_gate` — both expressible in
   accel, and neither reachable through `nn.Attention` today (§4).

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
  $T$ independent rows, and accel expresses the whole scan as **one node over a
  segmented extent** rather than as $T$ dependent graph nodes. The chunked
  parallel form — matmuls within a block, state carried between blocks — is
  recorded as deliberately unbuilt (§2), so the arithmetic is still one pass over
  `keyDim*valueDim` per token.
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

One of the three needs nothing new. The other two compose in accel and have no
field in `nn` to reach them through:

- `head_dim: 256` with $d/H = 213$ — a case [004 §5](004-model-graph.md) already
  refuses to infer, and reads from the config instead. **Built**, and the only
  bullet here that is.
- `partial_rotary_factor: 0.25` → `RoPE(b, x, 64, base, positions)`. `rotaryDim`
  is a parameter of accel's `tensor.RoPE` rather than the full width, and an ad
  hoc probe rotated 64 of 256. tgo cannot ask for it: `nn/attention.go:207`
  passes `cfg.HeadDim` as the rotary width and `AttentionConfig` carries no
  rotary-dim field, so a partial rotation is **inexpressible today** and no test
  in the tree would catch one breaking.
  [024](024-qwen3-5-architecture.md) adds the field.
- `attn_output_gate: true` → an elementwise multiply of the attention output by
  a projected gate. `tensor.Mul` composes, and `nn.Attention` has no gate to
  compose it with. [024](024-qwen3-5-architecture.md) again.

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
`[K-1+T_max, C]` `State` that each step scatters its rows into and slices $K$
windows out of. `ScatterRows` and `Slice` both exist.

What is built is **one slot**, one sequence's window: the block slices axis 0 of
the window (`nn/linear.go:220`) and the test builds the state two-dimensional
(`nn/linear_test.go:214`). A leading `slots` axis, which a batched step needs to
give two sequences their own windows, is not addressable through this block and
belongs to [023](023-cache-kinds.md).

So the convolution needed a cache row rather than an operator from accel, and
**it is built**. `nn.DepthwiseCausalConv` runs over that state: each
step scatters its rows in behind the $K-1$ already there, reads $K$ windows out,
and then writes the window's last $K-1$ rows to the front for the next step —
two writes to one state with an order that matters, which is what
`tensor.State`'s versions express. A write before the read would convolve rows
the step has not produced.

The carry is the half a padded operand could never have supplied, and it is what
the test has to reach for: a decode step has no earlier rows in its own tensor
at all, so a test that only ever prefills cannot tell a working carry from a
dropped one. The check is a five-token step followed by a one-token step against
a reference over all six at once.

**Two costs the dispatch count did not show.** The tap row needs `Contiguous`
*twice* — once because a slice at row $i$ is a view at an offset, and once
because `Mul` takes operands and not views — so every tap materialises a
$[T, C]$ copy. And the tap order is

$$y_t = \sum_i \texttt{taps}_i \cdot x_{t-K+1+i}$$

which is the convention the weights are trained under. Reversing it runs,
produces plausible numbers, and convolves the window backwards.

## 4.2 What a recurrent state needs, which is not what a cache needs

accel's answer to [#17](https://github.com/golang-design/accel/issues/17)
identified something this spec had only gestured at: `State` conflates a
per-**position** cache with a per-**sequence** recurrent state, and that is a
type distinction the library has no name for. accel recorded that the
distinction should be made first. **It was not**: `tensor.LinearAttention`
shipped over the undifferentiated `State`, which the KV path and
`nn.LinearAttention` both use, and the distinction is still owed.

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

Two blocks, both in `nn/linear.go`, both landed 2026-08-27:

- **`nn.LinearAttention`** (`nn/linear.go:67`) composes accel's gated delta scan
  over a `[slots, heads, ValueDim, KeyDim]` state and the same `QueryExtents`
  [008](008-scheduler.md) batches with, so one scheduler drives the linear layers
  and the softmax ones. `nn/linear_test.go:32` matches a float64 reference
  written from §2's equation; `nn/linear_test.go:82` records the refusals,
  including the transposed state that runs when `KeyDim` and `ValueDim` are equal
  and is wrong when they are not.
- **`nn.DepthwiseCausalConv`** (`nn/linear.go:178`) runs
  [§4.1.1](#411-the-correction-the-probe-padded-an-input-port-and-a-graph-cannot)'s
  rolling window. `nn/linear_test.go:246` matches a reference written from the
  definition, and `nn/linear_test.go:274` checks the carry across steps — five
  tokens then one, against a reference over all six — which a test that only
  prefills cannot see.

Nothing else. There is no `qwen3_5` architecture: `model/qwen3.go:28` registers
`Qwen3ForCausalLM` and nothing more, and `model/config.go` parses none of
`layer_types`, `full_attention_interval` or the `linear_*` fields. No code holds
a KV state beside a recurrent one. And no block assembles a gated delta layer
from a hidden state — $\alpha$, $\beta$, the q/k/v projections and the output
gate have no producer.

**The position on the target list, corrected.** accel closed
[#17](https://github.com/golang-design/accel/issues/17) and nothing upstream
blocks this spec ([011](011-sequencing.md)). What Qwen3.8-27B waits on is tgo's
own hybrid graph, and §6 says who owns each part of it.

## 6. The four children

The two blocks above are what this spec kept. The rest was four independent
scopes landing in four packages and touching three other specs, which no one
takes from nothing to done in one pass, so it is four specs. This section
indexes them and stops describing the work.

**[023](023-cache-kinds.md) — a cache that is per layer type.**
[005](005-kv-cache.md) gains a state per layer *kind* rather than one shape for
all: the recurrent `[slots, heads, valueDim, keyDim]` beside the paged KV, and
the convolution's rolling window — including the leading `slots` axis
`nn.DepthwiseCausalConv` cannot address today ([§4.1.1](#411-the-correction-the-probe-padded-an-input-port-and-a-graph-cannot)).
It is the allocation and addressing layer. Both blocks it serves already exist
and are tested, so it can be proved against them with no model graph present,
and the other three children wait on it.

**[024](024-qwen3-5-architecture.md) — the `qwen3_5` architecture.** Config
parsing for `layer_types`, `full_attention_interval` and the `linear_*` fields;
a registry entry with its weight map; the schedule that runs three linear layers
to one full one; the gate and projection block that feeds `nn.LinearAttention`;
and the two fields `nn.Attention` still lacks, a rotary width below `head_dim`
and `attn_output_gate` (§4). One architecture end to end, the shape
`model/qwen3.go` already established for the dense family. After 023.

**[025](025-recurrent-snapshot.md) — snapshot and restore for recurrent state.**
[016](016-prefix-cache.md) gains ollama's shape for the layers that have no
positions to page ([§4.2](#42-what-a-recurrent-state-needs-which-is-not-what-a-cache-needs)):
copy a sequence's state out, keep it against a prefix key, restore it into a
slot. Prefix reuse is orthogonal to whether the model runs at all, so the graph
is correct without it and slower. After 023.

**[026](026-image-tokens.md) — image tokens on the text path.** The tokenizer
and chat template tolerate a multimodal vocabulary: `image_token_id` 248056 and
`video_token_id` 248057 pass through or are refused with a reason, and the text
path does not break on a checkpoint that carries them. No image path. After 024,
which is where the checkpoint arrives.

Order: 023, then 024 and 025 in either order, then 026.

## Outcome

The operator this spec was written around exists, and so does the composition
beside it. `nn.LinearAttention` and `nn.DepthwiseCausalConv` landed on
2026-08-27 with value tests against float64 references, and that is the whole of
what 018 kept for itself. What it did not keep — the per-layer cache, the
`qwen3_5` architecture, snapshot and restore, and the multimodal vocabulary — is
[023](023-cache-kinds.md) through [026](026-image-tokens.md). This spec stays the
parent that indexes them.

**What shipped**, section by section:

| section | what landed | where |
| --- | --- | --- |
| 2 | accel's gated delta scan as one graph node over a segmented extent, with the state shape and the `QueryExtents` §2 describes | `nn/linear.go:67`, tested at `nn/linear_test.go:32` |
| 2.1 | the recurrence's refusals: heads and both widths at least one, k the same shape as q, v the same tokens and heads as q and differing only in width | `nn/linear.go:73-96`, tested at `nn/linear_test.go:82` |
| 4.1 | the depthwise causal convolution as $K$ scaled window slices summed, causality structural | `nn/linear.go:178`, tested at `nn/linear_test.go:246` |
| 4.1.1 | the rolling window: this step's rows scattered in, the window read, then its last $K-1$ rows written to the front for the next step — two writes to one state, ordered by its versions | `nn/linear.go:207`, `:220`, `:243`, carry tested at `nn/linear_test.go:274` |
| 6 | rows one and two of the original list; the remaining four moved to 023 through 026 | `specs/023-cache-kinds.md` … `specs/026-image-tokens.md` |

**What diverged** from the design, and why the code is right:

- **`nn.LinearAttention` returns accel's unflattened `[T, heads, ValueDim]`**,
  not the `[T, heads*ValueDim]` every projection around it is
  (`nn/linear.go:43-50`, `:96-105`). Flattening would put a `Reshape` in front of
  the result, and a `Reshape` in front of a *graph output* silently produces
  zeros — correct shape, no refusal, all zeros
  ([C25](010-conformance.md), accel#26). So every caller reshapes at its own call
  site, where the result is an operand and a view is ordinary. This binds the
  hybrid graph 024 builds.
- **`nn.ConvState` is a four-field contract**, not a state
  (`nn/linear.go:120-145`): the window, plus three caller-supplied u32 index
  ports — `Write` at $K-1+i$, `Carry` at $T..T+K-2$, `CarryWrite` at $0..K-2$.
  The step's index arithmetic lives in the caller. §4.1.1 argued for the rolling
  state and not for who computes the indices, and the code puts it outside the
  block because only the caller knows how many rows this step contributed.
  Getting `Carry` wrong is a silently wrong convolution, so 023 owns filling
  these ports.
- **§2.1 said the chunked form *is* the kernel.** accel shipped the sequential
  scan and recorded the chunked form as deliberately unbuilt. The scan is right
  first: chunking reassociates the recurrence and therefore needs its own numeric
  bound, which has to be derived against a scan that exists.
- **§4.1's "14 dispatches" was a $K=4$ probe figure.** `nn/linear.go:168-176`
  states ~$3K+5$, which at $K=4$ is 17 dispatches and 3 packing copies of a
  $[T, C]$ tensor per layer. The code is right because it counts the two scatters
  and the gather the rolling window added after the probe was run.
- **§4.2 expected accel to split `State` before building the operator.**
  `tensor.LinearAttention` shipped over the undifferentiated `State`, which
  exports `NewState`, `ReadState`, `LayerState` and `ScatterRows` and no
  snapshot. That is right for the operator, which needs no new type to run: the
  distinction buys snapshot and restore, which is 025's scope.
- **`DepthwiseCausalConv` refuses a non-f32 input and refuses $K \le 1$**
  (`nn/linear.go:189-196`, tested at `nn/linear_test.go:315`), neither of which
  this spec named. One tap is an elementwise scale, and Qwen3.8's
  `mamba_ssm_dtype: float32` makes f32-only plausible — but the dtype is a
  decision 023 has to record rather than inherit.

**Not built.** Four children carry what is left: [023](023-cache-kinds.md), the
per-layer cache kinds the other three wait on; [024](024-qwen3-5-architecture.md),
the `qwen3_5` config, weight map and hybrid graph;
[025](025-recurrent-snapshot.md), snapshot and restore for recurrent state; and
[026](026-image-tokens.md), image tokens on the text path. Two fields inside them
are worth naming on their own, because each is a silent wrong answer rather than
a refusal: `AttentionConfig` has no rotary width, so `nn/attention.go:207`
rotates the whole `HeadDim` and `partial_rotary_factor: 0.25` is inexpressible
(024 owns it); and `nn.ConvState`'s window has no `slots` axis, so a batched step
cannot give two sequences their own windows (023 owns it). Nothing in either
case refuses — the graph builds and the numbers are wrong.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 018-D1 | Qwen3.8-27B is out of v0, and said so publicly | list it as a target and hope | the hybrid graph, the per-layer cache and the `qwen3_5` registry entry are unbuilt (§6). The 48 gated-delta layers themselves are expressible — `nn.LinearAttention` and `nn.DepthwiseCausalConv` express them — so the overclaiming this decision avoids is about the *model*, not about the layers |
| 018-D2 | file the operator as a question, not a proposal | propose a kernel shape | tgo does not know accel's kernel constraints; #17 asked whether it is in scope first. **Outcome:** accel answered *in scope, recorded, not scheduled*, and found a deeper problem tgo could not see — that `State` conflates two types ([§4.2](#42-what-a-recurrent-state-needs-which-is-not-what-a-cache-needs)) |
| 018-D5 | check what composes before asking for a kernel | ask for the convolution too | the depthwise causal convolution composes over a **rolling state** ([§4.1](#41-the-depthwise-convolution-composes-over-a-port-and-not-over-a-projection)), which is a cache row rather than a kernel, so the ask stayed one kernel rather than two |
| 018-D3 | a hybrid cache is per layer *type*, not one shape | force the recurrent state into `State`'s per-position model | a recurrent state has one value per sequence and no positions to index |
| 018-D4 | record that ollama's snapshot design is forced here | keep treating it as a workload difference | [016 §10.1](016-prefix-cache.md) read it as a choice about concurrency; for a recurrent layer it is the only option |
