---
title: "The qwen3_5 architecture: forty-eight linear layers and sixteen softmax ones in one pass"
status: drafted
layer: graph
depends_on:
  - 000-decisions.md
  - 004-model-graph.md
  - 018-hybrid-models.md
  - 023-cache-kinds.md
---

# The `qwen3_5` architecture

[018](018-hybrid-models.md) §6 lists five things that must be built before
Qwen3.8-27B runs. Two are built, one is [023](023-cache-kinds.md), one is
[025](025-recurrent-snapshot.md), and one is
[026](026-image-tokens.md). This spec is the sixth line of that table — *the
`qwen3_5` registry entry: config parsing, the weight map, the layer-type
schedule* — written end to end, in the shape `model/qwen3.go` and
`model/qwen3_graph.go` already established for the dense family.

**Read [§11](#11-scope-this-is-not-one-pass) first if you are about to build it.**
This spec is three sub-scopes and one of them is gated on an accel answer that
does not exist yet ([§4.4](#44-the-gate-is-per-head-and-accels-is-per-token)).

## 1. What is there today, checked

Nothing in `model/` knows this architecture, and the refusal is already correct.

| claim | where it is checked |
| --- | --- |
| `ParseConfig` reads twelve fields and no more | `model/config.go:95` — `rawConfig` has no `layer_types`, no `full_attention_interval`, no `linear_*`, no `attn_output_gate`, no `partial_rotary_factor` |
| an unknown `architectures[0]` is refused with the known list | `model/model.go:167` — `unknown`, reached from `New` at `model/model.go:145` |
| `Architectures()` returns one name today | `model/qwen3.go:28` is the only `Register` call in the tree |
| the rotary width is **hardcoded** to `HeadDim` | `nn/attention.go:207-208` — `tensor.RoPE(g.B, qh, cfg.HeadDim, …)` twice; `AttentionConfig` has no rotary-dim field |
| the gated delta recurrence exists | `nn/linear.go:67` `LinearAttention`, over `tensor.LinearAttention` |
| the depthwise convolution exists | `nn/linear.go:178` `DepthwiseCausalConv`, over `nn.ConvState` |

So a `Qwen3_5ForConditionalGeneration` checkpoint fails at `model.Open` with the
list of architectures tgo knows, which is [004-D2](004-model-graph.md) working.

**[018 §4](018-hybrid-models.md) is three-quarters right and one-quarter wrong.**
It says `partial_rotary_factor: 0.25` is "already expressible" because
`rotaryDim` was always a parameter. That is true of `tensor.RoPE` and false of
`nn.Attention`, which passes `cfg.HeadDim` and offers the caller no way to pass
anything else. Adding a rotary-dim field to `AttentionConfig` is therefore a
prerequisite this spec owns, not a thing already done
([§5.1](#51-partial_rotary_factor-needs-a-field-attentionconfig-does-not-have)).

## 2. The config, and which field names are guesses

[018 §1](018-hybrid-models.md) and [011 §2.3](011-sequencing.md) quote the same
`config.json`. **They are the only source in this tree, and neither cites a
checkpoint.** No `qwen3_5` file has been opened here. Every name below is
therefore marked for confirmation against a real checkpoint before the parser is
written, because a parser keyed on an invented name reads zero and silently
builds a dense model.

| field | symbol | used for | confirmed? |
| --- | --- | --- | --- |
| `architectures[0]` = `Qwen3_5ForConditionalGeneration` | — | the registry key | **no** |
| `num_hidden_layers` = 64 | $L$ | the schedule's length | yes, §5 of 004 |
| `layer_types` | — | the schedule, verbatim | **no**, and its *shape* is also unknown ([§3.1](#31-layer_types-may-be-a-list-or-a-pattern-and-that-decides-the-check)) |
| `full_attention_interval` = 4 | $I$ | the schedule, derived | **no** |
| `hidden_size` = 5120 | $d$ | everything | yes |
| `head_dim` = 256, `num_attention_heads` = 24, `num_key_value_heads` = 4 | $d_h, H, H_{kv}$ | the full layers | yes |
| `partial_rotary_factor` = 0.25 | $\rho$ | rotary width $\rho d_h = 64$ | **no** |
| `attn_output_gate` = true | — | a fifth projection on the full layers | **no** |
| `linear_conv_kernel_dim` = 4 | $K$ | `DepthwiseCausalConv` | **no** |
| `linear_num_key_heads` = 16, `linear_key_head_dim` = 128 | $H_k, d_k$ | q and k of the linear layers | **no** |
| `linear_num_value_heads` = 48, `linear_value_head_dim` = 128 | $H_v, d_v$ | v, z and the state | **no** |
| `mamba_ssm_dtype` = `"float32"` | — | refused unless `float32` | **no** |
| `image_token_id`, `video_token_id` | — | [026](026-image-tokens.md); not read here | **no** |

Confirming them is one `curl` of a `config.json` and half an hour. It is the
first task of [§11](#11-scope-this-is-not-one-pass)'s sub-scope B, and every table
below is provisional until it is done.

**`Config` is not widened.** `model.Config` is [004 §5](004-model-graph.md)'s
table and is shared by every architecture. The `qwen3_5` fields go on a
`qwen35Config` that embeds it, reachable through the concrete builder, which is
what `Builder.Config` already promises at `model/model.go:55`: *"Fields a
specific architecture adds beyond §5's table are reachable through the concrete
builder type."*

## 3. The layer schedule

64 layers, one full-attention layer every $I = 4$, 16 full and 48 linear.

```mermaid
flowchart LR
  L0["ℓ0<br/>linear"] --> L1["ℓ1<br/>linear"] --> L2["ℓ2<br/>linear"] --> F3["ℓ3<br/>full"]
  F3 --> L4["ℓ4<br/>linear"] --> L5["ℓ5<br/>linear"] --> L6["ℓ6<br/>linear"] --> F7["ℓ7<br/>full"]
  F7 --> D["…"] --> L60["ℓ60<br/>linear"] --> L61["ℓ61<br/>linear"] --> L62["ℓ62<br/>linear"] --> F63["ℓ63<br/>full"]
```

The derived rule is $\ell$ is full attention iff $(\ell + 1) \bmod I = 0$, so
the full layers are $3, 7, \dots, 63$ and the *last* layer is full. The opposite
convention — $\ell \bmod I = 0$, so layer 0 is full and layer 63 is linear — is
equally plausible from the field name alone and produces a model with the same
shapes, the same parameter count, and different output. **The convention is not
derivable and is not guessed** ([024-D1](#decision-record)).

### 3.1 `layer_types` may be a list or a pattern, and that decides the check

[018 §1](018-hybrid-models.md) quotes

```json
"layer_types": ["linear_attention", "full_attention"],
```

which has **two** entries for a 64-layer model. Two readings:

- a **per-layer list** that the quote abbreviated, 64 entries long;
- a **repeating pattern**, in which case it says nothing `full_attention_interval`
  does not, and the two-entry form contradicts $I = 4$ outright — a period-2
  pattern is one full layer in two, not one in four.

The second reading makes the quoted config self-contradictory, which is itself
evidence the quote is an abbreviation. Either way the parser cannot tell without
a checkpoint, so it refuses a length it does not recognise rather than choosing
a reading.

### 3.2 The rule

```
schedule(cfg) -> []layerKind, error

  types  = cfg.LayerTypes            // present or absent
  I      = cfg.FullAttentionInterval // present or absent

  if types absent and I absent:      refuse: neither field says what the layers are
  if types present and len != L:     refuse: naming the length it has and L
  if I present and I <= 0:           refuse: naming the value
  derived = [ (ℓ+1) % I == 0 ? full : linear  for ℓ in 0..L-1 ]   // when I present
  if types present and I present and types != derived:
                                     refuse: naming the first ℓ they disagree at
  return types if present else derived
```

A checkpoint that says one thing in two places is **refused, not reconciled**.
[000 D6](000-decisions.md) is the authority: *"A model that the registry does not
know is refused at load with the list of what it does know, never guessed at"*,
and a schedule guessed from the field a reader happened to prefer is the same
failure one level in. [004 §7](004-model-graph.md)'s table is the precedent for
where the refusal goes and how it reads: at parse, naming the config field.

Refusing costs nothing if the two fields always agree. It costs everything if
they ever do not, because both schedules build, both run, and only one of them
is the model.

## 4. The gated delta block

One linear layer, end to end. $T$ tokens, $B$ sequences, one slot per sequence.

```mermaid
flowchart TB
  H["h &nbsp;[T, 5120]"] --> N["RMSNorm<br/>gain [5120]"]
  N --> QKVZ["in_proj_qkvz<br/>[5120, 16384]"]
  N --> BA["in_proj_ba<br/>[5120, 96]"]
  QKVZ --> SPLIT{"Slice"}
  SPLIT -->|"[T, 2048]"| QQ["q"]
  SPLIT -->|"[T, 2048]"| KK["k"]
  SPLIT -->|"[T, 6144]"| VV["v"]
  SPLIT -->|"[T, 6144]"| ZZ["z &nbsp;the gate"]
  QQ --> CV["DepthwiseCausalConv<br/>K=4, taps [4, 10240]<br/>over q, k, v joined"]
  KK --> CV
  VV --> CV
  CV --> SIL["SiLU"]
  SIL --> REP["Broadcast + Contiguous<br/>q,k: 16 heads → 48"]
  BA --> GATES["beta = sigmoid(b) &nbsp;[T, 48]<br/>alpha from A_log and dt_bias &nbsp;[T, 48]"]
  REP --> LA["nn.LinearAttention<br/>state [B, 48, 128, 128]"]
  GATES --> LA
  LA --> OUT["o &nbsp;[T, 48, 128]"]
  OUT --> RS["Reshape [T, 6144]"]
  RS --> GN["RMSNorm gain [128] per head<br/>times SiLU(z)"]
  ZZ --> GN
  GN --> OP["out_proj<br/>[6144, 5120]"]
  OP --> ADD["+ residual"]
```

### 4.1 The recurrence, restated from the code and not from the diagram

[018 §2](018-hybrid-models.md) gives accel's form as

$$u = S k, \qquad S \leftarrow \alpha S + \beta\,k\,(v-u)^\top, \qquad o = S q$$

and `nn/linear_test.go:154`'s `linearReference` — written from that equation and
not from the block — settles the two things the display math does not say:

- $u = S_{t-1} k_t$ reads the state **before** the decay
  (`nn/linear_test.go:177`, "the state's reading of this key, before the decay");
- $o_t = S_t q_t$ reads the state **after** the update
  (`nn/linear_test.go:193`, the accumulation runs over the just-written `S`).

A host reference written from the display math alone gets one of those wrong
half the time, so any reference this spec's tests use is written against
`linearReference`'s ordering.

**Where $\alpha$ sits is not re-decided here.** [018 §2](018-hybrid-models.md)
records that Qwen3.5's published form places $\alpha$ outside the whole bracket
and accel's does not, that the two differ in whether the correction term is
decayed, and that only the checkpoint settles it. That is carried forward
unchanged as a tier-3 check against real weights
([§8](#8-verification-without-a-50-gib-download)).

### 4.2 The projections, with every shape

Symbols from [§2](#2-the-config-and-which-field-names-are-guesses): $d = 5120$,
$H_k = 16$, $d_k = 128$, $H_v = 48$, $d_v = 128$, $K = 4$. Write
$W_k = H_k d_k = 2048$ and $W_v = H_v d_v = 6144$.

| step | operator | in | out |
| --- | --- | --- | --- |
| norm | `nn.RMSNorm` | `[T, d]`, gain `[d]` | `[T, 5120]` |
| qkvz | `nn.Linear` | `[T, 5120] × [5120, 16384]` | `[T, 16384]` |
| ba | `nn.Linear` | `[T, 5120] × [5120, 96]` | `[T, 96]` |
| split q | `Slice` → `Contiguous` | `[T, 16384]`, cols `[0, 2048)` | `[T, 2048]` |
| split k | `Slice` → `Contiguous` | cols `[2048, 4096)` | `[T, 2048]` |
| split v | `Slice` → `Contiguous` | cols `[4096, 10240)` | `[T, 6144]` |
| split z | `Slice` → `Contiguous` | cols `[10240, 16384)` | `[T, 6144]` |
| conv | `nn.DepthwiseCausalConv` | `[T, 10240]`, taps `[4, 10240]`, `ConvState` | `[T, 10240]` |
| activation | `tensor.SiLU` | `[T, 10240]` | `[T, 10240]` |
| repeat q, k | `Reshape` → `Broadcast` → `Contiguous` → `Reshape` | `[T, 16, 1, 128]` → `[T, 16, 3, 128]` | `[T, 6144]` |
| β | `Slice` → `Sigmoid` | `[T, 96]`, cols `[0, 48)` | `[T, 48]` |
| α | `Slice` → softplus → scale → `Exp` | cols `[48, 96)`, with `A_log` `[48]` and `dt_bias` `[48]` | `[T, 48]` |
| recurrence | `nn.LinearAttention` | q, k `[T, 6144]`, v `[T, 6144]`, state `[B, 48, 128, 128]` | `[T, 48, 128]` |
| flatten | `Reshape` | `[T, 48, 128]` | `[T, 6144]` |
| gated norm | `RMSNorm` per head × `SiLU(z)` | `[T·48, 128]`, gain `[128]`; z `[T, 6144]` | `[T, 6144]` |
| out | `nn.Linear` | `[T, 6144] × [6144, 5120]` | `[T, 5120]` |
| residual | `tensor.Add` | two `[T, 5120]` | `[T, 5120]` |

Note the `Reshape` on the recurrence's output is at the **caller's** call site
and not inside `nn.LinearAttention`, which is `nn/linear.go:106`'s deliberate
choice: a `Reshape` in front of a graph *output* silently produces zeros
([C25](010-conformance.md)), so the block returns accel's unflattened
`[T, heads, ValueDim]` and every caller flattens where the value is an operand.
This block is a caller, so it flattens.

### 4.3 q and k are 16 heads and the state is 48

`tensor.LinearAttention` requires `k.shape.Equal(q.shape)` and `v.shape[1] ==
heads` (accel `tensor/linear.go:107` and `:110`), so **one head count serves all
three**. Qwen3.5 has $H_k = 16$ against $H_v = 48$: the recurrence is per value
head and three value heads share a key head, which is grouped-query attention
inside a linear layer.

The expression is `Broadcast` over a size-one axis
(accel `tensor/views.go:178`), which gives repeat-each-3-times ordering and maps
value head $h$ to key head $\lfloor h/3 \rfloor$. Whether the checkpoint's
convention is repeat-each (interleave) or tile is the one thing the shapes
cannot distinguish, and it is a checkpoint question
([§8](#8-verification-without-a-50-gib-download)).

The cost is four extra nodes and one `[T, 6144]` copy for each of q and k. It is
not a state cost: the state is per value head either way,
$48 \times 128 \times 128 \times 4$ bytes $= 3$ MiB per sequence per layer, 144
MiB per sequence over 48 layers.

The comment at `nn/linear.go:15` explains the non-square state as *"16 key heads
of 128 against 48 value heads of 128"*, which reads a head-count difference as a
width difference. The widths are both 128. The state is non-square because
`KeyDim` and `ValueDim` are independently configurable, and the head counts are
a separate problem this section owns.

### 4.4 The gate is per head and accel's is per token

**This is what decides whether the block is buildable, and it is open.**

`tensor.LinearOptions.Alpha` and `.Beta` hold *"one entry per token, in the flat
order q is in"* (accel `tensor/linear.go:21`); the operator refuses anything else
with *"a gate is per token, not per sequence"* (accel `tensor/linear.go:127`),
and the kernel reads `alpha[tok]` with no head axis
(accel `internal/testkernels/linear_test.go:51`).

The projection this spec reads as `in_proj_ba` produces $2 H_v = 96$ values per
token, which is **one α and one β per value head**. If Qwen3.5's gates are per
head, accel's operator cannot express the layer, and

- averaging the 48 gates into one runs, produces plausible numbers, and is a
  different model;
- running the operator 48 times with one head each is 48 dispatches per layer
  over 48 layers and a state sliced per head, which is a workaround
  [000 D1](000-decisions.md) forbids for exactly the reason it gives.

So this is a report before it is a build: a test that names it, a new row in
[010 §2](010-conformance.md), and an issue on accel citing this spec — which is
[000 D1](000-decisions.md)'s sequence, unchanged. The row has no number yet
because the register numbers rows when they are added and this spec does not
edit 010.

**What must be established first is whether the gates are per head at all.**
`in_proj_ba`'s width is the evidence and it is an unconfirmed name
([§2](#2-the-config-and-which-field-names-are-guesses)). A checkpoint whose `ba`
projection is 2 wide rather than 96 makes this section moot and the block
buildable today. That check is one safetensors header read.

### 4.5 The weight map

The names below follow Qwen3-Next, the architecture `qwen3_5` succeeds.
**None is confirmed against a `qwen3_5` checkpoint.** The alternative naming a
reader should expect if these are wrong is a split projection — `q_proj`,
`k_proj`, `v_proj`, `z_proj`, `b_proj`, `a_proj` — which changes the table's
rows and none of its shapes, because a fused projection is the concatenation of
the split ones.

| checkpoint tensor | shape in file | port | transpose | permute | kind |
| --- | --- | --- | --- | --- | --- |
| `model.layers.ℓ.input_layernorm.weight` | `[d]` | `ℓ.attn_norm` | no | no | gain |
| `model.layers.ℓ.linear_attn.in_proj_qkvz.weight` | `[2W_k + 2W_v, d]` | `ℓ.lin_qkvz` | **yes** | no | projection |
| `model.layers.ℓ.linear_attn.in_proj_ba.weight` | `[2H_v, d]` | `ℓ.lin_ba` | **yes** | no | projection |
| `model.layers.ℓ.linear_attn.conv1d.weight` | `[2W_k + W_v, 1, K]` | `ℓ.lin_taps` | **reshaped** → `[K, 2W_k+W_v]` | no | gain |
| `model.layers.ℓ.linear_attn.dt_bias` | `[H_v]` | `ℓ.lin_dt` | no | no | gain |
| `model.layers.ℓ.linear_attn.A_log` | `[H_v]` | `ℓ.lin_alog` | no | no | gain |
| `model.layers.ℓ.linear_attn.norm.weight` | `[d_v]` | `ℓ.lin_norm` | no | no | gain |
| `model.layers.ℓ.linear_attn.out_proj.weight` | `[d, W_v]` | `ℓ.lin_out` | **yes** | no | projection |

Three rows are not ordinary:

- **`conv1d.weight` is not a transpose.** It is `[C, 1, K]` in the file and
  `DepthwiseCausalConv` wants `[K, C]` (`nn/linear.go:197`), which is a squeeze
  of the middle axis and then a transpose of a genuinely 2-D plane.
  `WeightSpec.Transpose` at `model/weights.go:97` reverses a **rank-2** shape and
  `weights.targetShape` refuses a transpose of any other rank
  (`weights/weights.go:762`). So the map cannot express this row today, and
  either `WeightSpec` gains a reshape or the loader gets a rank-3 case. A spec
  that stated the port shape and left this out would be a spec that does not
  build.
- **`A_log` and `dt_bias` are f32 and are not gains.** They are per-head
  constants inside an exponential, and `KindGain` is what routes a tensor through
  `loadGains` (`gains.go:40`) to an f32 device buffer, which is the width they
  need. Calling them gains is the right *mechanism* and the wrong *name*; the
  alternative is a `KindConstant` that does the same thing, and this spec takes
  the mechanism and records the mismatch rather than adding a kind for two
  tensors.
- **Nothing in a linear layer permutes.** `WeightSpec.Permute` exists for the
  rotary convention (`model/weights.go:108`, [004-D9](004-model-graph.md)) and a
  linear layer has no rotation, so the whole column is `no`. The full-attention
  layers keep 004's map exactly, including the two permuted QK-norm gains.

The full-attention layers' rows are [004 §4](004-model-graph.md)'s, unchanged,
plus one:

| checkpoint tensor | shape in file | port | transpose | permute |
| --- | --- | --- | --- | --- |
| `model.layers.ℓ.self_attn.gate_proj.weight` | `[H·d_h, d]` | `ℓ.wgate_attn` | **yes** | **no** — see below |

**This name is unconfirmed too**, and it has a second reading the others do not:
an exporter may fuse the gate into `q_proj` and ship it at `[2H·d_h, d]`, in
which case the row is a wider `wq` and a split, not a tensor of its own. The
header settles it — the width of `q_proj` says which — and until it is read the
weight map states both and builds neither.

The gate is **not** permuted. `Permute` reorders a projection's output channels
into accel's interleaved rotary pairing, and the gate's output multiplies the
attention result, which is a mixture of unrotated V rows — the same argument
`model/qwen3.go:69` makes for `v_proj` and `o_proj`.

Both layer kinds share the MLP rows and the embedding, final norm and head rows.
The weight map is therefore `[]WeightSpec` built by walking the schedule and
emitting one of two row sets per layer, which is `qwen3Weights`
(`model/qwen3.go:154`) with a branch. **`model/weights.go` does not change**:
`Check` already refuses a tensor the map does not name and a tensor the map names
and the file lacks, and a hybrid map is still a map.

## 5. What `nn` gains

Three changes, each defaulting to today's behaviour by zero value, so the dense
path is byte-identical.

### 5.1 `partial_rotary_factor` needs a field `AttentionConfig` does not have

```go
type AttentionConfig struct {
    QHeads, KVHeads, HeadDim int

    // RotaryDim is how many of a head's channels rotate, and zero is HeadDim.
    RotaryDim int
    …
}
```

and `nn/attention.go:207-208` becomes

```go
rot := cfg.RotaryDim
if rot == 0 {
    rot = cfg.HeadDim
}
qh = tensor.RoPE(g.B, qh, rot, cfg.RoPEBase, posQ)
kh = tensor.RoPE(g.B, kh, rot, cfg.RoPEBase, posK)
```

Zero rather than a `PartialRoPE` flag beside a width, for the reason `Step.Extents`
records at `nn/attention.go:44`: two fields that must agree are a pair a caller
can set half of. `qwen3` passes nothing and its graph does not change.

The refusal at `nn/attention.go:166` moves with it — the rotary dimension is what
must be even, not the head dimension — and it must also refuse `RotaryDim >
HeadDim`, which accel would refuse in its own terms one layer down.

`ParseConfig`'s odd-`head_dim` refusal (`model/config.go:171`) stays where it is;
`qwen35Config` adds the same check on $\rho d_h$, and refuses a $\rho$ that does
not produce an integer.

### 5.2 `attn_output_gate` is a fifth projection, and it multiplies before `O`

```go
type AttentionWeights struct {
    Q, K, V, O   Operand
    QNorm, KNorm *tensor.Tensor

    // Gate is attn_output_gate's projection, or the zero Operand.
    Gate Operand
}
```

`nn.Attention` ends with `Linear(g, Reshape(a, [T, H·d_h]), w.O)` at
`nn/attention.go:250` and `:261` — the ragged branch and the single-sequence
one. The gate multiplies the `[T, H·d_h]` value
**before** that projection:

$$\text{out} = \big(\,\mathrm{Attn}(q,k,v) \odot \sigma(xW_g)\,\big)\,W_O$$

Before, not after, because the gate's width is $H d_h$ and $W_O$ maps $H d_h \to
d$. A gate applied after $O$ would have to be $d$ wide, which is a different
projection shape and a different model. The width is what decides it, and the
zero `Operand` means no gate — `Operand.Form()` at `nn/nn.go:168` already
answers on a zero value, and `Operand.ok()` at `nn/nn.go:187` already refuses a
half-built one.

### 5.3 The gated delta block goes in `nn`

`nn.GatedDelta(g, x, w GatedDeltaWeights, st GatedDeltaStep, cfg LinearConfig)`,
composing `RMSNorm`, four `Linear`s, `DepthwiseCausalConv`, the gate arithmetic,
`LinearAttention` and the gated norm — [§4.2](#42-the-projections-with-every-shape)'s
table, one function.

In `nn` and not inline in `model/qwen3_5_graph.go` because
[004-D1](004-model-graph.md) puts composites in `nn` and because the block is
what [§9](#9-tests)'s per-block value test tests. A block inside a model file is
testable only through the whole graph, and a whole-graph test cannot say which of
sixteen nodes moved.

## 6. The graph, the registry, and the two ports that change

### 6.1 One `Register`, a second builder, and nothing else in `model/`

```go
// model/qwen3_5.go
const Qwen35Architecture = "Qwen3_5ForConditionalGeneration"

func init() { Register(Qwen35Architecture, newQwen35) }

type qwen35 struct {
    cfg   *qwen35Config
    kinds []layerKind // §3.2's schedule, computed once
}
```

`Register` (`model/model.go:95`), `Architectures` (`:114`), `New` (`:145`) and
`unknown` (`:167`) do not change. `Architectures()` returns two names and the
refusal message lists both, which is the only observable change to `model/` from
outside.

Two new files, `model/qwen3_5.go` and `model/qwen3_5_graph.go`, mirroring the
dense pair. `model/qwen3.go` is not touched: a second architecture is one file
and one `init` ([000 D6](000-decisions.md)), and threading a mode flag through
`qwen3` would make one builder answer for two weight maps and two forward passes,
which is the generic path [004-D2](004-model-graph.md) refuses in a different
costume.

`Template()` returns `chat.Qwen3()` until a `qwen3_5` chat template is read from
a checkpoint. That is [003](003-chat-template.md)'s and is not asserted here.

### 6.2 `Declare` must declare extents at batch one

`tensor.LinearAttention` refuses a nil `QueryExtents` unconditionally
(accel `tensor/linear.go:81`), and the state's slot count is read *from* the
extent count (accel `tensor/linear.go:116`, `:136`). `model.Declare` declares
`PortExtents` only when `batch > 1` (`model/graph.go:225`).

So a `qwen3_5` graph declares `extents` at every batch size, `[max(B,1)]` u32,
and a single sequence binds `[T]`. `GraphSpec` gains nothing: the architecture
decides, and `Declare` takes the config.

**The full-attention layers still get nil.** Passing the same extents into
`nn.Attention` sets `opts.BaseName = ""` and takes the ragged kernel
(`nn/attention.go:243`), and at batch 1 `Inputs.Validate` requires a non-empty
`Base` for $T > 1$ (`model/graph.go:402`). Those two are the same rule read from
both ends: a single-sequence prefill has a causal base and a ragged step does
not. A hybrid graph at batch 1 therefore hands `Step.Extents = nil` to the full
layers and the extents tensor to the linear ones, and at batch above one hands
the same tensor to both — which is what the dense batched path already does.

This is a wiring decision with no shape to catch it: extents into the full layers
compiles, runs, and masks the wrong keys. [§9](#9-tests) has the test.

### 6.3 The KV state is allocated at sixteen layers, not sixty-four

`Declare` allocates `[L, C, H_kv, d_h]` (`model/graph.go:230`) and
`Builder.Forward` takes layer $\ell$'s window with `tensor.LayerState(b, in.Keys,
l)` (`model/qwen3_graph.go:88`), indexed by **absolute** layer. With 16 full
layers of 64, an unchanged `Declare` reserves 48 layers of capacity nothing
writes: at $C = 8192$, $H_{kv} = 4$, $d_h = 256$, f16, that is 16 MiB per unused
layer per state, and **1.5 GiB per sequence** across the key and value states
together.

So the state is allocated at the *full-attention* count and the builder maps
absolute layer to dense index. The map is the schedule's prefix sum and is
computed once beside `kinds`.

[023](023-cache-kinds.md) owns the cache *kinds* — which state a layer type
carries. **The absolute-to-dense index is this spec's**, because it is a property
of the schedule and the schedule is here. If 023 lands with a different shape for
the same idea, 023 wins and this section is amended.

### 6.4 The convolution window is rank two, and a batch needs rank three

The block slices axis 0 of the window (`nn/linear.go:220`), which is the *time*
axis. The shipped test binds a rank-2 `[K-1+T, C]` state
(`nn/linear_test.go:214`), and that is the shape the block works over: on a
rank-3 state the slice would cut slots. `nn.ConvState`'s doc comment said
`[slots, K-1+T, C]` until 2026-08-27, which is what made this look done; it now
says what the block addresses, and so does [C26](010-conformance.md).

So the convolution is **single-sequence today**. A batched hybrid graph needs
either a slot axis the block slices past or one state per slot, and neither
exists. This is named here and owned by [023](023-cache-kinds.md), which is where
the per-slot state shapes are decided. Sub-scope C
([§11](#11-scope-this-is-not-one-pass)) is single-sequence for this reason.

## 7. Refusals

[004 §7](004-model-graph.md)'s table is the precedent: refuse at parse, name the
field, and refuse anything the graph cannot honour rather than approximating it.

| condition | why refusing beats approximating |
| --- | --- |
| `layer_types` and `full_attention_interval` disagree | two schedules build and run; only one is the model ([§3.2](#32-the-rule)) |
| both absent | nothing says which layers are which |
| `layer_types` has an unrecognised length | the list-or-pattern question is unanswerable from the file ([§3.1](#31-layer_types-may-be-a-list-or-a-pattern-and-that-decides-the-check)) |
| `layer_types` holds a string that is neither kind | a third layer type is a model this graph does not implement |
| `linear_num_value_heads` not a multiple of `linear_num_key_heads` | the heads do not group ([§4.3](#43-q-and-k-are-16-heads-and-the-state-is-48)) |
| $\rho \cdot d_h$ not an even integer | RoPE rotates pairs; `ParseConfig` already refuses this shape of mistake at `model/config.go:171` |
| `mamba_ssm_dtype` is not `float32` | accel's state is f32 (accel `tensor/linear.go:144`); a config asking for anything else asks for a precision that is not there |
| `attn_output_gate` true and no gate tensor in the checkpoint | `model.Check` already refuses it as a missing tensor, by name |
| **any `qwen3_5` field this graph does not implement** | see below |
| the per-head gate, until [§4.4](#44-the-gate-is-per-head-and-accels-is-per-token) resolves | a mean over 48 gates runs and is a different model |

The last general row is the one [004 §7](004-model-graph.md) does not have and
this architecture needs. `rawConfig` ignores unknown JSON keys, which is right
for a dense model whose extra fields are metadata. It is wrong here: a `qwen3_5`
config carries fields that change the arithmetic, this spec has read none of them
from a real file, and a field tgo silently ignores is a model tgo silently gets
wrong. So `qwen35Config` parses into a map as well as a struct and refuses any
key not on an explicit allow list, naming it.

That is stricter than `ParseConfig` and deliberately so. The dense parser earned
its tolerance by being run against real checkpoints; this one has not been run
against any.

## 8. Verification without a 50 GiB download

[000 D8](000-decisions.md): **no test downloads weights.** CI runs a synthetic
config, `TGO_MODEL` runs the real one by hand, and [011 §4](011-sequencing.md)
records the result.

The fixture is a **4-layer, $I=4$, $d=64$** hybrid — three linear layers and one
full — with $H_k = 2$, $d_k = 6$, $H_v = 6$, $d_v = 4$, $K = 3$, $H = 4$,
$H_{kv} = 2$, $d_h = 8$, $\rho = 0.5$, $V = 128$. Every pair of dimensions
differs, which is CONTRIBUTING's rule and `nn/linear_test.go:28`'s reason: at
$d_k = d_v$ a state with its last two axes transposed still runs, and at
$H_k = H_v$ [§4.3](#43-q-and-k-are-16-heads-and-the-state-is-48)'s replication is
the identity.

**What the fixture proves:** the schedule, every refusal, the weight map's
completeness against a synthetic checkpoint, each block against a host reference,
the wiring of extents to one layer kind and not the other, the residual stream's
shape through both kinds, decode equalling prefill's last row, and the state
carrying across steps.

**What only a real checkpoint proves:** every field name in
[§2](#2-the-config-and-which-field-names-are-guesses) and every tensor name in
[§4.5](#45-the-weight-map); the schedule's convention
([§3](#3-the-layer-schedule)); whether the gates are per head
([§4.4](#44-the-gate-is-per-head-and-accels-is-per-token)); the head-replication
convention ([§4.3](#43-q-and-k-are-16-heads-and-the-state-is-48)); and where
$\alpha$ sits ([018 §2](018-hybrid-models.md)). Each is a value the fixture
supplies to itself and therefore cannot check.

That list is the honest statement of what a green CI run means here: **the graph
is self-consistent, and whether it is Qwen3.5 is unproven.**

## 9. Tests

| test | what it asserts |
| --- | --- |
| `TestTheScheduleReadsLayerTypesVerbatim` | a 4-entry `layer_types` produces exactly those kinds, whatever the interval implies |
| `TestTheScheduleDerivesFromTheIntervalWhenTypesAreAbsent` | $(\ell+1) \bmod I = 0$ over $L$ layers, last layer full |
| `TestASchedulesTwoFieldsAreRefusedWhenTheyDisagree` | the error names the first $\ell$ they differ at, and neither reading is chosen |
| `TestAConfigWithNeitherScheduleFieldIsRefused` | both absent is a refusal, not a dense fallback |
| `TestAnUnknownQwen35FieldIsRefusedByName` | §7's general row: an unimplemented field is named |
| `TestGatedDeltaMatchesAHostReference` | **per-block value test.** The whole block of §4.2 in float64, written from the equation and `linearReference`'s ordering, within a tolerance derived from the contraction lengths and the scan depth ([010 §5.1](010-conformance.md)) |
| `TestTheConvolutionCarriesAcrossTwoSteps` | a 3-token step then a 1-token step equals a reference over all four; a dropped carry cannot be seen in a prefill-only test |
| `TestKeyHeadsAreReplicatedNotTiled` | value head $h$ reads key head $\lfloor h/3\rfloor$; a tile ordering has the same shape |
| `TestAPartialRotaryLeavesTheUpperChannelsAlone` | `RotaryDim` $< $ `HeadDim` rotates the first $\rho d_h$ and leaves the rest bit-identical |
| `TestTheDensePathIsUnchangedByTheRotaryField` | `qwen3`'s graph at `RotaryDim: 0` selects the same kernels and produces the same logits as before |
| `TestTheOutputGateMultipliesBeforeTheOutputProjection` | a gate of all-zeros gives all-zero output; a gate applied after $O$ has the wrong width and would refuse |
| `TestTheFullLayersDoNotReceiveTheLinearExtents` | at batch 1 the full layers keep their `Base` scalar; a graph that passed extents to both compiles and masks wrong |
| `TestTheKVStateIsAllocatedAtTheFullAttentionCount` | 16 layers of KV for 64 layers of model, and layer $\ell$ reads its dense index |
| `TestTheWholeHybridGraphAtTheFixtureSize` | **whole-graph test.** 4 layers, three kinds of state bound, logits `[1, V]` f32, finite, and equal to a float64 reference of the full pass |
| `TestDecodeEqualsPrefillsLastRow` | §3.1 of 004 for a hybrid: the recurrent state after a 3-token prefill plus one decode equals a 4-token prefill's last row |
| `TestEveryWeightTheMapNamesIsInTheFixtureAndNoOther` | `model.Check` over the hybrid map; a missing or extra tensor is refused by name |
| `TestTheGateIsPerHeadAndAccelTakesPerToken` | the named, skipping conformance test [000 D1](000-decisions.md) requires, pointing at the new row in [010 §2](010-conformance.md) |

## 10. What this spec does not own

| | owner |
| --- | --- |
| which state a layer *kind* carries, and its shape per slot | [023](023-cache-kinds.md) |
| snapshot and restore of a recurrent state, and prefix reuse over it | [025](025-recurrent-snapshot.md) |
| image and video tokens through the tokenizer and template | [026](026-image-tokens.md) |
| where $\alpha$ sits in the recurrence | [018 §2](018-hybrid-models.md), decided by a checkpoint |
| the chunked parallel form of the scan, and its cost | accel, recorded as deliberately unbuilt |
| a `qwen3_5` chat template | [003](003-chat-template.md) |

## 11. Scope: this is not one pass

**One person cannot execute this spec completely in one pass**, and the reason is
not size — it is that sub-scope A ends in a question tgo cannot answer alone.

| | sub-scope | executable now? |
| --- | --- | --- |
| **A** | the `nn` prerequisites: `AttentionConfig.RotaryDim`, `AttentionWeights.Gate`, and reading a real `config.json` and safetensors header to settle [§2](#2-the-config-and-which-field-names-are-guesses), [§4.4](#44-the-gate-is-per-head-and-accels-is-per-token) and [§4.5](#45-the-weight-map) | the `nn` half yes; the answer is a download somebody does by hand |
| **B** | `qwen35Config`, the schedule, its refusals, the weight map, the registry entry, `Declare`'s extents and the dense KV index | **yes, today**, and it needs nothing from A except the field names |
| **C** | `nn.GatedDelta` and `model/qwen3_5_graph.go` wired end to end against the host reference | **gated on A**: a per-head gate makes the block inexpressible |

Build **B** first. It is the half that is testable with a synthetic fixture, it
makes the refusals real, and it is what turns "tgo does not know this
architecture" into "tgo knows this architecture and says what it cannot do yet" —
which is [000 D1](000-decisions.md)'s output.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 024-D1 | read `layer_types` verbatim, derive from `full_attention_interval`, and **refuse when both are present and disagree** | trust `layer_types` alone; derive always; take the first field found | two schedules build, run, and produce fluent different text. [000 D6](000-decisions.md) forbids guessing an architecture and a schedule is an architecture ([§3.2](#32-the-rule)) |
| 024-D2 | one `Register` call in a second file, `model/qwen3_5.go`, with its own builder | a mode flag on `qwen3`; a shared builder over two weight maps | [000 D6](000-decisions.md)'s "one file and one init". A builder answering for two weight maps is the generic path [004-D2](004-model-graph.md) refuses |
| 024-D3 | the gated delta block is `nn.GatedDelta`, in `nn` | compose it inline in `model/qwen3_5_graph.go` | [004-D1](004-model-graph.md); and a block inside a model file is testable only through the whole graph, which cannot say which of sixteen nodes moved |
| 024-D4 | the per-head gate is **filed before it is built** — a named skipping test, a row in [010 §2](010-conformance.md), an accel issue | mean the 48 gates into one; run the operator once per head | a mean runs and is a different model; 48 dispatches per layer is the private workaround [000 D1](000-decisions.md) exists to forbid ([§4.4](#44-the-gate-is-per-head-and-accels-is-per-token)) |
| 024-D5 | replicate q and k from $H_k$ to $H_v$ heads with `Broadcast` + `Contiguous` | ask accel for grouped heads inside `LinearAttention` | it composes today at eight nodes and two `[T, W_v]` copies per layer, and [018-D5](018-hybrid-models.md) is the rule: check what composes before asking for a kernel |
| 024-D6 | `AttentionConfig.RotaryDim`, **zero means `HeadDim`** | a `PartialRoPE` bool beside a width; a separate block; passing the width at every call site | the dense path passes nothing and is unchanged. A flag and a width are a pair a caller can set half of, which `Step.Extents` records at `nn/attention.go:44` |
| 024-D7 | the output gate multiplies **before** $W_O$ | after $W_O$, on the `[T, d]` result | the gate projection is $H d_h$ wide and $W_O$ maps $H d_h \to d$; after-$O$ needs a $d$-wide gate, which is a different tensor in the checkpoint ([§5.2](#52-attn_output_gate-is-a-fifth-projection-and-it-multiplies-before-o)) |
| 024-D8 | `extents` is declared at batch one for this architecture, and the **full layers still receive nil** | give both layer kinds the same extents tensor | accel refuses a nil extent on the recurrence and refuses a base on a ragged attention. Passing extents to both compiles and masks the wrong keys, with nothing in the output to say so ([§6.2](#62-declare-must-declare-extents-at-batch-one)) |
| 024-D9 | the KV state is allocated at the **full-attention layer count** with an absolute-to-dense index | allocate `[L, …]` and leave 48 layers unwritten | 768 MiB per sequence per state of capacity nothing writes, at $C=8192$ f16 ([§6.3](#63-the-kv-state-is-allocated-at-sixteen-layers-not-sixty-four)) |
| 024-D10 | a `qwen3_5` config key this graph does not implement is **refused by name** | ignore unknown keys, as `rawConfig` does | the dense parser earned its tolerance against real checkpoints; this one has read none. A field tgo ignores here changes the arithmetic ([§7](#7-refusals)) |
| 024-D11 | ship sub-scope B before A and C, and say the spec is not one pass | one branch that lands the architecture whole | C is gated on an accel answer that does not exist. B is testable today and converts an unknown architecture into a stated, named gap, which is [000 D1](000-decisions.md)'s output ([§11](#11-scope-this-is-not-one-pass)) |
