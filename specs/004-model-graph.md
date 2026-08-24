---
title: "The model graph: nn blocks, the registry, Qwen3, and every shape between them"
status: drafted
layer: graph
depends_on:
  - 000-decisions.md
  - 001-weights.md
---

# The model graph

## 1. Where the line is

`tensor.Builder` records a DAG of operators. `nn` is a thin library of the
composites a transformer is made of, and a model is a function that calls them.
`nn` holds **no state, no device, and no weights** — a block takes tensors and
returns tensors, and weights arrive as `tensor.Weight` ports declared by name.

```mermaid
flowchart TB
  subgraph tgo
    M["model/qwen3<br/>config -> forward pass"] --> N["nn<br/>RMSNorm, Attention, MLP"]
    M --> R["model<br/>registry, Config, weight map"]
  end
  N --> T["accel/tensor<br/>Builder, operators, Plan"]
  T --> A["accel<br/>device, buffers, command graph"]
  A --> D["CPU backend / Metal"]
```

The arrow that must not exist is tgo → accel's device layer. If a model needs
it, that is [000 D1](000-decisions.md) and it becomes a
[010](010-conformance.md) register row.

## 2. The `nn` surface

```go
package nn

// Graph is what every block records into. It is tensor.Builder plus the
// per-model constants a block would otherwise take as six arguments.
type Graph struct {
    B      *tensor.Builder
    Eps    float32   // rms_norm_eps
    Prefix string    // weight-name prefix for the block being recorded
}

// Weight declares a f16 or quantized weight port by name, resolving which of
// the two from how the loader stored it.
func (g *Graph) Weight(name string, shape tensor.Shape) Operand

// Operand is a weight that is either f16 or int8+scales. It exists so a block
// writes one call rather than branching on precision at every projection.
type Operand struct {
    Dense *tensor.Tensor   // f16, [K, N]
    Quant tensor.Quantized // i8 quants + f16 scales
}

func Linear(g *Graph, x *tensor.Tensor, w Operand) *tensor.Tensor
func RMSNorm(g *Graph, x *tensor.Tensor, gain *tensor.Tensor) *tensor.Tensor
func SwiGLUMLP(g *Graph, x *tensor.Tensor, gate, up, down Operand) *tensor.Tensor

type AttentionConfig struct {
    QHeads, KVHeads, HeadDim int
    RoPEBase                 string // declared f32 scalar name
    ScaleName                string // declared f32 scalar, 1/sqrt(headDim)
    BaseName                 string // declared u32 scalar, the prefill's first position
    QKNorm                   bool
}

// posQ and posK are separate because under GQA the q and k row counts differ
// (T*QHeads against T*KVHeads), so one positions tensor cannot serve both.
func Attention(g *Graph, x *tensor.Tensor, w AttentionWeights,
    k, v *tensor.State, posQ, posK, lengths *tensor.Tensor,
    cfg AttentionConfig) *tensor.Tensor
```

Each block below is stated with the accel operators it lowers to. That is what
makes it reviewable, and what makes the dtype dance from
[000 §4](000-decisions.md) explicit rather than buried.

### 2.1 `Linear` — a projection

$$y = x W, \quad x \in \mathbb{R}^{M \times K},\ W \in \mathbb{R}^{K \times N}$$

| step | operator | in | out |
| --- | --- | --- | --- |
| multiply | `tensor.MatMul` or `tensor.QuantMatMul` | `[M,K]` **f32** × `[K,N]` f16 or int8 | `[M,N]` f32 |

f32 in, f32 out, **and no cast**: accel accepts f32 activations against narrow
weights directly ([C8](010-conformance.md) closed 2026-08-24). The weight stays
f16 or int8 for memory; the activation stays f32 for accuracy; the accumulator
was always f32. `MatMul` selects the
matrix-vector kernel at $M = 1$, which is every decode step, and
`Plan.Selections()` reports which and why. **`QuantMatMul` selects one too**, as
of [C15](010-conformance.md) — so the int8 path, which is what `auto` picks for
a large model, is no longer the one without a decode specialisation.

Qwen3 has no biases on its projections, so `tensor.Linear`'s fused epilogue is
unused here. It stays specified because [000 §4](000-decisions.md)'s table lists
it and because the composed `MatMul`-then-`Add` remains the reference a fused
form is checked against ([004-D5](#decision-record)).

### 2.2 `RMSNorm`

$$y_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^{n} x_j^2 + \varepsilon}} \cdot g_i$$

One `tensor.RMSNorm`, f32 throughout. It reduces over the **last axis** and
takes a gain of `[width]`, one value per feature — which is what makes §2.4's
per-head QK-norm a reshape rather than a different operator.

### 2.3 `SwiGLUMLP`

$$\text{MLP}(x) = \big(\text{SiLU}(x W_\text{gate}) \odot x W_\text{up}\big) W_\text{down}$$

| step | operator | in | out |
| --- | --- | --- | --- |
| gate | `Linear` | `[M,d]` | `[M,f]` |
| up | `Linear` | `[M,d]` | `[M,f]` |
| activate | `tensor.SwiGLU` | two `[M,f]` | `[M,f]` |
| down | `Linear` | `[M,f]` | `[M,d]` |

`tensor.SwiGLU` is a fused kernel. The composed `SiLU`-then-`Mul` form stays as
the reference; a fusion bug must be a test failure, not a quality loss nobody
can see.

### 2.4 Attention — GQA, QK-norm, RoPE

Qwen3 differs from Llama in one place that matters: it **normalises Q and K per
head, before RoPE**.

$$q_h = \text{RMSNorm}(x W_Q)_h,\qquad k_h = \text{RMSNorm}(x W_K)_h$$

over the head dimension $d_h$, with a learned gain of $d_h$ values shared across
heads. Getting the order wrong — normalising after RoPE, or over the full
$H\cdot d_h$ row instead of per head — gives a model that produces plausible
tokens and loses coherence after a few sentences. §7 has the test that fails if
the norm moves.

**Grouped-query attention**: $H_q$ query heads over $H_{kv}$ key/value heads,
$H_q / H_{kv}$ queries sharing each cache entry. accel derives the grouping from
the shapes and refuses a non-integer ratio.

$$\text{Attn}(q, K, V) = \text{softmax}\!\left(\frac{qK^\top}{\sqrt{d_h}}\right)V$$

with $1/\sqrt{d_h}$ bound as `AttentionOptions.ScaleName` — a model constant, so
a scalar, and one 043 explicitly leaves alone.

### 2.5 RoPE

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1}\end{pmatrix} =
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1}\end{pmatrix},\qquad
\theta_i = \text{base}^{-2i/d_h}$$

with $m$ the token's absolute position.

### 2.5.1 Which pairs rotate, which nothing in this tree used to say

**The formula above is ambiguous, and the ambiguity is a silent correctness
bug.** $\big(x_{2i}, x_{2i+1}\big)$ names *interleaved* pairs. There is a second
convention, and Qwen3 uses it.

| convention | pairs | used by |
| --- | --- | --- |
| **interleaved** (GPT-J) | $(x_0,x_1), (x_2,x_3), \dots$ | **accel** |
| **half-split** (NeoX) | $(x_0, x_{d_h/2}), (x_1, x_{d_h/2+1}), \dots$ | **Qwen3**, and every HF Llama-family checkpoint |

accel is interleaved, verified in the kernel — `internal/testkernels/elementwise.go`:

```go
lo := r*p.Width + 2*k
hi := lo + 1
```

and `tensor.RoPE` has **no style option**. Qwen3 is half-split: vLLM builds it
with `is_neox_style=True`, and ollama calls MLX's `RoPEWithBase(..., false, ...)`,
whose `traditional=false` is the same thing.

**Handing HF `q_proj` and `k_proj` output straight to accel's `RoPE` rotates the
wrong channel pairs.** Nothing refuses it. Every shape checks. The model produces
fluent text with degraded long-range coherence — the same failure signature as
the batched-RoPE bug in accel 043, arrived at from the other direction.

### 2.5.2 The fix is a load-time permutation, and its order is forced

The two conventions are related by a permutation of the head's channels, so
tgo pre-permutes the projection's **output** channels at load and accel's
interleaved kernel then computes the half-split rotation:

$$y[2i] = x[i], \qquad y[2i+1] = x[i + d_h/2], \qquad 0 \le i < d_h/2$$

Applied per head, to `q_proj` and `k_proj`, and **identically to the `q_norm`
and `k_norm` gains**, because QK-norm ([§2.4](#24-attention--gqa-qk-norm-rope))
is applied per channel before RoPE and its gain vector must follow its channels.

**Two ordering constraints, both forced:**

1. **After the transpose, before quantization.** `quant.Int8Quantize` blocks over
   the *flattened* matrix in runs of `quant.Int8Block = 32`, so permuting after
   quantizing would scatter each weight away from the scale that was computed
   for it. [001-D5](001-weights.md)'s "measured post-transpose" becomes
   post-transpose-**and-permute**.
2. **It is host-side and load-time**, so it costs nothing at run time and stays
   inside [000 D1](000-decisions.md) — it is a layout decision about bytes tgo
   owns, not a kernel.

`rotaryDim` is $d_h$: Qwen3 rotates the whole head.

> This is the single most valuable thing the design review found, and it was
> found by reading accel's kernel body against a reference implementation rather
> than by reading either one alone. It is [016 §10](016-prefix-cache.md)'s lesson
> again: the specs agreed with themselves and disagreed with the arithmetic.

`tensor.RoPE(b, x, rotaryDim, baseName, positions *Tensor)` takes the base as a
scalar — a model constant every row shares — and the **position as a u32 tensor,
one entry per row**. It refuses a positions tensor whose length is not the row
count. Landed upstream and verified by probe.

Since `x` is reshaped to `[rows, d_h]` with `rows = T·H`, the positions tensor
repeats each token's position $H$ times:

$$\text{positions} = [\,p_0^{\times H},\ p_1^{\times H},\ \dots,\ p_{T-1}^{\times H}\,]$$

> This signature is one day old. It took a scalar `Offset` and computed
> `row + Offset`, which is exactly right for one sequence and silently wrong for
> a batch: the row index is the *slot*, so only one member rotates at its own
> cache length. tgo filed
> [accel#2](https://github.com/golang-design/accel/issues/2); accel
> [043](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md)
> generalised it into one rule — *a scalar is a value every row shares; a value
> that differs per row is a tensor* — and changed the operator.
>
> The consequence for tgo is that **there is no batched RoPE to write later**.
> The one-row tensor a single sequence binds is the same mechanism, which is
> 043 §3's orthogonality test.

## 3. The forward pass, with every shape

Symbols: $T$ tokens this step, $d$ hidden size, $H$ query heads, $H_{kv}$
key/value heads, $d_h$ head dim, $f$ intermediate size, $V$ vocabulary, $L$
layers, $C$ cache capacity.

```
h = Embed(ids)
for ℓ in 0..L-1:
    h = h + Attention(RMSNorm(h))
    h = h + MLP(RMSNorm(h))
logits = LMHead(RMSNorm(h[-1]))
```

Pre-norm, with a residual around each sub-block. Expanded, one row per recorded
node:

**Ports and scalars the graph declares**, which an earlier draft under-counted:

| kind | name | shape | note |
| --- | --- | --- | --- |
| `Input` | `ids` | `[T]` u32 | the tokens |
| `Input` | `posq` | `[T·H]` u32 | RoPE positions for q, each token repeated $H$ times |
| `Input` | `posk` | `[T·H_kv]` u32 | **a separate tensor**: under GQA $H \ne H_{kv}$, so one positions tensor cannot serve both |
| `Input` | `slots` | `[T]` u32 | scatter destinations |
| `Input` | `lengths` | `[1]` u32 | `AttentionOptions.Lengths` |
| `Scalar` | `rope_base` | f32 | $10^6$ for Qwen3 |
| `Scalar` | `scale` | f32 | $1/\sqrt{d_h}$ |
| `Scalar` | `base` | u32 | the prefill's first position |

> `posq` and `posk` being separate is why `nn.Attention` cannot take one
> `positions` argument. [005 §1.1](005-kv-cache.md) already had this right and
> §2 of this spec did not.

| # | node | operator | inputs | output |
| --- | --- | --- | --- | --- |
| 1 | ids | `Input` u32 | — | `[T]` |
| 3 | h | `GatherRows` / `QuantGatherRows` | table `[V,d]`, ids `[T]` | `[T,d]` f32 |
| | | **per layer ℓ** | | |
| 4 | n | `RMSNorm` | `[T,d]`, gain `[d]` | `[T,d]` |
| 6 | q | `MatMul` | `[T,d]` f32 × `[d, H·d_h]` f16 | `[T, H·d_h]` f32 |
| 7 | k | `MatMul` | `[T,d]` × `[d, H_kv·d_h]` | `[T, H_kv·d_h]` f32 |
| 8 | v | `MatMul` | `[T,d]` × `[d, H_kv·d_h]` | `[T, H_kv·d_h]` f32 |
| 9 | q | `Reshape` | `[T, H·d_h]` | `[T·H, d_h]` |
| 10 | q | `RMSNorm` | `[T·H, d_h]`, gain `[d_h]` | `[T·H, d_h]` |
| 11 | k | `Reshape` → `RMSNorm` | `[T·H_kv, d_h]`, gain `[d_h]` | `[T·H_kv, d_h]` |
| 12 | q | `RoPE` | `[T·H, d_h]`, `posq` `[T·H]` | `[T·H, d_h]` |
| 13 | k | `RoPE` | `[T·H_kv, d_h]`, `posk` `[T·H_kv]` | `[T·H_kv, d_h]` |
| 14 | K_ℓ | `ScatterRows` | rows `[T, H_kv·d_h]`, ids `[T]` | `State` `[C, H_kv, d_h]` |
| 15 | V_ℓ | `ScatterRows` | rows `[T, H_kv·d_h]`, ids `[T]` | `State` `[C, H_kv, d_h]` |
| 16 | a | `Attention` | q `[T,H,d_h]`, K_ℓ, V_ℓ | `[T, H, d_h]` f32 |
| 17 | a | `Reshape` | `[T,H,d_h]` | `[T, H·d_h]` f32 |
| 18 | o | `MatMul` | `[T, H·d_h]` × `[H·d_h, d]` | `[T,d]` f32 |
| 19 | h | `Add` | `[T,d]`, `[T,d]` | `[T,d]` |
| 20 | n2 | `RMSNorm` | `[T,d]`, gain `[d]` | `[T,d]` f32 |
| 21 | g | `MatMul` | `[T,d]` × `[d,f]` | `[T,f]` f32 |
| 22 | u | `MatMul` | `[T,d]` × `[d,f]` | `[T,f]` f32 |
| 23 | s | `SwiGLU` | two `[T,f]` | `[T,f]` f32 |
| 24 | dn | `MatMul` | `[T,f]` × `[f,d]` | `[T,d]` f32 |
| 25 | h | `Add` | `[T,d]`, `[T,d]` | `[T,d]` |
| | | **after the loop** | | |
| 26 | last | `Slice` → `Contiguous` | `[T,d]`, axis 0, `[T-1,T)` | `[1,d]` |
| 27 | n | `RMSNorm` | `[1,d]`, gain `[d]` | `[1,d]` f32 |
| 28 | logits | `MatMul` | `[1,d]` × `[d,V]` | `[1,V]` f32 |
| 29 | | `Output("logits")` | | |

Roughly $21L$ nodes. Measured on the real Qwen3-4B graph — 36 layers,
$V=151936$, a 4096-position cache — **760 kernel selections** for prefill and
759 for decode, at both f16 and int8 weights. **Four** of the
per-layer nodes are `Cast` — rows 5, 17, 20 and 23 — which is
[010 C8](010-conformance.md): **144 dispatches per forward pass** that exist only
to satisfy a dtype check.

> An earlier draft said seven casts and 252 dispatches. It was wrong three ways:
> the enumeration summed to six, "the two inside `Linear`'s quantized form" had
> no referent — `QuantMatMul` *refuses* non-f16 activations rather than casting —
> and a probe of this graph counts 25 dispatches and 4 casts per layer. The cast
> is shared across the projections that consume the same normalized activation,
> which is why it is four and not seven. `nn.Linear` therefore must **not** cast
> per call; §2.1's description is the shared form.

> **The casts are gone.** An earlier draft of this table had four `Cast` nodes
> per layer — rows 5, 17, 20 and 23 — because `MatMul` required its two operands
> to share a dtype, and a transformer's activations are f32 while its weights are
> f16 or int8. tgo reported the cost; accel relaxed the rule
> ([C8](010-conformance.md)). Measured on the real graph: **1013 kernel
> selections became 760.**
>
> Rows 5, 17, 20 and 23 are struck from the table above. They are recorded here
> rather than deleted because the arithmetic that produced them is what closed
> the gap.

## 3.1 Decode is the same graph at $T = 1$

Every shape above with a leading $T$ becomes 1. Two things change beyond that:

- `MatMul` at $M = 1$ selects the matrix-vector kernel rather than the tiled
  GEMM, reported by `Plan.Selections()`.
- `Attention` takes rank-2 `q` (`[H, d_h]`) rather than rank-3, which is how
  accel distinguishes decode from prefill — *"a rank is not a hint, it is the
  shape of the computation"*. So the builder passes `[H, d_h]` at $T=1$ and
  `[T, H, d_h]` otherwise, and this is the one place the two graphs genuinely
  diverge rather than differing by a dimension.

### 3.2 Row 26 is the single largest avoidable cost

**Only the last position's logits are needed.** Running the LM head over all $T$
positions costs $T \times V$ f32 values: for a 2000-token prompt at Qwen3's
$V = 151936$ that is 304M values, **1.2 GB** — larger than the int8 weights of
the model producing it.

Slicing to the last row *before* the head makes it $V$ values, 608 KB. It is a
one-line mistake to make and it does not fail, it just allocates. §7 has the
test.

The `Contiguous` after the `Slice` is deliberate: `Slice` returns a view with a
non-zero offset, and accel refuses a strided operand into `MatMul` rather than
copying behind the caller's back. Calling `Contiguous` is how tgo says the copy
is worth it — and at `[1,d]` it is 10 KB, which it plainly is.

## 4. The weight map

[001 §4](001-weights.md) requires each model to declare which tensors transpose.
Qwen3, with the Hugging Face names on the left:

| checkpoint tensor | shape in file | port | transpose | **permute** | notes |
| --- | --- | --- | --- | --- | --- |
| `model.embed_tokens.weight` | `[V, d]` | `embed` | **no** | no | rows are gathered; `[V,d]` is already row-per-token. **f16 or int8** — accel gained an f16 `GatherRows` ([C14](010-conformance.md)), so the embedding no longer has to be f32 |
| `model.layers.ℓ.input_layernorm.weight` | `[d]` | `ℓ.attn_norm` | no | no | gain |
| `model.layers.ℓ.self_attn.q_proj.weight` | `[H·d_h, d]` | `ℓ.wq` | **yes** → `[d, H·d_h]` | **yes** | [§2.5.2](#252-the-fix-is-a-load-time-permutation-and-its-order-is-forced) |
| `model.layers.ℓ.self_attn.k_proj.weight` | `[H_kv·d_h, d]` | `ℓ.wk` | **yes** | **yes** | as above |
| `model.layers.ℓ.self_attn.v_proj.weight` | `[H_kv·d_h, d]` | `ℓ.wv` | **yes** | no | V is not rotated |
| `model.layers.ℓ.self_attn.o_proj.weight` | `[d, H·d_h]` | `ℓ.wo` | **yes** → `[H·d_h, d]` | no | reads attention output, which is unrotated |
| `model.layers.ℓ.self_attn.q_norm.weight` | `[d_h]` | `ℓ.qnorm` | no | **yes** | **Qwen3-specific**; the gain follows its channels |
| `model.layers.ℓ.self_attn.k_norm.weight` | `[d_h]` | `ℓ.knorm` | no | **yes** | **Qwen3-specific**; as above |
| `model.layers.ℓ.post_attention_layernorm.weight` | `[d]` | `ℓ.ffn_norm` | no | no | gain |
| `model.layers.ℓ.mlp.gate_proj.weight` | `[f, d]` | `ℓ.wgate` | **yes** | no | |
| `model.layers.ℓ.mlp.up_proj.weight` | `[f, d]` | `ℓ.wup` | **yes** | no | |
| `model.layers.ℓ.mlp.down_proj.weight` | `[d, f]` | `ℓ.wdown` | **yes** | no | |
| `model.norm.weight` | `[d]` | `final_norm` | no | no | gain |
| `lm_head.weight` | `[V, d]` | `lm_head` | **yes** → `[d, V]` | no | **absent when tied** |

**Tied embeddings.** When `tie_word_embeddings` is true — which it is for the
small Qwen3 sizes — there is no `lm_head.weight`. The LM head is the embedding
table transposed. Two consequences:

- the loader uploads **two** planes from one file tensor: `[V,d]` untransposed
  for `GatherRows`, and `[d,V]` transposed for the `MatMul`. They are different
  layouts, so they cannot share a buffer, and the "tied" saving is in the file,
  not on the device. Both may now be **f16** ([C14](010-conformance.md)), so the
  tie costs 2× the file tensor rather than the 3× it would have when
  `GatherRows` demanded f32;
- a checkpoint that is tied *and* ships an `lm_head.weight` needs the two planes
  compared, and **redundancy is not a contradiction**:

  | planes | verdict |
  | --- | --- |
  | byte-identical | **accept.** The exporter wrote the head out as well; the config and the weights agree. Load either |
  | differ | **refuse.** The config says the head is the embedding and the weights say it is not, and picking one is a guess about which the model was trained with |
  | no `lm_head.weight` | accept; the head is the embedding transposed |

  > **This rule was wrong when first written**, and the target checkpoint is what
  > proved it. An earlier draft refused *any* tied checkpoint that also shipped a
  > head. Qwen3-0.6B does exactly that, and its two planes hash identically
  > (`8f29acf5…`, verified over both 311 MB ranges) — so the rule refused the
  > model tgo exists to run. Shapes alone cannot tell the two cases apart, which
  > is why the comparison is on bytes and belongs to the loader rather than to
  > the header check.

> That first point corrects a natural assumption. "Tied" does not mean one
> device buffer here — it means one *source* tensor. Sharing one buffer would
> need `MatMul` to read a transposed view, which is [010 C9](010-conformance.md).

**Missing or extra tensors are refused, by name.** A checkpoint with a tensor
the map does not mention is not silently ignored: it means the architecture
string was right and the weights are not the ones the map was written for.

## 5. The config

Read from `config.json`. **tgo reads these values from the file; nothing depends
on a number written in this spec.** The symbolic shapes above are what the code
uses.

| field | symbol | required | notes |
| --- | --- | --- | --- |
| `architectures[0]` | — | yes | the registry key |
| `hidden_size` | $d$ | yes | |
| `num_hidden_layers` | $L$ | yes | |
| `num_attention_heads` | $H$ | yes | |
| `num_key_value_heads` | $H_{kv}$ | no | defaults to $H$; must divide it |
| `head_dim` | $d_h$ | no | defaults to $d/H$; **Qwen3 sets it explicitly and it is not always $d/H$** |
| `intermediate_size` | $f$ | yes | |
| `vocab_size` | $V$ | yes | must match the embedding table's rows |
| `rms_norm_eps` | $\varepsilon$ | yes | |
| `rope_theta` | base | yes | $10^6$ for Qwen3, against Llama's $10^4$ |
| `tie_word_embeddings` | — | no | defaults false |
| `max_position_embeddings` | — | yes | **advisory only**; capacity is a session parameter ([005-D2](005-kv-cache.md)) |

The `head_dim` row is the one that bites. Assuming $d_h = d/H$ is correct for
Llama and wrong for several Qwen3 sizes, and the failure is a shape mismatch at
graph-build time rather than bad output — which is the good case, and only
because accel checks shapes.

## 6. The registry

```go
package model

type Builder interface {
    Config() any                                   // the parsed config.json
    Weights() []WeightSpec                         // §4's map
    Forward(g *nn.Graph, in Inputs) *tensor.Tensor // §3
    Template() chat.Renderer                       // 003
}

func Register(architecture string, new func(json.RawMessage) (Builder, error))
func Open(dir string) (Builder, error)             // reads config.json, resolves
```

`config.json`'s `architectures[0]` selects. An unknown architecture is refused
**with the list of known ones** — never guessed at, never fallen back to a
"generic Llama" path, because a model that runs with the wrong architecture
produces fluent text and nobody finds out.

Adding a model is one file and one `init`. Nothing else changes.

## 7. Refusals

A config the builder cannot honour is refused at build time, with the field
named:

| condition | why refusing beats approximating |
| --- | --- |
| `rope_scaling` present and unsupported | a wrong scaling is fine for 4000 tokens and then is not |
| sliding-window attention configured | the graph has no window; output would be silently wrong beyond it |
| `head_dim` odd, or not positive | accel's `RoPE` refuses an odd `rotaryDim`; it rotates pairs |
| $H \bmod H_{kv} \ne 0$ | accel refuses it too; catching it here names the config field |
| `vocab_size` ≠ the embedding table's rows | the config and the weights are from different models |
| capacity $C > 128$ | [010 C11](010-conformance.md); refuse with accel's message and the issue |

## 8. Tests

Every one runs on a **synthetic 2-layer, $d=64$, $V=128$** config with seeded
weights, on both backends, against the [010 §5](010-conformance.md) oracle:

| test | what it catches |
| --- | --- |
| each block against the f64 oracle within a derived tolerance | any numerics error |
| **QK-norm placement**: fails if the norm moves after RoPE | the Qwen3-specific ordering |
| **rotary pairing**: q after RoPE matches an HF reference vector | [§2.5.1](#251-which-pairs-rotate-which-nothing-in-this-tree-used-to-say) — the permutation, which nothing else catches |
| the permutation runs before quantization, not after | [§2.5.2](#252-the-fix-is-a-load-time-permutation-and-its-order-is-forced)'s ordering constraint |
| **QK-norm axis**: fails if it normalises over `H·d_h` instead of `d_h` | the reshape in row 9–11 |
| RoPE positions repeat per head (row 12's formula) | a positions tensor built per token instead of per row |
| prefill transient memory does not scale with $V \times T$ | §3.2's slice |
| a tied head uploads two planes and refuses a checkpoint with both | §4's tie handling |
| decode at $T=1$ equals prefill's last row | §3.1's rank-2/rank-3 split |
| every §7 refusal | each names its config field |

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 004-D1 | `nn` is stateless; weights arrive as named ports | blocks that own their weights | blocks are testable with no loader; the registry owns naming |
| 004-D2 | registry keyed on `architectures[0]`, unknown is refused with the known list | a filename heuristic; a generic fallback path | a wrong model is refused, not guessed — a guess produces fluent wrong text. **Note 2026-08-24:** "additive" holds for the *graph* and not for the *cache* — a hybrid model's recurrent or sliding-window state cannot be sliced at an arbitrary position, which [016 §10.1](016-prefix-cache.md) records |
| 004-D3 | prefill and decode are separate plans, bucketed on $T$ | one dynamic-shape plan | bounded recompiles; §3.1 shows they differ only in `q`'s rank |
| 004-D4 | slice to the last row before the LM head | full-sequence logits | 1.2 GB → 608 KB for a 2000-token prompt |
| 004-D5 | fused kernels keep their composed form as the reference | fused only | a fusion bug is a test failure, not a silent quality loss |
| 004-D6 | `nn.Operand` carries f16-or-quantized, resolved once | branch on precision at each projection | one `Linear` call site; precision is a load-time decision, not a graph one |
| 004-D7 | a tied head uploads two planes | share one buffer | the two layouts differ; sharing needs [C9](010-conformance.md), which accel correctly refuses |
| 004-D10 | tied **and** shipped is refused only when the planes differ | refuse on the config/tensor mismatch alone | the first rule refused Qwen3-0.6B, which is tied, ships a head, and has identical planes. A header check cannot decide it, so the comparison is the loader's ([§4](#4-the-weight-map)) |
| 004-D8 | shapes in this spec are symbolic; values come from `config.json` | hardcode a size's constants | a spec cannot go stale against a checkpoint it does not contain |
| 004-D9 | permute q/k projection output channels and the QK-norm gains at load, after transpose and before quantization | ask accel for a NeoX RoPE; permute on device each step | accel is interleaved and Qwen3 is half-split; nothing refuses the mismatch. A load-time byte layout is tgo's to own and costs nothing per step ([§2.5.2](#252-the-fix-is-a-load-time-permutation-and-its-order-is-forced)) |
