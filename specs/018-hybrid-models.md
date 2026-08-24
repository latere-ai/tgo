---
title: "Hybrid attention: what Qwen3.8-27B is, and the operator it needs"
status: blocked
layer: graph
depends_on:
  - 000-decisions.md
  - 004-model-graph.md
  - 005-kv-cache.md
blocked_on:
  - "accel: no linear-attention operator, `gated delta` scan and per-sequence state (accel#17)"
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

## 2. The layer accel does not have

A gated delta layer carries a **matrix-valued recurrent state** per head,
`[key_head_dim, value_head_dim]` = 128×128 here, and updates it per token:

$$S_t = S_{t-1}\big(\alpha_t I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,
\qquad o_t = S_t\, q_t$$

with $\alpha_t$ a forget gate and $\beta_t$ a write strength, both produced from
the input. There is also a short depthwise causal convolution over the input
(`linear_conv_kernel_dim: 4`) before the recurrence.

**Why it is a kernel and not a composition of what exists:**

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
| `nn.LinearAttention` | over accel's scan operator, with the depthwise convolution |
| a per-layer cache kind | [005](005-kv-cache.md) gains a state per layer *type*, not one shape for all |
| snapshot/restore for recurrent state | [016](016-prefix-cache.md) gains ollama's shape for the layers that need it |
| the `qwen3_5` registry entry | config parsing, the weight map, the layer-type schedule |
| image-token tolerance | the text path must not break on a multimodal tokenizer |

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 018-D1 | Qwen3.8-27B is out of v0, and said so publicly | list it as a target and hope | 48 of 64 layers are inexpressible; implying otherwise is the overclaiming this tree exists to avoid |
| 018-D2 | file the operator as a question, not a proposal | propose a kernel shape | tgo does not know accel's kernel constraints; #17 asks whether it is in scope first |
| 018-D3 | a hybrid cache is per layer *type*, not one shape | force the recurrent state into `State`'s per-position model | a recurrent state has one value per sequence and no positions to index |
| 018-D4 | record that ollama's snapshot design is forced here | keep treating it as a workload difference | [016 §10.1](016-prefix-cache.md) read it as a choice about concurrency; for a recurrent layer it is the only option |
