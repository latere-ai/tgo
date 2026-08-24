---
title: "The KV cache: what a contiguous cache costs, and why tgo has no other kind"
status: drafted
layer: engine
depends_on:
  - 000-decisions.md
  - 004-model-graph.md
---

# The KV cache

## 1. What accel gives, exactly

`tensor.NewState(b, StateDesc{Name, DType, Shape})` declares caller-owned
mutable storage that the planner never aliases. `LayerState(b, s, layer)` slices
one layer's window out of a single state. `ScatterRows(b, s, rows, ids)` writes
rows and returns the **next version**; reading an older version is refused,
which is what makes the write-then-read ordering an ordinary DAG edge rather
than a rule the planner is told.

`tensor.Attention(b, q, k, v, opts)` reads two states. It requires **f32**.

That is the whole surface. There is no page table, no block pool, and no f16
cache.

## 2. The shape, and the number

One state per role, sliced per layer:

$$\text{Shape} = [L \cdot C,\; H_{kv},\; d_h]$$

with $L$ layers, context capacity $C$, $H_{kv}$ key/value heads, head dim $d_h$.
Two of them, K and V. In f32:

$$M_\text{kv} = 2 \cdot L \cdot C \cdot H_{kv} \cdot d_h \cdot 4 \ \text{bytes}$$

For Qwen3-4B — $L=36$, $H_{kv}=8$, $d_h=128$:

| context $C$ | f32 (what accel takes) | f16 (what it would cost) |
| --- | --- | --- |
| 2048 | 0.60 GB | 0.30 GB |
| 4096 | 1.21 GB | 0.60 GB |
| 8192 | 2.42 GB | 1.21 GB |
| 32768 | 9.66 GB | 4.83 GB |

At 32k context the cache for **one sequence** is larger than the int8 weights.
This is the number that decides how much of a model server tgo can be, and it is
entirely a consequence of two accel constraints:

- **f32 only** doubles it. accel `Attention` refuses f16 states.
- **contiguous only** means $C$ is reserved for the longest sequence the server
  will ever accept, whether or not anything is that long. Ten concurrent 200-token
  chats on a 32k-capable server pay for 320k positions and use 2k.

Paging fixes the second; an f16 state fixes the first. Both are
[010](010-conformance.md) entries against accel, with the arithmetic above as
the argument.

## 3. What tgo does with what it has

One state pair for the whole model, `LayerState` per layer. A `Session` owns a
cache and a length. Prefill scatters $T$ rows at positions $0..T-1$; each decode
step scatters one row at position $t$ and increments.

`C` is a **session parameter**, not a model parameter, and it defaults to
something small — 4096 — with the arithmetic above printed when it is raised.
A user who asks for 32k should be told what it costs before it is allocated,
not after it fails.

## 4. Position and the offset scalar

`ScatterRows` takes ids as device data, so where a row lands is a runtime value.
`RoPE` takes `Offset` as a scalar and computes `pos = row + Offset`. For a
single sequence these agree: prefill row $r$ is position $r$ with offset 0;
decode's single row is position $t$ with offset $t$.

They stop agreeing the moment two sequences share a dispatch. That is
[008](008-scheduler.md) gap 2, and it is why §3 is not a placeholder
implementation to be swapped out — it is the only correct one available.

## 5. Prefix reuse is deferred, and it is not free here

sglang's RadixAttention shares the KV of a common prefix across requests. Over a
contiguous per-session cache the sharing has nowhere to live: two sessions hold
two allocations, and making them share means copying, which is most of the cost
the sharing was meant to avoid.

So prefix reuse is specified in [008 §6](008-scheduler.md) and **depends on
paging**, not merely on scheduler policy. Recording that dependency now is the
point; it is otherwise easy to plan prefix caching as a tgo feature and discover
late that it is an accel one.

## 6. Tests

- A decode step attends to what prefill wrote: prefill $T$ tokens, decode one,
  and assert the attention output equals a host reference over the same $T+1$
  rows.
- Reading a stale state version is refused (this is accel's guarantee; the test
  is here because tgo depends on it).
- `LayerState` windows do not overlap: writing layer $i$ leaves layer $j$ bytes
  unchanged.
- The §2 arithmetic is a function, and the test checks it against the table.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 005-D1 | one state pair for the model, `LayerState` per layer | one state per layer | one allocation, one binding; layer windows must be proven disjoint |
| 005-D2 | context capacity is a session parameter defaulting to 4096 | model's `max_position_embeddings` | a 32k default would reserve 9.7 GB before the first token |
| 005-D3 | print the cache cost when capacity is raised | allocate and fail | the user learns the number before the allocation, not after |
| 005-D4 | no paging, no f16 cache; both filed upstream | a private page table in tgo | forbidden by [000 D1](000-decisions.md); the arithmetic is the filing |
