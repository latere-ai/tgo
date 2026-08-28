---
title: "Sequencing: the order the work is done in, and what actually landed"
status: record
layer: all
depends_on:
  - 000-decisions.md
---

# Sequencing

The other specs say what. This one says **when**, and afterwards, **what really
happened**. It is the only file here that is edited after its subject ships.

## 1. The rule that fixes the order

Nothing that can be wrong silently is built on top of something unverified. So
the order is: the pieces with a checkable ground truth first (a tokenizer has
fixed vectors; a template has a golden string), then numerics against an oracle,
then the loop, then the surfaces.

Writing the server early is the tempting mistake. A `framework` request invites
an API surface, and an API over logits that are subtly wrong is a working demo
of a broken model.

## 2. Readiness: build now, and what the targets actually are

**Verdict, 2026-08-24: ready to build the dense path.** The whole Qwen3-4B graph
compiles against accel, nothing in M1–M11 is blocked upstream, and further spec
polish would be writing against a design that has stopped moving.

**But one of the two named target models is not a dense transformer**, and that
was discovered by reading its config rather than assuming.

### 2.1 The two targets, corrected

| named | actual | status |
| --- | --- | --- |
| "Qwen3 3B" | **Qwen3-4B**. There is no 3B; the dense line is 0.6B, 1.7B, 4B, 8B, 14B, 32B | **buildable now** |
| "Qwen3.8 27B" | **Qwen3.8-27B**, Apache 2.0, released August 2026 | **one thing left, and it is the graph.** Both upstream blockers closed on 2026-08-27 and one of the two tgo halves shipped the same day: int4 storage, so the footprint is 13.4 GiB. What remains is [018](018-hybrid-models.md)'s hybrid graph — 48 gated-delta layers beside 16 softmax ones |

### 2.1.1 What "well tested" means for each, and what it cannot mean

The two targets are at different distances, and saying so precisely matters more
than restating the goal.

| | Qwen3 dense | Qwen3.8-27B |
| --- | --- | --- |
| architecture expressible | **yes** — the graph compiles at 36 layers and $V=151936$ | **the layer, yes; the graph, not yet.** `nn.LinearAttention` and `nn.DepthwiseCausalConv` are tgo's own blocks over accel's scan, each verified against a float64 reference (Wave 11). Nothing in `model/` reads `layer_types` or `full_attention_interval`, so there is no model-level graph to run them from ([024](024-qwen3-5-architecture.md)) |
| numerics verified | **yes** — against the f64 oracle, per block and end to end | not yet: no graph to check |
| real weights loaded | **yes** — Qwen3-0.6B, 311 tensors, 1.40 GiB at f16 | not yet, and no longer for want of a stored form: `weights.Precision` names `Int4` since Wave 10, so a 27B resolves to **13.4 GiB** |
| generation verified | **Wave 4** | not yet |
| blocked by | nothing | **nothing upstream, and nothing unowned.** [018](018-hybrid-models.md)'s remaining rows are [023](023-cache-kinds.md), [024](024-qwen3-5-architecture.md), [025](025-recurrent-snapshot.md) and [026](026-image-tokens.md), all tgo's |

**Two honest qualifications on the dense side.** The checkpoint on hand is
**0.6B**, not 4B — same architecture, same code path, one twelfth the
parameters. What that proves is the loader, the graph and the numerics; what it
does not prove is behaviour at 4B's memory footprint. And the 4B graph is
verified to *compile and to agree with the oracle*, not to have generated text,
until Wave 4 lands.

**On the 27B, the position is now "not yet" rather than "not by tgo".** It read
the other way while the operator was only scoped, and
[accel#17](https://github.com/golang-design/accel/issues/17) closed: accel
scheduled the kernel and shipped it, which [018-D2](018-hybrid-models.md)'s
outcome note records. [000 D1](000-decisions.md) still forbids tgo from writing
a kernel, and it no longer decides anything here. Every remaining piece is
tgo's, and each has a spec — [023](023-cache-kinds.md) for the per-layer cache
kinds, [024](024-qwen3-5-architecture.md) for the `qwen3_5` registry entry,
[025](025-recurrent-snapshot.md) for snapshot and restore, and
[026](026-image-tokens.md) for image-token tolerance.

Qwen3-4B: `Qwen3ForCausalLM`, $d=2560$, $L=36$, 32 query heads over 8 KV heads,
$d_h=128$, $f=9728$, $V=151936$, `rope_theta` $10^6$, tied embeddings. This is
exactly what [004](004-model-graph.md) specifies.

Qwen3.8-27B is a **different architecture family**:

```json
"architectures": ["Qwen3_5ForConditionalGeneration"],
"layer_types": ["linear_attention", "full_attention"],
"full_attention_interval": 4,
"num_hidden_layers": 64, "hidden_size": 5120, "head_dim": 256,
"num_attention_heads": 24, "num_key_value_heads": 4,
"partial_rotary_factor": 0.25, "attn_output_gate": true,
"linear_num_key_heads": 16, "linear_num_value_heads": 48,
"max_position_embeddings": 262144, "vocab_size": 248320
```

**Three of every four layers are linear attention** — a gated-delta recurrence,
not softmax attention — and it carries vision tokens. Filed as
[accel#17](https://github.com/golang-design/accel/issues/17), which closed:
`tensor.LinearAttention` shipped, and Wave 11 composed it as `nn.LinearAttention`
with the depthwise causal convolution in front of it. The 48 gated-delta layers
are expressible. What does not exist is the model-level graph that schedules
them beside the 16 softmax ones — [018](018-hybrid-models.md) is the design and
[024](024-qwen3-5-architecture.md) is the registry entry that builds it.

Two things about it that *do* work, checked rather than assumed:
`partial_rotary_factor: 0.25` is `RoPE(x, 64, …)` because `rotaryDim` was always
a parameter, and `attn_output_gate` is an elementwise multiply.

> **This is why the answer was "build now" rather than "spec more".** The dense
> path was fully specified and unblocked; the hybrid path waited on a kernel
> nobody had written. Building the dense path is what produced the evidence that
> made the hybrid ask concrete, and the ask was answered: the kernel shipped, so
> the hybrid path is now tgo's own work rather than an upstream decision.

### 2.2 Waves

Work is grouped by what can proceed in parallel. A wave starts when the one
before it is green.

```mermaid
flowchart TB
  subgraph W1["Wave 1 — no device, no dependencies"]
    T["tokenizer 002"]
    S["safetensors 001 §1"]
    C["chat templates 003"]
    B["bench harness 017"]
  end
  subgraph W2["Wave 2 — needs Wave 1"]
    L["weight loader 001"]
    N["nn blocks + oracle 004, 010 §5"]
  end
  subgraph W3["Wave 3 — needs Wave 2"]
    Q["Qwen3 forward pass"]
    K["KV cache + decode loop 005, 007"]
    P["sampling 006"]
  end
  subgraph W4["Wave 4 — needs Wave 3"]
    CLI["CLI"]
    SRV["server 009"]
    PC["prefix cache 016"]
  end
  W1 --> W2 --> W3 --> W4
  W4 --> W5["Wave 5 — real weights, batching 008, vLLM table 010 §3.1"]
```

**Wave 1 is four independent packages with no device and no network**, which is
where [000 D8](000-decisions.md) puts the coverage gate, and it is the wave that
parallelises cleanly. Wave 3 is the one that must be sequential: the forward
pass, the cache and the loop are one dependency chain.

## 2.3 What is missing that no spec covers

The waves above are the *known* work. This section is the other list: what an
inference framework has that this tree has never written down. It exists because
a spec tree makes its own gaps invisible — everything in `specs/` is accounted
for, so the absent things are absent from the accounting too.

**Ranked by what decides whether tgo is usable, not by effort.**

### 1. Four-bit weights — closed, except for the one plane that cannot pack

**Closed upstream on 2026-08-27** ([C21](010-conformance.md)):
`quant.Int4Quantize` packs eight codes per word with an fp16 scale *and zero*
per 128, and `Int4MatMul` and `Int4MatVec` are the prefill and decode shapes.
The arithmetic on the 27B target, with accel's actual representation rather than
the idealised one this section used to quote:

| stored as | bytes/weight | 27B resident | fits |
| --- | ---: | ---: | --- |
| bf16 | 2.0 | 50.3 GiB | a large workstation |
| **int8, what tgo stores** | 1.0625 | **26.7 GiB** | not a 24 GiB card |
| int4, scale + zero per 128 | 0.53125 | **13.4 GiB** | hardware people own |

**Built on 2026-08-27, so this row is no longer a gap.** `weights.Int4` is the
third stored form, `nn.Form` is the signal that names a representation rather
than a dtype, and `auto` narrows to it only where int8 does not fit. A 27B
checkpoint resolves to **13.4 GiB**. Wave 10 is the entry, and `--precision
int4` reaches it from the command line.

One thing does not pack: the embedding table is gathered rather than contracted
against, and accel registers no int4 gather, so it is capped at int8 — declared
per tensor in the loader rather than discovered as a refusal at record time,
because the footprint `tgo info` prints and the load itself are computed by two
different pieces of code ([001 §5.3](001-weights.md)).

The download is the other half and is unchanged: [001](001-weights.md) reads
full-precision safetensors and quantizes at load, so running a 27B still means
fetching **50 GiB** to produce 13.4, when the file the ecosystem publishes is a
13 GiB 4-bit checkpoint. That is §2 below, and it is now much closer than it
was.

### 2. Reading pre-quantized checkpoints — AWQ, GPTQ, GGUF

Distinct from the above: even with 4-bit kernels, tgo still quantizes from full
precision at load. AWQ and GPTQ dominate what is published for open weights, and
both use a group size of **128 with a zero point**.

**That is exactly the shape accel's int4 now has** — `quant.Int4Group` is 128
and `Int4Quantize` returns a scale *and* a zero per group — which it did not
when this section was written against a symmetric per-32 int8. So reading an AWQ
or GPTQ checkpoint is no longer a representation problem. What is left is the
file format and the layout each toolchain writes, which is a reader rather than
a kernel, and neither has a spec yet.

**Owner.** GGUF is [012](012-gguf.md), which is `blocked` for a different
reason: a Q4_K super-block is two levels of scale over eight sub-blocks with a
minimum each, which is a different shape and not a smaller one
([C17](010-conformance.md)), and accel closed
[#15](https://github.com/golang-design/accel/issues/15) as not planned. **AWQ
and GPTQ have no spec on disk.** They are a deferral rather than an omission,
and the trigger is stated so it can be met: write the reader when a checkpoint
tgo is asked to serve is published only in one of those layouts, since the
representation problem is already solved and what is left is a file format.

### 3. `rope_scaling` — decides context length

[004 §7](004-model-graph.md) **refuses** any `rope_scaling` it does not
implement, which is the right refusal and means tgo is capped at a checkpoint's
trained context. Qwen3 reaches its long-context modes through YaRN. Refusing is
correct; not having it is a gap, and it is entirely tgo's rather than accel's —
YaRN is a change to how $\theta_i$ is computed, and [004 §2.5](004-model-graph.md)
already binds the base as a scalar.

**Owner.** [004 §7](004-model-graph.md) owns the refusal. Nothing owns the
implementation, and no spec on disk covers it. Deferred, with the trigger: spec
YaRN when a target checkpoint's usable context is shorter than what a caller
asks for, which is when the refusal starts costing an answer rather than
preventing a wrong one.

### 4. Multi-device — a permanent ceiling on model size

accel opens one device. There is no tensor or pipeline parallelism anywhere in
either project, so **the largest model tgo can ever run is the largest that fits
one accelerator**. That is a legitimate scope decision and it should be a stated
one rather than an omission.

**Owner, recommended.** This is a permanent scope boundary rather than work
somebody will do, so it belongs in [000](000-decisions.md) as a decision with
its rejected alternative — sharding a model across devices, rejected because
tgo would be routing around accel's device model, which [000 D1](000-decisions.md)
forbids — and not as a spec that would never be built. Recorded here as a
recommendation; 000 is edited by whoever takes it.

### 5. Things with no spec and no blocker

Checked against the tree as it stands after the ten specs written on
2026-08-27.

| | why it matters | owner |
| --- | --- | --- |
| **speculative decoding** | the standard 2–3× decode win, and [017 §4.1](017-benchmarks.md) shows decode is 94.62% device — exactly the shape speculation attacks | none. No spec on disk |
| **embedding models** | a large share of what inference frameworks actually serve; `/v1/embeddings` is a 404 and there is no pooling | none. No spec on disk |
| **multimodal input** | [018 §1](018-hybrid-models.md) notes Qwen3.8 carries vision tokens and text-only is a coherent subset | half owned. [026](026-image-tokens.md) owns the tolerance — a multimodal vocabulary must not mis-embed on the text path — and the vision path itself is unspecced |
| **LoRA adapters** | [000](000-decisions.md) states v0's scope and does not include one; note [016 §10.3](016-prefix-cache.md) already records that an adapter must reach the prefix cache key | none. No spec on disk |

### What this section is not

A backlog. Several of these are correct things to *not* do — multi-device may
never be in scope, and refusing an unimplemented `rope_scaling` beats
approximating it. The point is that each should be a decision somebody made
rather than a question nobody asked. Four of the five now are: item 1 shipped,
item 2's GGUF half is [012](012-gguf.md) and its AWQ/GPTQ half is a deferral
with a trigger, item 3 is a deferral with a trigger, and item 4 is a
recommendation to [000](000-decisions.md). Item 5's four rows are the ones still
unanswered, and three of them have no spec at all.

## 3. Milestones

### 3.1 What is gated upstream, and what is not

```mermaid
flowchart LR
  subgraph free["unblocked — 90% of the coverage gate lives here"]
    M1["M1 tokenizer"] --> M2["M2 templates"] --> M3["M3 loader"]
    M3 --> M4["M4 nn + oracle"] --> M5["M5 forward pass"]
  end
  M5 --> M6["M6 decode loop"]
  M6 --> M7["M7 sampling"] --> M8["M8 CLI"] --> M9["M9 server"]
  M9 --> M10["M10 report"] --> M11["M11 real weights"]
  M11 --> M12["M12 batching"] --> M13["M13 vs vLLM"]
```

**M1 through M12 have shipped**, across Waves 1 to 11. M13, the vLLM
comparison, is the only milestone left, and [010 §3.1](010-conformance.md) is
the design nobody owns yet.

Where the work goes now is three groups, and each has a spec:

| | specs |
| --- | --- |
| the batched path the scheduler does not reach | [020](020-device-sampling.md) sampling on the device, [021](021-admission-queue.md) a queue in front of admission, [022](022-batched-serving.md) the server driving a scheduler — [008 §9](008-scheduler.md)'s three, which it has handed on |
| the hybrid graph, which is the last thing between tgo and the 27B | [023](023-cache-kinds.md) a cache per layer kind, [024](024-qwen3-5-architecture.md) the `qwen3_5` registry entry, [025](025-recurrent-snapshot.md) snapshot and restore, [026](026-image-tokens.md) image-token tolerance |
| the measurements M13 rests on | [027](027-batched-benchmarks.md) the throughput curve at batch, [028](028-performance-gate.md) a gate that fails a build which loses throughput |

[029](029-grammar-front-ends.md) is beside all three: the EBNF and regex front
ends over the grammar machine [015](015-structured-output.md) built.

M1–M5 remain where [000 D8](000-decisions.md) puts the device-free packages, so
they carry almost all of the coverage gate.

**M6 through M11 are unblocked.** [C11](010-conformance.md), the 128-position
cache cap that gated everything, closed on 2026-08-24 when accel shipped
[044](https://github.com/golang-design/accel/blob/main/specs/044-unbounded-context.md).
A 4096-position cache is verified. Nothing between here and serving a real model
is waiting on accel.

**Nothing is blocked upstream any more.** Batching was the last one and accel
closed it: a batched decode is expressible and verified, two sequences of
different lengths matching two single runs exactly. [C16](010-conformance.md)
closed with it, so a chunked prefill shares a dispatch with the decodes beside
it — measured at 8 prompt tokens and 2 decodes in one step (Wave 9). What
remains open is narrower — GGUF's super-blocks ([C17](010-conformance.md)) —
and it blocks no milestone.

### 2026-08-28 — Wave 12: the specs say what the code does

No new capability. This wave closed the gap between what shipped and what is
written down, which [§1](#1-the-rule-that-fixes-the-order)'s rule makes a
first-class piece of work rather than tidying: a spec that describes a design
the code no longer has is worse than no spec, because a contributor trusts it.

**Six specs reached `complete`**, each by writing what was already built rather
than by narrowing what was open:

- [003](003-chat-template.md) carried seven rules the renderer followed and the
  spec did not state. §3.3 gives the tools preamble byte for byte, §3.4 the
  Hugging Face `tojson` escaping Go's encoder does not produce, §3.5 the
  keep-thinking rule as a formula over the whole open round with its four
  whitespace trims, and §3.6 the refusal contract — which also became
  003-D9, so the second refuse-versus-warn choice in that spec has an id.
- [015](015-structured-output.md) documented three deliberate narrowings and the
  one behaviour that widens: `G` admits every number RFC 8259 does, so `1e999`
  is admissible JSON a `float64` field cannot hold.
- [016](016-prefix-cache.md) documented `Reserve`, the `Grow`/`Commit`/
  `Publish(written)` split, `Batch` and the hash encoding, and settled §6 — the
  cold-against-warm divergence stays asserted here rather than becoming a
  [010 §3](010-conformance.md) row, because accel could not act on the number.
- [001](001-weights.md) had eight items, and every one was a description of
  shipped code: the `weights` surface, the rotary permutation as a conversion
  in its own right — including that **accel refuses no mismatch**, so a
  permuted weight and an unpermuted one both compile and one produces fluent
  incoherence — the f32 gain path and its bytes missing from `Report.Bytes`,
  §3's NaN rule, seven refusal rows, the arena's four numbers, and 001-D10 for
  where the int8 bound is asserted.
- [005](005-kv-cache.md) documented the shared-pool addressing as addressing:
  the `rows`/`limit` split, which exists because a shared pool makes one
  session's limit a real row inside another conversation's block, so a dropped
  write's sentinel has to be the state's own extent. And 005-D8 decides where
  §7's capacity refusal lives — the device-memory one is the server's because
  the arithmetic needs the session count, the per-request one is the library's
  because it needs the request.
- [016 §8](016-prefix-cache.md)'s one uncovered row is covered:
  `TestPartialHitAttentionAtANonzeroBase` asserts a partial hit's attention
  **output** against the float64 oracle at causal base 9, and against a cold
  prefill of the whole prompt. Binding the base at 0 and changing nothing else
  fails at 160 elements of 160.

**[009-D14](009-server.md)'s footprint gate was the one decision row with
nothing behind it, and now has `internal/depcheck`.** It compares
`go list -deps ./server` against an allowlist carrying a reason per prefix. The
list is pinned both ways: a module nobody allowed fails, and so does an
allowance the build stopped using, because a stale prefix can only make the gate
greener while silently admitting whatever later appears under that path.

**[C27](010-conformance.md) is confirmed without the 50 GiB download.**
[024 §4.4](024-qwen3-5-architecture.md) had priced the answer as a safetensors
header read on a checkpoint nobody has. ollama's public `qwen3_5` implementation
answers it: `in_proj_ba` is permuted through a permutation of length
$2 H_v$, the native layout is beta then alpha per key head, and the whole width
reaches the gated delta kernel unreduced. The gate has a head axis, so the layer
is inexpressible on `tensor.LinearAttention` as it stands. Reported on
[accel#27](https://github.com/golang-design/accel/issues/27).

**Two gates were added because two bugs got past every existing one.** A pipe
inside a table cell splits the row — in a code span too — and the drift test for
the generated register compared the generated line against the file and found
them equal, because both carried it. `Validate` now rejects a pipe in any cell
that renders, and `speclint.CheckTables` checks every table in the tree. And a
semicolon inside a mermaid `Note` is a statement separator, so a sequence
diagram in [016](016-prefix-cache.md) did not parse and nothing but rendering it
would have said so.

### 2026-08-26 — Wave 7: the server reuses a conversation's prefix

[019](019-session-affinity.md), and it closes what Wave 6 opened: the prefix
cache worked and `tgo serve` could not reach it, because `generate.go` opened
one session per request and closed it on the way out.

`Model.NewPool(n)` holds N sessions' key/value cache for the process's life and
routes a request to the session already holding the longest matching prefix.
**It needs nothing from accel.** The refusal `CacheProcess` returns names a
missing page table, and that is the blocker for block-level sharing, not for
cross-request reuse: a session's cache is contiguous and single-owner, so a
row's index is its position. What 016 §4's pool would add over this is dedup
across *different* conversations, which affinity cannot do at any size.

**Measured, through the recorder rather than a clock.** Two turns of one
conversation, pool of two, 80-position context, greedy:

| | prompt | prefilled | prefill steps |
| --- | --- | --- | --- |
| turn 1 | 18 | 18 | 1 |
| turn 2 | 32 | **9** | 1 |
| turn 2, on a pool that never saw turn 1 | 32 | 32 | 1 |

The cold row is what makes 9 a number rather than a small integer. The 23
positions skipped are turn 1's 18-token prompt and the 5 tokens it generated.

**Three mutants survived the first suite**, all of them missing tests rather
than defects, and two are worth recording:

- **The unkeyed direction of the fail-closed rule.** Letting an unkeyed request
  match a *keyed* session passed everything. The table test established the
  unkeyed conversation first, so its later request always had its own session to
  hit and read the same reuse count either way. This is the security direction:
  a probe with no `cache_salt` could measure whether a salted caller's prompt
  exists.
- **The history truncation**, which no request can observe. `reusable` takes
  `min(s.length, len(s.history), ...)`, so a history longer than the length is
  invisible to routing; it reaches the sampler's repetition penalties instead.
  It needed a postcondition test that desynchronises the two the one way a
  caller cannot.

**A rule paid for twice.** The race step timed out on ubuntu and windows and
named a test that was not at fault -- `go test` reports whichever test was
running when the clock ran out. The root package runs a forward pass on the CPU
and the race detector costs about an order of magnitude on that arithmetic: 362s
wall but 1297s of CPU, so a runner with fewer, slower cores goes past the
ten-minute default. CONTRIBUTING now says to measure CPU time, because the wall
clock is what made it look green locally.

**Remaining**: [008](008-scheduler.md) continuous batching, which is the one
that turns "several conversations at once" into "several conversations in one
step", and [§2.3](#23-what-is-missing-that-no-spec-covers)'s unspecced gaps.

### 2026-08-27 — Wave 8: the cache is addressed through a page table

Two probes and two commits of code, in that order, because
[010 §2](010-conformance.md) says a row's state is decided by a value and not by
a commit message.

**The probes first.** accel closed [#16](https://github.com/golang-design/accel/issues/16)
with a ragged step and shipped `tensor.LinearAttention`. Both were checked by
binding real buffers rather than by reading `tensor/attention.go`:

| probe | what it asserts |
| --- | --- |
| [C16](010-conformance.md) | a mixed step — a 3-token chunk, a decode, and a sequence contributing nothing — matches a float64 reference that walks the page table itself, selects `AttentionRagged`, is **bit-identical** to the same tokens run as separate dispatches, and changes its output when the extents are re-split 2/1/1 |
| the gated delta scan ([018](018-hybrid-models.md)) | it matches a float64 reference, halving every $\alpha$ moves the output, and a state with `valueDim` and `keyDim` transposed is refused. Not a register row: the register is what tgo *cannot* do, and this it can |

Both closed. [008](008-scheduler.md) and [018](018-hybrid-models.md) are
therefore unblocked upstream, and **no spec in this tree is blocked upstream any
more.**

**And the probe found what it was not looking for.** The ragged kernel reads an
**f32** cache. [C5](010-conformance.md) closed on the argument that an f16 cache
halves the largest allocation a serving process has, and the operator that makes
batching possible gives that halving back — which by [008 §1](008-scheduler.md)
halves both the batch size worth reaching and the throughput ceiling. Filed as
[accel#23](https://github.com/golang-design/accel/issues/23) and recorded as
[C22](010-conformance.md). A consumer that reports the capability it wanted and
not the one it lost is reporting half.

**Then the port.** [016 §9](016-prefix-cache.md)'s third constraint was tgo's
own: the kernels honoured a page table and
[004 §3](004-model-graph.md)'s port table had none, so nothing here could pass
one. `GraphSpec.Block` declares `PortPages` and `NewPagedStep` maps a logical
position through it. The value test is a prefill over a **permuted** table
required to agree bit for bit with a contiguous run — an identity table would
pass whether the kernels read it or not, which is how
[accel#10](https://github.com/golang-design/accel/issues/10) stayed invisible —
with a negative control that writes contiguously and reads through the
permutation and requires the two to *disagree*.

**Then the pool.** `WithPrefixCache(CacheProcess, positions)` is available and
`internal/prefix` has an importer after two waves of having none. The key and
value states moved from the session to the model, which is what makes sharing
possible and also what bounds a server's memory: the cost is the pool, not
sessions times context. Measured on the fixture, a second conversation reused
**96 of 106** prompt tokens it never computed, and generated what a cold run
generates.

**One defect, found by the test written to find it.** `NewSession` sized the
page table and then replaced the whole step struct below it, dropping the slice.
`WriteBuffer` over an empty slice writes nothing and reports nothing, so the
port held whatever the allocation held and every step attended to blocks nobody
chose — fluent text, no error, and a pooled session generating different tokens
from a contiguous one at *identical* addressing.

Isolating it is the part worth keeping. A probe at the tensor layer proved
accel's paged prefill is bit-identical to the contiguous one at these shapes; a
second at the model layer proved the recorded graph is; which left the session,
and a diff of its bound ports showed the page table was empty. **Three layers,
each ruled out by a value.** The fix is one line in the constructor, so the
check at the seam is what makes the class visible next time.

**And one decision the shared pool forced that no per-session cache ever had
to make: how long a lease lives.** It was released at the next request, so an
idle conversation held a reference to blocks no live one could have — over $B$
blocks and $N$ sessions, [008 §3](008-scheduler.md)'s deadlock. The third
request of a four-request server test failed with six of eight blocks
referenced, which is the value of an end-to-end test over one that stubs the
engine: the flag-layer test could not have found it because it never serves a
request. [016-D12](016-prefix-cache.md) is the rule and the tail is its cost.

Over the wire, two different conversations sharing an opening turn: the second
reuses what the first paid for, and a third under a `cache_salt` reuses nothing.
No session-scoped cache produces that at any pool size.

**Remaining after Wave 8**: [008](008-scheduler.md), which Wave 9 built. One
smaller debt is named rather than hidden: `Pool.route` still asks
`Session.reusable`, which returns zero outside the session scope, so under a
process pool a request goes to the coldest session and the blocks do the
sharing. Nothing is lost, and [008-D9](008-scheduler.md) is where routing learns
to ask the pool.

### 2026-08-27 — Wave 9: several conversations in one step

[008](008-scheduler.md), which was `blocked` from the day it was written and is
`implemented`. [§8](008-scheduler.md) is what shipped, section by section;
what is worth recording here is the shape of the work and what it cost.

**The order was the page-table port, the shared pool, the batched graph, then
the policy** — the order 008 §8 planned, and each step's test was the same
shape: a batch of N must produce what N single steps produce, bit for bit, and
the page tables are permutations because an identity table passes whether the
kernel reads it or not.

**The mix is the deliverable.** A chunked prefill and the decodes beside it are
one dispatch, measured at 8 prompt tokens and 2 decodes in one step. That is
what makes chunking recover throughput rather than only bound latency: the
weights are read once for the chunk and its passengers together.

**The policy is a pure function and is tested without a device.** `nextStep` and
`victim` decide from slot state alone, so the cases are the ones that matter
rather than the ones a forward-pass fixture can afford. It is the first part of
this tree where the cheap test and the thorough test are the same test.

**Three defects, and none of them had a symptom a value test would catch.** A
batched step padded rows belonging to no sequence, which on a GPU is another
sequence's cache read back fluently — [C23](010-conformance.md), filed as
[accel#24](https://github.com/golang-design/accel/issues/24). `Admit` leased the
prompt and `Step` leased it again, chaining block hashes over `prompt+prompt`
while the step's own logits stayed correct. And a returned logits slice outlived
what it described, which the author's own test broke within minutes — a lifetime
nobody can keep is not a lifetime.

**Then a multi-agent adversarial review of the whole wave, and it is the part
worth carrying forward.** Five lenses over the diff, every candidate handed to a
skeptic told to default to refuted; ten survived. Four were defects and they
shared one root cause — **a lease covered positions no step had computed, and
everything downstream trusted its extent.**

| what | how it surfaced |
| --- | --- |
| a chunked prefill published all six blocks of a prompt 32 tokens in, so the next request reused 160 positions nobody wrote — under two salts, one tenant's hash naming another's rows | `Publish` walked every hash the lease held |
| a step that failed after an earlier slot reserved left that lease over a token nobody computed, and a retry with a different token wrote *its* state into a block named for the abandoned one | `reserve` recorded tokens before the submission |
| a caller carving prompts out of one buffer had the first request rewrite the first token of the second | `Admit` kept the caller's slice and `Feed` appended to it |
| `server.Wrap` dropped `cache_salt` while [009 §4](009-server.md)'s loss report said it was honoured | a session of its own shares nothing, so under `CacheSession` it reached nothing |

**None of them had a symptom a value test could see.** In every one the logits
of the step that caused the damage are correct, and the wrong answer goes to a
*later* request whose own caller did nothing unusual. That is the shape this
tree exists to catch in accel, found in tgo, and it is the argument for the
review rather than for more of the same tests.

The fix is one rule: **a lease grows before a step and records after it**, and
publishes only what its slot has written. `Grow` and `Commit` are those halves.

Three of the ten were weak tests, and each is the same mistake in a different
place: a fixture where two quantities are equal, so two hypotheses make the same
prediction. Two 4-token prompts summing to a chunk of 8; work always passed in
slot order; an assertion that could not fail because `InUse` cannot rise on an
exhausted pool.

**And two upstream reports closed within a day**, both filed from this wave.
[C22](010-conformance.md) — the ragged kernel read f32 only, giving back the
halving [C5](010-conformance.md) closed for — and
[C23](010-conformance.md), where accel took the shape the report argued for: a
query row past the last extent contributes nothing, rather than being clamped
into the last sequence, which would have turned an out-of-bounds read into a
wrong answer. That closure simplified tgo's padding back to what a single
sequence does.

**Remaining**: [008 §9](008-scheduler.md)'s three — sampling on the batched
path, a queue in front of admission, and the server actually using it — the rest
of [018 §6](018-hybrid-models.md)'s rows, and
[§2.3](#23-what-is-missing-that-no-spec-covers)'s unspecced gaps.

### 2026-08-27 — Wave 10: four-bit weights

[C21](010-conformance.md)'s tgo half, built the day the re-audit named it.
`weights.Int4` stores eight codes to a u32 word with an f16 scale and an f16
zero per 128, and a 27B checkpoint resolves to 13.4 GiB rather than 26.7.

**The signal was the expensive decision and it came first.** `nn.Graph.Stored`
returned an `accel.DType`, which has no int4 — so the only way to say it was to
overload `u32`, the code plane's dtype, and at that point the `shape` argument
stops describing the port: codes are $(K\cdot N+7)/8$ words and nothing about
them says $[K, N]$. `nn.Form` names the *representation*. Changing it touched
four files and would have touched the loader too if it had been written first.

**Two defects the tests caught, and neither was about int4.** `Set.Close` named
`Data` and `Scales` one at a time, so the third plane was allocated and closed
by nothing — surfacing as accel refusing to close a device under fifteen live
children, on a test about something else. And the weight binding used
`v.Elements` as the view count, which stops being the buffer's count at int4 by
a factor of eight.

**And one thing that was blocked for six hours.** [005 §3](005-kv-cache.md)
wants a narrow key/value cache, and the first refusal was tgo's own:
`nn.Attention` recorded no `Cast` on the scattered rows, so an f16 state could
not be written. It records one now and `model.GraphSpec` stopped refusing f16 —
and the pool still could not be narrow, because
[C24](010-conformance.md) found accel selecting the f16 prefill kernel and then
overwriting that selection whenever a page table was supplied. A pool is paged
by construction, so the narrow cache was reachable for a contiguous single
sequence and for nobody who shares blocks.
[accel#25](https://github.com/golang-design/accel/issues/25), fixed the same day
on the *pair* rather than as a fourth kernel name.

It is [C5](010-conformance.md)'s pattern a third time, and the third time is
what makes it a lesson rather than an incident: each of "`ScatterRows`, prefill
and paged decode all take f16" is true, and the *combination* has now failed at
the ragged kernel ([C22](010-conformance.md)) and at the paged prefill. **A
capability claim over three operators is three claims, and closing it on the
conjunction is what a register row is for.**

The pool is f16, which is twice the blocks, twice the prefixes worth keeping,
and by [008 §1](008-scheduler.md) twice the batch size worth reaching.

**Two claims withdrawn.** A first test compared int4's tokens against f16's and
required them to match; the fixture's weights are multiples of $1/8$ over a
range of 4, so a 4-bit step is $4/15$ and they diverge for a *correct*
implementation. What is checkable without a bound is that swapping the scale and
zero planes changes the answer — if it did not, neither would be read. And the
claim that int4 is *under* half of int8 is wrong: it is exactly half, because
the group doubles as the payload halves and both terms halve with it.

### 2026-08-27 — Wave 11: the gated delta layer, and the convolution in front of it

[018 §6](018-hybrid-models.md)'s first two rows, which everything else in that
spec waits on. `nn.LinearAttention` composes accel's scan and
`nn.DepthwiseCausalConv` runs over a rolling window, both checked against
float64 references written from §2 and §4.1's definitions rather than from the
blocks.

**The convolution is the one that had to be argued for first.**
[§4.1.1](018-hybrid-models.md) withdrew "it composes" — the probe had padded an
input *port*, and a real layer convolves a projection the graph computes — and
then said what to build instead: not a `Concat`, because a convolution running a
token at a time needs the K−1 inputs of the *previous step* rather than zeros. A
`[K−1+T, C]` state, scattered into, read K windows out of, and written back to
the front for the next step. The wave entry said `[slots, K−1+T, C]` until
2026-08-27: axis 0 is the time axis the taps slide along, so what shipped holds
one sequence and the slot axis is [023](023-cache-kinds.md)'s.

That last write is after the read and the state versions say so; a write before
it would convolve rows the step has not produced. And the carry is the half a
padded operand could never have supplied, so the test is a five-token step
followed by a one-token step against a reference over all six.

**Two accel defects, and one claim of this tree's own that had to go.**

- **[C25](010-conformance.md) is the worst kind and cost the most.** Declaring a
  *reshaped* result as a graph output produces all zeros — correct shape, no
  refusal, `Contiguous` in front does not help. An hour went into a recurrence
  that was correct; the operator's result unreshaped matched the reference to
  eleven digits. Reproduced on two harnesses,
  [accel#26](https://github.com/golang-design/accel/issues/26), with the
  argument that `Output` holds the tensor and can see it is a view.
- **[C26](010-conformance.md) is [§4.1](018-hybrid-models.md)'s claim withdrawn**
  and then answered in the same wave, which is why it is a `won't fix` rather
  than a filing: the refusal was the right one and what it pointed at is built.

**The block returns accel's shape rather than the one every projection around it
has**, and that is C25's consequence rather than a preference: flattening would
put a `Reshape` in every caller's path whether they output the result or not.

**Three more things the tests caught, none of them about the recurrence.** The
convolution's tap order was reversed against its own doc comment, and reversing
it runs and produces plausible numbers. `Mul` takes operands and not views, so a
broadcast tap row needs `Contiguous` around it as well as under it. And the `nn`
rig discarded `buf.Close()`'s error while binding with a batched write it never
flushed — so any test that binds a buffer and never submits leaked it, surfacing
as accel refusing to close the device from a cleanup, about something the test
was not testing.

### 2026-08-26 — Wave 6: structured output is reachable from a request

`internal/grammar` shipped with 97.8% coverage and **no caller**, which the
per-package gate cannot see: a package nothing imports reads green. This wave is
the wiring, and [015 §5](015-structured-output.md) is the shape of it.

`Policy.Schema` carries a schema, `Model.CheckSchema` compiles one on its own,
`Stream.advance` masks before the draw and advances after it, and `adapt.go`
accepts `response_format`, `output_format` and `text.format` instead of refusing
them. A schema the compiler will not compile is a 400 carrying the keyword and
the obstruction, answered before a session is allocated.

**Three joins the grammar package could not test, because it never sees a real
tokenizer**, and each one fails silently:

- **The bytes.** `Vocab.Bytes` is `Tokenizer.TextBytes`, new here: the decoded
  bytes, not the vocabulary file's spelling. A byte-level BPE stores `" the"` as
  `"Ġthe"`, and a mask built from the surface form constrains a different
  language and reports nothing.
- **The width.** The mask is applied to a logits row, so the vocabulary is the
  checkpoint's `vocab_size` and not the tokenizer's id space. The fixture pads
  640 over 582, which is what makes the confusion visible in a test.
- **The stop set.** `Options.Stop` is exactly what `Stream.isStop` ends on. Empty,
  `Mask` returns `ErrNoToken` the moment the document is complete -- so every
  constrained generation would fail on its last step with the right answer in
  hand. The test that catches it is a generation that runs to a complete
  document and *stops*, asserted as `finish_reason: stop` rather than as text.

**A fixture hazard worth writing down.** JSON admits whitespace before every
token, so the grammar does too, and a synthetic checkpoint's weights draw a
space as readily as a brace: an object schema ran a 600-token budget out on
indentation. Both end-to-end tests therefore use a schema whose language is
finite and ban whitespace through `logit_bias`, so the budget is a bound rather
than a hope. An `"integer"` property is not finite -- a magnitude bound is
`"maximum"`, which the compiler refuses as arithmetic on the value -- and that
narrowing is the package's, documented, not the mask failing.

**The other half of this wave was [016](016-prefix-cache.md), and it landed
short of the product.** `WithPrefixCache(CacheSession, n)` reuses a session's
own prefix and is tested at the library surface. It reuses nothing from `tgo
serve`, because the server opens one session per request and closes it on the
way out, so a session never sees a second turn. `CacheProcess` is refused: it
needs a page table, and [004 §3](004-model-graph.md) declares no port for one.
[016 §7.2](016-prefix-cache.md) carries the correction and the three options;
the decision taken was **session affinity** — pool the sessions and route a
request to the one already holding the longest matching prefix, which needs no
page table at all.

**Two packages have no importer, and the coverage gate cannot say so.**
`internal/prefix` is 100% covered and called by nothing: it is the block pool
for the sharing the missing port blocks, so it waits. `internal/grammar` was in
the same state until this wave, which is what the wave was. A per-package floor
measures the packages that exist, not the ones that are reached, so "15
packages at or above 90%" is true of code no request can run.

**Remaining**: session affinity for [016](016-prefix-cache.md),
[008](008-scheduler.md) continuous batching, and
[§2.3](#23-what-is-missing-that-no-spec-covers)'s unspecced gaps.

### 2026-08-26 — Wave 5 shipped: tgo serves

`server`, `internal/prefix`, `internal/hub`, and `serve`/`pull` in `cmd/tgo`.
**Seventeen packages**, every gate green, the coverage floor measuring fourteen.

**Verified against the real model, not against a fake.** `tgo serve` on
Qwen3-0.6B, Metal, 217 admitted sessions at 256 positions:

| | |
| --- | --- |
| OpenAI Chat | answers, `finish_reason`, usage |
| Anthropic Messages | answers, thinking typed as a `thinking` block |
| OpenAI Responses | answers, reasoning as a `summary_text` |
| SSE | streams token by token, `reasoning_content` deltas |
| `X-Tgo-Loss` | `service_tier, user` — advisory fields ran and were reported |
| `n=4` | refused by name, with the reason and a remedy |
| `/metrics` | in-flight per dialect, queue depth, wait histogram |
| SIGINT | graceful, in-flight requests given 30s |

That is [009-D2](009-server.md)'s rule working end to end: refuse what changes
the answer, report what cannot.

**What the reviews found, and none of it was in production code.** Every defect
this wave was a *test* that appeared to cover a property and did not:

- **`internal/hub`**: the lock suite proved the lock was *acquired* and never
  that it was *held while bytes landed* — which is the entirety of
  [013 §3](013-distribution.md)'s claim. The parallel bound was unproven because
  the fake released its barrier before the held handlers could be counted. And
  `openPart`'s truncate guarded a real corruption the fixture masked: a partial
  longer than the body leaves stale bytes on the end, and the rename publishes
  them.
- **`internal/prefix`**: the chained-hash test — the one named for
  [016-D2](016-prefix-cache.md) — was **insensitive to an unchained hash**. A
  match loop stops at the first miss, so an interior block is never looked up;
  the collision does its damage at **publish**, where the second prompt adopts
  the first's physical block. 016 §8 now names publish.
- **`server`**: [009-D12](009-server.md) was dialect-blind and shipped the
  defect it caused. Amended.
- **`cmd/tgo`**: the exactly-one-session admission boundary, where a machine
  that can *just* run the model would be turned away by a message telling it to
  lower a context that already fits.

**The header drop was proved by mutation rather than by reading**, and the kill
is not deleting the `Header.Del` line — it is changing the host comparison to
Go's own `Hostname()`, which strips the port. Both test servers sit on
127.0.0.1, so a port-stripped comparison sees one domain and forwards the token
to the CDN, which 403s.

**And a fourth degenerate fixture**, after three earlier waves: a page table that
was the identity, so `Row` could be replaced by `return t` and everything passed
— `Row` being the only consumer-facing arithmetic in that package.

**Remaining**: [008](008-scheduler.md) continuous batching,
[015](015-structured-output.md) structured output, and
[§2.3](#23-what-is-missing-that-no-spec-covers)'s unspecced gaps — of which
4-bit weights is the one that decides whether a 27B-class model is reachable.

### 2026-08-25 — Wave 4 shipped: the framework runs, and measures itself

`tgo` (the public API and decode loop), `cmd/tgo`, and the conformance
register. Twelve packages, every gate green, the coverage floor measuring
eleven.

**`tgo run`, `tgo bench` and `tgo info` work against the real 596M-parameter
Qwen3-0.6B checkpoint.** The first benchmark, and the reason
[017-D1](017-benchmarks.md) exists:

| batch | tokens/s | steps | p50 | host | submit | device | readback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | 1 | 3560.8s | 0.00% | 0.01% | **99.99%** | 0.00% |

**99.99% device.** A throughput number alone could not have said that; the
breakdown attributes the cost to accel's kernels rather than to tgo's loop,
which is the question [010 §1](010-conformance.md) says this project exists to
answer. The JSON record carries [017-D4](017-benchmarks.md)'s full conditions
and **names what it cannot measure as missing rather than printing zeros** —
plan compile time, the batch curve, the vLLM comparison.

**Two upstream findings, both from running a real model rather than reading
one.** This is the second time that method has beaten inspection:

- [accel#19](https://github.com/golang-design/accel/issues/19) — `Contiguous`
  was the **only kernel in the corpus with no MSL artifact**, so any graph that
  slices could not run on Metal, and [004 §3.2](004-model-graph.md) *requires*
  slicing before the LM head. **Filed and fixed upstream the same day**, which
  is what turned this wave from a CPU-only result into a working model.
- [accel#20](https://github.com/golang-design/accel/issues/20) — accel's CPU
  backend dispatches workgroups **serially**, by its own documentation. 32
  minutes for a 4-token prefill of 596M parameters. Not a bug; the serial loop
  is simply now on the critical path of a real workload.

**One gap closed by hand.** The engine recorded all four terms and exported no
way to set or read a recorder, so `tgo bench` printed the breakdown as *missing*.
The implementer reported that rather than printing zeros, which was right;
`WithRecorder` now threads it through, and the table above is the result.

**And a rule, now written where it will be read: no two dimensions in a test
config may be equal.** The root package had `synthLayers` and `synthKVHeads`
both 2 — the identity for every confusion between them. That exact shape cost
[Wave 2](#2026-08-24--wave-2-shipped-and-the-target-checkpoint-corrected-a-spec)
twelve surviving mutants and Wave 3 its whole f16 permutation path.

**tgo generates coherent text on Metal.** After
[accel#19](https://github.com/golang-design/accel/issues/19) closed:

```
$ tgo run --prompt "The capital of France is" --max-tokens 12 --temp 0
Okay, the user is asking about the capital of
13 prompt tokens, 12 generated, 4.17 tokens/second
```

Byte-identical across runs, which is [006-D3](006-sampling.md)'s greedy
determinism holding on a real model. Measured properly at 64 prompt tokens and
32 decode steps: **12.57 tokens/s decode, 379 tokens/s prefill**, 169ms warm
time to first token. [017 §4.1](017-benchmarks.md) has the breakdown and the
three findings it produced — the sharpest being that **submit is 15.61% of a
decode step**, read at the time as the largest non-kernel cost tgo has and the
first one that is not upstream.

**It was upstream after all, and that is the third finding this wave earned.**
Filed as [accel#21](https://github.com/golang-design/accel/issues/21) and fixed:
not the per-*step* submission cost this instrument assumed, but a per-*node*
one — a reflection frame rebuilt on every call into Objective-C, about five
message sends per dispatch over a ~790-node graph. Re-measured on the same
prompt and the same 32 decode steps:

| decode | tokens/s | p50 | p99 | submit | device |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 12.57 | 61.6ms | 474.7ms | 15.61% | 81.88% |
| **after** | **17.97** | 54.2ms | **76.5ms** | **3.34%** | **94.62%** |

**+43% throughput from one upstream change, and the p99 fell 84%** — the tail
collapsed by more than the median moved, because a per-call frame rebuild
produces occasional very slow calls rather than a uniform tax. It is the largest
single performance change this project has made and the second one tgo's
instrumentation earned rather than its code. A decode step is now 94.62% device,
which is the shape a decode step should have.

**Deviation from plan: RMSNorm gains are loaded outside the `weights`
package.** [004 §3](004-model-graph.md) declares a norm gain as an **f32** port
and [001 §2](001-weights.md)'s pipeline ends at f16-or-int8, so `weights.Load`
has no f32 output at all and the two shipped packages cannot be composed. accel
binds by exact dtype, so a gain that came through the loader is refused at the
first submission. The engine is the first place the two specs meet, so the
engine widens between them: `loadGains` uploads all 113 of Qwen3-0.6B's gains as
f32 and 198 of its 311 tensors reach the loader. It is 256 KB against 1.4 GB, so
the cost is the seam and not the bytes. What it must not skip is
[004-D9](004-model-graph.md)'s rotary permutation on the `q_norm` and `k_norm`
gains: getting that wrong scales the wrong channels and produces fluent text
that loses coherence, with nothing to report it. There is a test named for that
reason.

**Two gates caught their author, which is the point of having them.** The Metal
test written to assert *"this cannot work"* began failing because it works; its
own comment said to delete it, and it was replaced by its positive form instead,
since a backend that worked once and quietly stopped is what tier 2 exists to
catch. And the register generator rejected a hand-edit of
[010 §2](010-conformance.md) — including a row-order difference no reader would
have flagged — which is [010-D6](010-conformance.md) doing exactly its job.

**And one class of bug that only CI could find.** Five tests passed on macOS and
Linux and failed on Windows, whose timer granularity is about 15ms: a fake that
returned instantly made the wall clock around it zero, and an assertion that a
readback duration was positive was too strong for a 2-layer fixture whose real
readback is a few hundred microseconds. Both now assert **observation counts
rather than durations**, and `CONTRIBUTING.md` carries the rule. Worth recording
because the failure mode is invisible on the two platforms most of this was
developed on.

**Wave 5 next**: the OpenAI/Anthropic/Responses server ([009](009-server.md))
and prefix caching ([016](016-prefix-cache.md)). Host overhead per decode token
stays the axis [000 §11](000-decisions.md) says tgo should win: after accel#21
the `host` term is 0.64% of a step and submit 3.34%, so submit remains the
largest of the non-device terms.

### 2026-08-25 — Wave 3 shipped: the forward pass agrees with the oracle

`model`'s Qwen3 graph and KV cache, `sample`, and `internal/conformance`. Ten
packages, every gate green including `-race`, coverage floor measuring all ten.

**The forward pass runs on the CPU backend and matches a float64 reference**
derived from the mathematics rather than from the graph code
([010-D2](010-conformance.md)). That is the check that proves the model rather
than proving it compiles.

**Each of the three tests that prove the model was mutation-tested**, because a passing
test proves nothing until it has been shown it can fail:

- moving QK-norm to after RoPE kills two tests by a factor of **15,000 over
  tolerance** — the silent-coherence bug [004 §2.4](004-model-graph.md) warns
  about, which nothing downstream would report;
- decode at $T=1$ against prefill's last row, over one cache both plans bind;
- the last-row slice — see below.

**Writing the §3.2 test found that the obvious test does not work.** Transient
bytes do *not* catch an LM head running over every position, because
`PortLogits` is an **output** port: its buffer is the caller's and never appears
in `Plan.Memory()`. Measured by mutation — slicing `[0,t)` instead of
`[t-1,t)` left transients identical at 38912 bytes across a 36× vocabulary
change. The claim is now checked where it lives, on the declared shape of the
port, with the reason in the comment so the next reader does not "fix" it back.

**The conformance harness is what makes the rest checkable**, and its tolerances
are a **type rather than a number** — [010-D3](010-conformance.md) enforced by
the compiler instead of by discipline. A caller composes the stages its
computation actually has and the bound falls out, so widening one means adding a
term somebody has to argue for out loud. Six parity tests exercise it: RMSNorm,
SwiGLU, MatMul at f32 and at f16, int8, and RoPE.

**Two things this session got wrong and corrected.** `PrimitiveAbs` enters the
budget *relatively*, not absolutely — a cosine's absolute error becomes relative
to whatever that cosine multiplies — and the first test asserted the opposite.
And the tier rule's device branches are exempted from the coverage floor with a
stated reason rather than covered by a mock, because a mock device would cover
the statements and prove nothing about accel.

> **Wave 3 was interrupted.** All three agents hit an account limit mid-write.
> The partial work was salvaged rather than rerun: the packages were
> substantially complete and needed one API fix (`Float16.F32`, not
> `.Float32`), the parity tests, and the conformance tests. Recorded because
> "the wave was rebuilt from scratch" and "the wave was finished by hand" are
> different provenances and the second is this one.

**Wave 4 next**: the engine and the decode loop ([007](007-engine.md)), then the
CLI and the server.

### 2026-08-24 — Wave 2 shipped, and the target checkpoint corrected a spec

`weights`, `nn`, `internal/oracle` and `model`. Nine packages now, every gate
green including `-race`, and the coverage floor measures all nine: `bench`,
`chat`, `internal/oracle`, `model` and `nn` at 100.0%, `tokenizer` 99.1%,
`speclint` 98.0%, `safetensors` 97.7%, `weights` 93.7%.

**The real checkpoint loads.** Qwen3-0.6B onto an accel device: *311 tensors,
0.75e9 parameters, 1.40 GiB resident at f16*.

**A spec was wrong, and the checkpoint is what proved it.**
[004 §4](004-model-graph.md) said a checkpoint that is tied *and* ships an
`lm_head.weight` is a contradiction to refuse. Qwen3-0.6B does exactly that —
and both planes hash to `8f29acf5…` over their full 311 MB. The rule refused the
model tgo exists to run.

Corrected as [004-D10](004-model-graph.md): **redundancy is not a
contradiction.** Identical planes load; differing planes are the real
contradiction and stay refused; and because a safetensors header carries shapes
and shapes cannot tell the two apart, the comparison takes a
`WithPlaneComparator` and belongs to whoever holds the file. Writing the test
then exposed the second half — an accepted alias was still reported as a tensor
the map does not name.

**What the review pass earned this time**, all found by mutation rather than by
reading:

- `model`: forty-five mutants, **twelve survived**, from one root cause — the
  synthetic fixture had `d = H_kv·d_h` and `V = f`, so six wrong weight maps
  passed. Renumbered so no two dimensions collide, and the fixture now documents
  the rule.
- `weights`: every f16 test used `head_dim = 2`, where the rotary permutation
  **is the identity** — so the whole f16 path's permutation rested on a single
  int8 test. Widened to 8, with a degeneracy floor that fails if the expectation
  stops being sensitive to the permutation.
- `internal/oracle`: `mustPanic` accepted *any* panic, so nine deleted guards
  still "passed" by panicking one statement later out of a slice bound. Now
  asserts the message.
- `oracle` also settled a question the implementer left open, by reading accel's
  kernel: RoPE's theta divides by `rotaryDim`, not by the row width. Dividing by
  width would put the oracle out of parity with the device on every partial
  rotation — which is exactly what Qwen3.8-27B's `partial_rotary_factor` needs.

**Wave 3 next**: the Qwen3 forward pass, the KV cache and the decode loop —
[§2.2](#22-waves)'s one sequential chain.

### 2026-08-24 — Wave 1 shipped, Wave 2 started

**Verdict at the start of the wave: build.** [§2](#2-readiness-build-now-and-what-the-targets-actually-are)
has the evidence. Re-checked before Wave 2: accel has moved on to graphics work
so the tensor layer is stable, tgo builds and tests clean against it, and the
four open upstream issues
([#6](https://github.com/golang-design/accel/issues/6),
[#16](https://github.com/golang-design/accel/issues/16),
[#17](https://github.com/golang-design/accel/issues/17),
[#18](https://github.com/golang-design/accel/issues/18)) block neither Wave 2
nor Wave 3.

**What the field survey found**, since the targets turned out to be misnamed:
the Qwen3 dense line has no 3B (the target is 4B); Qwen3.8-27B is real and is
**not dense** — 48 of its 64 layers are linear attention ([018](018-hybrid-models.md),
[accel#17](https://github.com/golang-design/accel/issues/17)); and MoE is most of
what ships at single-machine sizes, which accel has no routed GEMM for
([accel#18](https://github.com/golang-design/accel/issues/18)). Both were filed
as questions rather than proposals.

**Wave 1 shipped**: `bench` 100.0%, `chat` 100.0%, `tokenizer` 99.1%,
`safetensors` 97.7%. Every gate green including `-race`, and the coverage floor
now measures five packages.

**What the parallel review pass earned**, since it doubles the cost and has to
pay for itself:

- `chat`'s reviewer mutation-tested fourteen edits, found three blind spots, and
  verified the goldens came from the reference by rendering the real Qwen3
  template with Jinja2 over 47 cases — all byte-identical.
- `tokenizer`'s reviewer found `post_processor` and `decoder` were never read. A
  Llama-3-style file would have loaded and encoded every prompt **without its
  BOS token**, silently. Both are now refused by name.
- One real defect, escalated rather than hidden: NFC was an identity seam while
  the loader *refused* NFKC because it "changes ids". Accepting a declared
  normalizer and not running it is the same silent divergence that refusal
  exists to prevent. Resolved by taking `x/text/unicode/norm`
  ([002-D10](002-tokenizer.md)) — pure Go, no cgo, so
  [000 D2](000-decisions.md) is untouched.

**A real checkpoint is on disk**: Qwen3-0.6B at `~/.cache/openllms-e2e/model`,
1.5 GB of bf16 safetensors. It carries `head_dim: 128` against
`hidden_size/num_attention_heads` of 64 — exactly the case
[004 §5](004-model-graph.md) says never to infer, now testable rather than
argued. Tests that read it are gated behind `TGO_MODEL` and never run in CI
([000 D8](000-decisions.md)).

**Wave 2 started**: `weights`, `nn`, `internal/oracle`, `model`.

### 2026-08-24 — the forward pass compiles, and the casts are gone

Not a milestone; a measurement, taken because the README claimed "nothing between
here and serving a model is waiting on accel" and nothing had checked it.

The whole Qwen3-4B graph from [004 §3](004-model-graph.md) — $d=2560$, $H=32$,
$H_{kv}=8$, $d_h=128$, $f=9728$, $V=151936$, $L=36$, a 4096-position cache —
compiles against accel HEAD:

| plan | selections | transients |
| --- | --- | --- |
| prefill 512, f16 weights | 760 | 62.0 MB |
| decode 1, f16 weights | 759 | 0.1 MB |
| prefill 512, int8 weights | 760 | 62.0 MB |
| decode 1, int8 weights | 759 | 0.1 MB |

**It was 1013 selections a day earlier.** The difference is 253 `Cast` nodes
that existed only because `MatMul` required both operands to share a dtype;
[010 C8](010-conformance.md) closed and they are gone.

Two things the numbers show that the specs only argued:

- **[004 §3.2](004-model-graph.md)'s last-row slice is worth what it claimed.**
  Prefill transients are 62 MB. Running the LM head over all 512 positions would
  add $512 \times 151936 \times 4 = 311$ MB of logits alone.
- **The cast cost was real and is now zero**, which is the first upstream change
  tgo can price exactly rather than estimate.

**What this does not show** is that the numbers are *correct* — that needs
[010 §5](010-conformance.md)'s oracle, which needs implementation. Compiling
proves the graph is expressible, not that tgo would build it right. The
[§2.5.1](004-model-graph.md) rotary permutation is exactly the kind of error
that compiles cleanly.

### 2026-08-24 — M0 — **done**

Module, CI, and the spec tree. CI mirrors accel's gates — build, vet, test,
race, cgo-free, cross-compile, gofmt — with two additions: a **per-package**
coverage floor rather than a repository average, and `speclint`, which checks
frontmatter shape, that dependency edges resolve, that the tree is acyclic, that
a `blocked` spec names what blocks it, that every spec carries a decision
record, and that the index cannot go stale.

**Deviation from plan:** none in scope, but the tree was written twice. It was
first written against accel's signatures as they stood, and then reconciled
against accel
[043](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md),
which landed mid-drafting in answer to seven reports tgo filed. Five register
rows changed state in one commit.

**Findings, which are the actual output of M0:** nine issues on accel. Seven
before the reconciliation, two after — and the two found last are the ones that
matter most. [accel#8](https://github.com/golang-design/accel/issues/8) caps the
KV cache at 128 positions — since closed by accel 044, which tgo also designed.
[accel#9](https://github.com/golang-design/accel/issues/9) refuses a
`LayerState` view, which corrected a decision this tree had already recorded
([005-D1](005-kv-cache.md)).

That last point is worth keeping. **A spec written against a library's
documentation was wrong about that library within a day.** Reading the
signatures is not the same as reading the refusals, and the refusals are where
the design actually lives.

**Two gates found their own bugs before any product code existed**, which is the
argument for building them first:

- The coverage gate reported *"every package at or above 90%"* over a tree whose
  only measured package was the one exempt from the check. It now fails when no
  non-exempt package was measured, and `speclint` moved from a test-only package
  into real code so the floor has something to stand on — 97.9%, measured.
- `speclint`'s negative tests immediately found a bug in its own citation regex,
  which captured the digits without the `C` and would have reported every valid
  register citation as dangling.
- CI's Windows row found that a CRLF checkout made **every spec in the tree**
  fail to parse. Nothing local would have.

CI is green on both workflows: `ci` across Linux, macOS and Windows with the
race detector, the fuzz seed corpus, the per-package coverage floor, the
cgo-free grep, ten cross-compile targets, gofmt and `speclint`; and `ci-metal`
on Apple silicon.

**M1 is next and is unblocked.**

## 4. The release-gate record

[000 D8](000-decisions.md) keeps real weights out of CI and calls the
`TGO_MODEL` run the release gate instead: run by hand, before a release, and
recorded here. This section is that record. [010 §4](010-conformance.md) points
at it for the same reason and adds the shape of an entry.

**It is not the wave log.** The dated entries under [§3](#3-milestones) say
what shipped.
This says what a number *was*, on a stated day, on a stated device, against a
stated checkpoint — so that the next release has something to differ from.

**What one entry carries**, and an entry missing any of these is not one:

- **the date** the run was taken, not the date it was written down;
- **the accel revision** it ran against, because most of a step is accel's and a
  number without one attributes a change to the wrong project;
- **the device**, named — a tier 2 number and a tier 1 number are different
  measurements of the same quantity;
- **the tier**, 1, 2 or 3, as `internal/conformance` defines them: tier 1 needs
  nothing, tier 2 needs a Metal device, tier 3 needs real weights and never runs
  in CI;
- **the checkpoint**, because a number measured on one is not comparable to a
  number measured on another;
- **the number and its tolerance.** A measurement with no tolerance cannot be
  compared to the next one, and [010-D3](010-conformance.md) makes the bound a
  type rather than a preference.

**Nothing is recorded here yet.** No release-gate run has been taken. The first
entries are [010 §3](010-conformance.md)'s five measurements, each of which is a
question accel cannot answer about itself and none of which has been run:

1. **CPU/Metal divergence** — greedy, same prompt: the first differing token
   index and the logit gap there.
2. **The readback share of a decode step**, which needs $V=151936$ rather than a
   fixture's vocabulary.
3. **Quantization error against `Int8ErrorBound`** on real blocks, which
   synthetic weights flatter because they have no outlier channels.
4. **Plan compile time per bucket**, and the plan cache hit rate over a session.
5. **Transient bytes** from `Plan.Memory()` against the hand-computed working
   set.

The table is empty and the columns are the shape to fill:

| date | accel revision | device | tier | checkpoint | measurement | value | tolerance |
| --- | --- | --- | --- | --- | --- | ---: | --- |
