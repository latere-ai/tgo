---
title: "Sequencing: the order the work is done in, and what actually landed"
status: living
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
| "Qwen3.8 27B" | **Qwen3.8-27B**, Apache 2.0, released August 2026 | **blocked**, [018](018-hybrid-models.md) |

### 2.1.1 What "well tested" means for each, and what it cannot mean

The two targets are at different distances, and saying so precisely matters more
than restating the goal.

| | Qwen3 dense | Qwen3.8-27B |
| --- | --- | --- |
| architecture expressible | **yes** — the graph compiles at 36 layers and $V=151936$ | **no** — 48 of 64 layers need an operator accel does not have |
| numerics verified | **yes** — against the f64 oracle, per block and end to end | not reachable |
| real weights loaded | **yes** — Qwen3-0.6B, 311 tensors, 1.40 GiB at f16 | not reachable |
| generation verified | **Wave 4** | not reachable |
| blocked by | nothing | [accel#17](https://github.com/golang-design/accel/issues/17) |

**Two honest qualifications on the dense side.** The checkpoint on hand is
**0.6B**, not 4B — same architecture, same code path, one twelfth the
parameters. What that proves is the loader, the graph and the numerics; what it
does not prove is behaviour at 4B's memory footprint. And the 4B graph is
verified to *compile and to agree with the oracle*, not to have generated text,
until Wave 4 lands.

**On the 27B, the position is not "not yet" but "not by tgo".** 018 states it,
accel has [scoped the operator](https://github.com/golang-design/accel/issues/17)
as in-scope-not-scheduled, and [000 D1](000-decisions.md) forbids tgo from
writing the kernel itself. Testing it is downstream of an upstream decision this
project deliberately does not control. Claiming otherwise would be the
overclaiming this tree exists to prevent.

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
not softmax attention — and it carries vision tokens. accel has no
linear-attention operator, so 48 of its 64 layers are inexpressible. Filed as
[accel#17](https://github.com/golang-design/accel/issues/17), and
[018](018-hybrid-models.md) is the design.

Two things about it that *do* work, checked rather than assumed:
`partial_rotary_factor: 0.25` is `RoPE(x, 64, …)` because `rotaryDim` was always
a parameter, and `attn_output_gate` is an elementwise multiply.

> **This is why the answer is "build now" rather than "spec more".** The dense
> path is fully specified and unblocked; the hybrid path is blocked on a kernel
> nobody has written, and no amount of tgo spec work moves it. Building the
> dense path is also what produces the evidence that makes the hybrid ask
> concrete.

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

M1–M5 are entirely unblocked and are where the work goes now. They are also
where [000 D8](000-decisions.md) puts the device-free packages, so they carry
almost all of the coverage gate: a tree that reaches M5 with the gate green is a
tree whose remaining risk is concentrated in the parts a device decides.

**M6 through M11 are unblocked.** [C11](010-conformance.md), the 128-position
cache cap that gated everything, closed on 2026-08-24 when accel shipped
[044](https://github.com/golang-design/accel/blob/main/specs/044-unbounded-context.md).
A 4096-position cache is verified. Nothing between here and serving a real model
is waiting on accel.

What remains blocked is post-v0 and narrower:

| | blocked on |
| --- | --- |
| chunked prefill recovering throughput | [C16](010-conformance.md) — a batched step takes one token per sequence, so a prefill runs alone |

**Nothing is blocked upstream any more.** Batching was the last one and accel
closed it: a batched decode is expressible and verified, two sequences of
different lengths matching two single runs exactly. What remains open is
narrower — no sampling operator at the tensor layer, no batched *prefill*, and
GGUF — and none of it blocks a milestone.

### 2026-08-25 — Wave 3 shipped: the forward pass agrees with the oracle

`model`'s Qwen3 graph and KV cache, `sample`, and `internal/conformance`. Ten
packages, every gate green including `-race`, coverage floor measuring all ten.

**The forward pass runs on the CPU backend and matches a float64 reference**
derived from the mathematics rather than from the graph code
([010-D2](010-conformance.md)). That is the check that proves the model rather
than proving it compiles.

**Each of the three load-bearing tests was mutation-tested**, because a passing
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
