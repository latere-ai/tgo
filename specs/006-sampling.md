---
title: "Sampling: the composition order, and reproducibility as a stream"
status: drafted
layer: engine
depends_on:
  - 000-decisions.md
  - 004-model-graph.md
---

# Sampling

## 1. Where it runs

accel 028 puts argmax, categorical sampling and top-k/top-p truncation on the
device, with the random draw as an **input**. accel 039 specifies temperature,
penalties and the composition order, and is **drafted, not built**. accel 043 §4
moves the draw from a scalar to a per-row tensor, which tgo asked for in
[accel#3](https://github.com/golang-design/accel/issues/3) and which matters
only once there is a batch.

The split for v0:

| stage | where | why |
| --- | --- | --- |
| logits | device | it is the model |
| logit bias, penalties, temperature | **host** | accel 039 is unbuilt; $O(V)$ over 152k floats is microseconds |
| top-k, top-p, categorical, argmax | **host** | the composition around accel's kernels does not exist, and splitting the pipeline mid-way would cost a second readback |

The host path is the **reference** in both cases. When 039 lands, the stages
move down and this implementation stays as what the device path is checked
against — so this is not throwaway code, it is [010 §5](010-conformance.md)'s
oracle for sampling.

The cost is [010 C6](010-conformance.md): a 608 KB logits readback per token,
which is the floor on a decode step ([007 §5.1](007-engine.md)).

## 2. The Go surface

```go
package sample

type Policy struct {
    Temperature       float32          // 0 means greedy, exactly
    TopK              int              // 0 means the stage is absent; 1..TopMaxRounds
    TopP              float32          // 0 means the stage is absent; otherwise (0,1]
    RepetitionPenalty float32          // 1 means none
    PresencePenalty   float32
    FrequencyPenalty  float32
    PenaltyWindow     int              // tokens back; 0 means the whole context
    LogitBias         map[int]float32
}

// Sampler holds the seeded draw stream. Not safe for concurrent use: it is
// per sequence by construction.
type Sampler struct{ /* ... */ }

func New(seed uint64) *Sampler

// Next consumes exactly one draw, whatever the policy. See section 4.
// history is the token window the penalties read; logits is modified in place.
func (s *Sampler) Next(logits []float32, history []int, p Policy) int

// Probs returns the post-policy distribution without consuming a draw,
// for logprobs. It must not disturb the stream.
func (s *Sampler) Probs(logits []float32, history []int, p Policy) []float32
```

## 3. The composition order, which is not arbitrary

$$\ell \xrightarrow{\text{bias}} \ell^{(1)} \xrightarrow{\text{penalties}} \ell^{(2)}
\xrightarrow{\;/T\;} \ell^{(3)} \xrightarrow{\text{softmax}} p
\xrightarrow{\text{top-}k} \xrightarrow{\text{top-}p} \xrightarrow{\;u\;} \text{token}$$

Each adjacency below is a decision, and §6 has a test that fails if it is
swapped.

> **Truncation moved after the softmax on 2026-08-25, and accel's argument is
> why.** This spec had top-$k$ and top-$p$ acting on logits, which is what vLLM
> does and is defensible. accel
> [039](https://github.com/golang-design/accel/blob/main/specs/039-sampling-policy.md)
> puts both after, on a sharper reason than "top-$p$ is a mass threshold":
> **f32 rounding can make two distinct logits equal probabilities**, so a top-$k$
> over logits keeps a different boundary entry than the cumulative walk later
> sees. The mask and the walk must agree about which entries exist, and they only
> do if both read the values the walk will read.
>
> tgo follows accel here because [006-D1](#decision-record) makes this package
> the *reference* for accel's device path: a reference that composes differently
> is not one. Recorded rather than silently rewritten, because the order was
> argued from vLLM's and the argument was not wrong so much as weaker.

- **Bias first.** `logit_bias` is a caller's absolute statement about a token; a
  penalty computed on a biased logit still means what it says, while biasing a
  penalised one does not.
- **Penalties before temperature.** A penalty is a logit adjustment with a fixed
  meaning. Applied after dividing by $T$, its strength depends on $T$, so the
  same policy behaves differently at every temperature.
- **Temperature before the softmax, and both before truncation.** Top-$p$ is a
  mass threshold and temperature is what changes the mass, so truncating first
  makes $p$ mean something different at each temperature. Truncating after the
  softmax additionally makes the mask and the walk agree on the boundary, per
  the note above.
- **Top-$k$ before top-$p$.** $k$ is a hard cap on the candidate count; $p$ then
  trims within it. The reverse lets $p$ admit more than $k$, since top-$p$ is
  relative to its own input's total — so each bound is one the other cannot
  violate.
- **Nothing renormalizes, and there is never a second softmax.** A mask leaves
  the weights summing below one, which invites a fix; the fix is wrong, because
  a softmax over a mask's output is near-uniform over the whole vocabulary
  ($e^0 = 1$ for every dropped entry). The walk compares against
  $u \times \text{total}$ instead.
- **Both stages are capped at `TopMaxRounds` = 128 candidates** by accel's
  kernels. A $k$ above that, or a nucleus wider than that, is expressible on the
  host and unrepresentable on the device — so the host reference must refuse it
  rather than compute something the device cannot reproduce. accel keeps exactly
  $k$, breaking ties lexicographically on (value, index); vLLM may return more
  than $k$. tgo follows accel, because accel is what it will be checked against.
- **$T = 0$ is greedy**, a distinct branch, not a division. It is also the
  branch that must be bit-exact.

> vLLM's sampler states this ordering as the same sequence, and names the
> property that justifies it: a stage is **argmax-invariant** if it cannot
> change which token is the maximum. Logit bias and the penalties are not —
> they reorder candidates — so they run before the greedy branch is taken.
> Temperature, `min_p`, top-$k$ and top-$p$ are, so they run after. That is a
> cleaner statement of why the greedy branch sits where it does than "it is a
> special case", and it is the test to apply when a new stage is added.

> accel 043 §6 adopts this ordering into
> [039](https://github.com/golang-design/accel/blob/main/specs/039-sampling-policy.md)'s
> scope, on the grounds that it is documentation rather than code. When 039 is
> built, this section stops being tgo's design and becomes tgo's reference
> implementation of accel's.

### 3.1 The penalties

With $c_v$ the count of token $v$ in the window and $\mathbb{1}_v$ whether it
appears at all:

$$\ell'_v = \begin{cases}
\ell_v / r & \ell_v > 0 \\
\ell_v \cdot r & \ell_v \le 0
\end{cases}
\qquad\text{then}\qquad
\ell''_v = \ell'_v - \alpha\,\mathbb{1}_v - \beta\,c_v$$

for repetition penalty $r$, presence $\alpha$, frequency $\beta$.

The asymmetry in the first is not a quirk: dividing a *negative* logit by $r>1$
would move it **toward** zero and make a penalised token more likely. Multiplying
instead keeps the penalty monotone in the right direction on both sides of zero.
This is the form llama.cpp and every stack since uses, and reproducing it
matters because callers tune $r$ against that behaviour.

The window is over **prompt and generated tokens together**. Penalising only
generated tokens lets the model repeat the prompt verbatim, which is the failure
users report as "it just echoes my question".

### 3.2 Top-$p$ ties

Top-$p$ keeps the smallest prefix of the sorted distribution whose mass reaches
$p$. Two subtleties:

- the token that **crosses** the threshold is kept, not dropped;
- **$p = 0$ means the stage is absent, not "keep one".** accel 039's contract:
  zero disables the stage, any other value must lie in $(0, 1]$, and anything
  else is refused. This matters because accel's kernel with $P = 0$ computes
  `target := best[0] * P` = 0 and keeps **nothing** — so a spec that said
  "$p=0$ keeps one token" would disagree with the device path it is supposed to
  be the reference for;
- ties at the boundary are broken by **token id**, ascending, so the result does
  not depend on the sort's stability. An unstable sort would make a supposedly
  reproducible run depend on the standard library's implementation.

## 4. Reproducibility is a stream property

The promise a user checks is: **the same prompt and the same seed give the same
completion.** accel 028 gives the token-level piece — a draw is an input — and
tgo owns the sequence of draws.

`Sampler` holds a `math/rand/v2` PCG seeded once. **Step $i$ consumes exactly
one $u_i \in [0,1)$, whether or not the policy is greedy.**

That last clause is the decision. Consuming a draw only when actually sampling
is the natural implementation, and it makes reproducibility hold in every test
and fail in production: a caller who changes temperature mid-request — a server
honouring a per-message policy, a retry with different settings — shifts every
subsequent draw, and the stream diverges from the same seed. The waste is one
PCG step per greedy token, which is nothing.

`Probs` must not consume a draw, for the same reason: computing logprobs is an
observation and must not move the stream.

### 4.1 What is and is not promised across devices

**Greedy decoding is bit-exact across runs on one device.**

It is **not** promised bit-exact between the CPU backend and Metal. Float
reduction order differs, so a near-tie at the argmax can resolve differently,
and one differing token diverges everything after it. [010 §3](010-conformance.md)
**measures** this — the first differing token index and the logit gap there —
rather than asserting a bound nobody verified.

## 5. Stopping

| condition | rule |
| --- | --- |
| EOS | `<\|im_end\|>` **and** `<\|endoftext\|>`, both, read from `generation_config.json` |
| max tokens | a count |
| stop strings | matched on the **decoded text**, not on ids |
| context exhausted | **refuse**, do not truncate |

Stop strings match on text because a stop string need not align to a token
boundary: `"\n\nUser:"` may arrive as three tokens with the boundary mid-string.
So the decoder holds back enough bytes to match the longest stop string, and
releases them when no match can complete — which is
[002-D8](002-tokenizer.md)'s second buffer, separate from the UTF-8 one.

When a stop string matches, the emitted text is truncated **at the match**, and
the tokens that produced the tail are still counted in usage. They were
generated.

Context exhaustion refuses because silently dropping a user's context is
unanswerable — the same reason accel 029 refuses to truncate a prompt.

## 6. Tests

| test | what it catches |
| --- | --- |
| each stage against an independent host reference, exactly | arithmetic |
| **order**: for each adjacent pair in §3, a case where swapping them changes the output | the whole of §3 |
| the §3.1 sign asymmetry: a negative logit is penalised downward | the classic bug |
| penalties read prompt tokens too | the "it echoes my question" failure |
| stream: the same seed gives the same completion **across a policy change mid-stream** | §4's decision |
| `Probs` does not move the stream | §4 |
| greedy determinism over 100 runs | §4.1 |
| degenerate policies: $k=1$ equals greedy; $p=1, T=1$ equals plain categorical; $k > V$; $p = 0$ keeps one token | boundaries |
| top-$p$ ties break by id | §3.2 |
| a stop string straddling a token boundary; one straddling a UTF-8 boundary | §5 |

Every one is host-only: no device, no weights, and therefore fully inside the
coverage gate.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 006-D1 | the whole policy on the host until accel 039 lands | wait for 039; reimplement 039 inside tgo | microseconds; and the host path becomes 039's reference rather than being deleted |
| 006-D2 | one draw consumed per step regardless of policy | draw only when sampling | a mid-stream policy change does not shift the stream; costs one PCG step per greedy token |
| 006-D3 | greedy bit-exact per device; cross-device divergence measured, not bounded | assert a cross-device tolerance | the number is reported rather than invented |
| 006-D4 | stop strings match decoded text, with a hold-back buffer separate from the UTF-8 one | match on token ids; one shared buffer | a stop string need not align to a token; the two buffers hold for different reasons |
| 006-D5 | context exhaustion refuses | truncate the prompt | matches accel 029; silent truncation is unanswerable |
| 006-D6 | top-$p$ keeps the crossing token; ties break by id | drop the crossing token; rely on sort stability | $p=0$ cannot empty the set, and reproducibility does not depend on the sort implementation |
| 006-D7 | `Probs` observes without consuming a draw | share the sampling path | logprobs do not perturb the completion they describe |
