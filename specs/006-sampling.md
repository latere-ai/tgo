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
penalties and composition order and is **drafted, not built**.

So the split for v0:

| stage | where | why |
| --- | --- | --- |
| logits | device | it is the model |
| penalties, temperature | **host** | 039 is unbuilt; these are $O(V)$ over 152k floats, which is microseconds |
| top-k, top-p, categorical | device where accel's kernels cover it, host otherwise | the kernels exist; the composition around them does not |

The host path is the reference in both cases, and the device path is checked
against it. When 039 lands, the host stages move down and the reference stays.

## 2. The composition order, which is not arbitrary

$$\ell \xrightarrow{\text{penalties}} \ell' \xrightarrow{\;/T\;} \ell'' \xrightarrow{\text{top-}k} \xrightarrow{\text{top-}p} \xrightarrow{\text{softmax}} p \xrightarrow{\;u\;} \text{token}$$

- **Penalties before temperature.** A penalty is a logit adjustment with a fixed
  meaning; applying it after dividing by $T$ makes its strength depend on $T$.
- **Temperature before truncation.** Top-$p$ is a mass threshold, and the mass
  is what temperature changes. Truncating first makes $p$ mean something
  different at every temperature.
- **Top-$k$ before top-$p$.** $k$ is a hard cap on the candidate count; $p$ then
  trims within it. The reverse lets $p$ admit more than $k$.
- **$T = 0$ means greedy**, not division by zero. It is a distinct branch, and
  it is the branch that must be bit-exact.

Repetition penalty applies to tokens in the window $[t-W, t)$ over both prompt
and generated tokens, dividing positive logits and multiplying negative ones —
the asymmetry is what keeps a penalised token from becoming *more* likely.
Presence and frequency penalties are additive and independent of it.

## 3. Reproducibility

The promise is at the **stream** level: a seed and a prompt give the same
completion. accel gives the token-level piece — a draw is an input — and tgo
owns the sequence of draws.

A `Sampler` holds a `math/rand/v2` PCG seeded once. Step $i$ consumes exactly
one $u_i \in [0,1)$, whether or not the policy is greedy, so **switching to
greedy and back does not shift the stream**. Consuming conditionally is the bug
that makes reproducibility hold in tests and fail in production, where a
policy changes mid-request.

Greedy is bit-exact across runs on one device. Across the CPU backend and Metal
it is not promised: floating-point reduction order differs, and a near-tie at
the argmax resolves differently. [010](010-conformance.md) **measures** that
divergence — how many tokens in and at what logit gap — rather than asserting a
bound nobody checked.

## 4. Stopping

- EOS: `<|im_end|>` and `<|endoftext|>`, both, from `generation_config.json`.
- Max tokens.
- Stop strings: matched on the **decoded text**, not on ids, because a stop
  string need not align to a token boundary. The decoder holds back enough
  bytes to match the longest stop string, and the held-back text is emitted if
  no match completes.
- Context exhausted: refuse, do not truncate. Silently dropping a user's context
  is the same failure accel 029 refuses.

## 5. Tests

- Each stage against a host reference, exactly.
- Order: a case where swapping any adjacent pair in §2 changes the output.
- Stream: same seed, same completion, across a policy change mid-stream.
- Greedy determinism over 100 runs.
- Degenerate policies: $k=1$ equals greedy; $p=1$ with $T=1$ equals plain
  categorical; $k > V$; $p = 0$.
- Stop strings that straddle a token boundary, and one that straddles a UTF-8
  boundary.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 006-D1 | penalties and temperature on the host until accel 039 lands | wait for 039; reimplement 039 in tgo | costs microseconds; the host stage is the reference afterwards |
| 006-D2 | one draw consumed per step regardless of policy | draw only when sampling | a mid-stream policy change does not shift the stream |
| 006-D3 | greedy is bit-exact per device, divergence measured across devices | assert a cross-device bound | the number is reported, not invented |
| 006-D4 | stop strings match decoded text with held-back bytes | match on token ids | a stop string need not align to a token |
| 006-D5 | context exhaustion refuses | truncate the prompt | matches accel 029; silent truncation is unanswerable |
