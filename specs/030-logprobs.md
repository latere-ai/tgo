---
title: "Logprobs: reporting the distribution a token was drawn from"
status: complete
layer: engine
depends_on:
  - 000-decisions.md
  - 006-sampling.md
  - 007-engine.md
  - 009-server.md
---

# Logprobs

`logprobs` and `top_logprobs` are accepted on all four routes, reported as an
advisory loss, and served by nothing. [009](009-server.md)'s Outcome carried
them as its own until 2026-08-28, when writing it out found they are not: the
obstruction is [007 §1](007-engine.md)'s surface, and this spec owns the change.

## 1. Why the server cannot do it alone

`sample.Sampler.Probs` already returns the post-policy distribution without
consuming a draw ([006-D7](006-sampling.md)). Nothing calls it, and the server
cannot: **a logprob is per token and a `tgo.Event` carries decoded text.**

```mermaid
flowchart LR
  T["tokens<br/>ids the sampler drew"] --> D["tokenizer.Decoder<br/>holds an incomplete rune"]
  D --> E["Event.Text<br/>0, 1 or several tokens"]
  E -.->|"no id, no probability"| X["server/"]
```

A byte-level vocabulary splits most non-ASCII characters across several tokens
([002 §6](002-tokenizer.md)), so the decoder holds back a valid-but-incomplete
prefix. One `Event` can therefore be the tail of one token, one whole token, or
several — and it carries no id and no probability. `Policy` has no field to ask
for them either.

## 2. The shape: a side accessor, not a field on the event

**030-D1: `Stream.LogProbs()` returns what the last `Next` produced.**

```go
// TokenProb is one token and what the sampler's distribution gave it.
type TokenProb struct {
    ID      int         // the token id
    Text    string      // TextBytes(ID) as a string; empty for a control token
    LogProb float64     // ln p, and math.Inf(-1) for a token the policy masked
    Top     []TokenProb // the Policy.TopLogProbs most likely, descending; Top is nil in these
}

// Policy
LogProbs    bool  // report the drawn token's logprob
TopLogProbs int   // and this many alternatives per step; 0 for none

// Stream
func (s *Stream) LogProbs() []TokenProb
```

It reads like `Usage()` and `StopReason()`, which are already side accessors on
the stream, and it leaves two things alone that a field on `Event` would not:

| | why it matters |
| --- | --- |
| `Event` keeps its four fields | a `BlockStart` would otherwise carry an empty `[]TokenProb` that means something different from a text delta's empty one |
| the `EventKind` set does not grow | [009 §3.2](009-server.md) maps kinds one-to-one onto the blocks the wire dialects encode, and a kind that is not a block breaks that mapping for every consumer's switch |

**The slice is the tokens of the last step**, which is normally one. It is a
slice and not a value because a prefill emits no token and a future batched step
may emit more than one, and a caller reading a length is a caller that does not
have to change when either happens.

**It is valid until the next `Next`.** The stream reuses the backing array, for
the same reason [017-D3](017-benchmarks.md) gives about instruments: a
per-token allocation in the decode loop is a cost the measurement imposes on
what it measures. A caller keeping them appends.

## 3. Which distribution, and the answer that is not obvious

**030-D2: the post-policy distribution — the one the token was actually drawn
from.**

[006 §3](006-sampling.md) composes bias, penalties, temperature, then top-*k*
and top-*p*. `Probs` returns the distribution after all of it, normalized over
what survived. So for a policy with $\mathcal{K}$ the kept set:

$$
p(v) = \frac{\exp\!\left((\ell_v + b_v)'/\tau\right)}{\sum_{u \in \mathcal{K}} \exp\!\left((\ell_u + b_u)'/\tau\right)}
\quad v \in \mathcal{K},
\qquad p(v) = 0 \quad v \notin \mathcal{K}
$$

with $(\cdot)'$ the penalised logit. The alternative — reporting the raw softmax
over the untruncated vocabulary — describes a distribution **nothing sampled
from**, and would tell a caller a token had a 3% chance when top-*k* had already
given it zero.

**A masked token's logprob is $-\infty$**, and that is the honest value rather
than a floor. Three things reach it: a token outside the top-*k* or the nucleus,
a token a grammar masked ([015-D2](015-structured-output.md) adds $-\infty$
before the penalties), and every non-argmax token under `Temperature: 0`, where
the distribution is one at the argmax. It is never the *drawn* token's value,
because the draw walks the kept set.

**§4 says what the wire does with it**, since JSON has no infinity.

## 4. The wire, and the loss table

`logprobs` comes off the loss table **for the routes that serve it and no
other**: `server/loss.go`'s `honoured` set names it, and `honouredHere` lists it
under `dialectLegacy` and `ir.DialectOpenAIChat`. A field reported as a loss
where it is served is worse than one that was never claimed; a field
subtracted where it is *not* served is [009-D12](009-server.md)'s defect,
which is why the subtraction is per dialect.

**Two routes can serve them, through two codecs.** `/v1/completions` is tgo's
own `Frontend` ([009 §3.3](009-server.md)). `/v1/chat/completions` is
`latere.ai/x/pkg/llmdialect`'s, whose `ir` carries `Request.LogProbs` and
`TopLogProbs`, `Response.LogProbs` and `Event.LogProbs` on a text delta. The
handler converts the engine's `TokenProb` to `ir.TokenLogProb` once, sets it on
the response or the event, and every `Frontend` reads it from there. The
Anthropic and Responses surfaces have no member for one, so the ask stays a
loss there whatever tgo could compute.

| route | encoder | shape |
| --- | --- | --- |
| `/v1/completions` | tgo's | `choices[].logprobs` with the parallel `tokens`, `token_logprobs` and `top_logprobs` arrays it has always declared and answered `null` for |
| `/v1/chat/completions` | `llmdialect` | `choices[].logprobs.content[]` with `token`, `logprob`, `bytes` and `top_logprobs`, on the body and on each streamed delta |
| `/v1/messages` | `llmdialect` | Anthropic's shape carries none; a loss |
| `/v1/responses` | `llmdialect` | the ask reaches the IR, and the encoder tgo answers through has no member; a loss the handler adds |

**030-D5 was the gap, and 030-D6 is what replaced it.** While the `ir` had no
logprobs shape, `/v1/chat/completions` kept the loss entry and the gap was
reported upstream rather than worked around by appending a member to a body
`llmdialect` encoded, which is what [009-D10](009-server.md) exists to prevent.
The field landed in `llmdialect`, and this route now serves through it.

**$-\infty$ encodes as JSON `null`.** Not as a large negative number, which a
consumer would average; not as the string `"-Infinity"`, which is not JSON.
`null` is what "this token could not be drawn" means, and it is what the field
already answers today for every token.

## 5. The cost, and why it is off by default

`Probs` copies the logits and runs the whole policy a second time: a
vocabulary-length `exp` pass, over $V = 151936$, per step. That is real next to
a decode step and it buys nothing for a caller who did not ask, so
`Policy.LogProbs` is false by default and the work is skipped entirely.

`Top` costs one selection of `TopLogProbs` over the same weights, which is
[006](006-sampling.md)'s `topN` and is not a second sort.

**It runs on the same logits the draw used, before `Next` consumes them.** The
stages write in place, so the order in `Stream.advance` is: grammar mask, then
`Probs` on a copy, then `Next`. Taking it after the draw would read logits the
policy has already rewritten, and taking it before the mask would report a
distribution the grammar had not yet cut.

## 6. Tests

| test | what it catches |
| --- | --- |
| the drawn token's `LogProb` is $\ln$ of its post-policy probability, against a float64 oracle | §3, and the difference between the two distributions |
| the reported top-*k* weights sum to one | §3's normalization over the kept set; the untruncated softmax sums to less |
| a token top-*k* excluded reports $-\infty$, and the drawn token never does | §3's masked case |
| under `Temperature: 0` the argmax reports $\ln 1 = 0$ and any other token $-\infty$ | the greedy branch, which is where a raw-softmax implementation would disagree most visibly |
| a grammar-masked token reports $-\infty$ | §5's ordering: taking `Probs` before the mask would report a positive probability for a token the grammar forbids |
| `Top` is descending, has length `TopLogProbs`, and its entries have nil `Top` | §2 |
| `LogProbs()` is empty when `Policy.LogProbs` is false, and `Probs` is not called | §5 |
| the same seed produces the same completion with and without `LogProbs` | [006-D7](006-sampling.md): an observation that perturbed what it describes would not be describing it |
| `logprobs` is absent from `X-Tgo-Loss` on `/v1/completions` and present on the other three | §4, and the per-dialect table; subtracting it everywhere is 009-D12's defect |
| a $-\infty$ encodes as `null` in the `token_logprobs` array | §4 |
| a request with no `logprobs` still answers `logprobs: null`, not an empty object | the shape the route has always declared |

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 030-D1 | a side accessor on `Stream` | a field on `Event`; a new `EventKind` | `Event` keeps meaning one thing, and 009 §3.2's kind-to-block mapping stays one-to-one |
| 030-D2 | report the **post-policy** distribution | the raw softmax over the untruncated vocabulary | the number describes the distribution the token was drawn from; a raw softmax describes one nothing sampled |
| 030-D3 | $-\infty$ for a masked token, `null` on the wire | a floor, or omitting the entry | a floor is a number a consumer averages; omitting breaks the per-token parallel arrays the legacy shape declares |
| 030-D5 | the three `llmdialect` routes keep the loss and the gap is **reported** | append the member to the encoded body from outside the `Frontend` | superseded by 030-D6 once the field landed; while it held, tgo never knew a dialect's JSON ([§4](#4-the-wire-and-the-loss-table)) |
| 030-D6 | logprobs travel on `ir.Response` and `ir.Event`, and every `Frontend` reads them there | a side interface only tgo's codec implements | one conversion in the handler, no optional interfaces, and `/v1/chat/completions` serves what `/v1/completions` does ([§4](#4-the-wire-and-the-loss-table)) |
| 030-D4 | off by default, and the pass is skipped | always compute and let the caller ignore it | a whole-vocabulary `exp` per step is what [017-D3](017-benchmarks.md) calls an instrument that changes what it measures |

## Outcome

Built and running as of 2026-08-28. `Policy.LogProbs` and `Policy.TopLogProbs`
ask for them, `Stream.LogProbs()` returns the last step's, and
`/v1/completions` serves them whole-body and streaming.

**What shipped**, section by section:

| section | what landed | where |
| --- | --- | --- |
| 2 | `TokenProb`, `Stream.LogProbs()`, and the reused backing array | `stream.go:65,464`, `stream.go:130` |
| 3 | `Probs` on a copy of the masked logits, before the draw | `stream.go:281`, `logprobs_test.go:22,63` |
| 4 | the parallel-array shape with `text_offset`, and the loss subtracted on `dialectLegacy` alone | `server/legacy.go:198,222`, `server/loss.go:104` |
| 5 | skipped entirely when off, and asked for only on the route that can encode it | `stream.go:281`, `server/adapt.go:288` |
| 6 | every row, plus the two the ordering needed | `logprobs_test.go`, `sample/stages_test.go`, `server/legacy_test.go` |

**What diverged** from the design, and why the code is right:

- **§4 named two routes and there is one.** The draft said
  `/v1/chat/completions` and `/v1/completions`. `latere.ai/x/pkg/llmdialect`'s
  `ir` carries no logprobs shape at all — not on `ir.Response`, not on
  `ir.Event` — so the three dialects it encodes cannot express one whatever tgo
  computes. Corrected before a line was written, and it became
  [030-D5](#decision-record).
- **The engine is not asked for work no encoder can carry.** `mapPolicy` sets
  `Policy.LogProbs` only for `dialectLegacy`, so a `logprobs` on the other three
  routes costs nothing and is reported as a loss. Computing them there would
  run a whole-vocabulary `exp` per step and throw the answer away, which is what
  §5 exists to avoid.
- **`TopLogProbs`'s wire name is `logprobs`, not `top_logprobs`.** The one route
  that serves them spells the count in `logprobs` itself. `top_logprobs` reaches
  no output and stays a loss everywhere, and `server/loss.go`'s per-dialect
  table says so — `honoured` maps both `Policy` fields to the same member.
- **Two tests in §6's table did not discriminate, and two more were added.** The
  sampled cases pass whether `Probs` runs before the draw or after it: the
  distribution is normalized over the kept set either way. What catches the
  ordering is the grammar-masked case (`logprobs_test.go:200`), where taking
  `Probs` before the mask reports a chance for a token that cannot be drawn —
  and `sample/stages_test.go`'s `TestNextRewritesTheRowItWasGiven`, which states
  why the order matters at all and fails if `Next` ever stops writing in place.

**Second route, 2026-09-02.** `llmdialect` gained the logprobs shape on its
IR, so `/v1/chat/completions` now serves them whole-body and streaming
([030-D6](#decision-record)). The side interfaces tgo's codec alone
implemented are gone: `server/generate.go` converts once to `ir.TokenLogProb`,
`server/legacy.go` reads `resp.LogProbs` and `ev.LogProbs` like the dialect
encoders do, and `server/loss.go` honours `logprobs` and `top_logprobs` on both
routes while adding the loss on the Responses route, whose frontend reads
`top_logprobs` into the IR without filing one. `/v1/messages` and
`/v1/responses` keep the loss entry, because their wire has no member for a
logprob. `server/chat_logprobs_test.go` pins the shape, the stream, and the
null answer for a caller that did not ask.

**Not built.** Nothing this spec designs. The two routes §4 named both serve
logprobs. `/v1/messages` and `/v1/responses` report a loss because their wire
has no member for one, which is a dialect's shape and not tgo's work.

**Filed as [latere-ai/pkg#7](https://github.com/latere-ai/pkg/issues/7)** on
2026-08-28, with the shape that would close it and the two details that bit tgo:
that a masked token's $-\infty$ needs a JSON answer, and that the number has to
say which distribution it is of. The loss report is what tells a caller in the
meantime.
