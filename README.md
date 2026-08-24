<h1 align="center">tgo</h1>

<p align="center">
  <strong>Run open-weight LLMs from Go. No cgo, no Python, no vendor runtime.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/go-1.27+-00ADD8.svg" alt="Go 1.27+">
  <img src="https://img.shields.io/badge/cgo-free-success.svg" alt="cgo-free">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/status-design-orange.svg" alt="Status: design">
</p>

---

tgo is an inference framework built on
[accel](https://github.com/golang-design/accel), which runs compute on the GPU
from Go with `CGO_ENABLED=0`. One binary, cross-compiled anywhere, with no
runtime to install beside it.

> [!IMPORTANT]
> **Design complete. Nothing is implemented.** Every spec in [`specs/`](specs/)
> is written and reviewable; there is no working code yet.
>
> v0 *was* blocked upstream — accel's attention refused a KV cache longer than
> 128 positions, shorter than a system prompt. tgo filed it, wrote the design,
> and accel shipped it.
>
> **Nothing between here and serving a model is waiting on accel**, and that is
> measured rather than asserted: the whole Qwen3-4B graph — 36 layers, a 151936
> vocabulary, a 4096-position cache — compiles against accel today, as four
> plans (prefill and decode, f16 and int8) of up to 730 nodes and 1013 kernel
> selections. Prefix caching is expressible too. The only thing still blocked
> upstream is continuous batching, which is post-v0. What is *not* proven is
> that the numbers would be right; that needs the parity oracle, which needs
> code. [`specs/011-sequencing.md`](specs/011-sequencing.md) has the table.

## Why this exists

Two reasons, and the second is the load-bearing one.

**Go should be able to run a model.** Today that means shelling out to a C++
process or binding to one. tgo is the same job in one static binary.

**tgo is accel's validating consumer.** accel is a young library, and the way a
young library finds out which of its abstractions survive is that something real
tries to use them. So tgo writes **no kernels and no device code**. When accel
cannot express something, tgo does not route around it — it files the gap, keeps
a named failing test, and waits.

That is not a policy for its own sake. In the first days of design it produced
**ten issues** on accel. Five turned out to be
[one decision seen five times](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md):
*a scalar is a value every row of a dispatch shares; a value that differs per
row is a tensor.* No single one of those five looked like a design decision from
inside accel. Together they were one, and the fix removed API surface rather
than adding it.

A sixth — attention capped at 128 positions, which made no model servable — tgo
filed, designed, and accel implemented. The tenth is the interesting one:
`Attention` *accepted* a page table on a prefill and silently ignored it. Every
finding before it was a refusal; that one returned a fluent wrong answer, and it
took a probe asserting a value rather than reading an error to see it. It is
fixed, and it is why tgo's register is decided by measurement.

**Ten of the eleven issues are now closed, and six register rows are still
open.** That is not a complaint: each fix matched its issue's title, and four of
those titles named a symptom rather than the cost. It is the reason the register
records *capabilities* rather than tickets — a title is a summary, and only a
capability is testable.

The register of what accel cannot do yet — with the arithmetic for each — is
[`specs/010-conformance.md`](specs/010-conformance.md). **It is this project's
primary output.**

## What is planned

Lessons taken deliberately from [ollama](https://github.com/ollama/ollama),
[vLLM](https://github.com/vllm-project/vllm) and
[sglang](https://github.com/sgl-project/sglang):

| | | status |
| --- | --- | --- |
| Qwen3 dense, from safetensors | f16 or int8, chosen by what fits | designed |
| Byte-level BPE, streaming decode | pure Go, no `tokenizers` dependency | designed |
| Chat templates | per model, with user text that cannot forge a turn | designed |
| OpenAI, Anthropic and Responses APIs | three wire dialects, one adapter, via `llmdialect` | designed |
| Paged KV cache | vLLM's contribution | designed |
| Continuous batching | many sequences per step | blocked on accel |
| Prefix caching | reuse an earlier turn's KV, and a system prompt across requests | designed |
| Constrained decoding | a JSON schema compiled to a token mask | designed, after batching |
| GGUF | needs a super-block kernel accel does not register | blocked |

## The shape it will have

Designed in [`specs/007-engine.md`](specs/007-engine.md). **This does not run
yet.**

```go
m, err := tgo.Open("./Qwen3-4B", tgo.WithPrecision(tgo.Int8))
if err != nil {
	log.Fatal(err)
}
defer m.Close()

s, _ := m.NewSession()
defer s.Close()

stream, _ := s.Chat(ctx, []chat.Message{
	{Role: chat.User, Blocks: []chat.Block{{Type: chat.BlockText, Text: "Why is the sky blue?"}}},
}, tgo.Policy{Temperature: 0.7, TopP: 0.8, MaxTokens: 512})

for stream.Next() {
	fmt.Print(stream.Text())
}
if err := stream.Err(); err != nil {
	log.Fatal(err)
}
```

## Reading it

| you are | start at |
| --- | --- |
| evaluating the design | [`specs/000-decisions.md`](specs/000-decisions.md) — ten decisions, each with what was rejected |
| wondering what works | [`specs/011-sequencing.md`](specs/011-sequencing.md) |
| interested in the accel side | [`specs/010-conformance.md`](specs/010-conformance.md) |
| going to contribute | [`specs/README.md`](specs/README.md), then [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| going to use it | [`docs/`](docs/) — an index and an orientation page today; guides arrive with the code |

## The one rule

tgo does not write kernels. A patch that works around a missing accel operator
with private device code will be turned down however good it is, because the gap
it hides is the output this project exists to produce.

The path for a missing operator is: a test that names it, a row in
[`specs/010-conformance.md`](specs/010-conformance.md), and an issue on
[accel](https://github.com/golang-design/accel) citing the spec that owns it.

## License

Apache 2.0. See [LICENSE](LICENSE).
