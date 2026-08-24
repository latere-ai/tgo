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
> is written and reviewable; there is no working code yet. v0 is also **gated
> upstream**: `accel`'s attention operator refuses a KV cache longer than 128
> positions, which is shorter than a system prompt. That is
> [accel#8](https://github.com/golang-design/accel/issues/8), it has no
> workaround, and it is the single thing between here and serving a model.
> [`specs/011-sequencing.md`](specs/011-sequencing.md) is where things actually
> stand.

## Why this exists

Two reasons, and the second is the load-bearing one.

**Go should be able to run a model.** Today that means shelling out to a C++
process or binding to one. tgo is the same job in one static binary.

**tgo is accel's validating consumer.** accel is a young library, and the way a
young library finds out which of its abstractions survive is that something real
tries to use them. So tgo writes **no kernels and no device code**. When accel
cannot express something, tgo does not route around it — it files the gap, keeps
a named failing test, and waits.

That is not a policy for its own sake. In the first day of design it produced
**nine issues** on accel, five of which turned out to be
[one decision seen five times](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md):
*a scalar is a value every row of a dispatch shares; a value that differs per
row is a tensor.* No single one of those five looked like a design decision from
inside accel. Together they were one, and the fix removed API surface rather
than adding it.

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
| Paged KV, continuous batching | vLLM's contribution | blocked on accel |
| Prefix caching | reuse the KV of a shared system prompt or an earlier turn | designed |
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
