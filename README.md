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

tgo runs open-weight language models from Go. It builds to **one static binary**
with `CGO_ENABLED=0` — no C++ runtime, no Python, no vendor SDK, and nothing to
install beside it. You cross-compile it the way you cross-compile any Go
program.

> [!IMPORTANT]
> **tgo does not run yet.** The design is complete and reviewable; the code is
> not written. There is nothing to install and nothing to try.
>
> This page describes what tgo will do. [`docs/`](docs/) explains how it fits
> together, and [`specs/`](specs/) is the design itself.

## Why you might want it

If you already run Python happily, [vLLM](https://github.com/vllm-project/vllm)
is faster and far more complete today, and you should use it.

tgo is for the case where that is awkward: shipping a model inside a Go service,
running on a machine you cannot install a toolchain on, or cross-compiling a
binary for a platform you do not build on. One file, no runtime, no version
matrix.

## What it will do

| | |
| --- | --- |
| **Models** | Qwen3 dense, from Hugging Face safetensors |
| **Precision** | f16 or int8, chosen by what fits your machine, and always overridable |
| **Devices** | CPU everywhere, Metal on Apple silicon; more as they arrive |
| **APIs** | OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses, so most clients work unchanged |
| **Serving** | streaming, logprobs, seeded reproducible output, and prompt caching that reuses work across turns and requests |
| **As a library** | open a model, hold a conversation, stream tokens |

Continuous batching — running many conversations in one step — is designed and
waiting on work in the layer below. Until it lands, tgo serves one conversation
at a time well rather than many badly, and says so in its metrics.

## What using it will look like

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

Or as a server, which speaks three APIs on the same model:

```sh
tgo serve ./Qwen3-4B
```

## How it is built, and why that matters to you

tgo does the model; [accel](https://github.com/golang-design/accel) does the
GPU. tgo contains no GPU code at all — when it needs something accel cannot do,
it reports the gap upstream and waits rather than working around it.

That is worth knowing for two reasons. It is why tgo gains a backend the moment
accel does, without changes. And it is why the status above is honest: a limit
you meet in tgo is a real limit, written down with the reason, rather than a
sharp edge nobody mapped.

## Documentation

- **[Orientation](docs/orientation.md)** — what tgo is, what runs where, and what
  it costs in memory. Written for people running models.
- **[docs/](docs/)** — the index. Quickstart, model and serving guides arrive
  with the code they describe.
- **[specs/](specs/)** — the design, written for contributors: what was decided,
  what was rejected, and why.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — start here to work on tgo.

## License

Apache 2.0. See [LICENSE](LICENSE).
