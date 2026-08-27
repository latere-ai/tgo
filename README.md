<h1 align="center">tgo</h1>

<p align="center">
  <strong>Run open-weight LLMs from Go. No cgo, no Python, no vendor runtime.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/go-1.27+-00ADD8.svg" alt="Go 1.27+">
  <img src="https://img.shields.io/badge/cgo-free-success.svg" alt="cgo-free">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/status-early-orange.svg" alt="Status: early">
</p>

---

tgo runs open-weight language models from Go. It builds to **one static binary**
with `CGO_ENABLED=0` — no C++ runtime, no Python, no vendor SDK, and nothing to
install beside it. You cross-compile it the way you cross-compile any Go
program.

> [!IMPORTANT]
> **Early, and it works.** tgo loads a real Qwen3 checkpoint and generates text
> on Apple silicon today — about 18 tokens a second on a 0.6B model, with a
> 170ms wait for the first one.
>
> One caveat worth knowing before you plan around it. `tgo serve` holds many
> conversations at once but still gives each request its own forward pass, so
> its total throughput is close to what one conversation gets. The engine below
> it does batch — see the table — and connecting the two is the next thing.
> [`docs/orientation.md`](docs/orientation.md) explains what runs where.

## Why you might want it

**Deployment.** One file. Ship a model inside a Go service, run it on a machine
you cannot install a toolchain on, cross-compile it for a platform you do not
build on. No runtime, no version matrix, no container to keep in step with a
driver.

**Speed, and this is the goal rather than a claim.** tgo aims to be **faster
than vLLM**, not to trade speed for convenience. The parts of serving that are
not matrix multiplication — scheduling a step, sampling a token, turning it back
into text, deciding what runs next — are pure overhead on every token, and they
are where a compiled language with no interpreter and no global lock should win.
Starting up is the same story: tgo builds its compute plan in milliseconds
rather than loading a Python stack.

Today [vLLM](https://github.com/vllm-project/vllm) is faster, because tgo does
not run yet. When it does, the honest position will be a table of measurements
rather than a claim, and we will publish the ones we lose.

**Hardware.** vLLM serves NVIDIA extremely well and other hardware less so. tgo
runs wherever its compute layer runs, which today means CPU everywhere and Metal
on Apple silicon.

## What it will do

| | |
| --- | --- |
| **Models** | Qwen3 dense — 0.6B through 32B — from Hugging Face safetensors. The hybrid-attention models (Qwen3.5, Qwen3.8) need work in the layer below first |
| **Precision** | f16 or int8, chosen by what fits your machine, and always overridable |
| **Devices** | Metal on Apple silicon, and CPU everywhere. Use a GPU if you have one: the CPU path works and is currently far slower |
| **APIs** | OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses, so most clients work unchanged |
| **Serving** | streaming, logprobs, seeded reproducible output, and JSON-schema output that parses every time |
| **Reuse** | a conversation's next turn prefills only what is new, and with `--prefix-cache process` two conversations sharing a system prompt prefill it once between them; `cache_salt` bounds who shares with whom |
| **As a library** | open a model, hold a conversation, stream tokens, reuse the prompt a conversation has already paid for, and run several conversations in one forward pass |

**Continuous batching runs, and `tgo serve` does not use it yet.** The engine
puts several conversations in one forward pass, and puts a long prompt's next
chunk in that same pass rather than making everyone wait for it — so the weights
are read once for all of them, which is where a server gets most of its
throughput. Reach it with `Model.NewScheduler`. The server still gives each
request its own conversation slot, so its throughput is close to what one
conversation gets; wiring the two together is the next thing.

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
tgo pull Qwen/Qwen3-0.6B     # fetch a checkpoint into the cache
tgo serve ./Qwen3-0.6B       # then serve it
```

It answers OpenAI Chat Completions, Anthropic Messages and OpenAI Responses on
the same model, streaming, so most clients work unchanged.

`--sessions N` sets how many conversations it holds at once, and
`--prefix-cache` lets a conversation's next turn skip the transcript it already
paid for. `--prefix-cache process` goes further and shares that state *between*
conversations, so a fleet of agents on one system prompt pays for it once — for
the same memory, because the pool replaces the per-session caches rather than
adding to them. Both cost memory that is reserved at startup and held for the
life of the process; `tgo serve` prints the arithmetic before it listens.
[Session pooling](docs/orientation.md#session-pooling-and-what-it-costs) has the
numbers.

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
