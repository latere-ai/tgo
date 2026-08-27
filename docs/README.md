# tgo documentation

**Audience: people running open-weight models with tgo.** Design documents for
contributors live in [`../specs/`](../specs/) and are written for a different
reader — they argue about tradeoffs, and this does not.

> [!NOTE]
> **tgo runs, and most of these pages are not written yet.** `tgo pull`,
> `tgo run` and `tgo serve` all work — the [README](../README.md) shows each one
> as a command you can run. Until the guides below arrive, that page and
> [Orientation](orientation.md) are what exist, and `tgo <command> --help` is
> accurate because it comes from the code.

## Now

- **[Orientation](orientation.md)** — what tgo is, what runs where, what it
  costs in memory, and how it relates to accel.
- **[README](../README.md)** — installing it, pulling a model, generating, and
  serving.

## Still to write

| page | what it will cover |
| --- | --- |
| Quickstart | install, fetch a model, generate your first tokens |
| Models | which models run, choosing precision, how much memory each needs |
| Serving | the HTTP API, streaming, running it behind something |
| Performance | what to expect, and how to measure it on your own machine |
| Troubleshooting | when a model will not load, or the output looks wrong |

Each describes behaviour that already exists, so each can be checked against the
code as it is written.
