# tgo documentation

**Audience: people running open-weight models with tgo.** Design documents for
contributors live in [`../specs/`](../specs/) and are written for a different
reader — they argue about tradeoffs, and this does not.

> [!NOTE]
> **tgo does not run yet.** The pages that would tell you how to install it,
> pull a model and serve it are missing because the commands they describe do
> not exist. Writing them now would give you instructions nobody could check.
>
> [Orientation](orientation.md) is here today, because what tgo is and what it
> costs to run are true regardless of how much of it is built.

## Now

- **[Orientation](orientation.md)** — what tgo is, what runs where, and how it
  relates to accel.

## With the code

| page | what it will cover |
| --- | --- |
| Quickstart | install, fetch a model, generate your first tokens |
| Models | which models run, choosing precision, how much memory each needs |
| Serving | the HTTP API, streaming, running it behind something |
| Performance | what to expect, and how to measure it on your own machine |
| Troubleshooting | when a model will not load, or the output looks wrong |

Each arrives with the code it describes.
