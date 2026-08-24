# tgo documentation

**Audience: people running open-weight models with tgo.** Design documents for
contributors live in [`../specs/`](../specs/) and are written for a different
reader — they argue about tradeoffs, and this does not.

> [!NOTE]
> **tgo is not implemented yet.** The pages that would tell you how to install
> it, pull a model and serve it do not exist, because the commands they would
> describe do not exist. Writing them now would produce instructions that are
> wrong in ways nobody could check.
>
> [`orientation.md`](orientation.md) is here today, because what tgo *is* and
> how it fits together is true regardless of how much of it runs.
> [`../specs/011-sequencing.md`](../specs/011-sequencing.md) says what is
> finished.

## Now

- **[Orientation](orientation.md)** — what tgo is, what runs where, and how it
  relates to accel.

## With the code

| page | arrives at |
| --- | --- |
| Quickstart — install, pull a model, generate | M8 |
| Models — what is supported, precision, memory | M8 |
| Serving — the HTTP API, streaming, concurrency | M9 |
| Performance — what to expect and how to measure it | M10 |
| Troubleshooting | M10 |

The milestones are in
[`../specs/011-sequencing.md`](../specs/011-sequencing.md).
