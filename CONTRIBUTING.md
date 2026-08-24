# Contributing

Thank you for looking. This project has one unusual rule and a few ordinary
ones; the unusual one is first because it decides whether a patch is wanted.

## The rule: tgo does not write kernels

tgo is [accel](https://github.com/golang-design/accel)'s validating consumer.
Every operation that touches a device goes through accel's tensor layer, and
tgo contains **no kernels, no backend code, and no device-conditional
numerics.**

When accel cannot express something, the sequence is fixed:

1. write the test that fails, or the operation you cannot record;
2. file it on [accel](https://github.com/golang-design/accel/issues), citing the
   spec that owns it, with the cost measured rather than asserted;
3. add a row to [`specs/010-conformance.md`](specs/010-conformance.md) §2 and a
   named test that skips with the reason;
4. wait.

**A patch that works around a missing accel operator with private device code
will be turned down however good it is.** The gap it hides is the output this
project exists to produce. See
[`specs/000-decisions.md`](specs/000-decisions.md) decision 1.

This has already paid: nine reports in the first day of design, five of which
were [one decision seen five times](https://github.com/golang-design/accel/blob/main/specs/043-per-row-values.md).

## Specs come before code

Nothing is implemented before the spec that owns it is written and its decisions
recorded. [`specs/README.md`](specs/README.md) has the lifecycle, the
frontmatter shape, and the decision-record format.

If you are changing behaviour, change the spec in the same series of commits —
amending the decision record **in place**, with the new reasoning, rather than
deleting the old row. The value of a decision record is that a later reader can
see what was considered.

`go test ./internal/speclint/` checks the tree: frontmatter, dependency edges,
acyclicity, blocked specs naming what blocks them, a decision record in every
spec, an index that cannot go stale, and no citation of a register row that does
not exist.

## Tests

- **Every bug fix has a test that fails without the fix.** No exceptions.
- **No test downloads weights.** CI runs on synthetic configurations — 2 layers,
  hidden size 64, vocab 128, seeded weights. Real weights run behind `TGO_MODEL`
  and never in CI ([000 D8](specs/000-decisions.md)).
- **Tolerances are derived, and carry a comment naming the term that produced
  them.** A tolerance raised to make a test pass is a finding, not a fix
  ([010-D3](specs/010-conformance.md)).
- **A linter or checker must be negative-tested.** One that passes vacuously is
  the failure it exists to catch.

The coverage floor is 90% per package, not per repository — an average lets a
well-tested package carry an untested one. Exemptions live in
`internal/covercheck` with a reason attached.

## Running the gates locally

```sh
go build ./...
go vet ./...
go test ./...
CGO_ENABLED=1 go test -race ./...
gofmt -l .
go test ./internal/speclint/
go test -coverprofile=cover.out -coverpkg=./... ./... && go run ./internal/covercheck -profile=cover.out
```

CI runs all of these plus a cgo-free grep and a cross-compile across ten
GOOS/GOARCH pairs.

## Dependencies

tgo's core is stdlib, `golang.design/x/accel`, and
`golang.org/x/text/unicode/norm` for Unicode NFC — which the tokenizer cannot be
correct without and the standard library does not provide
([002-D10](specs/002-tokenizer.md)). `tgo/server` adds
`latere.ai/x/pkg/llmdialect` and nothing else.

That module carries a large dependency set of its own, and none of it reaches
tgo — Go's module graph pruning keeps a consumer's build list to llmdialect's
stdlib-only subtree ([009 §2.1](specs/009-server.md)). **This holds because of
what llmdialect currently imports, not because of a guarantee**, so from M9 CI
checks the non-stdlib build list against an allowlist.

Adding a dependency means editing that allowlist, in the same commit, with the
reason. A `go.sum` that grows without one is the review comment.

## Commits

One logical change per commit, staged explicitly — not `git add -A`. The message
says **why**, not what; the diff already says what. Where a change reverses an
earlier decision, say which and what changed your mind.

## Style

- Comments explain the reasoning a reader cannot recover from the code. Prefer
  one paragraph on why an approach was chosen over three lines restating it.
- User-facing text, package docs and specs each address a different reader.
  Documentation aims at value and usage, API docs at precision, code comments at
  technical precision.
- No em dashes in prose. Write plainly.
