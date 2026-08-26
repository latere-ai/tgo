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
- **A test that drives a fake through a channel must keep driving it until the
  thing it waits for happens**, not until the thing it watches *appears*. A
  generator blocked on a gate between tokens only reaches its cancellation check
  after taking one, so a loop that stops nudging early parks it forever and then
  reports the timeout as a behaviour failure. This failed only under `-race`,
  where the timing is slow enough to lose a race the test did not know it had.
  Yield in such a loop, too — a hot spin starves the goroutine under test.
- **Do not `Sleep` to reach a state.** "Give it time to get to the queue" is a
  guess that holds on a fast machine and fails under `-race` on a loaded CI
  runner — and it fails as a *behaviour* failure, blaming the code for a state
  the test never reached. Wait for the state itself: poll the gauge, the
  counter, the channel. Every timing bug this project has hit has been one of
  these.
- **Do not assert that a duration is positive.** Windows' timer granularity is
  about 15ms, so a real interval of a few hundred microseconds measures as
  exactly zero, and every rate derived from it is zero. Five tests passed on
  macOS and Linux and failed on Windows for this reason. Assert the
  *observation count* — a term recorded on every step, a wall clock that
  advanced — and where a fixture needs measurable time to pass, spend it
  deliberately.
- **No two dimensions in a test fixture may be equal.** A config where the layer
  count equals the key/value head count, or the vocabulary equals the
  intermediate size, is the *identity* for every confusion between them, so a
  wrong shape reads as correct. This has cost three waves: twelve surviving
  mutants in `model`, the whole f16 permutation path in `weights` (where
  `head_dim = 2` makes the rotary permutation the identity), and a
  layer/kv-head swap in `cmd/tgo`. Where a collision is unavoidable, say so in
  the fixture's comment and name what it cannot discriminate.

The coverage floor is 90% per package, not per repository — an average lets a
well-tested package carry an untested one. Exemptions live in
`internal/covercheck` with a reason attached.

## Running the gates locally

Cheapest first, which is the order CI runs them in and the reason it is worth
copying. A list that puts a one-second check behind a nine-minute one is a list
whose tail gets run and whose head gets skipped, and `gofmt` sat fifth here
until it failed a build that every other gate had passed.

```sh
gofmt -l .
go build ./...
go vet ./...
go test ./internal/speclint/
go test ./...
go test -coverprofile=cover.out -coverpkg=./... ./... && go run ./internal/covercheck -profile=cover.out
CGO_ENABLED=1 go test -race -count=1 -timeout 45m ./...
```

CI runs all of these plus a cgo-free grep and a cross-compile across ten
GOOS/GOARCH pairs.

**Before you push, build a clean clone.** Every gate above reads your working
tree; CI reads what you committed. A file you forgot to stage passes locally and
fails everywhere else:

```sh
T=$(mktemp -d) && git clone -q . $T/tgo && (cd $T/tgo && go build ./... && go test ./...)
```

This has cost a red build once already — `go.mod` was left out of a commit that
staged its packages by name, so nine green packages built against a module file
CI did not have.

**A green local `-race` run does not mean a green CI one.** The root package
runs a synthetic model's forward pass on the CPU and the race detector costs
about an order of magnitude on that arithmetic, so the binding number is CPU
time, not wall time: your machine divides it by its cores and a CI runner has
fewer and slower ones. Measure it rather than trusting the wall clock:

```sh
/usr/bin/time -p go test -race -count=1 -timeout 45m ./...
```

`user` is the number, and it is a **budget rather than a warning line**, because
a ceiling that rises whenever a wave needs it stops being a gate. What it has
actually been:

| | `user` | ceiling | why it moved |
| --- | --- | --- | --- |
| Wave 7 | 1297s | 10m, then 30m | the ten-minute default was never raised from `go test`'s |
| Wave 8 | 2547s | 45m | one shared block pool exercised by real forward passes in two packages |

**The budget is 3500s.** Past it, the answer is to make the suite cheaper and
not the ceiling higher: no single test above is large — the heaviest is 22s and
the top twelve are 155s between them — so the growth is a suite that does more
each wave, and the lever is fixtures sized to their assertions rather than to
what was convenient.

This has cost a red build once: the suite passed locally at 362s wall and timed
out on ubuntu and windows on no single test, because `go test` reports whichever
test was running when the clock ran out. The package had simply run out of the
ten minutes `go test` allows by default.

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
