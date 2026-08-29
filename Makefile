GO ?= go

# CGO_ENABLED=0 is the default for every target here. specs/000-decisions.md
# decision 2 makes cgo-free the point rather than an optimization, so the build
# that must pass is the one without it. test-race turns it back on, because the
# race detector needs cgo on most platforms and the guarantee is about what tgo
# compiles to, not about what its test harness may use to find a race.
export CGO_ENABLED = 0

.PHONY: build test cover test-hermetic test-race fmt fmt-check lint-modernize spec-lint validate dist fuzz cgo-free deps

build:
	$(GO) build ./...

test:
	$(GO) vet ./...
	$(GO) test ./...

# Per package, not one repository average: an average lets a well-tested
# package carry an untested one and reports a number nobody can act on. The
# floor and the exemptions, each with its reason, live in .lateregate.yaml.
cover:
	$(GO) test -coverprofile=cover.out -coverpkg=./... ./...
	@$(GO) tool lateregate cover -profile=cover.out

# The suite with only the Go toolchain on PATH. A test that depends on what
# happens to be installed passes locally and fails on a runner, which is the
# worst order to find out.
test-hermetic:
	@$(GO) tool lateregate hermetic

# The timeout is a budget, not a guard against a hung test. The root package
# runs a synthetic model's forward pass on the CPU and the race detector costs
# roughly an order of magnitude on exactly that arithmetic: 2547s of CPU as of
# 2026-08-27, with a shared block pool exercised by real forward passes in two
# packages. CONTRIBUTING tracks that as a 3500s budget rather than a line to
# move again -- past it the answer is a cheaper suite, because no single test
# here is large and the growth is the suite doing more each wave.
test-race:
	CGO_ENABLED=1 $(GO) test -race -timeout 45m ./...

fmt:
	gofmt -w .

fmt-check:
	@$(GO) tool lateregate fmt-check

lint-modernize:
	@$(GO) tool lateregate modernize

# specs/README.md documents a lifecycle and a frontmatter shape. A spec tree
# nobody checks drifts from the code within a milestone. The vocabulary, the
# status-to-field rules, the required sections and the id scoping are all in
# .lateregate.yaml.
spec-lint:
	@$(GO) tool lateregate spec-lint

# The three gates that defend a promise tgo makes and most repos do not. Run
# as one job; each target keeps its own name, so a failure says which promise
# broke.
validate: deps cgo-free fuzz

# specs/009-server.md 009-D14. The server carries three wire dialects because
# llmdialect's subtree is stdlib-only, which is a property of its current
# imports rather than a promise. One upstream import would arrive on the next
# `go get`, and without this the first symptom is a slower build.
deps:
	@$(GO) tool lateregate depcheck

# Greps rather than relying on the build: a file can import "C" behind a build
# tag this platform does not select, and still be a violation.
cgo-free:
	@if grep -rn --include='*.go' '^import "C"\|^\s*"C"$$' . ; then \
		echo "found cgo usage; specs/000-decisions.md decision 2 makes cgo-free a hard requirement"; \
		exit 1; \
	fi; \
	echo "no cgo found"

# The seed corpus only, not a fuzzing campaign: a regression gate over the
# inputs that have already found a bug, and it must stay fast.
fuzz:
	$(GO) test -run='Fuzz' ./...

# The README tells a reader they can cross-compile with CGO_ENABLED=0. A claim
# in a user-facing document with no gate behind it goes stale silently.
dist:
	@set -e; for pair in \
	  linux/amd64 linux/arm64 linux/arm \
	  darwin/amd64 darwin/arm64 \
	  windows/amd64 windows/arm64 \
	  freebsd/amd64 openbsd/amd64 netbsd/amd64; do \
	  echo "--- $$pair"; \
	  GOOS=$${pair%/*} GOARCH=$${pair#*/} $(GO) build ./...; \
	done
