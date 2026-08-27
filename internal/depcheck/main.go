// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Command depcheck gates what a package's build actually pulls in.
//
// specs/009-server.md 009-D14. The server serves three wire dialects through
// latere.ai/x/pkg/llmdialect, and 009 §2 argues that costs almost nothing
// because llmdialect's subtree is stdlib-only. That is a property of its
// current imports, not a promise it makes: one OTEL import added upstream would
// arrive on the next `go get`, and the first symptom would be a slower build
// rather than an error.
//
// So the build list is compared against a list written down here, and growing
// it is a decision someone makes on purpose. The check is on `go list -deps`
// rather than on go.mod, because a module graph says what could be reached and
// this says what is.
//
// It is asked once per GOOS/GOARCH tgo supports, and the answer is the union.
// A build list is platform-dependent -- purego is in the build on darwin, where
// accel loads Metal, and not on linux -- so a gate that asks the host is a gate
// that sees whatever the developer happens to be running. A linux-only
// dependency would have reached CI unremarked, and the first run of this on
// linux said as much.
//
//	go run ./internal/depcheck
//
// It is a command rather than only a test because the thing being gated is the
// build, and a build is what a command line reports. The test beside it drives
// the same decision logic and pins the list itself: every prefix in `gated` is
// one the build actually reaches, so a stale allowance cannot sit there
// admitting whatever later moves under it.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// gated maps a package to the module prefixes its build may reach.
//
// A prefix rather than a package: llmdialect has internal subpackages that are
// its own business, and pinning each would make an upstream refactor a tgo
// failure without any new dependency. What is gated is which *modules* are
// reachable, which is the thing 009 §2's argument is about.
//
// accel is listed even though 000-D1 makes reaching it the project's premise.
// The list is exhaustive rather than interesting: an entry left out because
// everybody knows about it is an entry the test below cannot tell from an entry
// nobody decided on, and the gate's answer is which modules are reachable, not
// which ones are news.
var gated = map[string][]entry{
	// Import paths rather than ./relative ones. `go list` resolves a relative
	// path against its own working directory, so a gate written that way
	// passes from the module root and fails from the package's own directory
	// -- which is where `go test` runs it.
	"github.com/latere-ai/tgo/server": {
		{"golang.design/x/accel", "the compute layer; 000 D1 makes every package here reach it"},
		{"golang.org/x/text", "Unicode normalisation for the tokenizer, 002-D10"},
		{"latere.ai/x/pkg/llmdialect", "the three wire dialects, 009-D10; this is the one 009-D14 exists to watch"},
		{"github.com/ebitengine/purego", "accel's cgo-free dynamic loading of Metal, reached through accel on darwin only"},
	},
}

// platforms are the GOOS/GOARCH pairs the build list is taken on.
//
// The same set .github/workflows/ci.yml cross-compiles, for the same reason:
// what tgo claims to build for is what has to be checked. Asking `go list` is
// far cheaper than building, so the whole set is affordable here.
var platforms = [...]struct{ goos, goarch string }{
	{"linux", "amd64"}, {"linux", "arm64"}, {"linux", "arm"},
	{"darwin", "amd64"}, {"darwin", "arm64"},
	{"windows", "amd64"}, {"windows", "arm64"},
	{"freebsd", "amd64"}, {"openbsd", "amd64"}, {"netbsd", "amd64"},
}

// entry is one allowed module prefix and why it is allowed.
//
// The reason is required in the same way covercheck requires one for an
// exemption: a prefix nobody wrote a reason for is one nobody decided to allow.
type entry struct {
	prefix string
	why    string
}

// module is tgo's own path. Its packages are this module, not dependencies of
// it.
const module = "github.com/latere-ai/tgo"

func main() {
	verbose := flag.Bool("v", false, "print the allowed set and what matched it")
	flag.Parse()
	os.Exit(run(gated, *verbose, os.Stdout, os.Stderr))
}

// run checks every gated package and returns the process exit code: 0 when
// each one reaches only what a decision allows, 1 when one does not, and 2 when
// the build list could not be read at all.
//
// A read failure is a third code rather than a failure, because it says nothing
// about the dependency footprint. Reporting a broken `go list` as a violation
// would send a reader to the allowlist for an answer that is not there.
func run(gates map[string][]entry, verbose bool, out, errw io.Writer) int {
	pkgs := make([]string, 0, len(gates))
	for p := range gates {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)

	failed := false
	for _, pkg := range pkgs {
		unexpected, matched, err := checkAll(pkg, gates[pkg])
		if err != nil {
			fmt.Fprintf(errw, "depcheck: %v\n", err)
			return 2
		}
		if verbose {
			for _, e := range gates[pkg] {
				if where := matched[e.prefix]; where != "" {
					fmt.Fprintf(out, "       %s -- %s [%s]\n", e.prefix, e.why, where)
				}
			}
		}
		if len(unexpected) == 0 {
			fmt.Fprintf(out, "ok   %s reaches only what 009-D14 allows, on %d platforms\n",
				short(pkg), len(platforms))
			continue
		}
		failed = true
		fmt.Fprintf(out, "FAIL %s reaches %d module(s) no decision allows:\n", short(pkg), len(unexpected))
		for _, u := range unexpected {
			fmt.Fprintf(out, "       %s\n", u)
		}
	}
	if failed {
		fmt.Fprintln(errw, "\nA new dependency in a gated package is a decision, not an "+
			"upgrade. Either drop it, or add it to `gated` with the reason it is "+
			"worth what it costs -- specs/009-server.md §2 is the argument the "+
			"list defends.")
		return 1
	}
	return 0
}

// checkAll takes the build list on every platform and returns the union.
//
// unexpected names the platform each module appeared on, because a dependency
// that arrives only on one is the case a host-only check misses, and saying
// which one is the difference between a report and a puzzle. matched maps an
// allowed prefix to the first platform that reached it, so a prefix reached
// nowhere is distinguishable from one reached somewhere.
func checkAll(pkg string, allow []entry) (unexpected []string, matched map[string]string, err error) {
	matched = map[string]string{}
	seen := map[string]bool{}
	for _, p := range platforms {
		where := p.goos + "/" + p.goarch
		got, hit, err := check(pkg, allow, p.goos, p.goarch)
		if err != nil {
			return nil, nil, err
		}
		for _, prefix := range hit {
			if matched[prefix] == "" {
				matched[prefix] = where
			}
		}
		for _, u := range got {
			if !seen[u] {
				seen[u] = true
				unexpected = append(unexpected, u+" (on "+where+")")
			}
		}
	}
	sort.Strings(unexpected)
	return unexpected, matched, nil
}

// check returns the modules pkg's build reaches on one platform that no entry
// allows, and the prefixes that admitted something.
func check(pkg string, allow []entry, goos, goarch string) (unexpected, matched []string, err error) {
	cmd := exec.Command("go", "list", "-deps", pkg)
	cmd.Env = append(os.Environ(), "GOOS="+goos, "GOARCH="+goarch)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, nil, fmt.Errorf("go list -deps %s on %s/%s: %w: %s", pkg, goos, goarch,
			err, strings.TrimSpace(stderr.String()))
	}

	hit := map[string]bool{}
	seen := map[string]bool{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		p := strings.TrimSpace(line)
		if p == "" || !external(p) {
			continue
		}
		if prefix := allowed(p, allow); prefix != "" {
			if !hit[prefix] {
				hit[prefix] = true
				matched = append(matched, prefix)
			}
			continue
		}
		if !seen[p] {
			seen[p] = true
			unexpected = append(unexpected, p)
		}
	}
	sort.Strings(unexpected)
	sort.Strings(matched)
	return unexpected, matched, nil
}

// short trims tgo's own module path off a gated package, so the report names
// the package the way a reader of specs/009-server.md does.
func short(pkg string) string {
	return "./" + strings.TrimPrefix(pkg, module+"/")
}

// external reports whether a package is outside the standard library and
// outside this module.
//
// The standard library has no dot in its first path element, which is the same
// rule the go command uses. tgo's own packages are this module and are not a
// dependency of it.
func external(p string) bool {
	if p == module || strings.HasPrefix(p, module+"/") {
		return false
	}
	first, _, _ := strings.Cut(p, "/")
	return strings.Contains(first, ".")
}

// allowed returns the prefix that admits p, or "".
func allowed(p string, allow []entry) string {
	for _, e := range allow {
		if p == e.prefix || strings.HasPrefix(p, e.prefix+"/") {
			return e.prefix
		}
	}
	return ""
}
