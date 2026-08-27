// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package conformance

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

// specFile is the spec whose §2 table this package generates.
const specFile = "../../specs/010-conformance.md"

// specTable extracts the register table from the spec: the header line and
// every table line that follows it without a break.
//
// It is deliberately dumb. A Markdown parser would tolerate a table that had
// drifted into a different shape, and the point of this test is that nothing
// about the published table is tolerated.
func specTable(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(specFile)
	if err != nil {
		t.Fatalf("read %s: %v", specFile, err)
	}
	table, err := registerBlock(strings.Split(string(b), "\n"))
	if err != nil {
		t.Fatalf("%s: %v", specFile, err)
	}
	return table
}

// registerBlock is specTable's reading, separated from the file so it can be
// given a document rather than only the one on disk.
//
// The block runs from the header to the next blank line, and every line in it
// has to be a table row. Stopping at the first line that is not a row would be
// the obvious reading and it is the wrong one: it makes anything spliced
// directly under the table invisible, which is how three runs of `go test`
// output came to sit inside §2. A generated block's extent is decided by the
// document, not by how far the generated part happens to reach.
func registerBlock(lines []string) (string, error) {
	start := -1
	for i, l := range lines {
		if l == header {
			start = i
			break
		}
	}
	if start < 0 {
		return "", fmt.Errorf("no register table starting with\n\t%s\n"+
			"specs/010-conformance.md §2 is generated from Register(); a table "+
			"with a different header is a table this package no longer produces",
			header)
	}
	var table strings.Builder
	for i, l := range lines[start:] {
		if l == "" {
			return table.String(), nil
		}
		if !strings.HasPrefix(l, "|") {
			return "", fmt.Errorf("line %d is inside the register table and is "+
				"not a table row:\n\t%s\n"+
				"specs/010-conformance.md §2 is generated from Register(); the "+
				"block runs to the next blank line and holds nothing else",
				start+i+1, l)
		}
		table.WriteString(l + "\n")
	}
	return "", fmt.Errorf("the register table runs to the end of the file with no " +
		"blank line after it; §2 is a generated block and needs an end")
}

// TestSplicedOutputIsInsideTheTable is the regression for what was on main:
// three runs of `go test` output sat between the last register row and the
// paragraph under it, and every gate passed.
//
// The bug was in the reading, not in the register, so this gives the reading a
// document instead of the file. The corrupted shape is the exact one that
// shipped — no blank line before the spliced text, none after.
func TestSplicedOutputIsInsideTheTable(t *testing.T) {
	rows := []string{"| C1 | a thing | 040 | — | closed | none |"}
	clean := append([]string{"# spec", "", header}, append(rows, "", "prose")...)

	got, err := registerBlock(clean)
	if err != nil {
		t.Fatalf("a clean document: %v", err)
	}
	if want := header + "\n" + rows[0] + "\n"; got != want {
		t.Errorf("a clean document read as %q, want %q", got, want)
	}

	spliced := append([]string{"# spec", "", header}, append(append(append([]string{},
		rows...), "FAIL", "FAIL\tgithub.com/latere-ai/tgo/internal/conformance\t0.245s"),
		"", "prose")...)
	if _, err := registerBlock(spliced); err == nil {
		t.Fatal("output spliced under the last register row read as a clean table; " +
			"specs/010-conformance.md §2 is the project's primary output and a " +
			"guard blind to text inside it is not guarding it")
	} else if !strings.Contains(err.Error(), "FAIL") {
		t.Errorf("the refusal does not quote the offending line: %v", err)
	}
}

// TestTheRegisterTableIsTerminated is the other end of the same reading: a
// generated block with no blank line after it has no end, and reading to EOF
// would make the last row's neighbour a matter of luck.
func TestTheRegisterTableIsTerminated(t *testing.T) {
	unterminated := []string{header, "| C1 | a thing | 040 | — | closed | none |"}
	if _, err := registerBlock(unterminated); err == nil {
		t.Fatal("a table running to the end of the file read as terminated")
	}
}

// TestTheSpecTableIsGenerated is 010-D6.
//
// The register lives in Register() and specs/010-conformance.md §2 is its
// output. A hand-maintained register drifts within one milestone -- which is
// the exact failure this project exists to catch in accel -- so the two are
// pinned to each other here, and editing either one alone is a red test.
//
// On a mismatch the generated table is printed in full, because the fix for a
// drifted table is to paste this into §2, and a tripwire with no fix path is a
// test somebody deletes.
func TestTheSpecTableIsGenerated(t *testing.T) {
	got, want := Document(Register()), specTable(t)
	// The comparison has to be able to fail. A tripwire that would pass over
	// two empty strings is not a tripwire, so one row is mutated and the
	// mismatch is required.
	//
	// The mutation is *away from* whatever the row says rather than to a fixed
	// verdict. Setting the last row to Closed stopped mutating anything the day
	// that row closed, and the tripwire went off -- correctly, and for a reason
	// that had nothing to do with the table. A mutation that depends on the
	// data it mutates is one that keeps working as the data moves.
	t.Run("a mutated register no longer matches the spec", func(t *testing.T) {
		rows := Register()
		last := &rows[len(rows)-1]
		if last.State == Closed {
			last.State = Open
		} else {
			last.State = Closed
		}
		if Document(rows) == want {
			t.Fatal("the spec table matched a register with a different verdict in it")
		}
	})
	if got == want {
		return
	}
	gl, wl := strings.Split(got, "\n"), strings.Split(want, "\n")
	for i := 0; i < len(gl) || i < len(wl); i++ {
		g, w := "", ""
		if i < len(gl) {
			g = gl[i]
		}
		if i < len(wl) {
			w = wl[i]
		}
		if g != w {
			t.Errorf("line %d of the register table has drifted.\n"+
				"\tRegister(): %s\n\t%s: %s", i+1, g, specFile, w)
		}
	}
	t.Errorf("specs/010-conformance.md §2 is generated from Register() "+
		"(010-D6). Replace the table in §2 with:\n\n%s", got)
}

// TestEveryOpenRowHasASkippingTest is 010-D1: one skipping test per open
// register row, naming the reason and the accel spec that owns it.
//
// The subtests are the rows rather than a hand-written function each, so the
// set of named skips cannot fall behind the set of open rows. A row leaves the
// register when its test stops skipping, and this is that test.
func TestEveryOpenRowHasASkippingTest(t *testing.T) {
	open := 0
	for _, r := range Register() {
		if r.State != Open {
			continue
		}
		open++
		t.Run(r.ID, func(t *testing.T) {
			// Checked before skipping, because a skipped subtest asserts
			// nothing afterwards and 010-D1 is about what the skip says.
			why := r.SkipReason()
			for _, want := range []string{r.ID, r.Cannot, r.Specs[0], r.Cost} {
				if !strings.Contains(why, want) {
					t.Fatalf("the skip reason does not mention %q:\n%s", want, why)
				}
			}
			t.Skip(why)
		})
	}
	// 010-D1 is a table of rows that skip. A register with nothing open is a
	// register that stopped tracking, not a project with nothing left to
	// report, so the empty case is a failure rather than a quiet pass.
	if open == 0 {
		t.Fatal("no register row is open, so 010-D1's skipping tests are all gone; " +
			"specs/010-conformance.md §2 says a row leaves only when its test stops skipping")
	}
	t.Logf("%d open register rows, each skipping with its reason", open)
	// §1: the suite prints the table, and the table is the deliverable. This
	// is where a run emits it -- go test -v ./internal/conformance/ -- so §6's
	// generated document is produced by the tests rather than described by
	// them.
	t.Logf("specs/010-conformance.md §6, generated:\n\n%s",
		Publish(Register(), Measurements{}))
}

func TestTheRegisterObeysItsOwnRules(t *testing.T) {
	for _, bad := range Validate(Register()) {
		t.Errorf("register: %s", bad)
	}
}

// TestValidateCatchesABrokenRow negative-tests every rule, because a checker
// nobody has seen fail is a checker nobody has checked.
func TestValidateCatchesABrokenRow(t *testing.T) {
	sound := Row{ID: "C1", Cannot: "a thing", Specs: []string{"010"},
		Issue: 1, State: Open, Cost: "host-side"}
	cases := []struct {
		name string
		row  Row
		want string
	}{
		{"an open row citing nothing", func() Row {
			r := sound
			r.Issue = 0
			return r
		}(), "cites nothing upstream"},
		{"a correct refusal that was filed anyway", func() Row {
			r := sound
			r.State = WontFix
			return r
		}(), "correct refusal and cites an issue"},
		{"a row with no state", func() Row {
			r := sound
			r.State = 0
			return r
		}(), "has no state"},
		{"a row owned by no accel spec", func() Row {
			r := sound
			r.Specs = nil
			return r
		}(), "names no accel spec"},
		// The one a reader finds and no test did: a pipe inside a cell splits
		// the row. It reached §2 once, in a code span, where markdown breaks
		// it just the same -- and the drift test could not see it, because it
		// compares the generated line against the file and both carried it.
		{"a cost cell that splits its own row", func() Row {
			r := sound
			r.Cost = "the native layout is `[beta | alpha]` per key head"
			return r
		}(), "contains a pipe"},
		{"a capability cell that splits its own row", func() Row {
			r := sound
			r.Cannot = "a gate of shape [tokens | heads]"
			return r
		}(), "contains a pipe"},
		{"a row with no capability", func() Row {
			r := sound
			r.Cannot = ""
			return r
		}(), "leaves a cell empty"},
		{"a row with no cost", func() Row {
			r := sound
			r.Cost = ""
			return r
		}(), "leaves a cell empty"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			bad := Validate([]Row{c.row})
			if len(bad) != 1 {
				t.Fatalf("Validate reported %d findings, want 1: %v", len(bad), bad)
			}
			if !strings.Contains(bad[0], c.want) {
				t.Fatalf("Validate said %q, want it to mention %q", bad[0], c.want)
			}
		})
	}
	if bad := Validate([]Row{sound}); len(bad) != 0 {
		t.Fatalf("a sound row was rejected: %v", bad)
	}
	// An open row whose upstream record is an artifact rather than an issue
	// is what 010-D8 was widened to accept.
	artifact := sound
	artifact.Issue, artifact.IssueNote = 0, "`quant_matmul_superblock` in accel's kernel corpus"
	if bad := Validate([]Row{artifact}); len(bad) != 0 {
		t.Fatalf("an open row citing a durable artifact was rejected: %v", bad)
	}
}

func TestStateNamesItself(t *testing.T) {
	for _, c := range []struct {
		state State
		want  string
	}{
		{Closed, "closed"},
		{Open, "open"},
		{WontFix, "won't fix, correctly"},
		{State(9), "state 9"},
	} {
		if got := c.state.String(); got != c.want {
			t.Errorf("State(%d) = %q, want %q", int(c.state), got, c.want)
		}
	}
}

func TestACellRendersAsTheTablePublishesIt(t *testing.T) {
	for _, c := range []struct {
		name         string
		row          Row
		filed, state string
	}{
		{"filed and open", Row{Issue: 6, State: Open},
			"[#6](https://github.com/golang-design/accel/issues/6)", "**open**"},
		{"filed, closed upstream, and narrowed here",
			Row{Issue: 15, IssueNote: "not planned", State: Open, StateNote: "not scheduled"},
			"[#15](https://github.com/golang-design/accel/issues/15), not planned",
			"**open, not scheduled**"},
		{"not filed and should not be", Row{State: WontFix}, "—", "won't fix, correctly"},
		{"recorded upstream without an issue", Row{IssueNote: "accel's kernel corpus",
			State: Open}, "accel's kernel corpus", "**open**"},
		{"closed", Row{Issue: 2, State: Closed},
			"[#2](https://github.com/golang-design/accel/issues/2)", "**closed**"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := c.row.Filed(); got != c.filed {
				t.Errorf("Filed() = %q, want %q", got, c.filed)
			}
			if got := c.row.Status(); got != c.state {
				t.Errorf("Status() = %q, want %q", got, c.state)
			}
		})
	}
}

// TestASkipReasonSaysWhatAndWhose checks the message a skipping test carries.
// "Unsupported" sends the reader back to the register to find out what is
// unsupported and by whom, which is the round trip 010-D1 exists to remove.
func TestASkipReasonSaysWhatAndWhose(t *testing.T) {
	var c17 Row
	for _, r := range Register() {
		if r.ID == "C17" {
			c17 = r
		}
	}
	why := c17.SkipReason()
	for _, want := range []string{
		"C17", "GGUF's K-quant super-blocks", "accel spec 010",
		"https://github.com/golang-design/accel/issues/15", "not planned",
		"read safetensors and quantize at load", "stops skipping",
	} {
		if !strings.Contains(why, want) {
			t.Errorf("the skip reason does not mention %q:\n%s", want, why)
		}
	}
	unfiled := Row{ID: "C9", Cannot: "a strided view", Specs: []string{"025"},
		State: WontFix, Cost: "a host-side transpose"}
	if strings.Contains(unfiled.SkipReason(), "filed as") {
		t.Errorf("an unfiled row claims an issue: %s", unfiled.SkipReason())
	}
}

// TestTheRegisterIsACopy: a caller that sorts the rows must not be able to
// reorder the register, because the register's order is the table's order.
func TestTheRegisterIsACopy(t *testing.T) {
	rows := Register()
	rows[0].State = WontFix
	rows[0].Cost = "edited"
	if again := Register(); again[0].State == WontFix || again[0].Cost == "edited" {
		t.Fatal("editing the returned rows edited the register")
	}
}

func TestDocumentOfNoRowsIsStillATable(t *testing.T) {
	want := header + "\n" + divider + "\n"
	if got := Document(nil); got != want {
		t.Fatalf("Document(nil) = %q, want the header and divider alone", got)
	}
}
