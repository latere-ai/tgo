// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package conformance

import (
	"fmt"
	"strconv"
	"strings"
)

// State is what specs/010-conformance.md §2's "States" paragraph says about a
// row: whether accel's exported surface does the thing.
//
// Three states and not four. An operator that accepts a binding and computes
// the wrong thing -- C13, before it closed -- is not a state here. §2 puts it
// in §1's downward direction, where it is checked against the oracle, because a
// register row is a claim about what tgo can express and a wrong answer is a
// claim about what accel computes.
type State int

const (
	// Closed means accel's exported surface does the thing, verified by a
	// probe that asserted a value (010-D7). Not "the issue is closed": §2
	// records four rows that closed upstream and stayed open here.
	Closed State = iota + 1

	// Open means it does not, and the row cites the issue that tracks it.
	Open

	// WontFix means accel's refusal is the correct answer and the row stays
	// in the table because it still constrains what tgo can do at graph
	// time. C9 is the only one: silently copying a strided view into MatMul
	// would hide a real cost behind an operator that looks free.
	WontFix
)

func (s State) String() string {
	switch s {
	case Closed:
		return "closed"
	case Open:
		return "open"
	case WontFix:
		return "won't fix, correctly"
	}
	return "state " + strconv.Itoa(int(s))
}

// issueURL is where a filed row points.
const issueURL = "https://github.com/golang-design/accel/issues/"

// Row is one row of the register: a thing tgo cannot express, or could not.
//
// The prose cells are opaque Markdown and not structured text. They carry
// inline links, backticked identifiers, bold spans and LaTeX, and a type that
// modelled them would be a Markdown parser with a register attached. What has
// structure here is the id, the accel specs, the issue and the state, and those
// are the fields anything programmatic reads.
type Row struct {
	// ID is "C" and the row's number. Rows are numbered without gaps and a
	// number is never reused; internal/speclint enforces both over the spec
	// tree, and [Document] plus the drift test carry the enforcement here.
	ID string

	// Cannot is what tgo cannot do, in the register's voice: a capability,
	// not a symptom. §2.2 is about the difference -- four rows closed
	// upstream against issues whose titles named a symptom, and the
	// capability stayed absent.
	Cannot string

	// Specs are the accel spec numbers that own the capability.
	Specs []string

	// Issue is the accel issue number, or 0 for a row that is not filed and
	// should not be. 010-D8: an open row cites an open issue, and when accel
	// closes one whose capability is still absent, tgo files a new one
	// rather than commenting on the closed thread.
	Issue int

	// IssueNote qualifies the citation where the issue alone would mislead.
	// C17's issue is closed as not planned and the gap is recorded in
	// accel's kernel corpus instead, which 010-D8 accepts as a durable
	// upstream record.
	IssueNote string

	// State is the verdict.
	State State

	// StateNote narrows it: "narrowed" for a row whose scope shrank as accel
	// moved, "not scheduled" for one nobody is working on.
	StateNote string

	// Cost is the workaround and what it costs. "none needed" for a closed
	// row, and for an open one the price tgo is paying now.
	Cost string
}

// Register is the register of specs/010-conformance.md §2.
//
// It is the source and the spec's table is the output. 010-D6: a
// hand-maintained register drifts within one milestone, which is the exact
// failure this project exists to catch in accel, so the table is generated from
// here by [Document], and TestTheSpecTableIsGenerated fails when the two part.
//
// A row is added when tgo hits something it cannot express, and it leaves only
// when its test stops skipping -- not when an issue closes, not when a spec is
// written, and never because it was worked around.
//
// The rows returned are a copy, so a caller that sorts or filters them cannot
// edit the register by accident.
func Register() []Row {
	rows := []Row{{
		ID:     "C1",
		Cannot: "a **batched** decode",
		Specs:  []string{"040"},
		Issue:  12,
		State:  Closed,
		Cost: "none needed. Verified: two sequences of lengths 96 and 32 " +
			"batched match two single runs to `0.00e+00`",
	}, {
		ID:     "C2",
		Cannot: "RoPE at per-row positions",
		Specs:  []string{"025", "043"},
		Issue:  2,
		State:  Closed,
		Cost:   "none needed",
	}, {
		ID:     "C3",
		Cannot: "sampling of any kind at the `tensor` layer",
		Specs:  []string{"028", "039"},
		Issue:  6,
		State:  Closed,
		Cost: "none needed. `tensor.Sample` composes the whole policy on the " +
			"device — penalties, temperature, softmax, top-k, top-p and the " +
			"categorical walk — and returns a token id",
	}, {
		ID:     "C4",
		Cannot: "a paged KV **decode**",
		Specs:  []string{"030", "043"},
		Issue:  1,
		State:  Closed,
		Cost:   "none needed",
	}, {
		ID:     "C5",
		Cannot: "an f16 KV cache that can be **written**, or paged",
		Specs:  []string{"007", "010"},
		Issue:  13,
		State:  Closed,
		Cost: "none needed; `ScatterRows`, prefill and paged decode all take " +
			"f16. **Halves the cache**",
	}, {
		ID:     "C6",
		Cannot: "penalties and temperature on device",
		Specs:  []string{"039"},
		Issue:  6,
		State:  Closed,
		Cost: "none needed. The policy runs on the device, so a step can return " +
			"a token id rather than reading back 608 KB of logits",
	}, {
		ID:        "C7",
		Cannot:    "a **bf16 GEMM**",
		Specs:     []string{"002", "010"},
		Issue:     14,
		State:     Open,
		StateNote: "narrowed",
		Cost: "convert on the host at load. `Cast` now widens bf16 to f32, so " +
			"only the GEMM is missing, and a weight would not want a per-step " +
			"cast anyway",
	}, {
		ID:     "C8",
		Cannot: "f32 activations against f16 or int8 weights",
		Specs:  []string{"010"},
		Issue:  14,
		State:  Closed,
		Cost: "none needed. **The cast chain is gone**: 1013 selections → 760 " +
			"on the Qwen3 graph",
	}, {
		ID:     "C9",
		Cannot: "a strided view into `MatMul`",
		Specs:  []string{"025"},
		State:  WontFix,
		Cost:   "host-side transpose at load ([001 §4](001-weights.md))",
	}, {
		ID:     "C10",
		Cannot: "avoiding a host copy of every converted weight",
		Specs:  []string{"001"},
		Issue:  7,
		State:  Closed,
		Cost:   "none needed; `Buffer.Access`",
	}, {
		ID:     "C11",
		Cannot: "a KV cache longer than 128 positions",
		Specs:  []string{"007", "010", "044"},
		Issue:  8,
		State:  Closed,
		Cost:   "none needed",
	}, {
		ID:     "C12",
		Cannot: "binding a `LayerState` view",
		Specs:  []string{"007", "030"},
		Issue:  9,
		State:  Closed,
		Cost:   "none needed. 2 states, not 72",
	}, {
		ID:     "C13",
		Cannot: "a paged **prefill**",
		Specs:  []string{"010", "030"},
		Issue:  10,
		State:  Closed,
		Cost:   "none needed; verified by reversing the page table",
	}, {
		ID:     "C14",
		Cannot: "an f16 `GatherRows`",
		Specs:  []string{"010"},
		Issue:  11,
		State:  Closed,
		Cost:   "none needed",
	}, {
		ID:     "C15",
		Cannot: "a quantized matrix-vector kernel at $M=1$",
		Specs:  []string{"010"},
		Issue:  11,
		State:  Closed,
		Cost:   "none needed",
	}, {
		ID:     "C16",
		Cannot: "a **batched prefill**, or prefill and decode in one dispatch",
		Specs:  []string{"040"},
		Issue:  16,
		State:  Open,
		Cost: "prefills run alone. Chunked prefill bounds latency and recovers " +
			"no throughput ([008 §5](008-scheduler.md))",
	}, {
		ID:        "C17",
		Cannot:    "GGUF's K-quant super-blocks",
		Specs:     []string{"010"},
		Issue:     15,
		IssueNote: "not planned",
		State:     Open,
		StateNote: "not scheduled",
		Cost:      "read safetensors and quantize at load ([012](012-gguf.md))",
	}, {
		ID:     "C18",
		Cannot: "`Contiguous` on Metal",
		Specs:  []string{"010", "021"},
		Issue:  19,
		State:  Closed,
		Cost: "none needed. It was the only kernel in the corpus with no MSL " +
			"artifact, so every graph that slices — which " +
			"[004 §3.2](004-model-graph.md) requires — was refused at compile. " +
			"Fixed upstream the day it was filed",
	}, {
		ID:     "C19",
		Cannot: "a CPU backend that dispatches in parallel",
		Specs:  []string{"006"},
		Issue:  20,
		State:  Closed,
		Cost: "none needed. The worker pool landed: 19.5x per prompt token on a " +
			"real model, and device is 99.98% of a step, so nothing measurable " +
			"remains between dispatches. The residual gap to Metal is kernel " +
			"throughput rather than a missing capability",
	}, {
		ID:     "C21",
		Cannot: "**4-bit weights**",
		Specs:  []string{"027", "010"},
		Issue:  22,
		State:  Open,
		Cost: "int8, so a 27B model is 25.1 GiB rather than 12.6 GiB and does " +
			"not fit a 24 GiB device. A second blocker on the 27B target, " +
			"independent of the linear attention in #17 " +
			"([011 §2.3](011-sequencing.md))",
	}, {
		ID:     "C20",
		Cannot: "a decode step whose submit cost is amortised",
		Specs:  []string{"021"},
		Issue:  21,
		State:  Open,
		Cost: "none. Submit is 15.61% of a decode step against 1.12% of a " +
			"prefill: a fixed per-dispatch cost over a ~790-node graph, paid in " +
			"full by a one-token step ([017 §4.1](017-benchmarks.md))",
	}}
	return rows
}

// Filed renders the register's "filed" cell: the issue, or an em dash for a
// row that is not filed and should not be.
func (r Row) Filed() string {
	if r.Issue == 0 {
		if r.IssueNote != "" {
			return r.IssueNote
		}
		return "—"
	}
	cell := fmt.Sprintf("[#%d](%s%d)", r.Issue, issueURL, r.Issue)
	if r.IssueNote != "" {
		cell += ", " + r.IssueNote
	}
	return cell
}

// Status renders the register's "state" cell.
//
// Bold for a state a probe decided -- closed and open are the two the table is
// scanned for -- and plain for "won't fix, correctly", which is an argument
// rather than a verdict about what accel does.
func (r Row) Status() string {
	s := r.State.String()
	if r.StateNote != "" {
		s += ", " + r.StateNote
	}
	if r.State == WontFix {
		return s
	}
	return "**" + s + "**"
}

// SkipReason is what a row's test skips with (010-D1).
//
// It names the capability, the accel specs that own it and the issue that
// tracks it, because a skip message that says only "unsupported" sends the
// reader back to the register to find out what is unsupported and by whom.
func (r Row) SkipReason() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: tgo cannot express %s. accel spec %s",
		r.ID, r.Cannot, strings.Join(r.Specs, ", "))
	if r.Issue != 0 {
		fmt.Fprintf(&b, "; filed as %s%d", issueURL, r.Issue)
		if r.IssueNote != "" {
			b.WriteString(" (" + r.IssueNote + ")")
		}
	}
	fmt.Fprintf(&b, ". Today: %s. This test skips until the capability exists; "+
		"specs/010-conformance.md §2 says a row leaves the register only when "+
		"its test stops skipping.", r.Cost)
	return b.String()
}

// header and divider are the register table's two fixed lines.
const (
	header  = "| # | what tgo cannot do | accel spec | filed | state | workaround, and what it costs |"
	divider = "| --- | --- | --- | --- | --- | --- |"
)

// Document renders rows as the Markdown table specs/010-conformance.md §2
// publishes (010-D6, 010 §6).
//
// The table and not the section: the prose around it is an argument somebody
// made, and generating an argument from a slice of structs would be a way of
// deleting it. What is generated is the part that goes stale -- the rows.
func Document(rows []Row) string {
	var b strings.Builder
	b.WriteString(header + "\n" + divider + "\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %s |\n",
			r.ID, r.Cannot, strings.Join(r.Specs, ", "), r.Filed(), r.Status(), r.Cost)
	}
	return b.String()
}

// Validate reports the rows that break specs/010-conformance.md's own rules,
// one finding per line and none when the register is sound.
//
// It checks the half of 010-D8 a program can see: that an open row cites
// something upstream. It cannot check that the citation is *open* -- C17 cites
// an issue accel closed as not planned, and 010-D8 was widened to accept the
// kernel-corpus row that replaced it, which is a better record than an issue
// with no plan and is not visible from here.
//
// Numbering and cross-spec citations are internal/speclint's, over the spec
// text. Since the spec text is generated from these rows, checking them twice
// would be checking the same thing twice.
func Validate(rows []Row) []string {
	var bad []string
	for _, r := range rows {
		switch r.State {
		case Open:
			if r.Issue == 0 && r.IssueNote == "" {
				bad = append(bad, r.ID+" is open and cites nothing upstream. "+
					"010-D8: an open row cites an open issue, or the named "+
					"artifact upstream that records the gap; a row tracked "+
					"only here is a belief about whose problem it is")
			}
		case WontFix:
			if r.Issue != 0 {
				bad = append(bad, r.ID+" is a correct refusal and cites an "+
					"issue. §2: it is not filed and should not be, because "+
					"the refusal is the right answer and the row stays only "+
					"because it constrains what tgo can do at graph time")
			}
		case Closed:
		default:
			bad = append(bad, r.ID+" has no state; §2 has three")
		}
		if len(r.Specs) == 0 {
			bad = append(bad, r.ID+" names no accel spec, so nothing upstream owns it")
		}
		if r.Cannot == "" || r.Cost == "" {
			bad = append(bad, r.ID+" leaves a cell empty; a row with no "+
				"capability or no cost is not a row")
		}
	}
	return bad
}
