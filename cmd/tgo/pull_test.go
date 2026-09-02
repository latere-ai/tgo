// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/latere-ai/tgo/internal/hub"
)

// fakePuller is internal/hub as `tgo pull` uses it: a listing, a directory, and
// whatever progress the test wants reported.
//
// It touches no network and writes no cache, which is what lets every refusal
// and the whole progress path run in the default test run.
type fakePuller struct {
	rev      *hub.Revision
	revErr   error
	dir      string
	fetchErr error

	// events is reported to the progress reporter during Fetch, which is where
	// hub.Client calls it from.
	events []hub.Event
	pr     *progress

	asked []hub.Ref
}

func (f *fakePuller) Revision(ctx context.Context, ref hub.Ref) (*hub.Revision, error) {
	f.asked = append(f.asked, ref)
	return f.rev, f.revErr
}

func (f *fakePuller) Fetch(ctx context.Context, ref hub.Ref) (string, error) {
	for _, e := range f.events {
		f.pr.Event(e)
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return f.dir, f.fetchErr
}

// useFakePuller installs a puller for the duration of a test.
func useFakePuller(t *testing.T, f *fakePuller) *fakePuller {
	t.Helper()
	prev := newPuller
	newPuller = func(o pullOptions, pr *progress) puller {
		f.pr = pr
		return f
	}
	t.Cleanup(func() { newPuller = prev })
	return f
}

// noToken clears the environment a token is read from, so that a developer's
// own token does not decide what a test observes.
func noToken(t *testing.T) {
	t.Helper()
	for _, name := range hfTokenEnv {
		t.Setenv(name, "")
	}
}

func TestParsePullDefaults(t *testing.T) {
	noToken(t)
	o, err := parsePull([]string{"Qwen/Qwen3-0.6B"})
	if err != nil {
		t.Fatalf("parsePull: %v", err)
	}
	if o.Ref.Org != "Qwen" || o.Ref.Repo != "Qwen3-0.6B" {
		t.Errorf("ref = %+v, want the org and repo the argument named", o.Ref)
	}
	if o.Ref.Revision != "main" {
		t.Errorf("the default revision is %q, want main", o.Ref.Revision)
	}
	if o.Token != "" {
		t.Errorf("a token appeared from nowhere: %q", o.Token)
	}
	// A bare id, which the hub files under a sentinel org.
	o, err = parsePull([]string{"gpt2"})
	if err != nil {
		t.Fatalf("parsePull on a bare id: %v", err)
	}
	if o.Ref.ID() != "gpt2" {
		t.Errorf("ref id = %q, want gpt2", o.Ref.ID())
	}
}

func TestParsePullFlags(t *testing.T) {
	noToken(t)
	o, err := parsePull([]string{"--revision", "refs/pr/1", "--token", "hf_flag", "Qwen/Qwen3-0.6B"})
	if err != nil {
		t.Fatalf("parsePull: %v", err)
	}
	if o.Ref.Revision != "refs/pr/1" {
		t.Errorf("revision = %q, want the one --revision named", o.Ref.Revision)
	}
	if o.Token != "hf_flag" {
		t.Errorf("token = %q, want the one --token named", o.Token)
	}
	// A token pasted into a shell or exported from a file arrives with the
	// newline that was copied with it, and a bearer header carrying one is a
	// 401 that names nothing an operator can act on.
	o, err = parsePull([]string{"--token", "  hf_padded\n", "Qwen/Qwen3-0.6B"})
	if err != nil {
		t.Fatalf("parsePull with a padded token: %v", err)
	}
	if o.Token != "hf_padded" {
		t.Errorf("token = %q, want it trimmed", o.Token)
	}
}

// TestParsePullReadsTheTokenFromTheEnvironment: a gated repo is fetched without
// the token appearing in a shell history or the process table, and --token
// still wins over the environment.
func TestParsePullReadsTheTokenFromTheEnvironment(t *testing.T) {
	noToken(t)
	for i, name := range hfTokenEnv {
		t.Run(name, func(t *testing.T) {
			noToken(t)
			t.Setenv(name, "hf_env")
			o, err := parsePull([]string{"Qwen/Qwen3-0.6B"})
			if err != nil {
				t.Fatalf("parsePull: %v", err)
			}
			if o.Token != "hf_env" {
				t.Errorf("token = %q, want the one $%s carries (variable %d)", o.Token, name, i)
			}
			o, err = parsePull([]string{"--token", "hf_flag", "Qwen/Qwen3-0.6B"})
			if err != nil {
				t.Fatalf("parsePull: %v", err)
			}
			if o.Token != "hf_flag" {
				t.Errorf("$%s beat --token; the flag is what the user typed last", name)
			}
		})
	}
}

func TestParsePullRefusals(t *testing.T) {
	noToken(t)
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"no repo id", nil, "no repo id"},
		{"two repo ids", []string{"a/b", "c/d"}, "one repo id"},
		{"an empty repo id", []string{"  "}, "the repo id is empty"},
		{"a flag after the id", []string{"a/b", "--revision", "main"}, "flags go before it"},
		{"an unknown flag", []string{"--nope", "a/b"}, "nope"},
		{"a relative directory", []string{"./models/qwen"}, "nothing"},
		{"an absolute directory", []string{"/opt/qwen"}, "nothing"},
		{"a three-part path", []string{"a/b/c"}, "nothing"},
		{"a name a repo cannot have", []string{"a b"}, "has ' ' in it"},
		{"two revisions", []string{"--revision", "dev", "a/b@main"}, "already names a revision"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parsePull(tc.args)
			if err == nil {
				t.Fatalf("parsePull(%v) was accepted", tc.args)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want one containing %q", err, tc.want)
			}
			if !errors.Is(err, errUsage) {
				t.Error("the refusal does not wrap errUsage, so main would not print the usage")
			}
		})
	}
}

// listing is a revision whose files are a real repo's shape: the checkpoint,
// its tokenizer, and the other runtimes' copies of the same weights.
func listing() *hub.Revision {
	return &hub.Revision{
		SHA: "0123456789abcdef0123456789abcdef01234567",
		Files: []hub.File{
			{Path: "config.json", Size: 900},
			{Path: "model.safetensors", Size: 1_400_000_000},
			{Path: "tokenizer.json", Size: 7_000_000},
			{Path: "onnx/model.onnx", Size: 2_300_000_000},
			{Path: "coreml/weights.bin", Size: 1_100_000_000},
		},
	}
}

// TestWantedFilesIsTheFilterTheFetchApplies is the trap a total walks into: a
// real repo carries onnx/, coreml/ and ggml copies of the same weights, and a
// total taken over the whole listing announces a download several times the
// size of the one that follows.
func TestWantedFilesIsTheFilterTheFetchApplies(t *testing.T) {
	files, total := wantedFiles(listing())
	if len(files) != 3 {
		t.Errorf("wantedFiles kept %d of 5 entries, want the 3 a safetensors checkpoint needs: %v", len(files), files)
	}
	if want := int64(900 + 1_400_000_000 + 7_000_000); total != want {
		t.Errorf("total = %d, want %d: the other runtimes' copies are not downloaded", total, want)
	}
	for _, f := range files {
		if !hub.Wanted(f.Path) {
			t.Errorf("wantedFiles kept %s, which the fetch will not download", f.Path)
		}
	}
	// A file the API published no size for contributes nothing, which makes
	// the total a lower bound rather than a wrong number.
	_, total = wantedFiles(&hub.Revision{Files: []hub.File{
		{Path: "config.json", Size: 0}, {Path: "model.safetensors", Size: 512},
	}})
	if total != 512 {
		t.Errorf("total = %d, want 512: an unpublished size is not a size", total)
	}
}

func TestPullPrintsTheDirectoryOnStdoutAndProgressOnStderr(t *testing.T) {
	noToken(t)
	f := useFakePuller(t, &fakePuller{
		rev: listing(),
		dir: "/cache/models/Qwen/Qwen3-0.6B/0123456789abcdef0123456789abcdef01234567",
		events: []hub.Event{
			{Path: "config.json", Done: 900, Total: 900, Complete: true},
			{Path: "model.safetensors", Done: 1_400_000_000, Total: 1_400_000_000, Complete: true},
			{Path: "tokenizer.json", Done: 7_000_000, Total: 7_000_000, Resumed: true, Complete: true},
		},
	})

	var stdout, stderr strings.Builder
	if err := cmdPull([]string{"Qwen/Qwen3-0.6B"}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdPull: %v", err)
	}
	// Exactly the path and a newline, so that `tgo run "$(tgo pull ...)"`
	// works: a progress line on stdout would become part of the path.
	if got, want := stdout.String(), f.dir+"\n"; got != want {
		t.Errorf("stdout = %q, want %q", got, want)
	}
	out := stderr.String()
	// The header, as one whole line rather than as three substrings: the
	// progress that follows carries "3/3 files", which contains "3 files", so a
	// header that announced the 5 entries the API listed would go unnoticed by
	// a containment check.
	header, _, _ := strings.Cut(out, "\n")
	if want := "Qwen/Qwen3-0.6B at 0123456789ab: 3 files, 1.31 GiB"; header != want {
		t.Errorf("the header is %q, want %q: the count and the total are the files that will be\n"+
			"fetched, not the five the listing carries", header, want)
	}
	for _, want := range []string{
		"fetched config.json",
		"resumed tokenizer.json", // a resumed file says so rather than reading as a fresh fetch
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the progress does not carry %q:\n%s", want, out)
		}
	}
	if len(f.asked) != 1 || f.asked[0].Revision != "main" {
		t.Errorf("the listing was asked for as %v, want the parsed ref", f.asked)
	}
}

func TestPullRefusals(t *testing.T) {
	noToken(t)
	onnxOnly := &hub.Revision{SHA: "abc123", Files: []hub.File{{Path: "onnx/model.onnx", Size: 5}}}
	for _, tc := range []struct {
		name string
		fake *fakePuller
		want error
		says string
	}{
		{"a repo that is not there", &fakePuller{revErr: hub.ErrNotFound}, hub.ErrNotFound, "no such repo"},
		{"a repo that is gated", &fakePuller{revErr: hub.ErrUnauthorized}, hub.ErrUnauthorized, "gated"},
		{"a revision with nothing loadable", &fakePuller{rev: onnxOnly}, hub.ErrNoFiles, "safetensors"},
		{"a download that fails", &fakePuller{rev: listing(), fetchErr: hub.ErrChecksum},
			hub.ErrChecksum, "sha256"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			useFakePuller(t, tc.fake)
			var stdout, stderr strings.Builder
			err := cmdPull([]string{"Qwen/Qwen3-0.6B"}, &stdout, &stderr)
			if !errors.Is(err, tc.want) {
				t.Fatalf("cmdPull = %v, want %v", err, tc.want)
			}
			if !strings.Contains(err.Error(), tc.says) {
				t.Errorf("error = %v, want one containing %q", err, tc.says)
			}
			// Nothing on stdout, so a shell that captured it gets an empty
			// string rather than half a path.
			if stdout.Len() != 0 {
				t.Errorf("a failed pull wrote a path: %q", stdout.String())
			}
		})
	}
}

// TestProgressRedirectedIsOneLinePerFile is the "not noisy when redirected to a
// file" half: a log keeps every line, so a download that takes ten minutes
// leaves a record of what arrived and not a megabyte of repainting.
func TestProgressRedirectedIsOneLinePerFile(t *testing.T) {
	var b strings.Builder
	p := newProgress(&b)
	if p.tty {
		t.Fatal("a strings.Builder was taken for a terminal")
	}
	if p.interval != logInterval {
		t.Errorf("the redirected interval is %s, want %s", p.interval, logInterval)
	}
	p.start(2, 1000)
	// Partial events inside one interval print nothing: the interval is 30
	// seconds and this test takes microseconds.
	for i := range 20 {
		p.Event(hub.Event{Path: "model.safetensors", Done: int64(i * 25), Total: 800})
	}
	if b.Len() != 0 {
		t.Errorf("20 progress events inside one interval wrote %d bytes:\n%s", b.Len(), b.String())
	}
	p.Event(hub.Event{Path: "model.safetensors", Done: 800, Total: 800, Complete: true})
	p.Event(hub.Event{Path: "config.json", Done: 200, Total: 200, Complete: true})
	p.stop()

	lines := strings.Split(strings.TrimSuffix(b.String(), "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("a two-file download wrote %d lines, want one per file and one summary:\n%s",
			len(lines), b.String())
	}
	if !strings.HasPrefix(lines[0], "fetched model.safetensors") {
		t.Errorf("the first line is %q", lines[0])
	}
	if !strings.Contains(lines[2], "100.0%") || !strings.Contains(lines[2], "2/2 files") {
		t.Errorf("the summary does not say the download finished: %q", lines[2])
	}
	if strings.Contains(b.String(), "\r") {
		t.Error("a redirected progress repainted with a carriage return")
	}
}

// TestProgressAtATerminalRepaintsOneLine is the other half: a terminal shows
// motion, in place, without scrolling a log's worth of lines past.
func TestProgressAtATerminalRepaintsOneLine(t *testing.T) {
	var b strings.Builder
	// Constructed rather than detected: a test's writer is never a character
	// device, and this is the branch a person actually sees.
	p := &progress{w: &b, done: map[string]int64{}, tty: true, begun: time.Now().Add(-2 * time.Second)}
	p.start(2, 1000)
	p.begun = time.Now().Add(-2 * time.Second)

	p.Event(hub.Event{Path: "model.safetensors", Done: 100, Total: 800})
	first := b.String()
	if !strings.HasPrefix(first, "\r") {
		t.Errorf("the terminal line does not begin with a carriage return: %q", first)
	}
	if strings.Contains(first, "\n") {
		t.Errorf("the terminal line ended, so the next one scrolls it away: %q", first)
	}
	// Padded, so that a shorter line does not leave the tail of a longer one
	// on screen -- "10.0%" over "100.0%" reads as a download going backwards.
	if len(strings.TrimPrefix(first, "\r")) < 78 {
		t.Errorf("the terminal line is %d characters and is not padded: %q", len(first), first)
	}
	// Two seconds of elapsed time were arranged above, so the rate and the
	// estimate are computed rather than skipped. Their values are not
	// asserted: what is under test is that the line carries them.
	if !strings.Contains(first, "/s") || !strings.Contains(first, "left") {
		t.Errorf("the terminal line reports no rate or estimate: %q", first)
	}
	// A file completing does not write a line of its own at a terminal: it
	// moves the counter on the line that is already there.
	p.Event(hub.Event{Path: "model.safetensors", Done: 800, Total: 800, Complete: true})
	if strings.Contains(strings.TrimPrefix(b.String(), first), "fetched") {
		t.Errorf("a terminal wrote a per-file line as well as repainting:\n%q", b.String())
	}
	p.stop()
	if !strings.HasSuffix(b.String(), "\n") {
		t.Error("the terminal line was never ended, so a shell prompt lands on top of it")
	}
	if !strings.Contains(b.String(), "1/2 files") {
		t.Errorf("the final paint does not carry the file count:\n%q", b.String())
	}
}

// TestProgressSumsFilesRatherThanEvents is the arithmetic trap: an event
// reports a file's bytes so far, not a delta, and it is re-reported at every
// threshold. Adding them counts the same bytes many times over, and a download
// reports 400% complete.
func TestProgressSumsFilesRatherThanEvents(t *testing.T) {
	var b strings.Builder
	p := &progress{w: &b, done: map[string]int64{}, tty: true, begun: time.Now()}
	p.start(2, 1000)
	for _, done := range []int64{100, 200, 300, 400} {
		p.Event(hub.Event{Path: "model.safetensors", Done: done, Total: 400})
	}
	p.Event(hub.Event{Path: "config.json", Done: 600, Total: 600})
	p.stop()
	if !strings.Contains(b.String(), "100.0%") {
		t.Errorf("400 + 600 of 1000 is not reported as complete:\n%q", b.String())
	}
	if strings.Contains(b.String(), "1000.0%") || strings.Contains(b.String(), "160.0%") {
		t.Errorf("the events were added as deltas:\n%q", b.String())
	}
}

// TestProgressWithNoPublishedTotal: the API publishes no size for some files,
// so a percentage of the total would be a number somebody would read.
func TestProgressWithNoPublishedTotal(t *testing.T) {
	var b strings.Builder
	p := &progress{w: &b, done: map[string]int64{}, tty: true, begun: time.Now()}
	p.start(1, 0)
	p.Event(hub.Event{Path: "model.safetensors", Done: 4096})
	p.stop()
	if strings.Contains(b.String(), "%") {
		t.Errorf("a share of an unknown total was printed:\n%q", b.String())
	}
	if !strings.Contains(b.String(), "4.00 KiB") {
		t.Errorf("the bytes that did arrive were not reported:\n%q", b.String())
	}
}

// TestProgressStopIsQuietWhenNothingHappened: a pull whose files were all on
// disk already writes no progress line, because there was no progress.
func TestProgressStopIsQuietWhenNothingHappened(t *testing.T) {
	var b strings.Builder
	p := newProgress(&b)
	p.start(1, 100)
	p.stop()
	if b.Len() != 0 {
		t.Errorf("a download with no events wrote %q", b.String())
	}
}

// TestProgressIsSafeForConcurrentUse: hub.Client calls Progress from every
// download goroutine, so this is the contract its documentation states. The
// race detector is what makes the assertion.
func TestProgressIsSafeForConcurrentUse(t *testing.T) {
	var b strings.Builder
	p := &progress{w: &b, done: map[string]int64{}, begun: time.Now()}
	p.start(4, 4000)
	var wg sync.WaitGroup
	for i := range 4 {
		wg.Go(func() {
			for done := int64(100); done <= 1000; done += 100 {
				p.Event(hub.Event{Path: string(rune('a' + i)), Done: done, Total: 1000})
			}
			p.Event(hub.Event{Path: string(rune('a' + i)), Done: 1000, Total: 1000, Complete: true})
		})
	}
	wg.Wait()
	p.stop()
	if !strings.Contains(b.String(), "4/4 files") {
		t.Errorf("four files completed and the summary says otherwise:\n%s", b.String())
	}
}

// TestShortSHA: the abbreviation is for a line a human reads, and the whole sha
// is in the path this command prints, so nothing here has to be pasted back.
func TestShortSHA(t *testing.T) {
	if got := shortSHA("0123456789abcdef0123456789abcdef01234567"); got != "0123456789ab" {
		t.Errorf("shortSHA = %q, want the first twelve characters", got)
	}
	if got := shortSHA("abc"); got != "abc" {
		t.Errorf("shortSHA truncated a sha that was already short: %q", got)
	}
}

// TestPullStopsOnAnInterrupt: Ctrl-C cancels the download rather than killing
// the process mid-write. 013-D3 is what makes that safe -- a partial file keeps
// its temporary name, so the next pull resumes from it.
func TestPullStopsOnAnInterrupt(t *testing.T) {
	noToken(t)
	useFakePuller(t, &fakePuller{rev: listing(), dir: "/cache/qwen"})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	prev := interrupts
	interrupts = func() (context.Context, func()) { return ctx, func() {} }
	t.Cleanup(func() { interrupts = prev })

	var stdout, stderr strings.Builder
	err := cmdPull([]string{"Qwen/Qwen3-0.6B"}, &stdout, &stderr)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cmdPull = %v, want the cancellation", err)
	}
	if stdout.Len() != 0 {
		t.Errorf("an interrupted pull printed a path: %q", stdout.String())
	}
}

// charDevice is a writer that says it is a terminal, which is how a test
// reaches the branch a person sees: a test's own stdout is a pipe under
// `go test`, and no portable stdlib call opens a pty.
type charDevice struct {
	strings.Builder
	mode os.FileMode
	err  error
}

func (c *charDevice) Stat() (os.FileInfo, error) {
	if c.err != nil {
		return nil, c.err
	}
	return fakeFileInfo{mode: c.mode}, nil
}

type fakeFileInfo struct{ mode os.FileMode }

func (f fakeFileInfo) Name() string       { return "fake" }
func (f fakeFileInfo) Size() int64        { return 0 }
func (f fakeFileInfo) Mode() os.FileMode  { return f.mode }
func (f fakeFileInfo) ModTime() time.Time { return time.Time{} }
func (f fakeFileInfo) IsDir() bool        { return false }
func (f fakeFileInfo) Sys() any           { return nil }

// TestNewProgressPicksTheOutputsShape pins the rule that decides between the
// two renderings. Getting it backwards writes a megabyte of carriage returns
// into a CI log, or shows a person one line every thirty seconds.
func TestNewProgressPicksTheOutputsShape(t *testing.T) {
	for _, tc := range []struct {
		name     string
		w        io.Writer
		wantTTY  bool
		interval time.Duration
	}{
		{"a terminal", &charDevice{mode: os.ModeCharDevice | 0o620}, true, ttyInterval},
		{"a file", &charDevice{mode: 0o644}, false, logInterval},
		{"a writer that cannot be asked", &charDevice{err: os.ErrInvalid}, false, logInterval},
		{"a buffer, which is not a file at all", &strings.Builder{}, false, logInterval},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newProgress(tc.w)
			if p.tty != tc.wantTTY {
				t.Errorf("tty = %v, want %v", p.tty, tc.wantTTY)
			}
			if p.interval != tc.interval {
				t.Errorf("interval = %s, want %s", p.interval, tc.interval)
			}
		})
	}
}
