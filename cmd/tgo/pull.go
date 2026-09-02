// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/latere-ai/tgo/internal/hub"
	"github.com/latere-ai/tgo/weights"
)

// hfTokenEnv are the environment variables a Hugging Face token is read from
// when --token is not given, in the order huggingface_hub itself reads them.
//
// specs/013-distribution.md names no convention, and internal/hub takes the
// token as a field rather than reading the environment. Reading it here is what
// lets a gated repo be fetched without the token appearing in a shell history
// or in the process table, which is the reason the convention exists. See this
// package's reported discrepancies.
var hfTokenEnv = []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}

// pullOptions is `tgo pull`'s command line, parsed.
type pullOptions struct {
	Ref   hub.Ref
	Token string
}

// pullFlagSet declares what `tgo pull` accepts. See [runFlagSet] for why
// declaring is separate from parsing.
func pullFlagSet() (*flag.FlagSet, *pullFlags) {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	return fs, &pullFlags{
		revision: fs.String("revision", "", "branch, tag or commit sha; the default is the repo's main"),
		token:    fs.String("token", "", "Hugging Face access token; the default reads $HF_TOKEN"),
	}
}

// pullFlags holds `tgo pull`'s flag values.
type pullFlags struct{ revision, token *string }

// parsePull parses and checks `tgo pull`'s arguments.
func parsePull(args []string) (pullOptions, error) {
	fs, f := pullFlagSet()
	id, err := onePositional(fs, args, "repo id")
	if err != nil {
		return pullOptions{}, err
	}
	rev := strings.TrimSpace(*f.revision)
	if rev != "" && strings.Contains(id, "@") {
		return pullOptions{}, fmt.Errorf("%w: %q already names a revision and --revision %s names another; "+
			"give one of them", errUsage, id, rev)
	}
	ref, err := hub.ParseRef(id)
	if err != nil {
		return pullOptions{}, fmt.Errorf("%w: %w", errUsage, err)
	}
	if ref.IsLocal() {
		return pullOptions{}, fmt.Errorf("%w: %s is a directory on this machine, and there is nothing "+
			"to fetch; `tgo pull` takes a Hugging Face repo id such as Qwen/Qwen3-0.6B", errUsage, id)
	}
	if rev != "" {
		ref.Revision = rev
	}
	token := *f.token
	for _, name := range hfTokenEnv {
		if token != "" {
			break
		}
		token = os.Getenv(name)
	}
	return pullOptions{Ref: ref, Token: strings.TrimSpace(token)}, nil
}

// puller is what `tgo pull` needs from [hub.Client].
//
// An interface rather than the concrete client, for the reason [engine] is one:
// every line below -- the listing, the refusals, the progress and the report --
// is then reachable from a test that touches no network and writes no cache.
type puller interface {
	Revision(ctx context.Context, ref hub.Ref) (*hub.Revision, error)
	Fetch(ctx context.Context, ref hub.Ref) (string, error)
}

// newPuller builds the client one `tgo pull` uses. It is a variable so that the
// tests can replace it.
var newPuller = func(o pullOptions, pr *progress) puller {
	return &hub.Client{Token: o.Token, Progress: pr.Event}
}

// cmdPull downloads a checkpoint and prints where it landed.
//
// The path goes to stdout and everything else to stderr, so that
// `tgo run "$(tgo pull Qwen/Qwen3-0.6B)"` works and the progress an operator
// watches does not become part of the path.
func cmdPull(args []string, stdout, stderr io.Writer) error {
	o, err := parsePull(args)
	if err != nil {
		return err
	}
	pr := newProgress(stderr)
	c := newPuller(o, pr)

	// Ctrl-C cancels the download rather than killing the process mid-write.
	// specs/013-distribution.md 013-D3 makes that safe: a partial file keeps
	// its temporary name, so the next `tgo pull` resumes from it and never
	// leaves a file that looks whole.
	ctx, stop := interrupts()
	defer stop()

	// The listing is asked for before the fetch so that a repo that does not
	// exist, one that is gated, and one that carries nothing loadable are all
	// refused before a byte moves -- and so that the progress below has a
	// total. Fetch resolves the revision again, which is one extra API call
	// and the only way to know the size in advance; if a branch moves between
	// the two, the sha printed here is the one that was listed and the sha in
	// the printed path is the one that was fetched.
	rev, err := c.Revision(ctx, o.Ref)
	if err != nil {
		return err
	}
	files, total := wantedFiles(rev)
	if len(files) == 0 {
		return fmt.Errorf("%w: %s at %s lists %d files and none of them is a safetensors checkpoint "+
			"tgo can load", hub.ErrNoFiles, o.Ref.ID(), shortSHA(rev.SHA), len(rev.Files))
	}
	_, _ = fmt.Fprintf(stderr, "%s at %s: %d files, %s\n", o.Ref.ID(), shortSHA(rev.SHA), len(files), weights.HumanBytes(total))
	pr.start(len(files), total)

	dir, err := c.Fetch(ctx, o.Ref)
	pr.stop()
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintln(stdout, dir)
	return nil
}

// wantedFiles is the listing narrowed to what a fetch will actually download,
// and what those files weigh.
//
// The same [hub.Wanted] filter [hub.Client.Fetch] applies, applied here too:
// a real repo carries onnx/, coreml/ and ggml variants of the same weights, so
// a total taken over the whole listing announces a download several times the
// size of the one that follows.
//
// A file the API published no size for contributes nothing to the total, which
// is what makes the total a lower bound rather than a wrong number.
func wantedFiles(rev *hub.Revision) (files []hub.File, total int64) {
	for _, f := range rev.Files {
		if hub.Wanted(f.Path) {
			files = append(files, f)
			total += f.Size
		}
	}
	return files, total
}

// shortSHA abbreviates a commit for a line a human reads. The cache is keyed by
// the whole sha (013-D2) and the whole sha is in the path this command prints,
// so nothing here has to be pasted back.
func shortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// progressInterval is how often a progress line is written, at a terminal and
// redirected.
//
// The two differ by more than a factor of a hundred because the outputs are
// different things. A terminal repaints one line, so ten times a second reads
// as motion and costs nothing; a file or a CI log keeps every line, so ten a
// second is a megabyte of noise on a download that takes ten minutes and two a
// minute is a record of it.
const (
	ttyInterval = 100 * time.Millisecond
	logInterval = 30 * time.Second
)

// progress renders a download as it lands.
//
// [hub.Client] calls Progress from every download goroutine, so every field
// below is under the mutex. The per-file map rather than a running sum: an
// event reports a file's bytes-so-far, not a delta, and it is re-reported at
// each threshold, so adding them would count the same bytes many times over.
type progress struct {
	w        io.Writer
	tty      bool
	interval time.Duration

	mu       sync.Mutex
	done     map[string]int64
	files    int
	total    int64
	complete int
	begun    time.Time
	painted  time.Time
	dirty    bool
}

// newProgress builds a reporter for w, in the shape w can carry.
//
// A terminal is one whose file mode says character device. It is asked of the
// writer rather than of os.Stderr so that a test's buffer takes the redirected
// path, which is both the quiet one and the one whose output is deterministic.
func newProgress(w io.Writer) *progress {
	p := &progress{w: w, done: map[string]int64{}, interval: logInterval, begun: time.Now()}
	if f, ok := w.(interface{ Stat() (os.FileInfo, error) }); ok {
		if st, err := f.Stat(); err == nil && st.Mode()&os.ModeCharDevice != 0 {
			p.tty, p.interval = true, ttyInterval
		}
	}
	return p
}

// start records what the whole download weighs, so that a share of it can be
// reported rather than a running count of bytes.
//
// The paint clock starts here rather than at zero, which is what keeps the
// first line one interval away: the caller has just printed how many files and
// how many bytes are coming, and an immediate "0.0% of 1.31 GiB" restates it
// with less in it.
func (p *progress) start(files int, total int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.files, p.total = files, total
	p.begun, p.painted = time.Now(), time.Now()
}

// Event is [hub.Client.Progress]. It is safe for concurrent use.
func (p *progress) Event(e hub.Event) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.done[e.Path] = e.Done
	p.dirty = true
	if e.Complete {
		p.complete++
		if !p.tty {
			// One line per file, which is the whole record a log needs: what
			// arrived, how big it was, and whether it was resumed rather than
			// fetched.
			how := "fetched"
			if e.Resumed {
				how = "resumed"
			}
			p.line(fmt.Sprintf("%s %s (%s)\n", how, e.Path, weights.HumanBytes(e.Done)))
		}
		return
	}
	if now := time.Now(); now.Sub(p.painted) >= p.interval {
		p.painted = now
		p.paint()
	}
}

// stop writes the last state, so that a download which finished between two
// intervals does not end reading as if it had stalled.
func (p *progress) stop() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.dirty {
		return
	}
	p.paint()
	if p.tty {
		_, _ = fmt.Fprint(p.w, "\n")
	}
	p.dirty = false
}

// paint writes one progress line. The caller holds the mutex.
func (p *progress) paint() {
	var got int64
	for _, n := range p.done {
		got += n
	}
	var b strings.Builder
	if p.total > 0 {
		fmt.Fprintf(&b, "%5.1f%%  %s of %s", 100*float64(got)/float64(p.total), weights.HumanBytes(got), weights.HumanBytes(p.total))
	} else {
		// The API published no size. Bytes without a share, because a
		// percentage of an unknown total is a number somebody would read.
		fmt.Fprintf(&b, "%s of an unpublished total", weights.HumanBytes(got))
	}
	fmt.Fprintf(&b, ", %d/%d files", p.complete, p.files)
	if elapsed := time.Since(p.begun); elapsed > 0 && got > 0 {
		rate := float64(got) / elapsed.Seconds()
		fmt.Fprintf(&b, ", %s/s", weights.HumanBytes(int64(rate)))
		if left := p.total - got; left > 0 && rate > 0 {
			fmt.Fprintf(&b, ", %s left", humanDuration(time.Duration(float64(left)/rate*float64(time.Second))))
		}
	}
	if p.tty {
		// A carriage return and no newline: the line is repainted in place.
		// Padded because a shorter line would leave the tail of a longer one
		// on screen -- "10.0%" over "100.0%" reads as a download going
		// backwards.
		p.line(fmt.Sprintf("\r%-78s", b.String()))
		return
	}
	p.line(b.String() + "\n")
}

// line writes one string. The caller holds the mutex.
func (p *progress) line(s string) { _, _ = fmt.Fprint(p.w, s) }
