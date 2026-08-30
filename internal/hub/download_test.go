// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// fixture is a fake, a client, and the directory the revision will land in.
type fixture struct {
	f   *fake
	c   *Client
	ref Ref
	dir string
}

func newFixture(t *testing.T, files map[string]*fakeFile, order ...string) *fixture {
	t.Helper()
	f := newFake(t, files, order...)
	c := f.client(t)
	ref := testRef(t)
	dir, err := c.Dir(ref, fakeSHA)
	if err != nil {
		t.Fatal(err)
	}
	return &fixture{f: f, c: c, ref: ref, dir: dir}
}

// path is where a file of the revision lands.
func (x *fixture) path(name string) string { return filepath.Join(x.dir, name) }

// part is the temporary name that file downloads to.
func (x *fixture) part(name string) string { return x.path(name) + partSuffix }

// writePart stages a partial download, as an interrupted fetch would leave.
func (x *fixture) writePart(t *testing.T, name string, b []byte) {
	t.Helper()
	if err := os.MkdirAll(x.dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(x.part(name), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustNotExist(t *testing.T, path, why string) {
	t.Helper()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("%s still exists, and %s", filepath.Base(path), why)
	}
}

func mustHold(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s holds %d bytes, want %d, and they %s", filepath.Base(path),
			len(got), len(want), map[bool]string{true: "differ", false: "match"}[!bytes.Equal(got, want)])
	}
}

func TestAnInterruptedFetchLeavesNothingThatLooksWhole(t *testing.T) {
	// 013-D3, both halves. The final name never appears until the bytes are
	// checked, and the partial file DOES stay -- it is what the Range request
	// resumes from on the next attempt, which is the second half of this test.
	const prefix = 1129
	full := body(3701, 0xa5)
	var dropping atomic.Bool
	dropping.Store(true)

	files := repoFiles()
	files["model.safetensors"] = &fakeFile{
		body: full, digest: "auto",
		serve: func(w http.ResponseWriter, r *http.Request, f *fakeFile) {
			if !dropping.Load() {
				serveBytes(w, r, f.body)
				return
			}
			w.Header().Set("Content-Length", strconv.Itoa(len(f.body)))
			_, _ = w.Write(f.body[:prefix])
			w.(http.Flusher).Flush()
			<-r.Context().Done()
		},
	}
	x := newFixture(t, files, "model.safetensors")
	x.c.Parallel = 1

	// The cancel is driven by bytes reaching the file, not by the server
	// having written them: http.Client.Do returns when the HEADERS arrive, and
	// cancelling between that and the first body read aborts the request with
	// nothing on disk -- a different case from the one under test.
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var once sync.Once
	x.c.progressEvery = 1
	x.c.Progress = func(ev Event) {
		if ev.Path == "model.safetensors" && ev.Done >= prefix {
			once.Do(cancel)
		}
	}

	if _, err := x.c.Fetch(ctx, x.ref); err == nil {
		t.Fatal("an interrupted fetch reported success")
	}
	mustNotExist(t, x.path("model.safetensors"),
		"a reader would take a part of a shard for the whole one")
	st, err := os.Stat(x.part("model.safetensors"))
	if err != nil {
		t.Fatalf("the partial file is gone, so the resume has nothing to work from: %v", err)
	}
	if st.Size() != prefix {
		t.Fatalf("the partial file holds %d bytes, want %d", st.Size(), prefix)
	}

	// Now resume. The digest is published, so the rename happens only if the
	// prefix already on disk was folded into the hash rather than hashed away.
	dropping.Store(false)
	x.c.Progress = nil
	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatalf("the resume failed: %v", err)
	}
	mustHold(t, x.path("model.safetensors"), full)
	mustNotExist(t, x.part("model.safetensors"), "the rename should have consumed it")

	want := "bytes=" + strconv.Itoa(prefix) + "-"
	if !containsString(x.f.rangesSeen(), want) {
		t.Errorf("the CDN saw ranges %q, want one of them to be %q", x.f.rangesSeen(), want)
	}
}

func containsString(all []string, want string) bool {
	return slices.Contains(all, want)
}

func TestAServerThatIgnoresTheRangeRestartsTheHashToo(t *testing.T) {
	// A 200 to a Range request means the body is the whole file. The prefix on
	// disk is then worthless, and so is the hash of it: keeping either gives a
	// file of the right length whose digest matches nothing.
	full := body(43, 0x33)
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{
		body: full, digest: "auto",
		serve: func(w http.ResponseWriter, r *http.Request, f *fakeFile) {
			w.Header().Set("Content-Length", strconv.Itoa(len(f.body)))
			_, _ = w.Write(f.body) // the Range is ignored, as some mirrors do
		},
	}
	x := newFixture(t, files)
	x.writePart(t, "model.safetensors", bytes.Repeat([]byte{'X'}, 17))

	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	mustHold(t, x.path("model.safetensors"), full)
}

func TestAServerThatIgnoresTheRangeTruncatesALongerPartial(t *testing.T) {
	// The same 200-to-a-Range as above, with the partial file LONGER than the
	// body that arrives and no published length to catch the difference. The
	// temporary file has to be truncated back to zero: without that, the
	// forty-three bytes are written over a fifty-three byte file, ten stale
	// bytes survive on the end, and the rename publishes a file the download
	// counted as forty-three.
	full := body(43, 0x2b)
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{
		body: full, listSize: -1, // the API publishes no length for this one
		serve: func(w http.ResponseWriter, r *http.Request, f *fakeFile) {
			w.Header().Set("Content-Length", strconv.Itoa(len(f.body)))
			_, _ = w.Write(f.body) // the Range is ignored, as some mirrors do
		},
	}
	x := newFixture(t, files)
	x.writePart(t, "model.safetensors", bytes.Repeat([]byte{'Z'}, 53))

	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	mustHold(t, x.path("model.safetensors"), full)
}

func TestAPartialFileLongerThanTheServersIsDiscardedOnce(t *testing.T) {
	// 416: the file on disk is past the end of the file on the server, so
	// there is nothing to resume from. The partial file goes and the fetch is
	// tried once more from zero, which cannot reach this branch again.
	full := body(29, 0x71)
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{body: full, listSize: -1}
	x := newFixture(t, files)
	x.writePart(t, "model.safetensors", bytes.Repeat([]byte{'Y'}, 53))

	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	mustHold(t, x.path("model.safetensors"), full)
	if got := x.f.serveCount("model.safetensors"); got != 2 {
		t.Errorf("the CDN served the shard %d times, want 2: one refused range and one restart", got)
	}
}

func TestA206FromTheWrongOffsetIsRefused(t *testing.T) {
	// A 206 that starts somewhere else produces a file of the right length
	// with a hole in it. Length alone would not catch it.
	full := body(47, 0x19)
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{
		body: full, digest: "auto",
		serve: func(w http.ResponseWriter, r *http.Request, f *fakeFile) {
			w.Header().Set("Content-Range",
				fmt.Sprintf("bytes %d-%d/%d", 3, len(f.body)-1, len(f.body)))
			w.Header().Set("Content-Length", strconv.Itoa(len(f.body)-3))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(f.body[3:])
		},
	}
	x := newFixture(t, files)
	x.writePart(t, "model.safetensors", full[:19])

	_, err := x.c.Fetch(t.Context(), x.ref)
	if err == nil || !strings.Contains(err.Error(), "starts at 3") {
		t.Fatalf("Fetch = %v, want a refusal naming the offset", err)
	}
	mustNotExist(t, x.path("model.safetensors"), "the bytes were never verified")
}

func TestContentRangeIsParsedBeforeItIsTrusted(t *testing.T) {
	for _, v := range []string{"", "items 0-3/4", "bytes 12", "bytes abc-9/40"} {
		if err := checkContentRange(v, 12); err == nil {
			t.Errorf("checkContentRange(%q) was accepted", v)
		}
	}
	if err := checkContentRange("bytes 12-39/40", 12); err != nil {
		t.Errorf("a well-formed Content-Range was refused: %v", err)
	}
}

func TestAChecksumMismatchDeletesAndFails(t *testing.T) {
	// Section 2. Keeping it would let the next fetch resume a prefix that can
	// never hash to the published digest, so the download would never
	// converge.
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{
		body:   body(41, 0xa5),
		digest: strings.Repeat("de", 32), // not the digest of anything here
	}
	x := newFixture(t, files)

	_, err := x.c.Fetch(t.Context(), x.ref)
	if !errors.Is(err, ErrChecksum) {
		t.Fatalf("Fetch = %v, want ErrChecksum", err)
	}
	mustNotExist(t, x.part("model.safetensors"), "resuming it could never converge")
	mustNotExist(t, x.path("model.safetensors"), "the bytes are not what was published")
}

func TestADeclaredLengthThatDisagreesWithTheListingFails(t *testing.T) {
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{body: body(41, 0xa5), listSize: 97}
	x := newFixture(t, files)

	_, err := x.c.Fetch(t.Context(), x.ref)
	if !errors.Is(err, ErrSize) {
		t.Fatalf("Fetch = %v, want ErrSize", err)
	}
	if !strings.Contains(err.Error(), "97") {
		t.Errorf("Fetch = %v, want the published length named", err)
	}
	mustNotExist(t, x.path("model.safetensors"), "nothing was verified")
	// Refused from the response header, before the body. An eight gigabyte
	// shard whose declared length already disagrees must not be transferred
	// to find that out, and a fetch that read it anyway would leave the
	// bytes behind as something to resume.
	mustNotExist(t, x.part("model.safetensors"),
		"the disagreement was in the header, so no byte should have landed")
}

func TestATruncatedBodyKeepsThePartialAndALongOneDoesNot(t *testing.T) {
	// The two directions have to go opposite ways. If both deleted, a resume
	// is impossible; if both kept, a wrong file is appended to forever.
	short := &fakeFile{
		body: body(41, 0xa5), listSize: 41,
		serve: func(w http.ResponseWriter, r *http.Request, f *fakeFile) {
			w.(http.Flusher).Flush() // no Content-Length: the client cannot know
			_, _ = w.Write(f.body[:17])
		},
	}
	files := repoFiles()
	files["model.safetensors"] = short
	x := newFixture(t, files)
	if _, err := x.c.Fetch(t.Context(), x.ref); !errors.Is(err, ErrSize) {
		t.Fatalf("a short body = %v, want ErrSize", err)
	}
	st, err := os.Stat(x.part("model.safetensors"))
	if err != nil {
		t.Fatalf("the partial file of a truncated transfer is gone: %v", err)
	}
	if st.Size() != 17 {
		t.Errorf("the partial file holds %d bytes, want 17", st.Size())
	}

	long := &fakeFile{
		body: body(64, 0x11), listSize: 40,
		serve: func(w http.ResponseWriter, r *http.Request, f *fakeFile) {
			w.(http.Flusher).Flush()
			_, _ = w.Write(f.body)
		},
	}
	files2 := repoFiles()
	files2["model.safetensors"] = long
	y := newFixture(t, files2)
	if _, err := y.c.Fetch(t.Context(), y.ref); !errors.Is(err, ErrSize) {
		t.Fatalf("a long body = %v, want ErrSize", err)
	}
	mustNotExist(t, y.part("model.safetensors"), "there is nothing to resume from")
}

func TestAFileAlreadyAtThePublishedLengthIsNotFetchedAgain(t *testing.T) {
	x := newFixture(t, repoFiles())
	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatal(err)
	}
	before := x.f.serveCount("model.safetensors")
	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatal(err)
	}
	if after := x.f.serveCount("model.safetensors"); after != before {
		t.Errorf("the shard was served %d times and then %d; a cached file was re-fetched",
			before, after)
	}
}

func TestProgressReportsAsBytesLandAndOnceAtTheEnd(t *testing.T) {
	// Counts, never durations: a sub-millisecond interval measures as zero on
	// a runner whose clock ticks every 15 milliseconds.
	files := map[string]*fakeFile{"model.safetensors": {body: body(41, 0xa5), digest: "auto"}}
	x := newFixture(t, files)
	x.c.progressEvery = 8
	var events []Event
	var mu sync.Mutex
	x.c.Progress = func(ev Event) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, ev)
	}
	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatal(err)
	}
	if len(events) < 2 {
		t.Fatalf("saw %d progress events, want at least one partial and one final", len(events))
	}
	last := events[len(events)-1]
	if !last.Complete || last.Done != 41 || last.Total != 41 {
		t.Errorf("the final event is %+v, want 41 of 41 and complete", last)
	}
	for _, ev := range events[:len(events)-1] {
		if ev.Path != "model.safetensors" || ev.Done > 41 {
			t.Errorf("progress event %+v", ev)
		}
	}
}
