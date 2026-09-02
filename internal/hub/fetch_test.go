// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// shardedRepo is seven files of seven different lengths: no two dimensions
// here are equal, so a bound that is really the file count, or a length that
// is really the count, cannot hide.
func shardedRepo() (map[string]*fakeFile, []string) {
	sizes := []int{11, 13, 17, 19, 23, 29}
	files := map[string]*fakeFile{"config.json": {body: body(31, 0x01)}}
	order := []string{"config.json"}
	for i, n := range sizes {
		name := fmt.Sprintf("model-0000%d-of-00006.safetensors", i+1)
		files[name] = &fakeFile{body: body(n, byte(0x40+i)), digest: "auto"}
		order = append(order, name)
	}
	return files, order
}

func TestShardsDownloadInParallelUpToTheBound(t *testing.T) {
	// Section 3. The bound is observed, not timed: the CDN releases a handler
	// only once three are in flight, holds that batch in place while any
	// further request would arrive, and records the most that ever overlapped.
	// A bound smaller than three never trips the barrier; a bound larger than
	// three, or none at all, puts a fourth request in the count.
	const bound = 3
	files, order := shardedRepo()
	x := newFixture(t, files, order...)
	x.f.hold = bound
	x.c.Parallel = bound

	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatal(err)
	}
	if got := x.f.peakInflight(); got != bound {
		t.Errorf("%d downloads overlapped at most, want exactly %d out of %d files",
			got, bound, len(files))
	}
	for name, file := range files {
		mustHold(t, x.path(name), file.body)
	}
}

func TestABoundOfOneDownloadsOneAtATime(t *testing.T) {
	files, order := shardedRepo()
	x := newFixture(t, files, order...)
	x.f.hold = 1      // hold the first handler while a second would arrive
	x.c.Parallel = -1 // one at a time

	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatal(err)
	}
	if got := x.f.peakInflight(); got != 1 {
		t.Errorf("%d downloads overlapped, want 1", got)
	}
}

func TestOneFailureStopsTheRestAndIsWhatIsReported(t *testing.T) {
	// The cancellation a failure causes must not become the error the caller
	// sees: "context canceled" hides the checksum mismatch that started it.
	files, order := shardedRepo()
	files["model-00004-of-00006.safetensors"].digest = strings.Repeat("de", 32)
	x := newFixture(t, files, order...)
	x.c.Parallel = 2

	_, err := x.c.Fetch(t.Context(), x.ref)
	if !errors.Is(err, ErrChecksum) {
		t.Fatalf("Fetch = %v, want the ErrChecksum that stopped it", err)
	}
	if errors.Is(err, context.Canceled) {
		t.Error("the cancellation was reported instead of its cause")
	}
}

func TestACancelledFetchReportsTheCancellation(t *testing.T) {
	files, order := shardedRepo()
	x := newFixture(t, files, order...)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	if _, err := x.c.Fetch(ctx, x.ref); !errors.Is(err, context.Canceled) {
		t.Fatalf("Fetch = %v, want context.Canceled", err)
	}
}

func TestTheRevisionLockKeepsASecondProcessOut(t *testing.T) {
	// Section 3. A lock file per revision directory, taken with O_CREATE and
	// O_EXCL, which is atomic on every GOOS tgo builds for.
	x := newFixture(t, repoFiles())
	if err := os.MkdirAll(filepath.Dir(x.dir), 0o755); err != nil {
		t.Fatal(err)
	}
	held, err := acquireLock(t.Context(), x.dir+lockSuffix, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	x.c.LockWait = time.Millisecond

	_, err = x.c.Fetch(t.Context(), x.ref)
	if !errors.Is(err, ErrLocked) {
		t.Fatalf("Fetch against a held lock = %v, want ErrLocked", err)
	}
	if !strings.Contains(err.Error(), lockSuffix) {
		t.Errorf("Fetch = %v, want the lock file named so it can be removed", err)
	}
	if x.f.serveCount("model.safetensors") != 0 {
		t.Error("a locked fetch downloaded anyway")
	}

	held.release()
	held.release() // idempotent: the deferred release after an early return
	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatalf("Fetch after the lock was released: %v", err)
	}
	mustNotExist(t, x.dir+lockSuffix, "a finished fetch holds nothing")
}

func TestTheLockIsHeldForTheWholeDownloadNotJustItsStart(t *testing.T) {
	// Section 3's claim is that a lock keeps two tgo pull processes from
	// writing the same file, which needs the lock held WHILE the bytes land.
	// A lock taken and given straight back is acquired, never held, and every
	// other test here still passes: the second process is refused only if it
	// asks during the window a release-on-acquire has already closed.
	//
	// Observed from inside the download -- Progress runs on the download
	// goroutine -- and counted, never timed.
	x := newFixture(t, repoFiles())
	x.c.progressEvery = 1
	var mu sync.Mutex
	var held, gone int
	x.c.Progress = func(Event) {
		_, err := os.Stat(x.dir + lockSuffix)
		mu.Lock()
		defer mu.Unlock()
		if err == nil {
			held++
			return
		}
		gone++
	}

	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if held+gone == 0 {
		t.Fatal("no download reported progress, so nothing observed the lock")
	}
	if gone != 0 {
		t.Errorf("%d of %d reports from inside a download found no lock file; "+
			"a second process could have written the same file", gone, held+gone)
	}
	mustNotExist(t, x.dir+lockSuffix, "a finished fetch holds nothing")
}

func TestTheLockSitsBesideTheRevisionDirectoryNotInIt(t *testing.T) {
	// A checkpoint directory that holds a file no checkpoint has is a
	// directory safetensors.OpenRepo and this package disagree about.
	x := newFixture(t, repoFiles())
	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatal(err)
	}
	ents, err := os.ReadDir(x.dir)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, e := range ents {
		got[e.Name()] = true
	}
	if len(got) != 3 {
		t.Errorf("the revision directory holds %v, want only the three files", got)
	}
	for name := range repoFiles() {
		if !got[name] {
			t.Errorf("%s is missing from the revision directory", name)
		}
	}
}

func TestAWaiterTakesTheLockOnceItIsFree(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rev"+lockSuffix)
	first, err := acquireLock(t.Context(), path, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	released := make(chan struct{})
	go func() {
		defer close(released)
		first.release()
	}()
	<-released

	second, err := acquireLock(t.Context(), path, 5*time.Second)
	if err != nil {
		t.Fatalf("the lock was not free after it was released: %v", err)
	}
	second.release()
}

func TestAWaiterGivesUpWhenTheContextIsDone(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rev"+lockSuffix)
	held, err := acquireLock(t.Context(), path, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer held.release()

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := acquireLock(ctx, path, time.Minute); !errors.Is(err, context.Canceled) {
		t.Fatalf("acquireLock = %v, want context.Canceled", err)
	}
}

// A lock file the pid could not be written to is worse than no lock file: the
// refusal above names a holder nobody can identify, and an os.File is
// unbuffered, so the failed write never reappears at Close.
func TestALockThatCannotBeWrittenIsNotLeftBehind(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rev"+lockSuffix)
	saved := openLockFile
	t.Cleanup(func() { openLockFile = saved })
	openLockFile = func(p string) (*os.File, error) {
		fh, err := saved(p)
		if err != nil {
			return nil, err
		}
		// Closed before it is handed back, so every write to it fails.
		if err := fh.Close(); err != nil {
			return nil, err
		}
		return fh, nil
	}
	if _, err := acquireLock(t.Context(), path, 0); err == nil {
		t.Fatal("acquireLock reported a lock whose pid was never written")
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("the lock file outlived the write that could not fill it: %v", err)
	}
}

func TestALockPathThatCannotBeCreatedIsReported(t *testing.T) {
	_, err := acquireLock(t.Context(), filepath.Join(t.TempDir(), "no-such-dir", "x.lock"), 0)
	if err == nil || errors.Is(err, ErrLocked) {
		t.Fatalf("acquireLock = %v, want the create failure", err)
	}
	var nilLock *lock
	nilLock.release() // the deferred release of a lock that was never taken
}

func TestALocalPathNeedsNoNetworkAndNoCache(t *testing.T) {
	// 013-D4: a model is a repo id or a path. A path is already what
	// safetensors.OpenRepo wants, so nothing is copied.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	ref, err := ParseRef(dir)
	if err != nil {
		t.Fatal(err)
	}
	c := &Client{Endpoint: "http://127.0.0.1:1", Cache: t.TempDir()}
	got, err := c.Fetch(t.Context(), ref)
	if err != nil {
		t.Fatal(err)
	}
	if got != dir {
		t.Errorf("Fetch = %q, want %q", got, dir)
	}

	missing, err := ParseRef(filepath.Join(dir, "not-there"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Fetch(t.Context(), missing); err == nil {
		t.Error("a path that is not there was accepted")
	}

	file, err := ParseRef(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Fetch(t.Context(), file); err == nil ||
		!strings.Contains(err.Error(), "not a directory") {
		t.Errorf("a file = %v, want a refusal naming it", err)
	}
}

func TestARevisionWithNothingLoadableIsRefused(t *testing.T) {
	files := map[string]*fakeFile{
		"README.md":         {body: body(23, 0x02)},
		"pytorch_model.bin": {body: body(29, 0x03)},
	}
	x := newFixture(t, files)

	_, err := x.c.Fetch(t.Context(), x.ref)
	if !errors.Is(err, ErrNoFiles) {
		t.Fatalf("Fetch = %v, want ErrNoFiles", err)
	}
	if !strings.Contains(err.Error(), "2 files") {
		t.Errorf("Fetch = %v, want the count of what it did list", err)
	}
}

func TestAFilterChoosesWhatIsFetched(t *testing.T) {
	files, order := shardedRepo()
	x := newFixture(t, files, order...)
	x.c.Want = func(p string) bool { return p == "config.json" }

	dir, err := x.c.Fetch(t.Context(), x.ref)
	if err != nil {
		t.Fatal(err)
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(ents) != 1 || ents[0].Name() != "config.json" {
		t.Errorf("the directory holds %d entries, want only config.json", len(ents))
	}
}

func TestAnUnwritableCacheIsReportedRatherThanIgnored(t *testing.T) {
	x := newFixture(t, repoFiles())
	// A file where the cache root's directory must be. Every GOOS refuses to
	// make a directory under a regular file.
	blocker := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	x.c.Cache = blocker

	if _, err := x.c.Fetch(t.Context(), x.ref); err == nil {
		t.Fatal("a cache root that is a file was accepted")
	}
}

func TestAFetchIntoADirectoryThatCannotBeMadeFails(t *testing.T) {
	x := newFixture(t, repoFiles())
	if err := os.MkdirAll(filepath.Dir(x.dir), 0o755); err != nil {
		t.Fatal(err)
	}
	// The revision directory's name, taken by a regular file.
	if err := os.WriteFile(x.dir, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := x.c.Fetch(t.Context(), x.ref); err == nil {
		t.Fatal("a revision directory that is a file was accepted")
	}
}
