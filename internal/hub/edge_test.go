// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"crypto/sha256"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestFetchFileRefusesAPathTheListingShouldNeverHaveHad(t *testing.T) {
	// The second line of the defence. Revision refuses a traversing listing,
	// and this refuses one that reached the downloader anyway -- a filter or a
	// caller could hand one over.
	x := newFixture(t, repoFiles())
	err := x.c.fetchFile(t.Context(), x.ref, fakeSHA,
		File{Path: "../escape.json", Size: 3}, x.dir)
	if !errors.Is(err, ErrUnsafePath) {
		t.Fatalf("fetchFile = %v, want ErrUnsafePath", err)
	}
}

func TestFetchFileReportsADirectoryItCannotMake(t *testing.T) {
	x := newFixture(t, repoFiles())
	if err := os.MkdirAll(x.dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// "sub" is a regular file, so no directory of that name can be made.
	if err := os.WriteFile(filepath.Join(x.dir, "sub"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := x.c.fetchFile(t.Context(), x.ref, fakeSHA,
		File{Path: "sub/model.safetensors"}, x.dir)
	if err == nil {
		t.Fatal("a file in the way of a directory was accepted")
	}
}

func TestAPartialFileThatIsNotAFileIsRefused(t *testing.T) {
	// A directory where the temporary file belongs. It cannot be resumed from
	// and it cannot be removed, so the fetch says so rather than writing
	// somewhere else.
	x := newFixture(t, repoFiles())
	part := x.part("model.safetensors")
	if err := os.MkdirAll(filepath.Join(part, "occupied"), 0o755); err != nil {
		t.Fatal(err)
	}
	_, err := x.c.Fetch(t.Context(), x.ref)
	if err == nil {
		t.Fatal("a directory in the way of the temporary file was accepted")
	}
	if errors.Is(err, ErrChecksum) || errors.Is(err, ErrSize) {
		t.Errorf("Fetch = %v, want the filesystem failure", err)
	}
}

func TestAPartialFileThatCannotBeReadStartsAgain(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("chmod does not deny the owner a read on Windows")
	}
	x := newFixture(t, repoFiles())
	if err := os.MkdirAll(x.dir, 0o755); err != nil {
		t.Fatal(err)
	}
	part := x.part("model.safetensors")
	if err := os.WriteFile(part, []byte("unreadable"), 0o000); err != nil {
		t.Fatal(err)
	}
	digest := sha256.New()
	if got := resumeFrom(part, 41, digest); got != 0 {
		t.Errorf("resumeFrom = %d, want 0 for a file it cannot read", got)
	}
}

func TestA416WithNothingToResumeIsJustAFailure(t *testing.T) {
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{
		body: body(41, 0xa5),
		serve: func(w http.ResponseWriter, r *http.Request, _ *fakeFile) {
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
		},
	}
	x := newFixture(t, files)
	_, err := x.c.Fetch(t.Context(), x.ref)
	if err == nil || !strings.Contains(err.Error(), "416") {
		t.Fatalf("Fetch = %v, want the 416 reported", err)
	}
	if errors.Is(err, errStalePartial) {
		t.Error("a 416 with no partial file was retried as though there were one")
	}
	if got := x.f.serveCount("model.safetensors"); got != 1 {
		t.Errorf("the shard was served %d times, want 1: there was nothing to retry", got)
	}
}

func TestCacheDirAndDirReportAMachineWithNoHome(t *testing.T) {
	t.Setenv("TGO_CACHE", "")
	t.Setenv("XDG_CACHE_HOME", "")
	t.Setenv("HOME", "")
	t.Setenv("USERPROFILE", "")
	t.Setenv("LocalAppData", "")
	t.Setenv("home", "")
	if _, err := CacheDir(); err == nil {
		t.Skip("this machine has a cache directory without a home directory")
	}
	if _, err := (&Client{}).Dir(Ref{Org: "acme", Repo: "qwen-mini"}, fakeSHA); err == nil {
		t.Error("Dir reported a directory it cannot name")
	}
	if _, err := (&Client{}).Fetch(t.Context(), Ref{Org: "acme", Repo: "qwen-mini", Revision: "main"}); err == nil {
		t.Error("Fetch reported success with nowhere to write")
	}
}

func TestACDNRefusalIsNotReportedAsAGatedRepo(t *testing.T) {
	// The reader of "the repo is gated or private" goes looking for a token
	// they already have. A 403 from the second hop is about the redirect, and
	// says so.
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{
		body: body(41, 0xa5),
		serve: func(w http.ResponseWriter, r *http.Request, _ *fakeFile) {
			http.Error(w, "signature expired", http.StatusForbidden)
		},
	}
	x := newFixture(t, files)

	_, err := x.c.Fetch(t.Context(), x.ref)
	if err == nil {
		t.Fatal("a 403 from the CDN was accepted")
	}
	if errors.Is(err, ErrUnauthorized) {
		t.Errorf("Fetch = %v, want the CDN hop named rather than the repo blamed", err)
	}
	for _, want := range []string{"CDN", "signature expired", "Authorization"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Fetch = %v, want %q in it", err, want)
		}
	}
}

func TestAFinalFileOfTheWrongLengthIsReplaced(t *testing.T) {
	// The recovery story for a cache poisoned by an older run: the file has
	// the final name and the wrong length, so it is fetched again and the
	// rename replaces it. os.Rename over an existing file is also the call
	// whose behaviour differs most between GOOS.
	files := repoFiles()
	x := newFixture(t, files)
	if err := os.MkdirAll(x.dir, 0o755); err != nil {
		t.Fatal(err)
	}
	corrupt := body(23, 0x77) // not 41 bytes, and not the shard's bytes
	if err := os.WriteFile(x.path("model.safetensors"), corrupt, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := x.c.Fetch(t.Context(), x.ref); err != nil {
		t.Fatal(err)
	}
	mustHold(t, x.path("model.safetensors"), files["model.safetensors"].body)
	mustNotExist(t, x.part("model.safetensors"), "the rename should have consumed it")
}
