// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// body makes n bytes that depend on n and on seed, so that a file cut short or
// spliced from two responses does not hash to what a whole one hashes to.
func body(n int, seed byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = seed ^ byte(i*7+n)
	}
	return b
}

// repoFiles is the standard fake repo: three files, three different lengths,
// and only the shard carries a published digest -- which is what the API does,
// since only git-lfs tracked files have one.
func repoFiles() map[string]*fakeFile {
	return map[string]*fakeFile{
		"config.json":       {body: []byte(`{"architectures":["Qwen3ForCausalLM"],"hidden_size":11}`)},
		"tokenizer.json":    {body: body(37, 0x5a)},
		"model.safetensors": {body: body(41, 0xa5), digest: "auto"},
	}
}

func testRef(t *testing.T) Ref {
	t.Helper()
	ref, err := ParseRef(fakeRepo)
	if err != nil {
		t.Fatal(err)
	}
	return ref
}

func TestResolveDropsTheCredentialOnTheHopToTheCDN(t *testing.T) {
	// 013 section 1's trap. resolve answers 302 towards a CDN, and the CDN
	// answers 403 to a request that still carries the token. The two servers
	// are both on 127.0.0.1, so Go's own rule -- same domain or a subdomain,
	// port ignored -- would forward it. See the test below, which proves that.
	f := newFake(t, repoFiles())
	c := f.client(t)
	c.Token = "hf_secrettoken"

	dir, err := c.Fetch(t.Context(), testRef(t))
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}

	auths := f.authSeenByCDN()
	if len(auths) != len(repoFiles()) {
		t.Fatalf("the CDN saw %d requests, want %d", len(auths), len(repoFiles()))
	}
	for i, a := range auths {
		if a != "" {
			t.Errorf("CDN request %d carried Authorization %q", i, a)
		}
	}
	// The header must be dropped on the cross-host hop and nowhere else: an
	// API request without it reads only public repos.
	f.mu.Lock()
	resolve := append([]string(nil), f.resolveAuth...)
	f.mu.Unlock()
	if len(resolve) == 0 {
		t.Fatal("the resolve endpoint was never reached")
	}
	for i, a := range resolve {
		if a != "Bearer hf_secrettoken" {
			t.Errorf("resolve request %d carried %q, want the bearer token", i, a)
		}
	}
	for name, file := range repoFiles() {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if string(got) != string(file.body) {
			t.Errorf("%s: %d bytes on disk, want %d", name, len(got), len(file.body))
		}
	}
}

func TestGoDefaultRedirectPolicyForwardsTheCredential(t *testing.T) {
	// The reason dropAuthAcrossHosts exists, stated as a test rather than as a
	// comment. Go's shouldCopyHeaderOnRedirect compares url.Hostname(), which
	// strips the port, so two servers on 127.0.0.1 are one domain to it and
	// the token rides along to the CDN -- which answers 403.
	f := newFake(t, repoFiles())
	u := f.api.URL + "/" + fakeRepo + "/resolve/main/config.json"
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, u, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer hf_secrettoken")

	resp, err := (&http.Client{}).Do(req) // the default policy, on purpose
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)

	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("the default client got %s; the trap this package guards is gone "+
			"and dropAuthAcrossHosts may be testing nothing", resp.Status)
	}
	seen := f.authSeenByCDN()
	if len(seen) != 1 || seen[0] == "" {
		t.Fatalf("the CDN saw %q, want the forwarded credential", seen)
	}
}

func TestLFSPointerFromTheGitEndpointIsNamed(t *testing.T) {
	// The other trap. A pointer is about 130 bytes where the listing published
	// four megabytes, so a length check would fire first and report a
	// truncated download -- which is the confusing failure. The pointer is
	// detected before every other check and reported as itself.
	const pointer = "version https://git-lfs.github.com/spec/v1\n" +
		"oid sha256:9c3a4e1b0d2f5a687796b5c4d3e2f10112233445566778899aabbccddeeff00\n" +
		"size 4194304\n"
	files := repoFiles()
	files["model.safetensors"] = &fakeFile{
		listSize: 4194304,
		serve: func(w http.ResponseWriter, r *http.Request, _ *fakeFile) {
			w.Header().Set("Content-Length", strconv.Itoa(len(pointer)))
			io.WriteString(w, pointer)
		},
	}
	c := newFake(t, files).client(t)

	_, err := c.Fetch(t.Context(), testRef(t))
	if !errors.Is(err, ErrLFSPointer) {
		t.Fatalf("Fetch = %v, want ErrLFSPointer", err)
	}
	if errors.Is(err, ErrSize) {
		t.Fatal("the pointer was reported as a length mismatch, which is the " +
			"confusing failure 013 section 1 asks to name")
	}
	if len(pointer) > 200 {
		t.Fatalf("the fixture is %d bytes; a pointer is about 130", len(pointer))
	}
}

func TestRevisionResolvesTheShaAndTheDigests(t *testing.T) {
	f := newFake(t, repoFiles(), "config.json", "tokenizer.json", "model.safetensors")
	c := f.client(t)

	rev, err := c.Revision(t.Context(), testRef(t))
	if err != nil {
		t.Fatal(err)
	}
	if rev.SHA != fakeSHA {
		t.Errorf("SHA = %q, want %q", rev.SHA, fakeSHA)
	}
	if len(rev.Files) != 3 {
		t.Fatalf("listed %d files, want 3", len(rev.Files))
	}
	if rev.Files[0].Path != "config.json" || rev.Files[0].SHA256 != "" {
		t.Errorf("config.json = %+v, want no published digest", rev.Files[0])
	}
	shard := rev.Files[2]
	if shard.Size != 41 || len(shard.SHA256) != 64 {
		t.Errorf("model.safetensors = %+v, want 41 bytes and a digest", shard)
	}
}

func TestRevisionNamesWhatTheAPIRefused(t *testing.T) {
	f := newFake(t, repoFiles())
	c := f.client(t)

	missing, err := ParseRef("acme/not-there")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Revision(t.Context(), missing); !errors.Is(err, ErrNotFound) {
		t.Errorf("a missing repo = %v, want ErrNotFound", err)
	}

	badRev, err := ParseRef(fakeRepo + "@no-such-branch")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Revision(t.Context(), badRev); !errors.Is(err, ErrNotFound) {
		t.Errorf("a missing revision = %v, want ErrNotFound", err)
	}

	local := Ref{Local: t.TempDir()}
	if _, err := c.Revision(t.Context(), local); err == nil {
		t.Error("a local ref was given a revision")
	}
}

// statusServer answers every request with one status and body.
func statusServer(t *testing.T, code int, body string) *Client {
	t.Helper()
	srv := newStatusServer(t, code, body)
	return &Client{Endpoint: srv, Cache: t.TempDir()}
}

func TestRevisionSeparatesGatedFromMissingFromBroken(t *testing.T) {
	// A gated repo and a typo must not report the same thing: one is fixed
	// with a token and the other with a correction.
	for _, tc := range []struct {
		code int
		want error
	}{
		{http.StatusUnauthorized, ErrUnauthorized},
		{http.StatusForbidden, ErrUnauthorized},
		{http.StatusNotFound, ErrNotFound},
	} {
		c := statusServer(t, tc.code, `{"error":"nope"}`)
		_, err := c.Revision(t.Context(), testRef(t))
		if !errors.Is(err, tc.want) {
			t.Errorf("status %d = %v, want %v", tc.code, err, tc.want)
		}
	}

	c := statusServer(t, http.StatusInternalServerError, "upstream is unwell")
	_, err := c.Revision(t.Context(), testRef(t))
	if err == nil || errors.Is(err, ErrNotFound) || errors.Is(err, ErrUnauthorized) {
		t.Errorf("status 500 = %v, want an unclassified failure", err)
	}
	if !strings.Contains(err.Error(), "upstream is unwell") {
		t.Errorf("status 500 = %v, want the body quoted", err)
	}
}

func TestRevisionRefusesAListingItCannotTrust(t *testing.T) {
	c := statusServer(t, http.StatusOK, `{"sha": "0f1e`)
	if _, err := c.Revision(t.Context(), testRef(t)); err == nil ||
		!strings.Contains(err.Error(), "not JSON") {
		t.Errorf("a truncated listing = %v", err)
	}

	c = statusServer(t, http.StatusOK, `{"siblings":[{"rfilename":"config.json"}]}`)
	if _, err := c.Revision(t.Context(), testRef(t)); err == nil ||
		!strings.Contains(err.Error(), "commit sha") {
		t.Errorf("a listing with no sha = %v", err)
	}

	// The traversal. Without the check this writes outside the cache root.
	c = statusServer(t, http.StatusOK,
		`{"sha":"`+fakeSHA+`","siblings":[{"rfilename":"../../../.ssh/authorized_keys"}]}`)
	if _, err := c.Revision(t.Context(), testRef(t)); !errors.Is(err, ErrUnsafePath) {
		t.Errorf("a traversing listing = %v, want ErrUnsafePath", err)
	}

	if _, err := statusServer(t, http.StatusOK, `{"sha":"x"}`).Fetch(t.Context(), testRef(t)); err == nil {
		t.Error("a listing with no files was fetched")
	}
}

func TestRevisionTakesTheDigestTheAPIActuallyGave(t *testing.T) {
	// ?blobs=true publishes the LFS block, and which field carries the digest
	// has moved: oid, sometimes prefixed sha256:, and sometimes a sha256 field
	// beside it. Anything that is not 64 hex digits is no digest at all, and
	// turning the check off is better than comparing against a value that can
	// never match.
	digest := strings.Repeat("ab", 32)
	for _, tc := range []struct {
		json string
		want string
	}{
		{`{"oid":"` + digest + `","size":41}`, digest},
		{`{"oid":"sha256:` + digest + `","size":41}`, digest},
		{`{"oid":"whatever","sha256":"` + strings.ToUpper(digest) + `","size":41}`, digest},
		{`{"oid":"","size":41}`, ""},
		{`{"oid":"1234","size":41}`, ""},
		{`{"oid":"` + strings.Repeat("zz", 32) + `","size":41}`, ""},
	} {
		c := statusServer(t, http.StatusOK, `{"sha":"`+fakeSHA+
			`","siblings":[{"rfilename":"model.safetensors","size":7,"lfs":`+tc.json+`}]}`)
		rev, err := c.Revision(t.Context(), testRef(t))
		if err != nil {
			t.Fatalf("%s: %v", tc.json, err)
		}
		if got := rev.Files[0].SHA256; got != tc.want {
			t.Errorf("%s: digest %q, want %q", tc.json, got, tc.want)
		}
		// The LFS block's size wins over the sibling's, which is 7 here and
		// is the size of the pointer rather than of the file.
		if got := rev.Files[0].Size; got != 41 {
			t.Errorf("%s: size %d, want 41", tc.json, got)
		}
	}
}

func TestURLsEscapeWhatTheyMustAndKeepTheSlashesOfARef(t *testing.T) {
	c := &Client{Endpoint: "https://example.test/"}
	ref := Ref{Org: "acme", Repo: "qwen-mini", Revision: "refs/pr/1"}
	got := c.resolveURL(ref, ref.Revision, "sub dir/model.safetensors")
	want := "https://example.test/acme/qwen-mini/resolve/refs/pr/1/sub%20dir/model.safetensors"
	if got != want {
		t.Errorf("resolveURL = %q, want %q", got, want)
	}
	if got := escapeRevision("feature/a b"); got != "feature/a%20b" {
		t.Errorf("escapeRevision = %q", got)
	}
}

func TestEndpointFallsBackToTheEnvironmentThenTheDefault(t *testing.T) {
	t.Setenv("HF_ENDPOINT", "https://mirror.test/")
	if got, want := (&Client{}).endpoint(), "https://mirror.test"; got != want {
		t.Errorf("endpoint = %q, want %q", got, want)
	}
	t.Setenv("HF_ENDPOINT", "")
	if got, want := (&Client{}).endpoint(), DefaultEndpoint; got != want {
		t.Errorf("endpoint = %q, want %q", got, want)
	}
	if got, want := (&Client{Endpoint: "https://x.test"}).endpoint(), "https://x.test"; got != want {
		t.Errorf("endpoint = %q, want %q", got, want)
	}
}

func TestDirIsKeyedByTheResolvedShaNotTheRequestedRef(t *testing.T) {
	// 013-D2. main moves; the directory it resolved to does not, so two
	// revisions coexist and neither corrupts the other.
	c := &Client{Cache: filepath.FromSlash("/cache")}
	ref := Ref{Org: "acme", Repo: "qwen-mini", Revision: "main"}
	first, err := c.Dir(ref, "aaaa1111")
	if err != nil {
		t.Fatal(err)
	}
	second, err := c.Dir(ref, "bbbb2222")
	if err != nil {
		t.Fatal(err)
	}
	if first == second {
		t.Fatal("two revisions share a directory")
	}
	if want := filepath.FromSlash("/cache/models/acme/qwen-mini/aaaa1111"); first != want {
		t.Errorf("Dir = %q, want %q", first, want)
	}

	t.Setenv("TGO_CACHE", filepath.FromSlash("/env-cache"))
	got, err := (&Client{}).Dir(ref, "cccc3333")
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.FromSlash("/env-cache/models/acme/qwen-mini/cccc3333"); got != want {
		t.Errorf("Dir with no Cache = %q, want %q", got, want)
	}
}

func TestWantedTakesTheCheckpointAndLeavesTheRest(t *testing.T) {
	for _, name := range []string{
		"config.json", "generation_config.json", "tokenizer.json",
		"tokenizer_config.json", "special_tokens_map.json", "vocab.json",
		"merges.txt", "model.safetensors.index.json",
		"model.safetensors", "model-00002-of-00003.safetensors",
	} {
		if !Wanted(name) {
			t.Errorf("Wanted(%q) = false", name)
		}
	}
	// The variants a real repo carries beside the weights. Downloading them
	// multiplies an eight gigabyte fetch for bytes tgo cannot read.
	for _, name := range []string{
		"README.md", ".gitattributes", "pytorch_model.bin", "model.gguf",
		"onnx/model.onnx", "coreml/model.mlpackage", "LICENSE",
	} {
		if Wanted(name) {
			t.Errorf("Wanted(%q) = true", name)
		}
	}
}

func TestParallelAndLockWaitHaveDefaults(t *testing.T) {
	if got := (&Client{}).parallel(); got != defaultParallel {
		t.Errorf("parallel = %d, want %d", got, defaultParallel)
	}
	if got := (&Client{Parallel: 2}).parallel(); got != 2 {
		t.Errorf("parallel = %d, want 2", got)
	}
	if got := (&Client{Parallel: -3}).parallel(); got != 1 {
		t.Errorf("parallel = %d, want 1", got)
	}
	if got := (&Client{}).lockWait(); got != defaultLockWait {
		t.Errorf("lockWait = %v, want %v", got, defaultLockWait)
	}
	if got := (&Client{}).every(); got != defaultProgressEvery {
		t.Errorf("every = %d, want %d", got, defaultProgressEvery)
	}
}

func TestGetRefusesAnUnbuildableURL(t *testing.T) {
	c := &Client{Endpoint: "://not-a-url"}
	if _, err := c.get(context.Background(), "://not-a-url", nil); err == nil {
		t.Error("a malformed URL was accepted")
	}
	if _, err := c.Revision(t.Context(), testRef(t)); err == nil {
		t.Error("a malformed endpoint was accepted")
	}
}

func TestRedirectChainIsBounded(t *testing.T) {
	req, err := http.NewRequest(http.MethodGet, "https://a.test/", nil)
	if err != nil {
		t.Fatal(err)
	}
	via := make([]*http.Request, maxRedirects)
	for i := range via {
		via[i] = req
	}
	if err := dropAuthAcrossHosts(req, via); err == nil {
		t.Error("an unbounded redirect chain was allowed")
	}
	if err := dropAuthAcrossHosts(req, nil); err != nil {
		t.Errorf("the first request was refused: %v", err)
	}
}
