// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// The fake Hugging Face.
//
// Two servers, not one, because the failure this suite exists to catch is a
// header crossing between them: resolve answers 302 towards a CDN, and the CDN
// answers 403 to anything that still carries an Authorization header, exactly
// as the real one does. One server could not tell the two hops apart.
//
// Both listen on 127.0.0.1, which is the point rather than a convenience: Go's
// own redirect rule compares hostnames with the port stripped, so it treats
// the two as the same domain and forwards the credential. A fake on two
// different hostnames would let the default client pass and prove nothing.

// fakeFile is one file in the fake repo.
type fakeFile struct {
	// body is what the CDN serves.
	body []byte

	// listSize is the length the API publishes. Zero means len(body); a
	// different value is how a length disagreement is staged.
	listSize int64

	// digest is the sha256 the API publishes for this file. "auto" means the
	// digest of body; "" means the API publishes none, which is the case for
	// every file git-lfs does not track.
	digest string

	// serve overrides the CDN's handler for this file, which is how a dropped
	// connection and an LFS pointer are staged.
	serve func(w http.ResponseWriter, r *http.Request, f *fakeFile)
}

// fake is a Hugging Face API and its CDN.
type fake struct {
	t     *testing.T
	api   *httptest.Server
	cdn   *httptest.Server
	repo  string
	sha   string
	files map[string]*fakeFile
	order []string

	// hold releases CDN handlers only once this many are in flight at once,
	// which is how the parallel bound is observed rather than timed.
	hold        int
	reached     chan struct{}
	reachedOnce sync.Once

	// dwell holds the released batch in place once, so that a request the
	// bound should have prevented has time to arrive and be counted. Zero
	// means defaultDwell.
	dwell     time.Duration
	dwellOnce sync.Once

	mu          sync.Mutex
	resolveAuth []string
	cdnAuth     []string
	ranges      []string
	served      map[string]int
	inflight    int
	maxInflight int
}

const (
	fakeRepo = "acme/qwen-mini"
	fakeSHA  = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
)

// newFake starts the pair. order fixes the listing order so a test can rely on
// it; a file not in order is appended.
func newFake(t *testing.T, files map[string]*fakeFile, order ...string) *fake {
	t.Helper()
	f := &fake{
		t: t, repo: fakeRepo, sha: fakeSHA, files: files,
		served: map[string]int{}, reached: make(chan struct{}), order: order,
	}
	for name := range files {
		if !contains(order, name) {
			f.order = append(f.order, name)
		}
	}
	f.cdn = httptest.NewServer(http.HandlerFunc(f.serveCDN))
	f.api = httptest.NewServer(http.HandlerFunc(f.serveAPI))
	t.Cleanup(func() {
		f.api.Close()
		f.cdn.Close()
	})
	return f
}

func contains(s []string, v string) bool {
	return slices.Contains(s, v)
}

// client is a Client pointed at the fake, caching into a temporary directory.
func (f *fake) client(t *testing.T) *Client {
	t.Helper()
	return &Client{Endpoint: f.api.URL, Cache: t.TempDir(), LockWait: time.Second}
}

func (f *fake) serveAPI(w http.ResponseWriter, r *http.Request) {
	switch {
	case strings.HasPrefix(r.URL.Path, "/api/models/"):
		f.serveListing(w, r)
	case strings.Contains(r.URL.Path, "/resolve/"):
		f.serveResolve(w, r)
	default:
		http.NotFound(w, r)
	}
}

// serveListing answers /api/models/{repo}/revision/{rev}.
func (f *fake) serveListing(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/api/models/")
	repo, rev, ok := strings.Cut(rest, "/revision/")
	if !ok || repo != f.repo {
		http.Error(w, `{"error":"Repository not found"}`, http.StatusNotFound)
		return
	}
	if rev != "main" && rev != f.sha {
		http.Error(w, `{"error":"Invalid rev id"}`, http.StatusNotFound)
		return
	}
	if r.URL.Query().Get("blobs") != "true" {
		// Without it the API publishes no size and no digest, and every check
		// in section 2 silently turns off.
		f.t.Errorf("the listing was fetched without ?blobs=true: %s", r.URL)
	}
	type lfs struct {
		OID    string `json:"oid"`
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	}
	type sibling struct {
		Path string `json:"rfilename"`
		Size int64  `json:"size"`
		LFS  *lfs   `json:"lfs,omitempty"`
	}
	out := struct {
		SHA      string    `json:"sha"`
		Siblings []sibling `json:"siblings"`
	}{SHA: f.sha}
	for _, name := range f.order {
		file := f.files[name]
		s := sibling{Path: name, Size: file.published()}
		if d := file.publishedDigest(); d != "" {
			s.LFS = &lfs{OID: d, Size: s.Size}
		}
		out.Siblings = append(out.Siblings, s)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// published is the length the API publishes for this file.
func (f *fakeFile) published() int64 {
	switch {
	case f.listSize < 0:
		// The API publishes no length, which is what it does for a file
		// git-lfs does not track.
		return 0
	case f.listSize > 0:
		return f.listSize
	}
	return int64(len(f.body))
}

// publishedDigest is the sha256 the API publishes, or "".
func (f *fakeFile) publishedDigest() string {
	if f.digest == "auto" {
		sum := sha256.Sum256(f.body)
		return hex.EncodeToString(sum[:])
	}
	return f.digest
}

// serveResolve answers /{repo}/resolve/{rev}/{path} with the 302 to the CDN
// that the real endpoint answers with.
func (f *fake) serveResolve(w http.ResponseWriter, r *http.Request) {
	_, rest, _ := strings.Cut(r.URL.Path, "/resolve/")
	_, name, _ := strings.Cut(rest, "/")
	f.mu.Lock()
	f.resolveAuth = append(f.resolveAuth, r.Header.Get("Authorization"))
	f.mu.Unlock()
	if _, ok := f.files[name]; !ok {
		http.Error(w, `{"error":"Entry not found"}`, http.StatusNotFound)
		return
	}
	http.Redirect(w, r, f.cdn.URL+"/cdn/"+name, http.StatusFound)
}

// serveCDN is the second hop. It answers 403 to a request that still carries a
// credential, which is what the real CDN does and the reason the header has to
// be dropped on the way here.
func (f *fake) serveCDN(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/cdn/")
	auth := r.Header.Get("Authorization")
	f.mu.Lock()
	f.cdnAuth = append(f.cdnAuth, auth)
	f.ranges = append(f.ranges, r.Header.Get("Range"))
	f.served[name]++
	f.inflight++
	if f.inflight > f.maxInflight {
		f.maxInflight = f.inflight
	}
	reached := f.hold > 0 && f.inflight >= f.hold
	f.mu.Unlock()
	defer func() {
		f.mu.Lock()
		f.inflight--
		f.mu.Unlock()
	}()
	if f.hold > 0 {
		if reached {
			f.reachedOnce.Do(func() { close(f.reached) })
		}
		// A bound smaller than expected must fail the assertion rather than
		// hang the suite, so the barrier gives up. Generous, because it costs
		// nothing when the bound is right -- reached is already closed -- and
		// a loaded runner missing a short window would be a false red.
		select {
		case <-f.reached:
		case <-time.After(2 * time.Second):
		}
		// The barrier alone proves only that the bound is not SMALLER than
		// hold. The handlers it releases return in microseconds, so a request
		// the bound should have prevented arrives after they are counted out
		// again, and an unbounded download reads as a bounded one. The dwell
		// holds the whole released batch in place, once, which is what makes
		// the extra request observable. sync.Once blocks its other callers,
		// so the batch waits together rather than one member of it.
		//
		// It is a release delay, never an assertion: no test here measures it.
		f.dwellOnce.Do(func() { time.Sleep(f.dwellFor()) })
	}
	if auth != "" {
		http.Error(w, "credential is not allowed on the CDN", http.StatusForbidden)
		return
	}
	file, ok := f.files[name]
	if !ok {
		http.NotFound(w, r)
		return
	}
	if file.serve != nil {
		file.serve(w, r, file)
		return
	}
	serveBytes(w, r, file.body)
}

// serveBytes serves a body with the open-ended Range support a resume needs.
func serveBytes(w http.ResponseWriter, r *http.Request, body []byte) {
	start := int64(0)
	if spec, ok := strings.CutPrefix(r.Header.Get("Range"), "bytes="); ok {
		s, _, _ := strings.Cut(spec, "-")
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil || n < 0 {
			http.Error(w, "bad range", http.StatusBadRequest)
			return
		}
		if n >= int64(len(body)) {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", len(body)))
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		start = n
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(body)-int(start)))
	if start > 0 {
		w.Header().Set("Content-Range",
			fmt.Sprintf("bytes %d-%d/%d", start, len(body)-1, len(body)))
		w.WriteHeader(http.StatusPartialContent)
	}
	_, _ = w.Write(body[start:])
}

// defaultDwell is how long the released batch is held. It is generous
// because it costs nothing when the bound is right -- one batch pays it once
// -- and because the requests it waits for are already in flight over
// loopback.
const defaultDwell = 250 * time.Millisecond

// dwellFor is the dwell in force.
func (f *fake) dwellFor() time.Duration {
	if f.dwell > 0 {
		return f.dwell
	}
	return defaultDwell
}

// authSeenByCDN is every Authorization header the CDN was sent.
func (f *fake) authSeenByCDN() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.cdnAuth...)
}

// rangesSeen is every Range header the CDN was sent, in order.
func (f *fake) rangesSeen() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.ranges...)
}

// serveCount is how many times the CDN was asked for a file.
func (f *fake) serveCount(name string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.served[name]
}

// peakInflight is the most CDN requests that overlapped.
func (f *fake) peakInflight() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.maxInflight
}
