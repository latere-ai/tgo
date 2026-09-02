// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package hub

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// DefaultEndpoint is the Hugging Face API. HF_ENDPOINT overrides it, which is
// how a mirror or an on-premises deployment is reached.
const DefaultEndpoint = "https://huggingface.co"

// defaultParallel is how many files download at once (§3).
//
// Small on purpose: the disk is the limit, not the network, and eight parallel
// writes to one spinning disk are slower than two.
const defaultParallel = 4

// defaultLockWait is how long [Client.Fetch] waits for another process to
// finish with a revision directory before it refuses with [ErrLocked].
const defaultLockWait = 30 * time.Second

// maxRedirects bounds the resolve -> CDN chain. Go's own default is 10 and
// this restates it, because installing a CheckRedirect replaces that default
// rather than adding to it.
const maxRedirects = 10

// Event is one progress report. [Client.Progress] is called from every
// download goroutine, so an implementation must be safe for concurrent use.
type Event struct {
	// Path is the file's path within the repo.
	Path string

	// Done is the bytes on disk, the resumed prefix included.
	Done int64

	// Total is what the API published, or 0 when it published nothing.
	Total int64

	// Resumed reports that this file started from a partial download.
	Resumed bool

	// Complete reports the last event for this file.
	Complete bool
}

// Client fetches checkpoints from a Hugging Face-compatible API.
//
// The zero value works: every field has a default, filled on first use. A
// Client is safe for concurrent use.
type Client struct {
	// Endpoint is the API root. Empty means $HF_ENDPOINT, then
	// [DefaultEndpoint].
	Endpoint string

	// Token is the Hugging Face access token, sent as a bearer. It is set on
	// each request and dropped by [Client] on any redirect that changes
	// host:port — see the package comment for why that is not negotiable, and
	// why this must not become a RoundTripper.
	Token string

	// Transport carries the requests. Empty means [http.DefaultTransport].
	// The redirect policy is not configurable, so this is the only hook.
	Transport http.RoundTripper

	// Cache is the root of the download cache. Empty means [CacheDir].
	Cache string

	// Parallel bounds concurrent file downloads. Zero means
	// [defaultParallel]; a negative value means one at a time.
	Parallel int

	// LockWait bounds how long a fetch waits for another process's revision
	// lock. Zero means [defaultLockWait].
	LockWait time.Duration

	// Want selects the files to download from the listing. Empty means
	// [Wanted], which is what a safetensors checkpoint needs and nothing else.
	Want func(repoPath string) bool

	// Progress is called as bytes land, at most once per progressEvery bytes
	// per file, and once more when a file completes.
	Progress func(Event)

	// progressEvery is the reporting granularity in bytes. Tests set it; it
	// is not a knob a caller needs.
	progressEvery int64

	once sync.Once
	http *http.Client
}

// File is one entry of a revision's listing.
type File struct {
	// Path is the file's path within the repo, always slash-separated.
	Path string

	// Size is what the API published, or 0 when it published none. Zero
	// disables the length check for this file, which is what "where the API
	// gives one" means for the sha as well.
	Size int64

	// SHA256 is the lowercase hex digest the API published for an LFS-backed
	// file, or empty.
	SHA256 string
}

// Revision is a resolved revision: the commit the ref pointed at, and what is
// in it.
type Revision struct {
	// SHA is the commit the requested revision resolved to. It is the cache
	// key (013-D2), so that main moving does not overwrite what main used to
	// be.
	SHA string

	// Files is every entry the API listed, in the order it listed them.
	Files []File
}

// client builds the one http.Client this Client uses, with the redirect policy
// the CDN hop needs.
func (c *Client) client() *http.Client {
	c.once.Do(func() {
		c.http = &http.Client{
			Transport:     c.Transport,
			CheckRedirect: dropAuthAcrossHosts,
		}
	})
	return c.http
}

// dropAuthAcrossHosts removes the Authorization header on any redirect that
// changes host:port.
//
// Go's own rule keeps the header for a host that is the same domain or a
// subdomain of the first, comparing hostnames with the port stripped. A CDN
// under the API's own domain therefore keeps it under the default policy and
// answers 403; so do two test servers on 127.0.0.1 at different ports. This
// rule is coarser and holds in both cases: a different host:port is a
// different party, and it gets no credential.
func dropAuthAcrossHosts(req *http.Request, via []*http.Request) error {
	if len(via) >= maxRedirects {
		return fmt.Errorf("hub: stopped after %d redirects", maxRedirects)
	}
	if len(via) > 0 && req.URL.Host != via[0].URL.Host {
		req.Header.Del("Authorization")
	}
	return nil
}

// endpoint is the API root, with any trailing slash removed.
func (c *Client) endpoint() string {
	e := c.Endpoint
	if e == "" {
		e = os.Getenv("HF_ENDPOINT")
	}
	if e == "" {
		e = DefaultEndpoint
	}
	return strings.TrimRight(e, "/")
}

// cacheRoot is the cache directory this Client writes under.
func (c *Client) cacheRoot() (string, error) {
	if c.Cache != "" {
		return c.Cache, nil
	}
	return CacheDir()
}

// Dir is where a resolved revision lives: $TGO_CACHE/models/{org}/{repo}/{sha}.
func (c *Client) Dir(ref Ref, sha string) (string, error) {
	root, err := c.cacheRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "models", ref.Org, ref.Repo, sha), nil
}

// get issues one authenticated GET. The header is set here, per request, and
// never in a transport (see the package comment).
func (c *Client) get(ctx context.Context, url string, header http.Header) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("hub: %w", err)
	}
	for k, vs := range header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	resp, err := c.client().Do(req)
	if err != nil {
		return nil, fmt.Errorf("hub: GET %s: %w", url, err)
	}
	return resp, nil
}

// statusError names what an API status means, so that a gated repo and a typo
// in a repo id do not report the same thing.
func statusError(url string, resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	detail := strings.TrimSpace(string(body))
	if detail != "" {
		detail = ": " + detail
	}
	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%w: %s answered %s%s", ErrUnauthorized, url, resp.Status, detail)
	case http.StatusNotFound:
		return fmt.Errorf("%w: %s answered %s%s", ErrNotFound, url, resp.Status, detail)
	}
	return fmt.Errorf("hub: %s answered %s%s", url, resp.Status, detail)
}

// transferError names a refusal from the file transfer rather than from the
// API.
//
// A 403 from a host that is not the API is the CDN, and the reason the CDN
// refuses is almost always that a credential survived the redirect. Reporting
// that as "the repo is gated or private" sends the reader to look for a token
// they already have.
func (c *Client) transferError(u string, resp *http.Response) error {
	if resp.StatusCode == http.StatusForbidden && resp.Request != nil &&
		resp.Request.URL.Host != hostOf(c.endpoint()) {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return fmt.Errorf("hub: %s: the CDN at %s refused the transfer (%s: %s); "+
			"an Authorization header must not survive the redirect",
			u, resp.Request.URL.Host, resp.Status, strings.TrimSpace(string(body)))
	}
	return statusError(u, resp)
}

// hostOf is the host:port of a URL, or "" when it does not parse.
func hostOf(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return u.Host
}

// apiRevision is the shape of /api/models/{repo}/revision/{rev}.
//
// Everything below sha is optional on purpose. The size and the LFS digest
// arrive only with ?blobs=true, and only for the files git-lfs tracks, so the
// parser treats both as "where the API gives one" (§2) rather than as a
// contract.
type apiRevision struct {
	SHA      string `json:"sha"`
	Siblings []struct {
		Path string `json:"rfilename"`
		Size int64  `json:"size"`
		LFS  *struct {
			OID    string `json:"oid"`
			SHA256 string `json:"sha256"`
			Size   int64  `json:"size"`
		} `json:"lfs"`
	} `json:"siblings"`
}

// Revision lists a revision and resolves it to a commit sha.
func (c *Client) Revision(ctx context.Context, ref Ref) (*Revision, error) {
	if ref.IsLocal() {
		return nil, fmt.Errorf("hub: %s is a local path, which has no revision", ref.Local)
	}
	u := c.endpoint() + "/api/models/" + ref.ID() +
		"/revision/" + escapeRevision(ref.Revision) + "?blobs=true"
	resp, err := c.get(ctx, u, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, statusError(u, resp)
	}
	var api apiRevision
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<24)).Decode(&api); err != nil {
		return nil, fmt.Errorf("hub: %s: the listing is not JSON: %w", u, err)
	}
	if api.SHA == "" {
		return nil, fmt.Errorf("hub: %s: the listing names no commit sha", u)
	}
	rev := &Revision{SHA: api.SHA, Files: make([]File, 0, len(api.Siblings))}
	for _, s := range api.Siblings {
		if err := safePath(s.Path); err != nil {
			return nil, fmt.Errorf("hub: %s: %w", u, err)
		}
		f := File{Path: s.Path, Size: s.Size}
		if s.LFS != nil {
			if s.LFS.Size > 0 {
				f.Size = s.LFS.Size
			}
			f.SHA256 = normalizeDigest(s.LFS.SHA256)
			if f.SHA256 == "" {
				f.SHA256 = normalizeDigest(s.LFS.OID)
			}
		}
		rev.Files = append(rev.Files, f)
	}
	return rev, nil
}

// escapeRevision keeps a revision safe in a path segment while leaving the
// slashes of refs/pr/1 alone, which the API expects unescaped.
func escapeRevision(rev string) string {
	parts := strings.Split(rev, "/")
	for i, p := range parts {
		parts[i] = url.PathEscape(p)
	}
	return strings.Join(parts, "/")
}

// normalizeDigest accepts both "sha256:hex" and bare hex, and rejects anything
// that is not 64 hex digits rather than comparing against a value that cannot
// match.
func normalizeDigest(s string) string {
	s = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(s)), "sha256:")
	if len(s) != 64 {
		return ""
	}
	for _, r := range s {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return ""
		}
	}
	return s
}

// resolveURL is where a file's bytes are: the resolve endpoint, never the git
// one, which answers with an LFS pointer (see [ErrLFSPointer]).
func (c *Client) resolveURL(ref Ref, rev, repoPath string) string {
	segs := strings.Split(repoPath, "/")
	for i, s := range segs {
		segs[i] = url.PathEscape(s)
	}
	return c.endpoint() + "/" + ref.ID() + "/resolve/" + escapeRevision(rev) +
		"/" + strings.Join(segs, "/")
}

// Wanted is the default file filter: what a safetensors checkpoint needs, and
// nothing else.
//
// Top-level files only. A real repo carries onnx/, coreml/ and ggml variants
// of the same weights, and downloading them multiplies an 8 GB fetch for
// nothing tgo can read.
func Wanted(repoPath string) bool {
	if strings.Contains(repoPath, "/") {
		return false
	}
	switch path.Ext(repoPath) {
	case ".safetensors":
		return true
	}
	switch repoPath {
	case "config.json", "generation_config.json",
		"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
		"vocab.json", "merges.txt", "model.safetensors.index.json":
		return true
	}
	return false
}

// want is the filter in force.
func (c *Client) want() func(string) bool {
	if c.Want != nil {
		return c.Want
	}
	return Wanted
}
