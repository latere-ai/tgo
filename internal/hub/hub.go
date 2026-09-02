// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package hub turns a Hugging Face repo id into a local directory that
// [github.com/latere-ai/tgo/safetensors].OpenRepo can read.
//
// specs/013-distribution.md is the design. specs/001-weights.md takes a
// directory; this is how the directory appears.
//
// # The source
//
// Plain net/http against the Hugging Face API, no SDK (013-D1):
// GET /api/models/{repo}/revision/{rev} lists the files, then
// resolve/{rev}/{path} fetches each one. Two failures are specific to that
// pair and both are named here rather than left to confuse somebody:
//
// The resolve endpoint answers 302 with a CDN location, and the CDN rejects a
// request that still carries the Authorization header. Go's own redirect rule
// compares hostnames with the port stripped, so it forwards the header to any
// host that is the same domain or a subdomain of the first — which is exactly
// what a CDN under the API's domain is. [Client] therefore installs its own
// CheckRedirect that drops the header on ANY change of host:port, and sets the
// header per request. Do not move that header into a [http.RoundTripper]: a
// transport re-adds it on every hop and the redirect policy becomes
// decorative.
//
// Fetching an LFS-backed file from the git endpoint instead of resolve returns
// a ~130-byte pointer file, which parses as neither JSON nor safetensors and
// fails many layers later. The body is sniffed before anything else is checked
// and the failure is [ErrLFSPointer].
//
// # The cache
//
// Keyed by the RESOLVED commit sha rather than by the requested revision
// (013-D2), so that a moving ref such as main is harmless: two revisions
// coexist under
//
//	$TGO_CACHE/models/{org}/{repo}/{sha}/
//
// and neither corrupts the other. A repo id with no org, such as gpt2, uses
// the sentinel org "_", which no Hugging Face account can be called.
//
// A download writes to a temporary name and renames on completion (013-D3), so
// an interrupted fetch never leaves a file that looks whole. The temporary file
// survives a dropped connection — that is what makes the Range resume work —
// and is deleted when the bytes are known to be wrong, because resuming a
// corrupt prefix never converges.
package hub

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
)

// ErrLFSPointer reports a body that is a git-lfs pointer rather than the file
// it points at.
//
// It means the fetch reached the git endpoint instead of resolve. The pointer
// is about 130 bytes of text and parses as neither JSON nor safetensors, so
// without this the failure surfaces as a corrupt checkpoint several layers
// away from its cause.
var ErrLFSPointer = errors.New("hub: the body is a git-lfs pointer, not the file")

// ErrChecksum reports a downloaded file whose sha256 is not the one the API
// published. The partial file is deleted before this is returned.
var ErrChecksum = errors.New("hub: the sha256 does not match what the API published")

// ErrSize reports a downloaded file whose length is not the one the API
// published.
var ErrSize = errors.New("hub: the length does not match what the API published")

// ErrUnsafePath reports a listing entry that would write outside the cache
// directory.
//
// A file list is a server's word for what a repo contains, which makes it
// untrusted input: an entry of "../../../.ssh/authorized_keys" is a path
// traversal, and a fetch that took it would write there.
var ErrUnsafePath = errors.New("hub: the listing names a path outside the cache")

// ErrNotFound reports a repo or revision the API does not have.
var ErrNotFound = errors.New("hub: no such repo or revision")

// ErrUnauthorized reports a repo the API will not serve to this caller: gated,
// private, or needing a token that was not supplied.
var ErrUnauthorized = errors.New("hub: the repo is gated or private")

// ErrNoFiles reports a revision that lists nothing tgo can load.
var ErrNoFiles = errors.New("hub: the revision lists no file tgo can load")

// ErrLocked reports a revision directory another process is writing.
var ErrLocked = errors.New("hub: another process holds the revision lock")

// defaultRevision is what a ref with no @rev asks for. It is a moving ref,
// which is why the cache is keyed by what it resolves to (013-D2).
const defaultRevision = "main"

// noOrg is the directory that stands in for the org of a bare repo id such as
// gpt2. Hugging Face account names are alphanumeric with dashes, so no account
// can collide with it.
const noOrg = "_"

// Ref names a checkpoint: a Hugging Face repo id at a revision, or a local
// directory (013-D4 — there is no registry, and no Modelfile).
type Ref struct {
	// Org is the account the repo belongs to, or [noOrg] for a bare id.
	Org string

	// Repo is the repository name.
	Repo string

	// Revision is a branch, a tag, or a commit sha. It is what is ASKED for;
	// what it resolves to is [Revision.SHA].
	Revision string

	// Local is the directory a local ref names, and is empty for a repo id.
	Local string
}

// IsLocal reports whether the ref is a directory on this machine, which needs
// no network and no cache.
func (r Ref) IsLocal() bool { return r.Local != "" }

// ID is the repo id the API path uses: "org/repo", or "repo" for a bare id.
func (r Ref) ID() string {
	if r.Org == noOrg {
		return r.Repo
	}
	return r.Org + "/" + r.Repo
}

// String renders the ref the way [ParseRef] reads it.
func (r Ref) String() string {
	if r.IsLocal() {
		return r.Local
	}
	return r.ID() + "@" + r.Revision
}

// ParseRef reads a model argument: a Hugging Face repo id, optionally at a
// revision, or a path to a local directory.
//
//	Qwen/Qwen3-0.6B            org and repo, revision main
//	Qwen/Qwen3-0.6B@refs/pr/1  an explicit revision
//	gpt2                       a bare id, org "_"
//	./models/qwen  /opt/qwen   a local directory
//
// A path is anything that begins with a separator, a dot, or a tilde, anything
// that carries a Windows volume name or a backslash, and anything with more
// than the two components a repo id has.
func ParseRef(s string) (Ref, error) {
	if strings.TrimSpace(s) == "" {
		return Ref{}, errors.New("hub: the model ref is empty")
	}
	if isLocalPath(s) {
		return Ref{Local: filepath.Clean(s)}, nil
	}
	id, rev, ok := strings.Cut(s, "@")
	if !ok {
		rev = defaultRevision
	}
	if rev == "" {
		return Ref{}, fmt.Errorf("hub: %q names an empty revision", s)
	}
	org, repo := noOrg, id
	if before, after, found := strings.Cut(id, "/"); found {
		org, repo = before, after
	}
	if err := validName(org); err != nil {
		return Ref{}, fmt.Errorf("hub: %q: the org: %w", s, err)
	}
	if err := validName(repo); err != nil {
		return Ref{}, fmt.Errorf("hub: %q: the repo: %w", s, err)
	}
	return Ref{Org: org, Repo: repo, Revision: rev}, nil
}

// isLocalPath reports whether s names a directory rather than a repo id.
func isLocalPath(s string) bool {
	switch {
	case strings.HasPrefix(s, "."), strings.HasPrefix(s, "~"),
		strings.HasPrefix(s, "/"), strings.HasPrefix(s, `\`):
		return true
	case filepath.VolumeName(s) != "", strings.Contains(s, `\`):
		return true
	case strings.Count(strings.SplitN(s, "@", 2)[0], "/") > 1:
		return true
	}
	return false
}

// validName checks one component of a repo id. Hugging Face allows letters,
// digits, dot, dash and underscore; the check exists because the component
// becomes a directory name.
func validName(s string) error {
	if s == "" {
		return errors.New("is empty")
	}
	if s == "." || s == ".." {
		return errors.New("is a path element")
	}
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		case r == '.' || r == '-' || r == '_':
		default:
			return fmt.Errorf("has %q in it", r)
		}
	}
	return nil
}

// safePath checks one entry of a file listing before it becomes a path under
// the cache. The listing comes from a server, so it is untrusted (001 §6 takes
// the same posture towards the checkpoint itself).
//
// It refuses an absolute path, any ".." element, a Windows volume or
// separator, and anything path.Clean would rewrite — a rewrite means the name
// on disk would not be the name the API published.
func safePath(name string) error {
	switch {
	case name == "":
		return fmt.Errorf("%w: the name is empty", ErrUnsafePath)
	case strings.HasPrefix(name, "/"), filepath.VolumeName(name) != "":
		return fmt.Errorf("%w: %q is absolute", ErrUnsafePath, name)
	case strings.Contains(name, `\`):
		return fmt.Errorf("%w: %q has a backslash in it", ErrUnsafePath, name)
	case path.Clean(name) != name:
		return fmt.Errorf("%w: %q is not a clean relative path", ErrUnsafePath, name)
	}
	if slices.Contains(strings.Split(name, "/"), "..") {
		return fmt.Errorf("%w: %q climbs out of the directory", ErrUnsafePath, name)
	}
	for _, r := range name {
		if r < 0x20 || r == 0x7f {
			return fmt.Errorf("%w: %q has a control character in it", ErrUnsafePath, name)
		}
	}
	return nil
}

// CacheDir is where downloaded checkpoints live: $TGO_CACHE, else
// $XDG_CACHE_HOME/tgo, else ~/.cache/tgo.
//
// The last one is taken literally on every GOOS, which is what
// huggingface_hub does, so a checkpoint fetched by either tool is in the place
// the other looks for it. os.UserCacheDir is the fallback for a machine with
// no home directory.
func CacheDir() (string, error) {
	if d := os.Getenv("TGO_CACHE"); d != "" {
		return d, nil
	}
	if d := os.Getenv("XDG_CACHE_HOME"); d != "" {
		return filepath.Join(d, "tgo"), nil
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".cache", "tgo"), nil
	}
	d, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("hub: no cache directory: %w", err)
	}
	return filepath.Join(d, "tgo"), nil
}
