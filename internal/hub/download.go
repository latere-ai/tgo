// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// lfsMagic is the first line of a git-lfs pointer file. The whole file is
// about 130 bytes:
//
//	version https://git-lfs.github.com/spec/v1
//	oid sha256:4d5f...
//	size 4194304
//
// It is what the git endpoint returns for a file git-lfs tracks, which is
// every shard of every real checkpoint. Reaching it means the URL was built
// wrong, and the body parses as neither JSON nor safetensors.
const lfsMagic = "version https://git-lfs.github.com/spec/v1"

// partSuffix names the temporary file a download writes to (013-D3). The name
// is deterministic so that a later fetch can resume it, and it is not the
// final name so that nothing can mistake a partial file for a whole one.
const partSuffix = ".part"

// defaultProgressEvery is how many bytes land between progress reports.
const defaultProgressEvery = 1 << 20

// errStalePartial reports a partial file the server will not resume from,
// which is a 416: the file on disk is longer than the file on the server. The
// answer is to discard it and start again, once.
var errStalePartial = errors.New("hub: the partial file is stale")

// report delivers one progress event, if anybody asked for them.
func (c *Client) report(ev Event) {
	if c.Progress != nil {
		c.Progress(ev)
	}
}

// every is the progress granularity in bytes.
func (c *Client) every() int64 {
	if c.progressEvery > 0 {
		return c.progressEvery
	}
	return defaultProgressEvery
}

// fetchFile downloads one file into dir, resuming a partial download where one
// is on disk, and renaming into place only once the bytes are known to be
// right.
func (c *Client) fetchFile(ctx context.Context, ref Ref, rev string, f File, dir string) error {
	err := c.fetchOnce(ctx, ref, rev, f, dir, true)
	if errors.Is(err, errStalePartial) {
		// The one case worth retrying: the partial file was longer than the
		// server's file, so there is nothing to resume from. It is discarded
		// by fetchOnce, and the retry cannot reach this branch again because
		// it does not resume.
		err = c.fetchOnce(ctx, ref, rev, f, dir, false)
	}
	return err
}

// fetchOnce is one attempt. resume=false starts from zero whatever is on disk.
func (c *Client) fetchOnce(ctx context.Context, ref Ref, rev string, f File, dir string, resume bool) error {
	if err := safePath(f.Path); err != nil {
		return err
	}
	final := filepath.Join(dir, filepath.FromSlash(f.Path))
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return fmt.Errorf("hub: %s: %w", f.Path, err)
	}
	if st, err := os.Stat(final); err == nil && st.Mode().IsRegular() &&
		(f.Size == 0 || st.Size() == f.Size) {
		// Already here at the published length. The cache is keyed by commit
		// sha, so a file at this path in this directory is that commit's file.
		c.report(Event{Path: f.Path, Done: st.Size(), Total: f.Size, Complete: true})
		return nil
	}

	part := final + partSuffix
	digest := sha256.New()
	offset := int64(0)
	if resume {
		offset = resumeFrom(part, f.Size, digest)
	}
	if offset == 0 {
		// Nothing usable on disk: any leftover is discarded rather than
		// appended to, which would produce a file that is the right length and
		// the wrong bytes.
		if err := os.Remove(part); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("hub: %s: %w", f.Path, err)
		}
	}

	header := http.Header{}
	if offset > 0 {
		header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	u := c.resolveURL(ref, rev, f.Path)
	resp, err := c.get(ctx, u, header)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	switch resp.StatusCode {
	case http.StatusOK:
		// The server ignored the Range, so the body is the whole file and the
		// prefix on disk is worthless. Both the offset and the digest of that
		// prefix have to go, or the sha256 is computed over the file twice.
		offset, digest = 0, sha256.New()
	case http.StatusPartialContent:
		if err := checkContentRange(resp.Header.Get("Content-Range"), offset); err != nil {
			return fmt.Errorf("hub: %s: %w", f.Path, err)
		}
	case http.StatusRequestedRangeNotSatisfiable:
		if offset > 0 {
			if err := os.Remove(part); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("hub: %s: %w", f.Path, err)
			}
			return fmt.Errorf("%w: %s is %d bytes, which the server will not resume",
				errStalePartial, part, offset)
		}
		return c.transferError(u, resp)
	default:
		return c.transferError(u, resp)
	}

	body := bufio.NewReaderSize(resp.Body, 64<<10)
	if offset == 0 {
		// Before every other check. An LFS pointer is a short body where a
		// long one was published, so a length check would fire first and
		// report a truncated download -- which is the confusing failure this
		// exists to name.
		if head, _ := body.Peek(len(lfsMagic)); bytes.HasPrefix(head, []byte(lfsMagic)) {
			return fmt.Errorf("%w: %s served %d bytes of pointer; the URL must be "+
				"the resolve endpoint, not the git one", ErrLFSPointer, u, resp.ContentLength)
		}
	}
	if f.Size > 0 && resp.ContentLength >= 0 && offset+resp.ContentLength != f.Size {
		return fmt.Errorf("%w: %s: the listing says %d bytes and the response declares %d",
			ErrSize, f.Path, f.Size, offset+resp.ContentLength)
	}

	fh, err := openPart(part, offset)
	if err != nil {
		return fmt.Errorf("hub: %s: %w", f.Path, err)
	}
	pw := &progressWriter{c: c, path: f.Path, total: f.Size, done: offset,
		resumed: offset > 0, every: c.every()}
	n, copyErr := io.Copy(io.MultiWriter(fh, digest, pw), body)
	total := offset + n
	syncErr := fh.Sync()
	closeErr := fh.Close()
	if copyErr != nil {
		// The connection dropped or the context was cancelled. The partial
		// file STAYS: it is what the next Range request resumes from, and it
		// carries the temporary name, so no reader can pick it up.
		return fmt.Errorf("hub: %s: %d of %d bytes: %w", f.Path, total, f.Size, copyErr)
	}
	if syncErr != nil {
		return fmt.Errorf("hub: %s: %w", f.Path, syncErr)
	}
	if closeErr != nil {
		return fmt.Errorf("hub: %s: %w", f.Path, closeErr)
	}

	if f.Size > 0 && total != f.Size {
		if total > f.Size {
			// Longer than published: not a truncated transfer, so there is
			// nothing to resume and the bytes are wrong.
			_ = os.Remove(part)
		}
		return fmt.Errorf("%w: %s: the listing says %d bytes and %d arrived",
			ErrSize, f.Path, f.Size, total)
	}
	if f.SHA256 != "" {
		if got := hex.EncodeToString(digest.Sum(nil)); got != f.SHA256 {
			// A mismatch deletes and fails (§2). Keeping it would let the next
			// fetch resume a prefix that can never hash to the right digest.
			_ = os.Remove(part)
			return fmt.Errorf("%w: %s: published %s, got %s", ErrChecksum, f.Path, f.SHA256, got)
		}
	}
	if err := os.Rename(part, final); err != nil {
		return fmt.Errorf("hub: %s: %w", f.Path, err)
	}
	c.report(Event{Path: f.Path, Done: total, Total: f.Size, Resumed: offset > 0, Complete: true})
	return nil
}

// resumeFrom reports how many bytes of part can be resumed, having fed them to
// digest. It reports 0 when there is nothing usable, in which case digest is
// left as it was found.
func resumeFrom(part string, size int64, digest hash.Hash) int64 {
	st, err := os.Stat(part)
	if err != nil || !st.Mode().IsRegular() || st.Size() == 0 {
		return 0
	}
	if size > 0 && st.Size() >= size {
		// At or past the published length and not renamed, so the bytes never
		// passed a check. Start again rather than trust them.
		return 0
	}
	fh, err := os.Open(part)
	if err != nil {
		return 0
	}
	defer func() { _ = fh.Close() }()
	n, err := io.Copy(digest, fh)
	if err != nil || n != st.Size() {
		digest.Reset()
		return 0
	}
	return n
}

// openPart opens the temporary file positioned at offset, truncating anything
// past it.
func openPart(part string, offset int64) (*os.File, error) {
	fh, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	if err := fh.Truncate(offset); err != nil {
		_ = fh.Close()
		return nil, err
	}
	if _, err := fh.Seek(offset, io.SeekStart); err != nil {
		_ = fh.Close()
		return nil, err
	}
	return fh, nil
}

// checkContentRange verifies that a 206 starts where the Range asked it to. A
// server that answers 206 from a different offset would produce a file with a
// hole in it, at the right length, hashing to nothing anybody published.
func checkContentRange(v string, offset int64) error {
	spec, ok := strings.CutPrefix(strings.TrimSpace(v), "bytes ")
	if !ok {
		return fmt.Errorf("the 206 has no byte Content-Range: %q", v)
	}
	start, _, ok := strings.Cut(spec, "-")
	if !ok {
		return fmt.Errorf("the 206 has a malformed Content-Range: %q", v)
	}
	got, err := strconv.ParseInt(start, 10, 64)
	if err != nil {
		return fmt.Errorf("the 206 has a malformed Content-Range: %q", v)
	}
	if got != offset {
		return fmt.Errorf("asked to resume at %d and the 206 starts at %d", offset, got)
	}
	return nil
}

// progressWriter reports bytes as they land, every c.every() bytes.
type progressWriter struct {
	c        *Client
	path     string
	total    int64
	done     int64
	resumed  bool
	every    int64
	reported int64
}

func (w *progressWriter) Write(p []byte) (int, error) {
	w.done += int64(len(p))
	if w.done-w.reported >= w.every {
		w.reported = w.done
		w.c.report(Event{Path: w.path, Done: w.done, Total: w.total, Resumed: w.resumed})
	}
	return len(p), nil
}
