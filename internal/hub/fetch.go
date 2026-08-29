// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Fetch makes a checkpoint local and reports the directory it is in.
//
// A local ref is checked and returned; nothing is copied, because a directory
// on this machine is already what safetensors.OpenRepo wants (013-D4). A repo
// id is resolved to a commit sha, and the files that revision lists are
// downloaded into $TGO_CACHE/models/{org}/{repo}/{sha}.
//
// Fetch is resumable and re-entrant: a file already in the directory at the
// published length is left alone, a partial download continues from where it
// stopped, and a second process is held off by the revision lock (§3).
func (c *Client) Fetch(ctx context.Context, ref Ref) (string, error) {
	if ref.IsLocal() {
		st, err := os.Stat(ref.Local)
		if err != nil {
			return "", fmt.Errorf("hub: %s: %w", ref.Local, err)
		}
		if !st.IsDir() {
			return "", fmt.Errorf("hub: %s is not a directory", ref.Local)
		}
		return ref.Local, nil
	}

	rev, err := c.Revision(ctx, ref)
	if err != nil {
		return "", err
	}
	dir, err := c.Dir(ref, rev.SHA)
	if err != nil {
		return "", err
	}
	want := c.want()
	files := make([]File, 0, len(rev.Files))
	for _, f := range rev.Files {
		if want(f.Path) {
			files = append(files, f)
		}
	}
	if len(files) == 0 {
		return "", fmt.Errorf("%w: %s at %s lists %d files, none of them loadable",
			ErrNoFiles, ref.ID(), rev.SHA, len(rev.Files))
	}

	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return "", fmt.Errorf("hub: %w", err)
	}
	lk, err := acquireLock(ctx, dir+lockSuffix, c.lockWait())
	if err != nil {
		return "", err
	}
	defer lk.release()

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("hub: %w", err)
	}
	if err := c.fetchAll(ctx, ref, rev.SHA, files, dir); err != nil {
		return "", err
	}
	return dir, nil
}

// lockWait is how long a fetch waits for another process's lock.
func (c *Client) lockWait() time.Duration {
	if c.LockWait != 0 {
		return c.LockWait
	}
	return defaultLockWait
}

// parallel is how many files download at once. The bound exists because the
// disk is the limit, not the network (§3).
func (c *Client) parallel() int {
	switch {
	case c.Parallel > 0:
		return c.Parallel
	case c.Parallel < 0:
		return 1
	}
	return defaultParallel
}

// fetchAll downloads every file, at most parallel() at a time, and stops the
// rest as soon as one fails.
func (c *Client) fetchAll(ctx context.Context, ref Ref, sha string, files []File, dir string) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sem := make(chan struct{}, c.parallel())
	errs := make([]error, len(files))
	var wg sync.WaitGroup
	for i, f := range files {
		wg.Go(func() {
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				errs[i] = ctx.Err()
				return
			}
			defer func() { <-sem }()
			if errs[i] = c.fetchFile(ctx, ref, sha, f, dir); errs[i] != nil {
				cancel()
			}
		})
	}
	wg.Wait()

	// The first REAL failure, not the cancellation it caused: cancel() poisons
	// every download still running, and reporting "context canceled" would
	// hide the checksum mismatch that started it.
	var fallback error
	for _, err := range errs {
		switch {
		case err == nil:
		case errors.Is(err, context.Canceled) && ctx.Err() != nil:
			if fallback == nil {
				fallback = err
			}
		default:
			return err
		}
	}
	return fallback
}
