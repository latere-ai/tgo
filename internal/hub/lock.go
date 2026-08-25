// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"context"
	"fmt"
	"os"
	"time"
)

// lockSuffix names the lock file for a revision directory (§3).
//
// It sits BESIDE the directory, at {sha}.lock, not inside it. Inside, it would
// be an entry that safetensors.OpenRepo has to know to ignore, and a
// checkpoint directory that contains a file no checkpoint has is a directory
// two readers disagree about.
const lockSuffix = ".lock"

// lockPoll is how often a waiter re-tries the exclusive create.
const lockPoll = 20 * time.Millisecond

// lock is a held revision lock.
type lock struct{ path string }

// acquireLock takes the lock for a revision directory, waiting up to wait for
// whoever holds it.
//
// O_CREATE|O_EXCL is atomic on every GOOS tgo builds for and needs no syscall
// beyond what os offers, which is what keeps this cgo-free. There is
// deliberately no staleness timeout: a lock whose owner died is cleared by
// naming the file in the error, not by guessing from its age that the owner is
// gone. A guess that is wrong lets two processes write one file.
func acquireLock(ctx context.Context, path string, wait time.Duration) (*lock, error) {
	deadline := time.Now().Add(wait)
	for {
		fh, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			fmt.Fprintf(fh, "%d\n", os.Getpid())
			if err := fh.Close(); err != nil {
				return nil, fmt.Errorf("hub: %s: %w", path, err)
			}
			return &lock{path: path}, nil
		}
		if !os.IsExist(err) {
			return nil, fmt.Errorf("hub: %s: %w", path, err)
		}
		if !time.Now().Before(deadline) {
			return nil, fmt.Errorf("%w: %s; if no tgo is running, remove it",
				ErrLocked, path)
		}
		t := time.NewTimer(lockPoll)
		select {
		case <-ctx.Done():
			t.Stop()
			return nil, ctx.Err()
		case <-t.C:
		}
	}
}

// release gives the lock back. It is safe to call twice.
func (l *lock) release() {
	if l == nil || l.path == "" {
		return
	}
	os.Remove(l.path)
	l.path = ""
}
