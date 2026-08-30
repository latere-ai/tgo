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

// openLockFile is the exclusive create the lock is built on. It is a variable
// because the write that follows it fails only on a filesystem this suite
// cannot produce, and that write not being checked was a defect.
var openLockFile = func(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
}

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
		fh, err := openLockFile(path)
		if err == nil {
			// The pid is the whole content of the lock and an os.File is
			// unbuffered, so a failed write never reappears at Close. Left
			// unchecked it leaves an empty lock file behind: the refusal above
			// names a holder that cannot be identified, and the file outlives
			// the process that could not write it.
			_, werr := fmt.Fprintf(fh, "%d\n", os.Getpid())
			cerr := fh.Close()
			if werr == nil {
				werr = cerr
			}
			if werr != nil {
				_ = os.Remove(path)
				return nil, fmt.Errorf("hub: %s: %w", path, werr)
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
	_ = os.Remove(l.path)
	l.path = ""
}
