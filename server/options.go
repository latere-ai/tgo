// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"fmt"
	"io"
	"os"
	"time"
)

// Defaults, each chosen because it is the honest answer rather than the
// generous one.
const (
	// DefaultAddr is loopback (009-D8). A server with no authentication is not
	// exposed by omission.
	DefaultAddr = "127.0.0.1:11434"

	// DefaultConcurrency is one session at a time.
	//
	// Not timidity: without batching (specs/008-scheduler.md), two requests
	// generating at once interleave at submission granularity and the total
	// throughput is what one sequence gets, while the second session's KV
	// cache is a second fixed reservation. Raising it costs memory and buys
	// latency fairness, which is a trade a deployment makes on purpose.
	DefaultConcurrency = 1

	// DefaultQueue is how many requests wait for a slot.
	DefaultQueue = 32

	// DefaultQueueWait is how long one of them waits before it is refused.
	DefaultQueueWait = 30 * time.Second

	// DefaultMaxBodyBytes bounds a request body. A prompt that does not fit a
	// session's context is refused anyway, so this bounds the memory a body
	// takes before anything has looked at it.
	DefaultMaxBodyBytes = 8 << 20
)

// options are what [New] resolved.
type options struct {
	concurrency int
	queue       int
	queueWait   time.Duration
	maxBody     int64
	kvBudget    int64
	public      bool
	notice      io.Writer
}

func defaults() options {
	return options{
		concurrency: DefaultConcurrency,
		queue:       DefaultQueue,
		queueWait:   DefaultQueueWait,
		maxBody:     DefaultMaxBodyBytes,
		notice:      os.Stderr,
	}
}

// Option configures [New].
type Option func(*options) error

// WithConcurrency sets how many sessions may exist at once.
//
// [WithKVBudget] is the same number arrived at from memory, which is the
// quantity that actually bounds it for an engine that allocates per request.
// For a pooled engine ([WrapPool]) the bound is already a count -- the pool's
// size -- so pass that here, and [New] refuses a larger one
// (specs/019-session-affinity.md §4).
func WithConcurrency(n int) Option {
	return func(o *options) error {
		if n < 1 {
			return fmt.Errorf("server: concurrency must be at least 1, got %d", n)
		}
		o.concurrency = n
		return nil
	}
}

// WithKVBudget sets the concurrency from the device memory left for key/value
// caches, in bytes.
//
// specs/005-kv-cache.md makes each session's cache a fixed reservation, so
//
//	N_max = floor(budget / Engine.CacheBytesPerSession)
//
// and a budget that does not hold one session is an error at startup rather
// than an out-of-memory error under load.
//
// It is for an engine that allocates a session per request ([Wrap]). A pooled
// engine reserved its sessions already, so what bounds it is how many it
// reserved and not what the device has left; use [WithConcurrency] with the
// pool's size.
func WithKVBudget(bytes int64) Option {
	return func(o *options) error {
		if bytes <= 0 {
			return fmt.Errorf("server: the KV budget must be positive, got %d", bytes)
		}
		o.kvBudget = bytes
		return nil
	}
}

// WithQueue bounds how many requests wait for a session.
//
// The bound is the point (009-D3): an unbounded queue converts a load problem
// into an out-of-memory one, and a caller who will be served in an hour would
// rather be told now.
func WithQueue(n int) Option {
	return func(o *options) error {
		if n < 0 {
			return fmt.Errorf("server: the queue bound must not be negative, got %d", n)
		}
		o.queue = n
		return nil
	}
}

// WithQueueWait sets how long a queued request waits before it is refused with
// 429 and Retry-After.
func WithQueueWait(d time.Duration) Option {
	return func(o *options) error {
		if d <= 0 {
			return fmt.Errorf("server: the queue wait must be positive, got %s", d)
		}
		o.queueWait = d
		return nil
	}
}

// WithMaxBodyBytes bounds a request body.
func WithMaxBodyBytes(n int64) Option {
	return func(o *options) error {
		if n < 1 {
			return fmt.Errorf("server: the body bound must be at least 1 byte, got %d", n)
		}
		o.maxBody = n
		return nil
	}
}

// WithPublicBind allows [Server.Listen] to bind an address that is not
// loopback.
//
// It is a flag and not a default because this server has no authentication
// (009-D5, 009-D8), and binding it to the world by omission is the mistake the
// flag exists to make deliberate. Listening publicly prints a line saying so.
func WithPublicBind() Option {
	return func(o *options) error {
		o.public = true
		return nil
	}
}

// WithNotice redirects the lines this package prints -- the public-bind
// warning and the failures a request cannot be told about. The default is
// standard error; nil silences them.
func WithNotice(w io.Writer) Option {
	return func(o *options) error {
		o.notice = w
		return nil
	}
}
