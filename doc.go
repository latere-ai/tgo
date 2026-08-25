// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

/*
Package tgo runs a language model.

It is the whole of tgo's public surface. Everything under it -- the plan cache,
the KV layout, the graph builders and the scheduler -- is unexported, because
every one of them is a place accel's shape will move
(specs/000-decisions.md D10, specs/007-engine.md 007-D7).

	m, err := tgo.Open("/path/to/qwen3-0.6b")
	defer m.Close()

	s, err := m.NewSession()
	defer s.Close()

	st, err := s.Chat(ctx, []chat.Message{{
		Role:   chat.User,
		Blocks: []chat.Block{{Type: chat.BlockText, Text: "Why is the sky blue?"}},
	}}, tgo.Policy{Temperature: 0.7, Seed: 1})
	for st.Next() {
		fmt.Print(st.Text())
	}
	if err := st.Err(); err != nil { ... }

# Three objects

A [Model] owns the device, the uploaded weights and the plan cache. One per
process per model; weights are immutable and shared, and nothing about a request
touches it.

A [Session] owns one conversation's key/value cache, its position in that cache
and the buffers a step binds.

A [Stream] runs prefill and then the decode loop, one model step per
[Stream.Next].

# Concurrency

[Model] is safe for concurrent use, and that needs a lock rather than only a
claim: accel returns the same compiled plan for an identical graph and a plan
refuses a second submission while one is in flight, so two sessions decoding at
once would get a failed fence rather than a data race -- invisible to a -race
test, and visible to a server as errors under load (007-D9).

[Session] is deliberately not safe for concurrent use (007-D1). Two goroutines
decoding one session would interleave writes into one cache, and serialising
that internally would hide a caller's bug rather than report it. Use one session
per conversation and as many sessions at once as you like.

# Errors

A device failure mid-generation ends the stream with the error and leaves the
session unusable, not silently reset: the cache holds a partial write whose
extent is unknown, and continuing from it would produce plausible text from a
corrupt state. [Session.Reset] is the explicit recovery, and a failed session
refuses work with the original error attached until then (007-D5).

A request that does not fit the session's context is refused rather than
truncated, at the request and not partway through it (007 §7).

# What runs today, measured

Neither device this package can open runs a real model at a usable speed, and
that is the state to know before reaching for it rather than after.

accel's Metal backend cannot compile a tgo graph at all.
specs/004-model-graph.md §3.2 slices the last position out before the LM head
and packs the result, because accel refuses a strided operand into a matrix
multiply rather than copying behind the caller's back — and the packing kernel
carries no MSL artifact. Every forward pass contains one, so every compile is
refused.

accel's CPU backend runs, and it is a correctness oracle rather than an engine:
its own documentation calls it one. Measured on an Apple M2 on 2026-08-25 with
Qwen3-0.6B at f16, opening the checkpoint takes 11 s and one decode step takes
3 minutes 21 seconds.

So the loop this package implements is correct, instrumented and tested, on a
model whose arithmetic no available device does quickly. That gap is the
finding, and closing it is upstream work.

# What this package measures

A decode step transfers a whole row of logits back to the host to sample it:
608 KB for Qwen3, on a step whose useful output is four bytes. That readback is
the floor v0 measures and reports upstream, because "how much of a decode step
is the readback" is the question tgo exists to answer for accel (007-D4).
*/
package tgo
