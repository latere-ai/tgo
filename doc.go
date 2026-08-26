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

# Reusing a prompt's prefix

The key/value state at position t is a function of tokens 0..t and the weights
alone, so a prompt that begins with tokens a session already scored begins with
state it already holds. [WithPrefixCache] turns that into a shorter prefill:
turn n of a conversation prefills the new turn instead of the whole history,
which is specs/016-prefix-cache.md §1's 1-1/n. [Usage.CachedPromptTokens]
reports how many positions were reused, so a cache that stopped working reads as
a number rather than as "the framework got slower".

It is off by default, and turning it on is a decision rather than a default,
because a reused prefix was computed under a different prefill shape and
floating point is not associative: a warm answer equals a cold one in
distribution and not bit for bit (016-D6). Reuse also stops one token short of
the prompt, always -- the cache holds key/value state and not logits, and
sampling needs a forward pass over the last prompt position (016-D10).

Sharing across sessions ([CacheProcess]) is refused rather than approximated.
It needs the cache addressed through a page table, and tgo's graph declares no
page-table port.

# JSON that parses by construction

[Policy.Schema] carries a JSON Schema, and the completion is masked against it
at every step: a token that cannot continue a document matching the schema is
given probability zero, so the output parses and matches without a retry loop
(specs/015-structured-output.md §1). [Model.CheckSchema] is the same compilation
on its own, for a caller who wants the refusal before the request, and it keeps
what it compiled so the request that follows pays a map lookup.

A schema that cannot be turned into a mask is refused by name rather than
approximated -- "minimum" is arithmetic on a value and the automaton counts
characters -- because a keyword silently ignored produces a document that
validates against a schema the caller did not write (015-D4). Three narrowings
are deliberate and all three shrink the admitted language: an object's
properties are emitted in the schema's order, an object is closed, and
"integer" admits the plain spelling. A number's magnitude is not narrowed,
because a bound on it is spelled "maximum", so a caller who needs one checks it
after decoding.

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
