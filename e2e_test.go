// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tgo

import (
	"testing"
	"time"

	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/internal/conformance"
	"github.com/latere-ai/tgo/model"
)

// TestRealCheckpointEndToEnd is the one gated test in this package: a real
// Qwen3-0.6B, opened, prompted, and generated from.
//
// Tier 3, so it is skipped unless TGO_MODEL names a checkpoint directory and it
// is never in CI (specs/010-conformance.md 010-D4). It stays in the tree
// because everything else here runs on a fixture this package wrote, and the
// three things a fixture cannot check are all here: that the weight map meets a
// file nobody in this repository produced, that the loader's f16 planes and
// this package's f32 gains bind one graph, and that the same request twice
// gives the same answer.
//
// # Why it generates one token and not fifty
//
// Measured on an Apple M2, 2026-08-25: opening the checkpoint takes 11 s and
// **one decode step takes 3 minutes 21 seconds**. accel's CPU backend is an
// oracle rather than an engine — its own documentation calls it one — and it
// runs a 0.6B forward pass at roughly two million multiply-accumulates a
// second.
//
// The device that would run this in milliseconds cannot compile the graph:
// specs/004-model-graph.md §3.2's Slice-then-Contiguous puts a packing kernel in
// every forward pass and that kernel carries no MSL artifact, so no tgo graph
// compiles on Metal at all. [TestMetalCannotYetRunTheForwardPass] pins it.
//
// So this test is sized to the device that works. A one-token prompt in a
// two-position cache is one prefill submission per run, and the whole test is
// three submissions — measured at 11 minutes each on a loaded machine, so
// budget half an hour. Fifty tokens would be a day, and a gate nobody runs is
// not a gate. The test logs its own total; if that number falls, accel got
// faster.
//
// # And the two rows a fixture cannot reach
//
// The prefill runs at T=1 real token in a 2-row bucket, so row 1 is a pad row —
// and it is the row specs/004-model-graph.md §3.2 reads the logits from. The
// third submission below scores the same token as a one-row step, where there
// is no padding at all, and asserts the two choose the same token. That is
// specs/007-engine.md §4's guarantee on a real model's weights rather than on a
// fixture's.
func TestRealCheckpointEndToEnd(t *testing.T) {
	dir := conformance.ModelPath(t)

	// Two positions: one prompt token and one generated. bucketsFor gives a
	// single bucket of 2, so a prefill is two rows and the second is padding.
	const capacity = 2

	started := time.Now()
	// Auto rather than CPU: accel#19 is closed, so Metal runs the forward pass
	// and is what a user on this hardware gets. The CPU pin here was a
	// workaround for that gap and outlived it.
	m, err := Open(dir, WithContext(capacity))
	if err != nil {
		t.Fatalf("Open(%s): %v", dir, err)
	}
	// t.Cleanup and not defer: cleanups run last-in-first-out, so registering
	// the model's close before any session's is what closes the sessions first.
	// accel closes in order rather than recursively and refuses a device whose
	// buffers are still live.
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("Model.Close: %v", err)
		}
	})
	t.Logf("Open took %v", time.Since(started))

	info := m.Info()
	t.Logf("info: %+v", info)
	if info.Architecture != model.Qwen3Architecture {
		t.Fatalf("Architecture = %q", info.Architecture)
	}
	// The field that bites: Qwen3-0.6B states head_dim 128 where
	// hidden_size/num_attention_heads is 64 (specs/004-model-graph.md §5).
	if info.HeadDim == info.HiddenSize/info.Heads {
		t.Errorf("HeadDim = %d = hidden_size/heads; this checkpoint states it explicitly "+
			"and it is not that value", info.HeadDim)
	}
	if info.WeightBytes < 1<<29 {
		t.Errorf("WeightBytes = %d, too small for a 0.6B model at f16", info.WeightBytes)
	}
	// The structural markers the decode loop reads. A real Qwen3 tokenizer has
	// all of them; a checkpoint without them would stream as plain text.
	for name, id := range map[string]int{
		"<|im_end|>": m.special.imEnd, "<think>": m.special.think[0],
		"</think>": m.special.thinkEnd, "<tool_call>": m.special.toolCall,
	} {
		if id < 0 {
			t.Errorf("the tokenizer does not resolve %s", name)
		}
	}

	// The chat path's host half, which costs nothing: the real template
	// rendered against the real vocabulary, with every control token resolved
	// by id rather than encoded as characters (003-D4). Generating from it
	// would need a 32-row bucket, which is an hour.
	probe := session(t, m, WithSessionContext(capacity), WithThinking(false))
	prompt, err := m.renderer().Render([]chat.Message{{
		Role:   chat.User,
		Blocks: []chat.Block{{Type: chat.BlockText, Text: "Say hello."}},
	}}, chat.Options{AddGenerationHint: true})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	rendered, err := probe.m.encode(prompt)
	if err != nil {
		t.Fatalf("encoding the rendered prompt: %v", err)
	}
	if len(rendered) < 8 {
		t.Errorf("the rendered turn is %d tokens, which is too few to be a chat prompt",
			len(rendered))
	}
	if _, err := probe.start(t.Context(), rendered, greedy(1)); err == nil {
		t.Errorf("a %d-token prompt was accepted into a %d-position cache",
			len(rendered), capacity)
	}

	// One prompt token, one generated token, twice — through the public path,
	// so the prompt text has to be one whose ids are one token.
	ids := m.tok.Encode("Paris", false)
	if len(ids) == 0 {
		t.Fatal("the prompt encoded to nothing")
	}
	ids = ids[:1]
	text := m.tok.Decode(ids)
	if got := m.tok.Encode(text, false); len(got) != 1 || got[0] != ids[0] {
		t.Fatalf("%q encodes to %v rather than to the single token %d it decoded from; "+
			"pick a prompt that round-trips", text, got, ids[0])
	}
	t.Logf("prompt token %d = %q", ids[0], text)

	var runs []string
	for i := range 2 {
		s := session(t, m, WithSessionContext(capacity))
		step := time.Now()
		st, err := s.Complete(t.Context(), text, greedy(1))
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		out, evs := collect(t, st)
		if err := st.Err(); err != nil {
			t.Fatalf("stream: %v", err)
		}
		t.Logf("run %d: %q over %d events in %v", i, out, len(evs), time.Since(step))
		if u := st.Usage(); u.PromptTokens != 1 || u.CompletionTokens != 1 {
			t.Errorf("usage = %+v, want one prompt token and one completion token", u)
		}
		runs = append(runs, out)
	}
	if runs[0] != runs[1] {
		t.Errorf("two greedy runs of one prompt gave %q and %q", runs[0], runs[1])
	}
	if runs[0] == "" {
		t.Error("the completion is empty")
	}

	// specs/007-engine.md §4 on real weights: the padded prefill above read its
	// logits from row 1, which is a pad row. A one-row step has no padding, so
	// if the pad row were a pad *token* rather than the last real one, these two
	// would choose different tokens — and nothing anywhere would say so.
	s := session(t, m, WithSessionContext(capacity))
	logits, _, err := s.run(1, ids, 0)
	if err != nil {
		t.Fatalf("one-row step: %v", err)
	}
	unpadded := m.tok.Decode([]int{argmax(logits)})
	if unpadded != runs[0] {
		t.Errorf("the padded prefill chose %q and an unpadded one-row step chose %q; a "+
			"bucket's pad rows must reproduce the last real row, because §3.2 reads the "+
			"logits from the last row of the plan", runs[0], unpadded)
	}
	t.Logf("the whole gated test took %v", time.Since(started))
}
