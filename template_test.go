// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/latere-ai/tgo/chat"
)

// stubRenderer carries a checksum and renders nothing. The comparison under
// test reads only the checksum, and a real renderer would tie these cases to
// one architecture's template.
type stubRenderer struct{ sum string }

func (s stubRenderer) Render([]chat.Message, chat.Options) (chat.Prompt, error) {
	return chat.Prompt{}, nil
}
func (s stubRenderer) TemplateChecksum() string { return s.sum }

// TestACheckpointTemplateMismatchIsReported is 003-D2, which had no code path
// until 2026-08-27: Checksum and TemplateChecksum both existed and nothing
// called either against a checkpoint.
//
// The failure it catches is silent by construction. A checkpoint tuned on one
// template, rendered with another, produces a prompt in a shape the model was
// not trained on — and the symptom is answers that are slightly worse, which
// nobody attributes to a template.
func TestACheckpointTemplateMismatchIsReported(t *testing.T) {
	const tmpl = "{% for m in messages %}{{ m.content }}{% endfor %}"
	matching := stubRenderer{sum: chat.Checksum(tmpl)}
	other := stubRenderer{sum: chat.Checksum(tmpl + " ")}

	for _, tc := range []struct {
		name string
		raw  string
		r    chat.Renderer
		want string // "" means say nothing
	}{{
		name: "a template that differs is named with both checksums",
		raw:  `{"chat_template": "` + tmpl + `"}`,
		r:    other,
		want: chat.Checksum(tmpl),
	}, {
		name: "a template that matches is silent",
		raw:  `{"chat_template": "` + tmpl + `"}`,
		r:    matching,
	}, {
		// A base model ships without one. Warning here would fire on every
		// such checkpoint and teach the reader to ignore the message.
		name: "no template declared is silent",
		raw:  `{"model_max_length": 32768}`,
		r:    other,
	}, {
		name: "a named list is read for its default entry",
		raw: `{"chat_template": [{"name": "tool_use", "template": "x"},
		                        {"name": "default", "template": "` + tmpl + `"}]}`,
		r:    other,
		want: chat.Checksum(tmpl),
	}, {
		// Which of several is "the one" is not tgo's to guess, and guessing
		// wrong produces a warning about a template nobody renders with.
		name: "a named list with no default is silent",
		raw:  `{"chat_template": [{"name": "tool_use", "template": "x"}]}`,
		r:    other,
	}, {
		name: "a file that does not parse says so",
		raw:  `{"chat_template":`,
		r:    other,
		want: "not comparing the chat template",
	}, {
		name: "a chat_template of the wrong shape says so",
		raw:  `{"chat_template": 7}`,
		r:    other,
		want: "neither a string nor a list",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			got := templateWarning([]byte(tc.raw), tc.r)
			if tc.want == "" {
				if got != "" {
					t.Fatalf("warned when there was nothing to say: %q", got)
				}
				return
			}
			if got == "" {
				t.Fatal("said nothing")
			}
			if !strings.Contains(got, tc.want) {
				t.Errorf("warning is %q, want it to contain %q", got, tc.want)
			}
			// 003-D2 names *both* checksums, because one is unactionable: a
			// reader with only the checkpoint's hash cannot tell which
			// renderer tgo used.
			if tc.want == chat.Checksum(tmpl) && !strings.Contains(got, tc.r.TemplateChecksum()) {
				t.Errorf("warning names the checkpoint's checksum and not tgo's: %q", got)
			}
		})
	}
}

// TestAMissingTokenizerConfigIsNotAnError pins the rule that this check never
// stops a load. Every refusal that matters about a checkpoint is made
// elsewhere, and a model that loads and generates should not be blocked by a
// file read only to be helpful about.
func TestAMissingTokenizerConfigIsNotAnError(t *testing.T) {
	warnTemplateMismatch(t.TempDir(), stubRenderer{sum: "x"})
}

// TestTheRealQwen3TemplateIsSilent is the case that decides whether this check
// is useful or is noise.
//
// A warning that fires on every correct checkpoint teaches the reader to ignore
// it, and then it is worse than nothing. chat/testdata/qwen3_chat_template.jinja
// is the `chat_template` field of Qwen3's own tokenizer_config.json, byte for
// byte, so wrapping it back into that shape is the load a real checkpoint
// makes. It must say nothing.
//
// The test ties three things together that are each checked alone elsewhere:
// the vendored fixture, Qwen3TemplateChecksum, and the JSON path this file
// added. Editing any one of them without the others turns every real load
// noisy.
func TestTheRealQwen3TemplateIsSilent(t *testing.T) {
	tmpl, err := os.ReadFile(filepath.Join("chat", "testdata", "qwen3_chat_template.jinja"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(map[string]string{"chat_template": string(tmpl)})
	if err != nil {
		t.Fatal(err)
	}
	if got := templateWarning(raw, chat.Qwen3()); got != "" {
		t.Errorf("the checkpoint's own template warned:\n%s", got)
	}

	// And the negative control: one edited byte must warn. Without it this
	// test passes for a comparison that never fires.
	edited, err := json.Marshal(map[string]string{"chat_template": string(tmpl) + "\n"})
	if err != nil {
		t.Fatal(err)
	}
	if got := templateWarning(edited, chat.Qwen3()); got == "" {
		t.Error("a template with one byte added was accepted as the same template")
	}
}
