// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tgo

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/latere-ai/tgo/chat"
)

// 003-D2: a checkpoint carries the chat template it was tuned with, and tgo
// renders with a Go implementation of it. When the two disagree the model sees
// a prompt in a shape it was not trained on, and the symptom is not an error —
// it is answers that are slightly worse, which nobody attributes to a template.
//
// So the mismatch is reported and the render proceeds. It is a warning rather
// than a refusal because a mis-rendered chat template produces text a human can
// read and check, unlike a mis-split tokenizer, which 002-D7 refuses over.
//
// The pieces for this existed from Wave 1 — chat.Checksum, Renderer's
// TemplateChecksum, and the constant each renderer carries — and nothing called
// them, so the decision had no code path until 2026-08-27. The 2026-08-27 spec
// audit found it by looking for the call rather than for the constant.

// templateFile is where a Hugging Face checkpoint keeps its chat template.
const templateFile = "tokenizer_config.json"

// checkpointTemplate returns the chat template a checkpoint declares, or "" if
// it declares none.
//
// `chat_template` is a string in most checkpoints and a list of named templates
// in a few — the shape transformers added for models with a separate
// tool-calling template. A list is read for the entry named "default", which is
// the one a plain conversation renders with; a list with no such entry returns
// "" rather than guessing which of several is the one to compare against.
func checkpointTemplate(raw []byte) (string, error) {
	var cfg struct {
		ChatTemplate json.RawMessage `json:"chat_template"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return "", fmt.Errorf("parse %s: %w", templateFile, err)
	}
	if len(cfg.ChatTemplate) == 0 {
		return "", nil
	}

	var one string
	if err := json.Unmarshal(cfg.ChatTemplate, &one); err == nil {
		return one, nil
	}

	var many []struct {
		Name     string `json:"name"`
		Template string `json:"template"`
	}
	if err := json.Unmarshal(cfg.ChatTemplate, &many); err != nil {
		return "", fmt.Errorf("%s: chat_template is neither a string nor a list of "+
			"named templates", templateFile)
	}
	for _, t := range many {
		if t.Name == "default" {
			return t.Template, nil
		}
	}
	return "", nil
}

// templateWarning reports what to say about a checkpoint's template, or "" when
// there is nothing to say.
//
// Nothing to say covers two cases that are not the same and read the same from
// here: the checkpoint declares no template, and the checkpoint declares one
// that matches. Neither is a problem, and a message on the first would fire for
// every base model that ships without a template at all.
func templateWarning(raw []byte, r chat.Renderer) string {
	declared, err := checkpointTemplate(raw)
	if err != nil {
		// A tokenizer_config.json that does not parse is worth saying out loud
		// even though nothing here depends on it: the file is the tokenizer's
		// too, and a caller who sees this knows why a later surprise happened.
		return fmt.Sprintf("tgo: %v; not comparing the chat template", err)
	}
	if declared == "" {
		return ""
	}
	got, want := chat.Checksum(declared), r.TemplateChecksum()
	if got == want {
		return ""
	}
	return fmt.Sprintf("tgo: this checkpoint's chat template is not the one tgo "+
		"renders: the checkpoint's hashes to %s and tgo's to %s. Rendering with "+
		"tgo's anyway (003-D2) — the prompt may differ from what the model was "+
		"tuned on, which shows up as slightly worse answers rather than as an "+
		"error", got, want)
}

// warnTemplateMismatch runs the comparison for a checkpoint directory.
//
// A missing file is not an error. Every refusal that matters about a checkpoint
// is made elsewhere — the architecture, the weights, the tokenizer — and a
// model that loads and generates should not be stopped by the absence of a file
// this only reads to be helpful about.
func warnTemplateMismatch(dir string, r chat.Renderer) {
	raw, err := os.ReadFile(filepath.Join(dir, templateFile))
	if err != nil {
		return
	}
	if msg := templateWarning(raw, r); msg != "" {
		fmt.Fprintln(os.Stderr, msg)
	}
}
