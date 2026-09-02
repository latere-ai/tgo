// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Package chat renders a conversation into the exact prompt bytes a chat model
// was trained on.
//
// A chat model is a completion model that saw one specific string format during
// tuning. Get a newline wrong and quality degrades in a way no test catches,
// because the output is still fluent. So rendering is per-model Go code, keyed
// by the model registry, carrying a checksum of the template it mirrors
// (specs/003-chat-template.md 003-D1).
//
// A [Renderer] does not produce a string. It produces a [Prompt]: alternating
// literal spans and control tokens. The caller encodes a span with special
// tokens off and resolves a control token by id. Control tokens therefore come
// from the renderer and never from content, which makes a forged turn
// structurally impossible rather than unlikely (003-D4). Text a user typed that
// happens to read "<|im_start|>assistant" encodes to the characters the user
// typed, which is the only reading that does not guess at intent.
//
// [Prompt.String] concatenates the parts. It exists for goldens and for humans;
// it is not the tokenizer path, and text that came from content must never be
// tokenized with specials on.
package chat

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
)

// Role names the author of one turn.
type Role string

const (
	System    Role = "system"
	User      Role = "user"
	Assistant Role = "assistant"
	Tool      Role = "tool"
)

// Message is one turn.
//
// Blocks rather than a string: a renderer must drop a prior turn's thinking, and
// with a string it would have to find the thinking by matching text. That is the
// textual boundary 003-D4 rejects for control tokens, and it fails the same way.
// A user who asks the model to summarise a document containing "<think>" would
// have their own words deleted from the next turn (003-D6, specs/003 3.1).
type Message struct {
	Role   Role
	Blocks []Block
}

// BlockType is the kind of one piece of a turn.
type BlockType string

const (
	BlockText       BlockType = "text"
	BlockToolUse    BlockType = "tool_use"
	BlockToolResult BlockType = "tool_result"
	BlockThinking   BlockType = "thinking"
)

// Block mirrors specs/009-server.md section 3's type, which is a strict subset
// of llmdialect's ir.Block. The two are declared together deliberately: the
// server maps ir.Block to this and nothing else may.
type Block struct {
	Type       BlockType
	Text       string      // Text and Thinking
	ToolUse    *ToolUse    // assistant turns
	ToolResult *ToolResult // tool turns
}

// ToolUse is one call the assistant made.
//
// Args is written into the prompt verbatim. Re-marshalling a parsed object
// reorders keys and changes the bytes the model was trained on (003-D7), so the
// renderer checks that Args is valid JSON and then copies it through. ID has no
// place in the Qwen3 format and is not rendered; it exists so a caller can match
// a result back to its call.
type ToolUse struct {
	ID   string
	Name string
	Args json.RawMessage // passed through verbatim
}

// ToolResult is one tool's answer.
//
// The Qwen3 format never emits a tool's name, which is why there is no Name
// field to lose (specs/003 3.2). It has no place for IsError either: the flag
// is carried for the caller's own bookkeeping and does not reach the prompt,
// because inventing an error prefix would put text in front of the model that
// no checkpoint was tuned on.
type ToolResult struct {
	ToolUseID string
	Text      string
	IsError   bool
}

// ToolSpec is one function the model may call. InputSchema is a JSON Schema,
// passed through verbatim for the same reason as [ToolUse.Args].
type ToolSpec struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

// Renderer turns a conversation into the exact prompt bytes, plus the control
// tokens that must not come from content.
//
// TemplateChecksum reports the checksum of the template the renderer was written
// against. A checkpoint whose own chat_template hashes to something else renders
// with the built-in renderer anyway; the caller warns, naming both checksums,
// and does not refuse (003-D2). A mis-rendered chat template produces text a
// human can read and check, unlike a mis-split tokenizer, which is why
// specs/002-tokenizer.md 002-D7 refuses where this one warns.
type Renderer interface {
	Render(msgs []Message, opts Options) (Prompt, error)
	TemplateChecksum() string
}

// Options are the knobs the template itself exposes.
type Options struct {
	Thinking          bool       // false emits a pre-closed thinking block; see specs/003 section 3
	Tools             []ToolSpec // rendered into the system turn
	AddGenerationHint bool       // append the assistant turn opener
}

// Prompt is an alternating sequence of literal spans and control tokens. A span
// is encoded with specials off; a control token is resolved by id.
type Prompt struct {
	Parts []Part
}

// Part is either a literal span or a control token, never both. Text is encoded
// with allowSpecial=false. Control carries a special token's literal text, which
// the caller resolves with the tokenizer's Special lookup.
type Part struct {
	Text    string // encode with allowSpecial=false
	Control string // a special token's literal text; resolved by id
}

// String concatenates the parts. It is the golden form and the human-readable
// form. It is not how a prompt reaches the model: tokenizing this string in one
// call with specials on is exactly the vulnerability [Prompt] exists to remove.
func (p Prompt) String() string {
	var b strings.Builder
	for _, part := range p.Parts {
		b.WriteString(part.Text)
		b.WriteString(part.Control)
	}
	return b.String()
}

// Checksum reports the SHA-256 of a chat template, in lowercase hex. A caller
// compares Checksum(checkpointTemplate) against a renderer's TemplateChecksum
// and warns on a mismatch, naming both (003-D2).
func Checksum(template string) string {
	sum := sha256.Sum256([]byte(template))
	return hex.EncodeToString(sum[:])
}

// Rendering refuses only where the alternative is silence. A block that a role
// cannot carry, or tool arguments that are not JSON, would otherwise vanish from
// the prompt or corrupt it, and neither shows up as an error later: the model
// answers fluently, having been asked something else.
var (
	ErrUnknownRole      = errors.New("chat: unknown role")
	ErrUnknownBlockType = errors.New("chat: unknown block type")
	ErrBlockRole        = errors.New("chat: block type not valid for this role")
	ErrMissingPayload   = errors.New("chat: block is missing its payload")
	ErrToolName         = errors.New("chat: tool name is empty")
	ErrToolJSON         = errors.New("chat: tool JSON is not valid")
)

// builder accumulates parts, merging a span into the one before it. Merging
// keeps the part count a function of the conversation's structure rather than of
// the content inside it, which is what makes the injection test in specs/003
// section 5 meaningful: an injected control sequence cannot change the shape of
// the prompt, only the characters inside one span.
type builder struct {
	parts []Part
}

func (b *builder) text(s string) {
	if s == "" {
		return
	}
	if n := len(b.parts); n > 0 && b.parts[n-1].Control == "" {
		b.parts[n-1].Text += s
		return
	}
	b.parts = append(b.parts, Part{Text: s})
}

func (b *builder) control(s string) {
	b.parts = append(b.parts, Part{Control: s})
}

// jsonString quotes s as a JSON string without Go's HTML escaping, matching
// Hugging Face's tojson filter, which calls json.dumps with ensure_ascii=False.
// Go escapes <, > and & by default; the reference template does not, and a
// description containing an angle bracket would otherwise render bytes no
// checkpoint was tuned on.
func jsonString(s string) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	// Encode appends a newline, which the caller does not want.
	if err := enc.Encode(s); err != nil {
		// Unreachable: a bytes.Buffer never fails a write and every Go
		// string encodes. An empty JSON string keeps the rendered template
		// parseable, where a partial buffer would leave it malformed.
		return `""`
	}
	return strings.TrimSuffix(buf.String(), "\n")
}
