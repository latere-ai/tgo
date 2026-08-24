package chat

import (
	"encoding/json"
	"fmt"
	"strings"
)

// The control tokens the Qwen3 renderer emits. Every one of them is an entry in
// the tokenizer's added_tokens, so each must reach the tokenizer as a
// [Part.Control] and be resolved by id. Writing any of them into a [Part.Text]
// would encode it as characters, which is right for content and wrong for the
// renderer's own markup.
const (
	imStart           = "<|im_start|>"
	imEnd             = "<|im_end|>"
	thinkOpen         = "<think>"
	thinkClose        = "</think>"
	toolCallOpen      = "<tool_call>"
	toolCallClose     = "</tool_call>"
	toolResponseOpen  = "<tool_response>"
	toolResponseClose = "</tool_response>"
)

// Qwen3TemplateChecksum is the SHA-256, in lowercase hex, of the chat_template
// string this renderer was written against. testdata/qwen3_chat_template.jinja
// is that template, copied byte for byte out of a Qwen3 checkpoint's
// tokenizer_config.json, and a test keeps the two in step.
//
// A checkpoint whose template hashes to something else still renders, with a
// warning naming both checksums (003-D2). The warning is the caller's: a
// renderer that consulted the checksum would have two behaviours to test and
// would refuse work a human can verify by reading the prompt.
const Qwen3TemplateChecksum = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"

// Qwen3 returns the renderer for Qwen3's chat format (specs/003 section 3).
func Qwen3() Renderer { return qwen3{} }

type qwen3 struct{}

func (qwen3) TemplateChecksum() string { return Qwen3TemplateChecksum }

// Render walks the conversation once and emits the turn structure. It never
// inspects content to decide structure: a turn boundary comes from a role, a
// thinking block from a block type, a tool result from the Tool role (003-D8).
func (q qwen3) Render(msgs []Message, opts Options) (Prompt, error) {
	if err := validate(msgs, opts); err != nil {
		return Prompt{}, err
	}

	b := &builder{}
	// A system message is emitted only when the caller supplied one, and only
	// from position zero, which is where the template looks for it. Qwen3 has no
	// default system prompt and inventing one changes the tuned behaviour
	// (003-D5).
	leadSystem := len(msgs) > 0 && msgs[0].Role == System
	q.header(b, msgs, opts, leadSystem)

	lastQuery := lastQueryIndex(msgs)
	for i, m := range msgs {
		if i == 0 && leadSystem {
			continue // already in the header
		}
		switch m.Role {
		case User, System:
			b.control(imStart)
			b.text(string(m.Role) + "\n" + plainText(m))
			b.control(imEnd)
			b.text("\n")
		case Assistant:
			q.assistant(b, m, i, lastQuery, len(msgs))
		case Tool:
			// A tool result is not its own turn. A run of them merges into one
			// user turn, each result wrapped in <tool_response> (specs/003 3.2).
			if i == 0 || msgs[i-1].Role != Tool {
				b.control(imStart)
				b.text("user")
			}
			b.text("\n")
			b.control(toolResponseOpen)
			b.text("\n" + plainText(m) + "\n")
			b.control(toolResponseClose)
			if i == len(msgs)-1 || msgs[i+1].Role != Tool {
				b.control(imEnd)
				b.text("\n")
			}
		}
	}

	if opts.AddGenerationHint {
		b.control(imStart)
		// The newline after "assistant" is part of the prompt. Without it the
		// model's first generated token is that newline and every downstream
		// measurement is off by one.
		b.text("assistant\n")
		if !opts.Thinking {
			// Thinking off does not omit the block, it pre-closes one. Omitting
			// it leaves the model free to open its own, which is the behaviour
			// the flag exists to prevent (specs/003 section 3).
			b.control(thinkOpen)
			b.text("\n\n")
			b.control(thinkClose)
			b.text("\n\n")
		}
	}
	return Prompt{Parts: b.parts}, nil
}

// header renders the system turn, which is also where tool definitions go: the
// template gives tools no role of their own.
func (qwen3) header(b *builder, msgs []Message, opts Options, leadSystem bool) {
	var sys string
	if leadSystem {
		sys = plainText(msgs[0])
	}
	if len(opts.Tools) == 0 {
		if leadSystem {
			b.control(imStart)
			b.text("system\n" + sys)
			b.control(imEnd)
			b.text("\n")
		}
		return
	}

	b.control(imStart)
	b.text("system\n")
	if leadSystem {
		b.text(sys + "\n\n")
	}
	b.text("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
		"You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
	for _, t := range opts.Tools {
		b.text("\n" + toolJSON(t))
	}
	b.text("\n</tools>\n\nFor each function call, return a json object with function name and arguments within ")
	b.control(toolCallOpen)
	b.control(toolCallClose)
	b.text(" XML tags:\n")
	b.control(toolCallOpen)
	b.text("\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n")
	b.control(toolCallClose)
	b.control(imEnd)
	b.text("\n")
}

// assistant renders one assistant turn, keeping its thinking only if the turn
// belongs to the round now being generated.
func (qwen3) assistant(b *builder, m Message, i, lastQuery, n int) {
	var content, reasoning strings.Builder
	var calls []*ToolUse
	for _, bl := range m.Blocks {
		switch bl.Type {
		case BlockText:
			content.WriteString(bl.Text)
		case BlockThinking:
			reasoning.WriteString(bl.Text)
		case BlockToolUse:
			calls = append(calls, bl.ToolUse)
		}
	}
	c, r := content.String(), reasoning.String()

	b.control(imStart)
	b.text("assistant\n")
	// Turns up to the last user message are replayed without their thinking:
	// Qwen3 keeps it only for the round being generated, and replaying old
	// thinking wastes context and shifts the distribution the model was tuned
	// for. Turns after it, which is the rest of a multi-step tool round, keep
	// theirs. Nothing here reads the text; the thinking is gone by type.
	if i > lastQuery && (i == n-1 || r != "") {
		b.control(thinkOpen)
		b.text("\n" + strings.Trim(r, "\n") + "\n")
		b.control(thinkClose)
		b.text("\n\n" + strings.TrimLeft(c, "\n"))
	} else {
		b.text(c)
	}

	for j, call := range calls {
		if j > 0 || c != "" {
			b.text("\n")
		}
		b.control(toolCallOpen)
		b.text("\n{\"name\": " + jsonString(call.Name) + ", \"arguments\": " + arguments(call) + "}\n")
		b.control(toolCallClose)
	}

	b.control(imEnd)
	b.text("\n")
}

// arguments returns the call's arguments verbatim. Absent arguments render as an
// empty object rather than as nothing, which would leave the JSON malformed.
func arguments(call *ToolUse) string {
	if len(call.Args) == 0 {
		return "{}"
	}
	return string(call.Args)
}

// toolJSON is one tool in the shape Hugging Face's tojson filter produces for a
// normalised tool definition: a space after every colon and comma, no HTML
// escaping, and the schema copied through untouched.
func toolJSON(t ToolSpec) string {
	schema := "{}"
	if len(t.InputSchema) > 0 {
		schema = string(t.InputSchema)
	}
	return `{"type": "function", "function": {"name": ` + jsonString(t.Name) +
		`, "description": ` + jsonString(t.Description) +
		`, "parameters": ` + schema + `}}`
}

// plainText concatenates a turn's text, and a tool turn's results, in order. No
// separator is inserted between blocks: a separator would be bytes the caller
// did not write and the model was not tuned on.
func plainText(m Message) string {
	var b strings.Builder
	for _, bl := range m.Blocks {
		switch bl.Type {
		case BlockText:
			b.WriteString(bl.Text)
		case BlockToolResult:
			b.WriteString(bl.ToolResult.Text)
		}
	}
	return b.String()
}

// lastQueryIndex is the index of the last real user query. Assistant turns after
// it are the round now being generated and keep their thinking.
//
// The reference template answers "is this message a tool result" by testing
// whether a user message starts and ends with <tool_response>, which misfires on
// a user who quotes the tag. Here a tool result has the Tool role, so the scan
// is structural (003-D8). With no user message at all the template treats the
// whole conversation as prior context, and so does this.
func lastQueryIndex(msgs []Message) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == User {
			return i
		}
	}
	return len(msgs) - 1
}

// validate rejects a conversation the format cannot carry. Every case here would
// otherwise drop a block from the prompt or emit malformed JSON, and neither
// surfaces later: the model answers fluently, having been asked something else.
func validate(msgs []Message, opts Options) error {
	for i, m := range msgs {
		switch m.Role {
		case System, User, Assistant, Tool:
		default:
			return fmt.Errorf("%w: message %d: %q", ErrUnknownRole, i, m.Role)
		}
		for j, bl := range m.Blocks {
			where := fmt.Sprintf("message %d block %d", i, j)
			switch bl.Type {
			case BlockText:
			case BlockThinking:
				if m.Role != Assistant {
					return fmt.Errorf("%w: %s: %s on a %s turn", ErrBlockRole, where, bl.Type, m.Role)
				}
			case BlockToolUse:
				if m.Role != Assistant {
					return fmt.Errorf("%w: %s: %s on a %s turn", ErrBlockRole, where, bl.Type, m.Role)
				}
				if bl.ToolUse == nil {
					return fmt.Errorf("%w: %s: %s without a ToolUse", ErrMissingPayload, where, bl.Type)
				}
				if bl.ToolUse.Name == "" {
					return fmt.Errorf("%w: %s", ErrToolName, where)
				}
				if len(bl.ToolUse.Args) > 0 && !json.Valid(bl.ToolUse.Args) {
					return fmt.Errorf("%w: %s: arguments for %q", ErrToolJSON, where, bl.ToolUse.Name)
				}
			case BlockToolResult:
				if m.Role != Tool {
					return fmt.Errorf("%w: %s: %s on a %s turn", ErrBlockRole, where, bl.Type, m.Role)
				}
				if bl.ToolResult == nil {
					return fmt.Errorf("%w: %s: %s without a ToolResult", ErrMissingPayload, where, bl.Type)
				}
			default:
				return fmt.Errorf("%w: %s: %q", ErrUnknownBlockType, where, bl.Type)
			}
		}
	}
	for i, t := range opts.Tools {
		if t.Name == "" {
			return fmt.Errorf("%w: tool %d", ErrToolName, i)
		}
		if len(t.InputSchema) > 0 && !json.Valid(t.InputSchema) {
			return fmt.Errorf("%w: input schema for tool %q", ErrToolJSON, t.Name)
		}
	}
	return nil
}
