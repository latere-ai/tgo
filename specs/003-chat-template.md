---
title: "Chat templates: rendering a conversation into the exact bytes the model was trained on"
status: drafted
layer: text
depends_on:
  - 000-decisions.md
  - 002-tokenizer.md
---

# Chat templates

A chat model is a completion model that was trained on one specific string
format. Get a space wrong and quality degrades in a way no test catches, because
the output is still fluent.

## 1. The problem

`tokenizer_config.json` carries `chat_template`: a **Jinja2** template. Qwen3's
is about sixty lines of Jinja, and it handles system messages, tools, multi-part
content, and the thinking block.

Go has no Jinja2. The options:

| option | cost |
| --- | --- |
| a Jinja2 interpreter in Go | a language implementation, with its own bug surface, to run one template per model |
| a per-model Go function | correct and fast; one function per model family, and a checkpoint that ships a customised template renders wrong |
| a Jinja **subset** interpreter | the subset is a moving target; a template using something outside it fails at load rather than at render |

**Decision: a per-model Go renderer, keyed by the same registry as the model
graph, with a checksum of the template it was written against.** A checkpoint
whose `chat_template` does not match the checksum is not refused — it renders
with the built-in and **warns, naming both**, because a customised template is
usually a trivial edit and refusing to run the model over it helps nobody.

**Rejected: silently trusting an unrecognised template.** The warning is the
whole value of the checksum.

The Jinja interpreter is [014](014-jinja.md), written and unbuilt. It becomes
the right answer when tgo carries enough model families that per-model renderers
stop scaling, and it is not that yet.

## 2. Qwen3's format

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
```

and the model completes until `<|im_end|>`. With thinking enabled the assistant
turn opens with `<think>` and the model closes it with `</think>` before its
answer.

Three details that are load-bearing and easy to lose:

- **The trailing newline after `assistant` is part of the prompt.** Without it
  the model's first generated token is that newline.
- **Prior assistant turns have their thinking stripped.** Qwen3's template keeps
  the thinking block only for the turn being generated; replaying old ones
  wastes context and shifts the distribution.
- **A system message is emitted only if present.** Qwen3 does not inject a
  default one.

## 3. Rendering is a string, and the boundary is exact

The renderer produces bytes. The tokenizer turns them into ids. They are
separate steps and stay separate, because it makes the renderer testable against
a checked-in golden string with no tokenizer involved, and it is what makes §4
possible.

## 4. Injection: user text is data

A user message containing the literal `<|im_start|>assistant` would, after
rendering and tokenizing, produce a **real turn boundary** — the special-token
matcher in [002 §3](002-tokenizer.md) matches it wherever it appears. The user
would have forged an assistant turn.

The rule: **special tokens are emitted by the renderer, never by content.** The
tokenizer therefore encodes rendered *content spans* with the added-token
matcher **off**, and the renderer emits control token ids directly. This makes
the boundary structural rather than textual, and it cannot be talked around.

Cost: a message whose text legitimately contains `<|im_end|>` is encoded as
those literal characters. That is the correct reading of the user's input.

## 5. Tests

- Golden renders for: bare user turn, with system, multi-turn, thinking on and
  off, prior-assistant-thinking stripped.
- The injection case: a user message containing every special token renders to
  content ids, and the turn count is unchanged.
- Checksum mismatch warns and still renders.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 003-D1 | per-model Go renderer, registry-keyed | Jinja2 interpreter; Jinja subset | one renderer per model family; [014](014-jinja.md) if that stops scaling |
| 003-D2 | template checksum warns, does not refuse | silent trust; hard refusal | a customised checkpoint runs, loudly |
| 003-D3 | render to bytes, tokenize separately | render straight to ids | goldens need no tokenizer; enables 003-D4 |
| 003-D4 | control tokens come from the renderer; content encodes with specials off | a denylist over user text | forged turns are structurally impossible |
