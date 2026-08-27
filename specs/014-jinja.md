---
title: "A Jinja subset: when per-model renderers stop scaling"
status: deferred
layer: text
depends_on:
  - 003-chat-template.md
---

# A Jinja subset

**Status: deferred.** [003](003-chat-template.md) chose per-model Go renderers.
This states what replaces them and when, so the choice is revisitable rather
than permanent by neglect.

## 1. The trigger

Two of: more than roughly six model families; a family whose checkpoints
routinely ship customised templates; or a template feature that a Go renderer
cannot reasonably mirror. Until then a renderer is 60 lines and an interpreter
is a language.

Condition 2 is undetectable today. Nothing reads a checkpoint's
`chat_template`, so 003-D2's warn-on-mismatch has no code path even though
`chat.Checksum` and `Renderer.TemplateChecksum` both exist
(`chat/chat.go:119`, `chat/chat.go:158`). [003](003-chat-template.md) owns that
gap and names it in its Outcome.

## 2. The subset

Measured against `chat/testdata/qwen3_chat_template.jinja`, the only chat
template in the tree. Line numbers below are that file.

- Statements: `{{ }}`, `{% if %}` / `{% elif %}` / `{% else %}` (33, 72),
  `{% for %}`, `{% set %}`, and `{% set ns.field %}` mutating a `namespace()`
  binding (17, 21, 22).
- Expressions: attribute access; index access, including negative and computed
  indices (`messages[0]`, `[-1]`, `messages[loop.index0 - 1]`: 3, 39, 73); the
  reversed slice `messages[::-1]` (18); `+` and `-`; comparison; membership
  `in` (38); `not`, `and`, `or` (20, 31, 44); and the truth of a bare value
  (1, 52, 54).
- The `loop` object: `index0` (19, 43, 73, 79), `first` (31, 54, 73), `last`
  (44, 79).
- Tests: `is defined` (86), `is string` (26, 63), `is false` (86).
- Filters: `tojson` (9, 66) and `length` (17, 19). `tojson` is Hugging Face's,
  not Jinja's, and is already written: `jsonString` in `chat/chat.go:205`
  suppresses Go's `<`, `>` and `&` escaping to match `ensure_ascii=False`.
- String methods, by name: `startswith` and `endswith` (20), `split`, `rstrip`
  and `lstrip` (39, 40), `strip` (45).

Whitespace control (`{%-`, `-%}`) is **not optional**: chat templates use it to
control the exact newlines, and those newlines are the thing that must be
right. Qwen3 opens every tag, statement and expression alike, with `-`.

Not in scope: macros, `include`, `extends`, custom filters, and any method not
named above. A construct enters the subset when a template in the tree calls
it, not before. `default`, `join`, `trim` and `selectattr` were listed here and
are called by nothing.

## 3. The rule that makes it safe

**A template using anything outside the subset fails at load, naming the
construct and the line — never at render, and never by ignoring it.** A silently
skipped `{% if %}` produces a prompt that is wrong in a way no test written
against the subset will catch.

The Jinja template is the oracle, not the Go renderer. The goldens in
[003 §5](003-chat-template.md) are the reference template's own output
(`chat/qwen3_test.go:11`), and the Go renderer was checked against them over 47
cases, byte-identical ([011](011-sequencing.md)). An interpreter is correct
when it reproduces the same goldens, so it and the Go renderer are measured
against the same third thing.

Regenerating a golden means rendering the template the way transformers
configures Jinja2:

```python
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
env.from_string(template).render(messages=..., tools=..., enable_thinking=...,
                                 add_generation_prompt=...)
```

`trim_blocks` and `lstrip_blocks` are the two flags the whitespace rule above
depends on, and the `tojson` override is why tool JSON in the goldens has a
space after every colon and comma and no `<` escaping. The fixture is vendored
byte for byte from a published checkpoint (`chat/testdata/README.md`); editing
it to make a test pass is forbidden.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 014-D1 | a subset, not a full Jinja2 | full compatibility | bounded work; unknown constructs must fail loudly |
| 014-D2 | unknown construct fails at load | ignore or best-effort | a wrong prompt is impossible rather than unlikely |
| 014-D3 | the reference Jinja template is the correctness oracle | the Go renderers are | interpreter and Go renderer are checked against one set of goldens, not each other |
