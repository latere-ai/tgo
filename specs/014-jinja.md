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

## 2. The subset

What chat templates actually use: `{{ }}`, `{% if %}`, `{% for %}`, `{% set %}`,
attribute and index access, `is defined`, `is none`, the `+` and comparison
operators, and the filters `default`, `join`, `trim`, `tojson`, `length`,
`selectattr`. Whitespace control (`{%-`, `-%}`) is **not optional**: chat
templates use it to control the exact newlines, and those newlines are the
thing that must be right.

Not in scope: macros, `include`, `extends`, custom filters, arbitrary Python
expressions.

## 3. The rule that makes it safe

**A template using anything outside the subset fails at load, naming the
construct and the line — never at render, and never by ignoring it.** A silently
skipped `{% if %}` produces a prompt that is wrong in a way no test written
against the subset will catch.

The per-model renderers stay as the reference: the interpreter is correct when
it produces byte-identical output to the Go renderer on every golden in
[003 §5](003-chat-template.md).

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 014-D1 | a subset, not a full Jinja2 | full compatibility | bounded work; unknown constructs must fail loudly |
| 014-D2 | unknown construct fails at load | ignore or best-effort | a wrong prompt is impossible rather than unlikely |
| 014-D3 | Go renderers remain the correctness oracle | replace them | the interpreter has a ground truth to be checked against |
