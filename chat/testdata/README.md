# Third-party fixtures

## `qwen3_chat_template.jinja`

The `chat_template` field of **Qwen3's `tokenizer_config.json`**, taken verbatim
from the published checkpoints (for example
[`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)) and licensed by Qwen
under the **Apache License 2.0**, the same licence as this repository.

It is here byte for byte and is not edited. That is the point of it: the Go
renderer in [`chat`](..) is written from
[`specs/003-chat-template.md`](../../specs/003-chat-template.md) rather than
from this file, and the test renders both and compares them. Two independent
implementations of one specification agreeing is evidence; one implementation
compared against itself is not.

Editing it to make a test pass would remove the only thing it is for. If the
renderer and the template disagree, one of them is wrong about what Qwen3
consumes, and the checkpoint decides.
