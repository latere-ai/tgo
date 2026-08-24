---
title: "Tokenizer: byte-level BPE, and why the merge order is the whole algorithm"
status: drafted
layer: text
depends_on:
  - 000-decisions.md
---

# Tokenizer

Pure Go, no device, no network. This is the package where the coverage gate is
easiest to reach and where a bug is hardest to see, because a wrong tokenizer
produces fluent text that answers a different question.

## 1. What Qwen3 uses

`tokenizer.json`, the Hugging Face `tokenizers` serialisation, with:

- a **byte-level** pre-tokenizer: the input is split by a regex, then each byte
  is mapped into a printable Unicode range so that every byte sequence is
  representable and no token ever splits a byte;
- a **BPE** model: a vocabulary (token to id) and an ordered merge list;
- **added tokens**, the specials — `<|im_start|>`, `<|im_end|>`,
  `<|endoftext|>`, and Qwen3's `<think>` / `</think>` — which are matched before
  anything else and never merged into.

## 2. The algorithm, and the part that is easy to get subtly wrong

Encoding one pre-token:

1. Map each byte through the byte-level alphabet, giving a sequence of symbols.
2. Repeatedly find the adjacent pair with the **lowest merge rank** and merge
   it. Stop when no adjacent pair is in the merge table.
3. Look up each resulting symbol in the vocabulary.

Step 2 is the whole algorithm. The rank is the pair's **index in the merge
list**, and it is a global ordering, not a local preference:

$$\text{merge } \arg\min_{i}\ \operatorname{rank}(s_i, s_{i+1})$$

**Rejected:** greedy left-to-right longest-match against the vocabulary. It is
faster, it produces valid tokens, and it produces *different* tokens — which
means a prompt tokenized by tgo and the same prompt tokenized by the reference
put the model in different states. There is no partial credit here.

The naive implementation is $O(n^2)$ per pre-token; a linked list with a heap of
candidate pairs is $O(n \log n)$. Pre-tokens are short, so v0 is naive and
correct, and the heap is an optimisation with a benchmark behind it.

## 3. Specials are matched first, and never merged into

`<|im_start|>` must produce exactly one token. If it reached the BPE it would be
split into its bytes and merged into something else, and the model would never
see the turn boundary. So the encoder splits the input on the added-token set
first, emits those ids directly, and BPEs only the gaps.

This also means **user text containing `<|im_start|>` is a prompt injection
surface.** [003 §4](003-chat-template.md) owns that; the tokenizer's job is to
report faithfully what it was given.

## 4. Decoding, and the incremental case

Decoding is a table lookup and the inverse byte-level map. The hard part is
**streaming**: a single token is often an incomplete UTF-8 sequence, and
emitting it as a string produces `U+FFFD`. So the decoder is stateful — it holds
back a trailing partial sequence and emits it when the next token completes it.

This is not an edge case. It is every CJK character, every emoji, and most
accented Latin, in nearly every response.

## 5. What is tested, and against what

- **Round trip.** `decode(encode(s)) == s` for a corpus that includes CJK, emoji
  with modifiers, combining marks, lone surrogate-shaped bytes, and the empty
  string.
- **Fixed vectors.** Known strings against known id sequences, taken from the
  reference tokenizer and **checked in as testdata**, because they are the only
  thing that catches a merge-order bug. A round trip passes with a wrong merge
  order.
- **Fuzz.** `encode` never panics and never emits an out-of-range id, on
  arbitrary bytes, including invalid UTF-8.
- **Streaming equals batch.** Feeding tokens one at a time to the incremental
  decoder produces the same string as decoding them together.

All of this runs on a small checked-in tokenizer, so none of it needs a model.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 002-D1 | true BPE by global merge rank | greedy longest-match against the vocabulary | ids match the reference; there is no partial credit here |
| 002-D2 | naive $O(n^2)$ merge in v0 | a heap from the start | pre-tokens are short; the heap needs a benchmark behind it |
| 002-D3 | added tokens matched before BPE, never merged into | let specials reach the BPE | turn boundaries survive; see [003-D4](003-chat-template.md) |
| 002-D4 | a stateful streaming decoder holding back partial UTF-8 | decode each token independently | every CJK character and emoji would otherwise emit U+FFFD |
| 002-D5 | fixed vectors checked in as testdata | round-trip tests only | a round trip passes with a wrong merge order; the vectors do not |
