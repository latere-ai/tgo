---
title: "GGUF: what it would take, and what accel must register first"
status: blocked
layer: load
depends_on:
  - 000-decisions.md
  - 001-weights.md
blocked_on:
  - "accel specs/010-kernel-corpus.md"
---

# GGUF

**Status: blocked.** Written so the cost is known and so the trigger is
explicit.

## 1. Why it is wanted

GGUF is where the pre-quantized checkpoints are. A user with a `Q4_K_M` file has
a 4B model in 2.5 GB, already quantized by someone who measured the result.
Reading safetensors means quantizing at load, every load, from a much larger
download.

The container itself is easy: a header, a key-value metadata block, a tensor
index, aligned data. A day's work, and none of it is the problem.

## 2. Why it is blocked

The problem is the quantization formats. A `Q4_K` super-block is 256 weights as
eight sub-blocks of 32, each with a 6-bit scale and a 6-bit minimum, and two
fp16 super-scales:

$$w_i = d \cdot s_j \cdot q_i - d_\text{min} \cdot m_j, \quad j = \lfloor i/32 \rfloor$$

accel registers **one** quantized weight: int8 quants with one fp16 scale per
32, no minimum, one level of scale ([000 D3](000-decisions.md)). There is no
kernel that reads a super-block, and [000 D1](000-decisions.md) forbids tgo from
writing one.

The two ways around it are both bad. Dequantizing to f16 at load discards the
memory saving that is the entire reason to read GGUF, and needs a larger
resident model than the safetensors path. Requantizing to accel's int8 stacks
two lossy steps and gives a model measurably worse than either format alone.

## 3. The trigger

accel 010 registering a super-block GEMM. When that lands, this spec becomes
`drafted` and the work is: container reader, metadata mapping onto the same
`Config` the registry already uses, the ggml tokenizer variant (GGUF carries its
own vocabulary, not `tokenizer.json`), and the K-quant plane layouts.

The container reader is worth writing before then only if something else needs
it. Nothing does.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 012-D1 | blocked on an accel super-block kernel | dequantize on load; requantize to int8 | neither workaround is worth shipping; the trigger is one upstream change |
