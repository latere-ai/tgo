---
title: "GGUF: what it would take, and what accel must register first"
status: blocked
layer: load
depends_on:
  - 000-decisions.md
  - 001-weights.md
blocked_on:
  - "accel specs/010-kernel-corpus.md — `quant_matmul_superblock`, not registered (accel#15, closed as not planned)"
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

accel registers **two** quantized weights ([000 D3](000-decisions.md)): int8
quants with one fp16 scale per block, and int4 at `quant.Int4Group` of 128 with
an f16 scale **and** an f16 zero per group. Neither carries a second level of
scale, and that is the gap: the super-block's two levels over eight sub-blocks
with a minimum each are a different shape, not a smaller one
([011 §2](011-sequencing.md)). accel states the same from its own
side: a native int4 "would not read a published `Q4_K` file"
(`010-kernel-corpus.md`). There is no kernel that reads a super-block, and
[000 D1](000-decisions.md) forbids tgo from writing one.

Three ways around it exist.

- **Dequantize to f16 at load.** Discards the memory saving that is the entire
  reason to read GGUF, and needs a larger resident model than the safetensors
  path.
- **Requantize into accel's int8.** Stacks two lossy steps, and widens a 4-bit
  file into 8-bit storage, so it gives back part of the saving as well.
- **Requantize into accel's int4.** The memory objection does not apply: both
  sides are 4-bit, and group 128 with a zero is the nearest thing accel has to a
  sub-block of 32 with a minimum. It stacks two lossy steps like the int8 path,
  and that is unmeasured.

None of the three changes the status, because none of them is what this spec
specifies. 012 is **reading K-quants natively**: the container, the K-quant
plane layouts, and a GEMM that consumes a super-block. That is what
`quant_matmul_superblock` blocks. Requantizing at load is smaller, separate
work, and §3's second measurement decides whether it is worth doing.

> **"Stacks two lossy steps" is an argument, not a measurement**, and an earlier
> draft overstated it: it said requantizing "gives a model measurably worse than
> either format alone", which nobody had measured. That is the
> assertion-where-a-measurement-belongs that [010 §3](010-conformance.md) exists
> to catch, made here in tgo's own spec. accel's maintainer caught it while
> closing [accel#15](https://github.com/golang-design/accel/issues/15). Corrected
> rather than quietly deleted, and §3 now names the measurement that settles it.

## 3. The trigger, and the two numbers that decide it

accel closed [#15](https://github.com/golang-design/accel/issues/15) as **not
planned** and recorded the gap in its corpus instead: `010-kernel-corpus.md`
carries a `quant_matmul_superblock` row with the layout, the formula, and why
the two workarounds it considers are bad.

**That closure was right and this spec accepts it.** Nothing in tgo is blocked by
it, so it competed against three issues that block work in progress. And a corpus
row is a better record than an issue open with no plan: the corpus is what
someone adding a kernel reads, whereas an issue is what someone opening the
tracker reads.

**Two measurements decide whether this is ever worth building**, and tgo can
produce both where accel cannot:

| measurement | why it decides the case |
| --- | --- |
| **which K-quant formats actually circulate** for the models tgo targets — a count over real checkpoints | the corpus should register the two that matter, not six. That `Q4_K` and `Q6_K` cover most of what circulates is a **guess**: this spec has never counted. Producing the count over real checkpoint listings is the work |
| **int4-at-load against `Q4_K` on real weight blocks**, checked against `quant.Int4ErrorBound` | int4 is the deciding comparand, not int8: `Q4_K` is 4-bit with a per-sub-block minimum, so accel's asymmetric int4 is the near neighbour and carries its own bound ([001 §5.4](001-weights.md)). int8 stays the wider reference point. If the two are within noise on trained weights, **GGUF stops being a quality argument and becomes a download-size argument** — a much weaker case for a new kernel family, and worth knowing before anyone writes one |

Both belong with [010 §3](010-conformance.md)'s numbers, and the second is a
variant of one already there. That row measures int8 against
`quant.Int8ErrorBound` on real blocks. What changes is the width and the other
side: int4 against `quant.Int4ErrorBound`, and `Q4_K` as the comparand, which is
why it needs a block decoder that the existing row does not.

**Reopening happens on a number or on a user, not on a preference.** When a
super-block GEMM lands, this spec becomes `drafted` and the work is: container reader, metadata mapping onto the same
`Config` the registry already uses, the ggml tokenizer variant (GGUF carries its
own vocabulary, not `tokenizer.json`), and the K-quant plane layouts.

The container reader is worth writing before then only if something else needs
it. The second measurement does: **real** weight blocks means one real file, so
it needs the read-only container path — header, metadata, tensor index — and a
`Q4_K` block decoder. No kernel, no accel change, no loader integration. That is
the only part of the reader this spec justifies writing today.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 012-D1 | blocked on an accel super-block kernel | dequantize on load; requantize to int8 | neither workaround looks worth shipping. **Amended:** the requantization half was asserted, not measured; [§3](#3-the-trigger-and-the-two-numbers-that-decide-it) names the measurement that would settle it. **Amended 2026-08-27:** the rejected cell enumerates two workarounds and there are three. Requantizing into accel's int4 was not available when the row was written: it keeps the 4-bit footprint the int8 path gives back, so only the lossy-stacking argument counts against it, and that is still unmeasured. The decision stands because requantizing at load does not read K-quants natively, which is what this spec specifies |
| 012-D2 | accept accel's `not planned` closure; the corpus row is the record | press to keep an issue open | an issue open with no plan is a worse record than a corpus row carrying the reasoning, and the corpus is what a kernel author reads |
