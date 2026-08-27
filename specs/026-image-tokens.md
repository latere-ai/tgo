---
title: "Image tokens on the text path: a multimodal vocabulary a text-only model must not mis-embed"
status: drafted
layer: text
depends_on:
  - 000-decisions.md
  - 002-tokenizer.md
  - 003-chat-template.md
  - 024-qwen3-5-architecture.md
---

# Image tokens on the text path

[018 §6](018-hybrid-models.md)'s last row is "image-token tolerance — the text
path must not break on a multimodal tokenizer". This spec is that row, and it is
the smallest of [018](018-hybrid-models.md)'s four children.

tgo serves text. The checkpoint does not. The question is not whether tgo can
render a picture; it is what happens to a token that *stands for* one, on a path
that has no vision tower to fill it in.

## 1. What the checkpoint declares

[018 §1](018-hybrid-models.md) reads Qwen3.8-27B's `config.json` and records:

```json
"architectures": ["Qwen3_5ForConditionalGeneration"],
"vocab_size": 248320,
"image_token_id": 248056, "video_token_id": 248057
```

Three facts follow, and each decides one section below.

1. **The two ids are inside the vocabulary.** 248056 and 248057 are less than
   248320, so every range check tgo already runs admits them (§2).
2. **The ids are declared, the spellings are not.** `config.json` names two
   integers. A surface form such as `<|image_pad|>` is a checkpoint convention
   and is not what the head predicts; the integer is. So tgo must read the ids
   and must never look them up by text.
3. **The head is `ForConditionalGeneration`.** The LM head predicts over all
   248320 rows, the two placeholder rows included, so the model can emit one.

## 2. Where a placeholder id can enter a forward pass

Only `Inputs.IDs` reaches `GatherRows` over the embedding table
(`model/qwen3_graph.go:54-60`). Four paths write it, and they are the whole set.

| path | can it carry 248056? | evidence |
| --- | --- | --- |
| user text | **no** | `Encode(s, false)` never matches an added token: `matchAdded` returns early on `allowSpecial` false (`tokenizer/tokenizer.go:390-401`). This is [003-D4](003-chat-template.md), structural |
| a renderer control part | **no** | the Qwen3 renderer emits eight literal spellings and no other (`chat/qwen3.go:14-23`), and `Model.encode` resolves only those (`session.go:443-458`) |
| a caller's raw ids | **yes** | `plan.go:236` and `scheduler.go:254` check `0 <= id < VocabSize` and nothing else. 248056 passes both |
| a token the model drew | **yes** | `stream.go:281` sets `st.feed = tok` for every non-stop token, so the drawn id is the next step's input |

The first two rows are already correct and cost nothing to keep (§3, §4). The
last two are open, and §5 closes them.

## 3. The tokenizer: carry the ids, refuse nothing (026-D1)

`package tokenizer` needs no change, and that is a decision rather than an
absence.

It never reads `config.json`. `tokenizerFile` (`tokenizer/jsonfile.go:16-23`)
has no token-id member at all, so a placeholder id reaches the tokenizer only as
an ordinary `added_tokens` entry. `applyAdded` (`tokenizer/tokenizer.go:280-303`)
already holds the invariants a 248320-entry file needs: no id claimed twice, no
added token colliding with the BPE vocabulary, longest-first matching.
`VocabSize` returns one past the highest id and is documented as the id space
rather than the embedding row count (`tokenizer/tokenizer.go:316-322`), which is
the reading a padded multimodal vocabulary requires.

**Rejected: refuse at load a checkpoint that declares `image_token_id`.** It is
wrong twice. The tokenizer cannot see the declaration, so the refusal would have
to live somewhere else and would then be about the config rather than about the
tokenizer. And [§6](#6-a-text-only-prompt-is-a-coherent-subset-026-d4) makes the
text path *correct* on this checkpoint, so refusing throws away complete
inference to avoid an input that is already refused by name.
[002-D7](002-tokenizer.md) refuses what the tokenizer cannot reproduce exactly;
an added token is reproduced exactly, so there is nothing here it gets wrong.

**One asymmetry to carry into §5.** `TextBytes` returns nil for an added id
(`tokenizer/tokenizer.go:339-344`), but `Decode` and `Decoder.Push` go through
`bytesFor` (`tokenizer/tokenizer.go:429-434`) and **do** emit the added token's
literal content. A placeholder id that reaches the decoder is therefore printed
as its own spelling. That is why §5's guard is upstream of the decoder.

## 4. The chat renderer: refuse the picture, by name (026-D2)

The refusal path exists and is reached from two places.

- **At the wire.** `server/adapt.go:225` is `imageRefusal`, called from
  `mapTurn` on `ir.BlockImage` (`server/adapt.go:210-211`) and from
  `textOfChecked` for an image inside a tool result (`server/adapt.go:243-250`).
  It names the field `image` and gives the reason: "this model is text-only, and
  dropping the image would answer a different question." Tests cover an OpenAI
  and an Anthropic shape at `server/refuse_test.go:52` and `:57`.
- **In `chat`.** `validate` returns `ErrUnknownBlockType` for any block type the
  four constants do not name (`chat/qwen3.go:286-288`), tested at
  `chat/qwen3_test.go:433`.

Both, deliberately. The wire refusal names the dialect member the caller sent,
which `chat` cannot know. The `chat` refusal catches a caller using tgo as a
library with no server in front, which is most of `package tgo`'s API surface.

**Rejected: expand an image part into a run of placeholder ids.** This is what
the reference implementation's template does — it writes the placeholder token
once per image patch and the vision tower then overwrites those embedding rows
before the first layer runs. tgo has no vision tower. The same expansion here
produces exactly §5's failure, with a prompt that renders, tokenizes, and runs.

**Rejected: refuse in one place only.** Either half alone leaves a hole: the
wire alone leaves `Session.Chat` open, and `chat` alone reports an internal block
type to a caller who sent `content[0].type`.

## 5. The graph: a row nothing computed must be unreachable (026-D3)

A text-only forward pass that gathers the embedding row for 248056 gathers a
**placeholder**: a row the reference implementation overwrites with the vision
encoder's output before layer 0. Nothing overwrites it here. The gather succeeds,
the shapes check, every kernel runs, and the model continues fluently from a
vector nothing computed. There is no exception, no NaN and no shape mismatch —
the failure has no symptom. That is the class [C25](010-conformance.md) named at
the operator level, one layer up.

So the requirement is *impossible*, not *unlikely*, and §2's table is what makes
that provable: there are two open paths and each already has a check at the point
the id enters.

### 5.1 The config carries the ids

`model.Config` gains two fields, and `rawConfig` (`model/config.go:95-111`) gains
`image_token_id` and `video_token_id`. Today it parses **no** token-id field, so
this is the first.

Absent means −1, which is `resolveSpecials`' convention for a token this
checkpoint does not have (`model.go:547-557`): −1 equals no id, so a Qwen3-4B
config parses to two −1s and every check below is inert. Present and outside
`[0, vocab_size)` is refused at parse by field name, like every other row of
`ParseConfig`: an id the embedding table has no row for is a config that cannot
be honoured.

One predicate reads them — `Config.IsPlaceholder(id int) bool` — so the two guards
below are one rule stated once.

### 5.2 The input guard

`plan.go:236` and `scheduler.go:254` already refuse an id outside the vocabulary,
in the same sentence shape. A placeholder id is refused beside it, with the
reason rather than the range: this id stands for a picture, tgo computes no
embedding for it, and scoring it would run the model over a row nothing wrote.

It goes there and not at the server because both call sites are public API. A
guard at `server/adapt.go` would leave `Scheduler.Feed` and `Plan.Score`
reachable, and those are the two functions that take ids rather than text.

### 5.3 The output guard

The placeholder ids are masked to −∞ before the draw, beside the grammar mask at
`stream.go:271-277`, and at the point the batched path's logits reach a sampler.

−∞ is the correct mechanism and not a large negative number: `stream.go:268-271`
records that the penalties and the temperature are monotone in the logit with −∞
as a fixed point, so a token masked there cannot be brought back by anything
downstream. The mask runs whether or not a request carries a schema.

**Rejected: filter the id at the decoder instead.** It stops the text and not the
token. `stream.go:281` assigns `st.feed = tok` before `emit` runs, so the id
still becomes the next step's input and still gathers the placeholder row; §3's
asymmetry means the decoder would also have had to special-case an id whose
`bytesFor` is non-nil. A filter at the end of the pipeline cannot fix an input
at the start of the next one.

**Rejected: rely on the row being harmless.** The row is real, it is trained, and
its trained job is to be replaced. Reading it is not reading noise; it is reading
a value whose meaning is "the vision tower writes here".

## 6. A text-only prompt is a coherent subset (026-D4)

It is allowed, and it is correct rather than tolerated.

Every embedding row a text token gathers is a text row the checkpoint trained.
The 64 layers, the two cache kinds and the LM head are the same operators for a
text prompt whether or not the checkpoint can also see; the vision tower is an
*additional* input path to the residual stream, and a prompt that uses none of it
leaves nothing uncomputed. §5's guards are what turn "uses none of it" from a
property of the request into a property of the run.

**What tgo prints.** `renderInfo` (`cmd/tgo/info.go:426-434`) reports the
vocabulary size; beside it, a checkpoint whose config declares either placeholder
id gets one line naming the ids and stating that image and video input are
refused. A user then learns what this checkpoint can do elsewhere *before* they
send an image, rather than from a 400.

**Rejected: refuse the checkpoint.** 64 layers of text inference are complete and
correct without the vision tower. Refusing is the overclaim's mirror image — it
reports "cannot" where the honest answer is "this part, correctly".

**Rejected: say nothing.** The refusal in §4 is then the first time a user hears
that the checkpoint had a capability tgo declines. `docs/orientation.md`'s "What
tgo will not do" is a promise to say so in advance, and a per-checkpoint
capability is exactly the case that needs it.

## 7. The vocabulary size and the sampler (026-D5)

**No arithmetic in [006](006-sampling.md) changes.** `sample.TopMaxRounds = 128`
is accel's kernel round count restated (`sample/sample.go:48-56`) — how many
entries either mask walks — and it is not a function of $V$. `policy.go:138`
refuses `TopK > 128` because that is what the device can reproduce, and it is
equally right at $V = 248320$. The nucleus path already takes
`min(TopMaxRounds, len(logits))` (`sample/stages.go:120`), which is the same
128 candidates out of a larger vocabulary.

**What changes is the readback.** One row of logits is

$$248320 \times 4\ \text{B} = 993\ \text{KB}$$

against the 608 KB [C6](010-conformance.md) quotes for Qwen3's $V = 151936$: a
factor of 1.63, per step, per sequence. `session.go:248` allocates one row and
`batch.go:169-176` allocates $n$ of them, so a batch of 16 reads back 15.9 MB per
step. [017 §4.1](017-benchmarks.md) puts the readback at 807 MB/s off a mapped
buffer.

That does not change a decision here; it changes which [010](010-conformance.md)
row pays. [C3](010-conformance.md) closed with `tensor.Sample` composing the whole
policy on the device and returning a token id, so the readback is avoidable
already. For this checkpoint it stops being a nicety and becomes the largest
fixed cost of a decode step, and [017](017-benchmarks.md) is where that is
measured rather than asserted.

**Rejected: scale `TopMaxRounds` with the vocabulary.** It reads as the obvious
adjustment and it names a $k$ the device cannot walk, which is the refusal at
`policy.go:138` turned into silence.

## 8. Tests

| test | what it asserts |
| --- | --- |
| `TestConfigPlaceholderIDs` | `image_token_id` and `video_token_id` parse; absent yields −1; an id at or past `vocab_size` is refused by field name |
| `TestConfigNoPlaceholderIDs` | a Qwen3-4B config parses to two −1s and `IsPlaceholder` is false for every id in range |
| `TestImagePartRefusedByName` | an OpenAI `image_url` part and an Anthropic `image` block each return `invalid_request_error` naming the field `image`, and the message carries the reason ("text-only", "dropping the image would answer a different question"), not just a code. Extends `server/refuse_test.go:52` |
| `TestChatRefusesImageBlock` | `chat.Render` on a block type outside the four constants returns `ErrUnknownBlockType`, so a library caller with no server hits the same wall |
| `TestTextOnlyRoundTripOnMultimodalFixture` | a fixture tokenizer with the multimodal ids and $V = 248320$ encodes a plain user prompt, renders it, and encodes it to **exactly** the ids the same prompt produces on the text-only fixture; no placeholder id appears in the output |
| `TestPlaceholderIDRefusedAsInput` | `Scheduler.Feed` and `Plan.Score` refuse 248056 by id, with the placeholder reason and not the range message, on a config that declares it |
| `TestPlaceholderIDMasked` | with a logits row whose maximum is at 248056, the sampler draws something else at every temperature and with a grammar absent; the mask is −∞ and survives the penalties |
| `TestPlaceholderNeverFedBack` | a stream whose first step would draw 248056 never assigns it to `st.feed`, so the next step's `Inputs.IDs` holds no placeholder |
| `TestInfoNamesPlaceholderIDs` | `tgo info` on a multimodal config prints the ids and the refusal; on a text-only config it prints neither line |

The round-trip test is the one that would decay first. It must compare against a
**text-only fixture's ids for the same prompt**, not against a golden recorded
from the multimodal fixture: a golden recorded from the same file cannot tell a
correct encoding from a consistently wrong one.

## 9. What this spec does not own

- **Any image path.** No decoding, no patching, no vision encoder, no
  `<|vision_start|>` handling, no multimodal RoPE. There is no image path here,
  and that is the whole point of the spec: the placeholder is closed, not filled.
- **The `qwen3_5` registry entry, the layer-type schedule and the weight map.**
  [024](024-qwen3-5-architecture.md).
- **The recurrent state and its cache kind.** [018 §6](018-hybrid-models.md).
- **Whether the readback is worth moving to the device.**
  [006](006-sampling.md) decides, [017](017-benchmarks.md) measures; §7 only
  records the number that makes it urgent here.
- **`tokenizer.json` for this family.** §3 concludes the loader needs no change;
  if a real Qwen3.8 file proves otherwise, that is [002](002-tokenizer.md)'s
  amendment and not this spec's.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 026-D1 | the tokenizer carries the placeholder ids as ordinary added tokens; no change to `package tokenizer` | refuse at load a checkpoint that declares `image_token_id` | the tokenizer cannot see `config.json` and reproduces an added token exactly, so there is nothing here it gets wrong; and refusal would discard the complete text inference [026-D4](#decision-record) establishes |
| 026-D2 | an image part is refused by name at the wire **and** in `chat` | expand it into a run of placeholder ids, as the reference template does; or refuse in one place only | the expansion is [026-D3](#decision-record)'s failure with a prompt that renders and runs. One place alone leaves either `Session.Chat` or the dialect field name uncovered |
| 026-D3 | the placeholder ids are refused where an id enters (`plan.go:236`, `scheduler.go:254`) and masked to −∞ where a token is drawn (`stream.go:271`, and the batched path's sampler) | filter the id at the decoder; or rely on the placeholder row being harmless | `stream.go:281` feeds the drawn id back before the decoder sees it, so a decode-time filter stops the text and not the token. The row is overwritten by the vision encoder upstream and by nothing here, so gathering it is reading a value this run never computed |
| 026-D4 | a text-only prompt on a multimodal checkpoint runs, and `tgo info` names what is declined | refuse the checkpoint; or run it and say nothing | 64 layers of text inference are correct without the vision tower. Silence makes §4's refusal the first notice a user gets, which is what `docs/orientation.md` promises not to do |
| 026-D5 | the sampler is unchanged; only the readback grows | scale `TopMaxRounds` with the vocabulary | 128 is accel's round count and not a function of $V$, so scaling it names a $k$ the device cannot walk. The 993 KB row makes [C3](010-conformance.md)'s on-device sampling a throughput item for this checkpoint rather than a nicety |
