---
title: "Prefix caching: reusing the KV of a prompt somebody already paid for"
status: drafted
layer: engine
depends_on:
  - 000-decisions.md
  - 003-chat-template.md
  - 005-kv-cache.md
  - 007-engine.md
---

# Prefix caching

**This is tgo's job, not accel's**, and it is the largest single win available
to the framework. [005 §5](005-kv-cache.md) and [008 §6](008-scheduler.md) both
deferred it on the grounds that it depends on paging. Paging landed on
2026-08-24, so this spec exists and those two are corrected.

## 1. The observation

A prefill costs $O(T)$ forward passes' worth of arithmetic, and in an agent or
chat workload most of $T$ is **the same every time**. A system prompt with tool
definitions is 500–2000 tokens and is byte-identical across every request in a
session. A multi-turn conversation re-sends its entire history on every turn, so
turn $n$ re-prefills everything turns $1..n-1$ already prefilled.

$$\text{work saved} = \frac{|\text{shared prefix}|}{T_\text{prompt}}$$

For a 900-token system prompt and a 30-token question that is **97% of the
prefill**, every request after the first. For turn $n$ of a conversation it
approaches $1 - 1/n$.

**Nothing about this is an approximation.** The KV for position $t$ is a
function of tokens $0..t$ and the weights alone. Two requests with the same
token prefix have the same KV over that prefix, so reusing it is not a cache in
the lossy sense — it is not recomputing a pure function. §6 states the one place
that "same" needs care.

## 2. Where the key comes from, and why it is token ids

The cache is keyed on **token ids, not text**.

Text is the wrong key in both directions. Two different strings can tokenize to
the same ids (they cannot, for a bijective byte-level BPE — but two *renderings*
differing in whitespace the template normalises can), and more importantly one
string can be rendered into different ids by different template options. What
the model consumed is the id sequence, so that is what identifies the state.

```mermaid
flowchart LR
  M["messages + options"] --> R["003 renderer<br/>Prompt parts"]
  R --> T["002 tokenizer"]
  T --> I["token ids"]
  I --> K["the cache key"]
```

This falls out of [003-D3](003-chat-template.md) — render to parts, tokenize
separately — for free. The key is taken *after* tokenization, so it cannot
disagree with what the model was fed.

### 2.1 What the chat template guarantees

A generic radix trie **discovers** that requests share a prefix. tgo **knows**
it, structurally: [003 §3](003-chat-template.md) renders the system turn first
and never injects a default one, so two requests with the same system message
and tools produce the same leading ids by construction.

That is worth stating because it decides the block size in §3: the shared prefix
has a natural boundary at the end of the system turn, and a block size that
makes that boundary reachable matters more than one tuned for a synthetic
benchmark.

## 3. Blocks are the unit, so the key is block-aligned

A page table maps logical blocks to physical ones, and a block is the smallest
thing that can be shared. So the cache matches **whole blocks**:

$$\text{shared blocks} = \left\lfloor \frac{\text{common prefix length}}{B} \right\rfloor$$

and up to $B-1$ tokens of a genuine common prefix are re-prefilled. With
$B = 32$ that is at most 31 tokens against a prefix of hundreds — a rounding
error against the win, and the alternative (sharing partial blocks) would mean
two sequences writing into one block at different offsets, which is a
correctness problem rather than an optimisation.

### 3.1 A full hit must still prefill one token

**The reusable prefix is capped at $T-1$, never $T$.**

$$\text{reused} = \min\!\big(\text{hit}, \; T - 1\big)$$

The cache holds **KV, not logits**. Sampling the next token needs the logits at
the last prompt position, and those come from a forward pass over it
([004 §3](004-model-graph.md) rows 26–28 slice the last hidden state and run the
LM head). Reuse everything and there is no hidden state to slice — the request
has a warm cache and nothing to sample from.

This is not a rare path. An exact resubmission is a retry, an agent loop
re-sending an unchanged prompt, or `/v1/completions` called twice. In *chat* it
hides, because [003 §3](003-chat-template.md)'s rendered prompt always ends with
a fresh `<|im_start|>assistant\n`, so the suffix is naturally non-empty and the
bug never fires in the obvious test.

Taken from ollama, which does exactly this and says why:

```go
// Always keep at least one token to re-evaluate so the
// pipeline can seed token generation from it.
if matched == len(inputs) && matched > 0 {
    matchPath, matched = findBestMatch(c.root, keys[:matched-1])
}
```

§8 tests it with an identical prompt submitted twice, which is the case the
chat path cannot produce.

Each block gets a hash chained over its predecessor, so a block's identity
includes everything before it:

$$h_0 = H(\text{ids}[0{:}B]), \qquad h_i = H(h_{i-1} \parallel \text{ids}[iB{:}(i{+}1)B])$$

**The chain is what makes the hash a prefix key rather than a content key.** Two
different prompts that happen to contain the same 32 tokens in the middle have
different $h_i$ there, because their $h_{i-1}$ differ. Without the chain they
would collide and the second request would silently attend to the first's
context.

### 3.1 The hash is a security boundary, so $H$ is not free choice

A hash collision here does not corrupt data — it hands one request **another
request's KV**, and the output stays fluent. So $H$ must be chosen adversarially,
not for speed:

- **$H$ is SHA-256**, and the chain input includes the salt of §7.
- A fast non-cryptographic hash is permitted **only with a per-process random
  seed**, because a predictable one lets an attacker compute a colliding block
  offline and then submit a prompt that reads somebody else's cache. vLLM hit
  exactly this ([vllm#12621](https://github.com/vllm-project/vllm/issues/12621))
  and now keeps a per-process seed for non-cryptographic algorithms while
  deriving a fixed one for SHA-256, so that separate processes can share a cache
  without weakening collision resistance.

tgo takes the same split and the same reasoning. The earlier draft of this spec
wrote $H$ and said nothing about it, which is how the property gets lost.

## 4. The structure: a hash map, not a trie

vLLM hashes chained blocks into a map; sglang keeps a radix trie of token
sequences. Both work. tgo takes the map.

| | hash map | radix trie |
| --- | --- | --- |
| lookup | $O(\text{blocks})$ map probes | $O(\text{blocks})$ node walks |
| memory | one entry per cached block | nodes plus edges |
| partial-prefix match | falls out: walk until a probe misses | falls out |
| **eviction** | LRU over unreferenced entries; each is independent | evicting an interior node orphans its children |
| implementation | a map and a refcount | a tree with splitting and merging on insert |

The deciding row is eviction. A trie's nodes are entangled — freeing one must
consider its descendants — while chained hashes make each block independently
evictable, and correctness does not depend on getting the tree surgery right.
The trie's advantage is enumerating what shares a given prefix, which is a
scheduling question ([008](008-scheduler.md)) rather than a caching one.

**Rejected: a trie, for now.** If [008](008-scheduler.md) later wants
"which waiting requests share a prefix with this one", the trie earns its
complexity. Nothing today asks that.

## 5. Lifetime: refcounts, then LRU

```mermaid
flowchart LR
  A["block allocated<br/>refcount 1"] -->|another sequence hits| B["refcount 2"]
  B -->|a sequence finishes| A
  A -->|last sequence finishes| C["refcount 0<br/>cached, evictable"]
  C -->|a new request hits it| A
  C -->|pool pressure, LRU| D["freed<br/>hash entry removed"]
```

A block at refcount 0 is **not freed** — it is the cache. It is freed only under
pool pressure, least-recently-used first, and its hash entry is removed in the
same step. The invariant that must not break: **a hash entry never outlives the
block it names**, or a later request maps a hash to a physical block that now
holds another sequence's KV, and attends to somebody else's context.

That is the single most dangerous bug in this design, and it is silent: the
output stays fluent. §8 tests it directly rather than by exercising the paths
that would happen to produce it.

## 6. Correctness: two subtleties, one of which is real

**A reused prefix was computed under a different prefill shape.** A fresh
request prefills $T$ tokens in one dispatch; a hit prefills only the suffix. The
KV for the shared prefix was produced by a differently-shaped GEMM, and floating
point is not associative, so the bits can differ from what this request would
have computed alone.

This is the same class as the CPU/Metal divergence [006 §4.1](006-sampling.md)
already refuses to bound and instead measures. It is handled the same way:
[010 §3](010-conformance.md) gains a measurement — greedy, same prompt, cold
against warm, the first differing token index — rather than an assertion nobody
verified.

> It is worth saying plainly that this makes a cache hit **observable**, and
> therefore that "prefix caching is transparent" is not quite true. It is
> transparent in distribution and not bit-for-bit.

**Sampling reproducibility is unaffected.** A prefill consumes no draws
([006-D2](006-sampling.md) draws once per *step*), so a seeded stream produces
the same completion whether or not the prompt was cached. That is worth a test,
because it is the property a user checks and it would be easy to break by
threading the cache through the sampler.

## 7. Isolation: sharing KV across tenants is a side channel

**A cache hit is faster than a miss, and that timing is observable.** If tgo
shares blocks across users, a user can learn whether *somebody else* has
recently submitted a given prefix, by measuring their own first-token latency
against a prompt they construct. That is a membership oracle over other users'
prompts.

This is not hypothetical and it is not specific to tgo; it is inherent to
cross-request KV reuse, and most published inference stacks share by default.

### 7.1 Two mechanisms, and tgo takes both

vLLM and sglang both solve this with a **caller-supplied `cache_salt`**: an
opaque string mixed into the first block's hash, which the chain then propagates
to every block after it. Blocks match only within the same salt.

That is more expressive than a server-side scope and it is the right primitive
for the layer that knows who the caller is — a gateway can salt by tenant id,
which tgo cannot do because [009 §7](009-server.md) says tgo has no notion of a
tenant. But it **fails open**: a caller who sets no salt shares globally, so the
default is the unsafe one.

So tgo takes both, and they compose:

- **`cache_salt`** on the request, mixed into $h_0$ exactly as vLLM does. The
  layer with tenant identity supplies it.
- **scope** on the server, which bounds what a *missing* salt can reach.

The scope is what makes the default safe; the salt is what makes it precise.
Neither alone is enough: a scope cannot express "these two sessions are the same
customer", and a salt cannot protect a caller who forgot it.

**The decision: the cache is scoped, the default scope is the process, and a
request may narrow further with a salt.** [009 §7](009-server.md) says tgo serves one model with no
authentication and no tenancy — so within one tgo process there is no tenant
boundary to cross, and sharing is correct. The moment something in front of it
multiplexes users, the operator must scope the cache, and tgo makes that
possible rather than deciding it:

| scope | when |
| --- | --- |
| `process` (default) | single-tenant: a CLI, one team's server, an agent runtime |
| `session` | share within a conversation only; safe under multi-tenancy. Reachable from the server since [019](019-session-affinity.md) pooled the sessions; `tgo serve --prefix-cache` |
| `off` | measurement, and comparison against a cold baseline |

**`session` is the important row.** It is the scope that keeps the largest share
of the benefit with no cross-user leak at all, so an operator who cannot reason
about the risk has a safe setting that is not "off".

The documentation states the trade rather than burying it. An inference
framework that silently makes one user's prompts detectable by another has made
a security decision on the operator's behalf.

### 7.2 What shipped, and why the server gets none of it

Shipped 2026-08-26: `WithPrefixCache(CacheSession, n)` reuses a session's own
prefix. The default is `off`, not the `process` this section chose, because
`process` cannot be honoured at all: it needs the page-table port
[§9](#9-what-accel-gives-verified-by-value) records as missing.

**The larger correction is that `session` is unreachable from `tgo serve`, and
the table above says the opposite.** `server/generate.go` opens one session per
request and closes it on the way out — that is what returns the KV reservation
[§6](#6-correctness-two-subtleties-one-of-which-is-real) accounts for. A session
that is destroyed at the end of the request it was made for never sees a second
turn, so there is no own-prefix to reuse. The $1 - 1/n$ win this table credits
to `session` is a win only for a caller who holds a `Session` across
generations, which today means an embedded caller and not the server.

So: **no scope reuses anything from `tgo serve`.** The feature is real and
tested at the library surface, and the product does not reach it.

**Closed by [019](019-session-affinity.md), 2026-08-26.** `tgo serve` keeps a
pool of sessions and routes a request to the one already holding the longest
matching prefix, so a session sees a second turn and `session` reuses what it
holds. `--prefix-cache` turns it on; `--sessions N` is the pool. 019 §8.1 has
the measurement.

**And the refusal `CacheProcess` returns names the wrong obstruction.** It names
the missing page-table port, which is the blocker for *block-level* sharing —
arbitrary requests sharing arbitrary blocks, which is what
[§4](#4-the-structure-a-hash-map-not-a-trie)'s pool and `internal/prefix` are
for. It is **not** the blocker for cross-request reuse as such: a server that
kept a pool of sessions and routed a request to the session already holding the
longest matching prefix would get cross-request reuse with no page table and no
accel change, because `Session.reusable` is a token comparison against that
session's own history and each pooled session keeps its own contiguous,
single-owner cache. That design trades this section's isolation argument for a
different one — the affinity pool, not the block, becomes the thing a scope has
to bound — and it changes when the KV reservation is released, so it is a
decision to be made rather than a gap to be filled.

## 8. Tests

Every one is host-side logic — the trie, the refcounts, the eviction — plus a
small device test for the reuse itself. No weights.

**The rows below are the POOLED path, which has no code yet.** Everything that
mentions a block, a refcount, an eviction or a lease is waiting on the port in
[§9](#9-what-accel-gives-verified-by-value); a table that reads as a test plan
for shipped work would be the third false green in this document. What the
session-local path shipped with is the first row, the last row, the seeded row,
and one more that is not here: **a request refused after the rewind leaves the
session's history intact**, which is what stops a rejected request from
silently truncating the conversation it was rejected from.

| test | what it catches |
| --- | --- |
| a warm request produces the same tokens as a cold one (greedy) | the whole point |
| hit length is block-aligned and never exceeds the true common prefix | §3 |
| **the chained hash**: two prompts sharing an interior run do **not** share a block, asserted **at `Publish`** | §3's chain. The collision does its damage at publish, not at acquire: a match loop stops at the first miss, so an interior block is never looked up, and the second prompt instead *adopts* the first's physical block when it publishes. A test written against "they do not share" without publishing passes under an unchained hash |
| **a freed block's hash entry is gone**: force eviction, then request the evicted prefix, and assert a miss rather than a hit | §5's invariant, tested directly |
| refcount: a block shared by two sequences survives one of them finishing | §5 |
| eviction never frees a block at refcount > 0 | §5 |
| a seeded completion is identical cold and warm | §6 |
| `session` scope: two sessions with the same prefix do not share | §7 |
| a partial hit's attention **output** matches the host oracle, not merely its `base` value | §9 — asserting the base is what let C13 pass |
| concurrent identical-prefix inserts keep one block, under `-race` | §10.4 |
| **the identical prompt submitted twice** returns the same completion, and the second prefills exactly one token | §3.1 — the case the chat path cannot produce |

## 9. What accel gives, verified by value

Re-audited at accel HEAD by asserting outputs, not by reading refusals —
[010-D7](010-conformance.md), a rule this section is the reason for.

**Everything this spec needs is now reachable.** A cache of any capacity
([C11](010-conformance.md)), a paged decode ([C4](010-conformance.md)), per-row
positions ([C2](010-conformance.md)), `BaseName` to prefill a suffix at a
non-zero base, and — as of 2026-08-24 — a **paged prefill**
([C13](010-conformance.md)):

```
selection: Attention -- the paged causal prefill kernel: blocks of 32
           addressed through a page table, one workgroup per query position
reversing the page table moves the output, max diff 0.6057
```

The value test is the point. **For one day this section said the opposite, on
the strength of a probe that only checked the graph compiled.** `Attention`
accepted `Pages` on a prefill, dropped it, and read the cache contiguously;
tgo's probe recorded it as working and this spec claimed cross-request sharing
was expressible when it was not. tgo filed it as
[accel#10](https://github.com/golang-design/accel/issues/10), accel shipped the
kernel, and the same test now shows the table being honoured.

That episode is why [010-D7](010-conformance.md) exists, and it is left visible
rather than tidied away: **the spec was confidently wrong because its evidence
was the absence of an error.**

Three constraints remain, and the third is tgo's own.

- **tgo's graph declared no page-table port, and does now.** This section
  audits accel's kernels; for a week it did not audit the graph that would have
  to call them, and [004 §3](004-model-graph.md)'s port table had no page table
  in it while `nn.Attention` bound no `tensor.AttentionOptions.Pages` or
  `Block`. Nothing in tgo could pass a page table however capable C13 was.
  **That was the same defect as the one this section records below, one layer
  in**: the evidence was accel's behaviour and the question was about tgo's.

  `PortPages` landed on 2026-08-26, verified by a prefill over a *permuted*
  table required to agree bit for bit with a contiguous run, with a negative
  control that writes the key/value state contiguously and reads it through the
  permutation and requires the two to disagree. `CacheProcess` is available and
  §4's pool has an importer.
- **the block pool is tgo's.** `tensor/internal/pagetable` is unexported, and
  that is right: accel 030 declines to evict because choosing a victim is
  policy, and [§5](#5-lifetime-refcounts-then-lru) is that policy.
- **f16 is available on every path except the ragged one.**
  [C5](010-conformance.md) closed, so a paged prefill and a paged decode both
  read an f16 cache: half the memory, twice the blocks, twice the prefixes worth
  keeping. [C22](010-conformance.md) is the exception, and it decides the pool's
  dtype rather than being a detail of it — the ragged kernel
  [008](008-scheduler.md) batches with reads **f32**, and one pool cannot be
  both. So a build that batches runs an f32 pool and holds half the prefixes,
  and a build that does not keeps f16. Filed as
  [accel#23](https://github.com/golang-design/accel/issues/23); until it closes
  this is a configuration rather than a default.

## 10. Against vLLM and sglang

Read from their source rather than from their papers, at the versions checked
out on 2026-08-24.

| | vLLM | sglang | ollama | tgo |
| --- | --- | --- | --- | --- |
| structure | chained block hashes → map | radix trie | compressed prefix trie | chained block hashes → map |
| concurrent sharing | many sequences attend shared blocks | same | **one active path**; others live as snapshots | many |
| reclaim | LRU over unreferenced blocks | pluggable (LRU, LFU) over evictable leaves | **page out to host**, 8 GiB threshold | LRU over unreferenced blocks |
| isolation | `cache_salt` | `cache_salt` | — (single user) | **salt *and* server scope** |
| full-hit rule | — | — | **re-evaluate one token** | re-evaluate one token ([§3.1](#31-a-full-hit-must-still-prefill-one-token)) |
| non-sliceable state | — | — | **whole-state layers handled** | not yet ([§10.1](#101-what-tgo-does-not-have)) |
| batch determinism | opt-in `VLLM_BATCH_INVARIANT`, beta, SM 8.0+ | — | — | measured, not eliminated |
| preemption | recompute | retract (recompute) | page out | recompute |

**Where tgo agrees, it agrees for the same reasons**, and that is worth saying:
chaining, block alignment, refcounting and recompute-over-swap are not
independent inventions here. Two mature systems converged on them, and a design
that differed would need an argument this one does not have.

**Where it differs, three times:**

1. **Map over trie.** sglang's trie earns its complexity by answering "what
   shares this prefix", which a scheduler wants. Its cost is visible in its own
   eviction: the heap must re-examine a parent when its last child is freed
   (`if len(x.parent.children) == 0 and x.parent.lock_ref == 0`), because
   interior nodes are entangled. vLLM chose the map and tgo follows.
   [016-D3](#decision-record) records the trie as the answer if
   [008](008-scheduler.md) ever needs the query.

2. **Isolation defaults.** Both of them make isolation a caller's job and share
   globally when the caller says nothing. tgo adds a server scope *underneath*
   the salt, so the unsafe case requires a decision rather than an omission
   ([§7.1](#71-two-mechanisms-and-tgo-takes-both)). This is the one place tgo
   thinks the prior art has the default the wrong way round.

3. **Batch-shape determinism.** vLLM built a batch-invariant mode to *eliminate*
   the divergence a cache hit introduces. tgo **measures** it instead
   ([§6](#6-correctness-two-subtleties-one-of-which-is-real)). Not because
   measuring is better — invariance is strictly more useful — but because
   vLLM's mode is beta, hardware-gated, and costs performance, and tgo has no
   basis for a bound it has not taken. If accel later offers reduction-order
   guarantees, this becomes a real option rather than an aspiration.

### 10.1 ollama is the closest comparison and the least similar design

It is Go, single-binary and cgo-averse, so its constraints are tgo's. Its cache
is not.

**One active path.** Only one root-to-leaf path is backed by live cache arrays;
switching to another pages the new one in from snapshots. vLLM, sglang and tgo
instead let many sequences attend shared blocks concurrently. ollama's shape
follows from its workload — one user, one conversation at a time — and it buys
something the block designs cannot have: a cached branch costs *host* memory
rather than device memory, so the cache can far exceed the KV pool.

**It pages out rather than recomputing.** `maxPagedOutBytes = 8 GiB` of
snapshots before eviction. That is the opposite of
[008-D2](008-scheduler.md), and it is right *for a desktop*: host RAM is
plentiful, a memcpy is cheap against a prefill, and there is no second request
whose latency the transfer would hurt. Under concurrency the calculus inverts,
which is why vLLM and sglang both recompute. **The difference is workload, not
correctness**, and tgo serving concurrent requests puts it on their side.

**It handles state that cannot be sliced.** Its trie distinguishes sliceable KV
layers, which span a node's edge exactly, from *whole-state* layers — recurrent
and rotating (sliding-window) — which keep entries only at node ends. tgo's
design assumes every layer is sliceable KV, which is true for Qwen3 dense and
**false for the hybrid-attention successors** ([011 §3](011-sequencing.md)
lists them as out of scope). A recurrent state has no meaning at an arbitrary
position, so it cannot be resumed mid-edge.

That is worth recording now: [004-D2](004-model-graph.md) says a new
architecture is additive at the registry, and for a hybrid model **that is not
true of the cache**. ollama had to generalise its trie; tgo would too.

### 10.2 Dialect translation: three positions

ollama also answers [009](009-server.md)'s question, differently:
`FromChatRequest` and `FromMessagesRequest` both convert into **`api.ChatRequest`
— ollama's own public API**. Hub-and-spoke, like llmdialect, but the hub is the
product's native surface rather than a neutral IR.

| | hub | cost |
| --- | --- | --- |
| ollama | its own public API | every dialect feature must be expressible in ollama's API, so the API accretes other people's fields |
| llmdialect / tgo | a neutral IR, separate from both | one more type to maintain; the engine API stays free |

That accretion is precisely what [009-D1](009-server.md) rejects, and ollama is
the evidence that it happens rather than a hypothetical.

### 10.3 What tgo does not have

Multimodal and LoRA identity in the hash key. vLLM mixes image hashes and the LoRA id into `extra_keys` for the
obvious reason — the same tokens under a different adapter are different KV.
tgo has neither feature, and [004-D2](004-model-graph.md) makes both additive.
**Recorded here so that whoever adds one remembers this key exists**, because
forgetting it is silent: an adapter's KV would be served to a request that did
not ask for it.

### 10.4 Concurrency, which neither reference will warn you about

[§7](#7-isolation-sharing-kv-across-tenants-is-a-side-channel)'s default scope is
the **process**, so every session's goroutine reaches one pool — while
[007-D1](007-engine.md) deliberately leaves a `Session` unlocked. The pool is
therefore internally locked, and **lookup → allocate → insert is atomic**: two
concurrent misses on the same prefix must keep one block and drop the other's
refcount, or the loser leaks a block and the winner's refcount is short by one.

vLLM and sglang give no guidance here because both schedulers are
single-threaded loops; the question does not arise for them. It arises for tgo
because [007](007-engine.md) serves sessions from independent goroutines. A
`-race` test over concurrent identical-prefix inserts is in §8.

## 11. What this is not

**Not Anthropic's `cache_control`.** That is an explicit, caller-declared
breakpoint. tgo's caching is automatic and needs no annotation, so
[009 §4](009-server.md) keeps `cache_control` in the advisory-loss category:
the request runs, the caching happens anyway, and the field is reported as
unhonoured because tgo did not do what the caller literally asked — it did
something that makes the request faster regardless.

**Not a response cache.** Identical prompts still generate; only the prefill is
skipped. A response cache would change what the model does, and
[006 §4](006-sampling.md)'s seeded stream is the mechanism for a caller who
wants determinism.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 016-D1 | key on token ids, taken after tokenization | key on rendered text, or on messages | the key cannot disagree with what the model was fed; falls out of [003-D3](003-chat-template.md) |
| 016-D2 | chained block hashes | hash each block's contents alone | an unchained hash collides on a shared interior run and silently attends to another prompt's context |
| 016-D3 | a hash map with refcounts and LRU | a radix trie | chained hashes are independently evictable; a trie's interior nodes are entangled. Revisit if [008](008-scheduler.md) needs prefix-sharing queries |
| 016-D4 | block-aligned sharing only | share partial blocks | two sequences writing one block at different offsets is a correctness problem; the cost is ≤ 31 tokens |
| 016-D5 | a refcount-0 block is cached, not freed; freed LRU under pressure, hash entry removed in the same step | free at refcount 0 | the cache *is* the retained blocks. The paired removal is the invariant §8 tests directly |
| 016-D6 | measure cold-vs-warm divergence; do not claim bit-exactness | assert transparency | a reused prefix was computed under a different prefill shape, and floating point is not associative |
| 016-D7 | **both** a server-side scope (default `process`) and a request `cache_salt` | share globally and silently; a salt alone, as vLLM and sglang do | a salt is precise but fails open when omitted; a scope makes the default safe but cannot say "same customer". They compose ([§7.1](#71-two-mechanisms-and-tgo-takes-both)) |
| 016-D9 | $H$ is SHA-256; a fast hash needs a per-process random seed | any fast hash | a collision hands one request another's KV. vLLM shipped a predictable non-crypto hash and had to fix it ([§3.1](#31-the-hash-is-a-security-boundary-so-h-is-not-free-choice)) |
| 016-D8 | tgo owns the block pool | ask accel to export `pagetable` | accel 030 declines to evict because eviction is policy, and it is right; the policy is §5 |
| 016-D10 | reuse at most $T-1$ positions; a full hit still prefills one token | reuse the whole match | the cache holds KV, not logits, so a full reuse has nothing to sample from. Taken from ollama; the chat path hides it because the rendered prompt always ends with a fresh assistant opener ([§3.1](#31-a-full-hit-must-still-prefill-one-token)) |
| 016-D11 | many sequences share blocks concurrently; reclaim by recompute | ollama's single active path with snapshots paged to host | ollama's shape is right for one user and inverts under concurrency, which is what tgo serves. Recorded because the difference is workload, not correctness |
