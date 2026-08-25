---
title: "Distribution: fetching a checkpoint, and the cache it lands in"
status: implemented
layer: load
depends_on:
  - 000-decisions.md
  - 001-weights.md
---

# Distribution

[001](001-weights.md) takes a local directory. This is how a directory appears.

## 1. The source

Hugging Face, over HTTPS, no SDK: `GET /api/models/{repo}/revision/{rev}` for
the file list, then `resolve/{rev}/{path}` per file. Pure `net/http`.

**The trap:** `resolve` returns a 302 to a CDN, and a client that does not
follow it, or that follows it with the `Authorization` header attached, gets
either a redirect body or a 403. The header must be dropped on the cross-host
hop. A naive fetch of an LFS-backed file from the git endpoint instead of
`resolve` returns a 130-byte pointer file that parses as neither JSON nor
safetensors, which is the confusing failure worth naming here.

## 2. The cache

Content-addressed by revision, so two model versions share nothing and neither
is corrupted by the other:

```
$TGO_CACHE/models/{org}/{repo}/{revision}/
```

Default `$XDG_CACHE_HOME/tgo` or `~/.cache/tgo`. A download writes to a
temporary name and renames on completion, so an interrupted fetch never leaves a
file that looks whole. `Content-Length` and, where the API gives one, the sha256
are both checked; a mismatch deletes and fails.

**Resumable**, via `Range`, because a 8 GB download over a bad connection
otherwise starts over. The partial file keeps its temporary name, so resumption
never operates on something a reader might pick up.

## 3. Concurrency

Shards download in parallel with a small bound; the bound exists because the
disk is the limit, not the network, and eight parallel writes to one spinning
disk is slower than two. A lock file per revision directory keeps two `tgo pull`
processes from writing the same file.

## 4. Not a registry

tgo does not have its own model registry, does not host weights, and does not
have a `Modelfile`. A model is a Hugging Face repo id or a local path. Ollama's
registry is a real product decision; it is not this one, and adding it later
costs nothing that is decided here.

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 013-D1 | plain `net/http` against the HF API | an SDK dependency | no dependency; the 302 and LFS traps are ours to test |
| 013-D2 | cache keyed by revision | keyed by repo, overwritten | two revisions coexist; neither corrupts the other |
| 013-D3 | temp name plus rename, resumable via `Range` | write in place | an interrupted fetch is never mistaken for a whole one |
| 013-D4 | no registry, no Modelfile | an ollama-shaped registry | a model is a repo id or a path; nothing here forecloses more |
