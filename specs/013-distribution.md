---
title: "Distribution: fetching a checkpoint, and the cache it lands in"
status: complete
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

Keyed by the resolved commit sha, so two model versions share nothing and
neither is corrupted by the other:

```
$TGO_CACHE/models/{org}/{repo}/{sha}/
```

The key is what the revision resolved to, not what was asked for, so a moving
ref such as `main` does not overwrite what `main` used to be. A repo id with no
org, such as `gpt2`, lands under the sentinel org `_`, which no Hugging Face
account can be called.

Default `$XDG_CACHE_HOME/tgo` or `~/.cache/tgo`. A download writes to a
temporary name and renames on completion, so an interrupted fetch never leaves a
file that looks whole. `Content-Length` and, where the API gives one, the sha256
are both checked, and a mismatch is one of two kinds. Bytes that can never
become right are deleted: a wrong digest, and a body longer than the published
length. Bytes that are merely incomplete are kept under the temporary name,
which is what makes the resume below possible.

**Resumable**, via `Range`, because a 8 GB download over a bad connection
otherwise starts over. The partial file keeps its temporary name, so resumption
never operates on something a reader might pick up.

## 3. Concurrency

Shards download in parallel with a small bound; the bound exists because the
disk is the limit, not the network, and eight parallel writes to one spinning
disk is slower than two. A lock file beside each revision directory, at
`{sha}.lock`, keeps two `tgo pull` processes from writing the same file. It sits
beside rather than inside, because inside it would be an entry
`safetensors.OpenRepo` has to know to ignore.

## 4. Not a registry

tgo does not have its own model registry, does not host weights, and does not
have a `Modelfile`. A model is a Hugging Face repo id or a local path. Ollama's
registry is a real product decision; it is not this one, and adding it later
costs nothing that is decided here.

## Outcome

Checkpoint distribution is `internal/hub` plus the `tgo pull` command, and it
runs. It landed whole on 2026-08-26: the HF listing and resolve calls, the
sha-keyed cache, the resumable download, the parallel bound, the revision lock,
and the two ref forms. Every section of this spec has code behind it, at 95.4%
coverage in `internal/hub` and 90.3% in `cmd/tgo`.

**What shipped**, section by section:

| section | what landed | where |
| --- | --- | --- |
| 1 | `GET /api/models/{repo}/revision/{rev}?blobs=true`, then `resolve/{rev}/{path}` per file, on plain `net/http` | `internal/hub/client.go:278`, `client.go:347` |
| 1 | `Authorization` dropped on any `host:port` change, chain bounded at 10, header set per request rather than in a transport | `internal/hub/client.go:153`, `client.go:192` |
| 1 | the LFS pointer sniffed before any length check, as `ErrLFSPointer` | `internal/hub/download.go:150`, `hub.go:66` |
| 2 | `$TGO_CACHE/models/{org}/{repo}/{sha}`, with `_` for a bare repo id | `internal/hub/client.go:184`, `hub.go:104` |
| 2 | `TGO_CACHE`, else `XDG_CACHE_HOME/tgo`, else `~/.cache/tgo` | `internal/hub/hub.go:254` |
| 2 | the `.part` name and the rename, the `Range` resume, and both length and sha256 checks | `internal/hub/download.go:95`, `download.go:112`, `download.go:183`, `download.go:192`, `download.go:200` |
| 3 | a semaphore at `parallel()`, four by default | `internal/hub/fetch.go:102`, `client.go:29` |
| 3 | `O_CREATE`\|`O_EXCL` on `{sha}.lock`, held for the whole download | `internal/hub/lock.go:19`, `lock.go:35`, `fetch.go:61` |
| 4 | `Ref` is a repo id at a revision or a local path, and nothing else; a local ref is stat'd and returned | `internal/hub/hub.go:108`, `hub.go:154`, `fetch.go:27` |

**What diverged** from the design, and why the code is right:

- **The cache is keyed by the resolved commit sha, not by the requested
  revision.** `Fetch` resolves first and calls `Dir` with the sha
  (`client.go:184`, `fetch.go:42`). Keying by the requested ref would let a
  second fetch of `main` write into the directory the first fetch of `main`
  filled, which is the corruption this section set out to prevent. §2 and
  013-D2 are corrected above.
- **A truncated body keeps its partial file rather than being deleted.** The
  original rule deleted on any mismatch, which contradicted the Resumable
  paragraph three lines below it. The code splits the rule instead: a wrong
  digest and an over-long body are deleted because those bytes can never become
  right (`download.go:187`, `download.go:196`), and a short body stays under its
  temporary name because that is exactly what the next `Range` request resumes
  from (`download.go:174`).

**Not built.** Nothing in this spec's scope. Four areas are built and
undescribed here, which is description debt rather than a design gap: the `Wanted` file
filter, which takes top-level files only and, besides `.safetensors`, a fixed
allowlist of nine config and tokenizer names (`client.go:362`); the listing
treated as untrusted server input, where `safePath` refuses absolute paths, `..`
elements, backslashes, volume names and control characters, and reports
`ErrUnsafePath` (`hub.go:223`, `hub.go:82`); where the token comes from, which
is `--token`, then `$HF_TOKEN`, then `$HUGGING_FACE_HUB_TOKEN`, with `cmd`
reading the environment and `hub` taking the token as a field
(`cmd/tgo/pull.go:27`), and `$HF_ENDPOINT` overriding the API root for a mirror
or an on-premises deployment (`client.go:164`); and `tgo pull` itself, which
prints the path on stdout and progress on stderr so `tgo run "$(tgo pull ...)"`
composes, and which turns SIGINT into a cancelled download the next run resumes
(`cmd/tgo/pull.go:102`, `pull.go:114`).

## Decision record

| id | decision | rejected | consequence |
| --- | --- | --- | --- |
| 013-D1 | plain `net/http` against the HF API | an SDK dependency | no dependency; the 302 and LFS traps are ours to test |
| 013-D2 | ~~cache keyed by revision~~ → **keyed by the resolved commit sha** | keyed by repo, overwritten; keyed by the requested ref | **Amended 2026-08-27.** Two revisions coexist and neither corrupts the other, and a moving ref such as `main` cannot overwrite what it used to name |
| 013-D3 | temp name plus rename, resumable via `Range` | write in place | an interrupted fetch is never mistaken for a whole one |
| 013-D4 | no registry, no Modelfile | an ollama-shaped registry | a model is a repo id or a path; nothing here forecloses more |
