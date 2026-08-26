// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"errors"
	"fmt"

	"github.com/latere-ai/tgo/internal/grammar"
	"github.com/latere-ai/tgo/tokenizer"
)

// This file is the whole of the wiring between a request's schema and the
// per-step mask specs/015-structured-output.md describes. The compiler is
// internal/grammar and knows nothing about a tokenizer or a sampler; what
// belongs here is the three joins it deliberately left out: which vocabulary it
// is compiled against, which ids may end a generation, and where the compiled
// grammar is kept so the second request does not pay for the first one's walk.

// schemaCacheMax is how many compiled grammars a [Model] keeps.
//
// A cache is required rather than nice: 015-D1's per-state token sets are the
// design, and they are worth building only if a later request finds them. It is
// bounded because the key is a schema body that arrived over HTTP, and an
// unbounded map keyed on caller-supplied bytes is a memory leak a caller can
// drive. On overflow the whole map is dropped rather than one entry evicted:
// there is no access order to evict by that is not itself per-request
// bookkeeping, and a server serving a handful of schemas never reaches it.
const schemaCacheMax = 32

// vocabulary is [grammar.Vocab] over a real tokenizer, and it is the seam
// specs/015-structured-output.md §2 turns on.
//
// Two joins that a synthetic vocabulary cannot exercise, and each one is a
// silent failure rather than an error:
//
//   - The bytes are [tokenizer.Tokenizer.TextBytes] and not the vocabulary
//     file's spelling. A byte-level BPE stores " the" as "Ġthe", so a mask
//     built from the surface form would constrain a different language and
//     nothing would say so.
//   - size is the model's vocabulary and not the tokenizer's. A checkpoint
//     pads its embedding matrix past the last real token -- Qwen3 has 151669
//     tokens against 151936 rows -- and the mask is applied to a logits row of
//     the padded width. A grammar compiled at the tokenizer's width would be
//     handed a longer row and refuse it on every step.
type vocabulary struct {
	tok  *tokenizer.Tokenizer
	size int
}

func (v vocabulary) Size() int { return v.size }

func (v vocabulary) Bytes(id int) []byte { return v.tok.TextBytes(id) }

// stopIDs is the set a grammar admits a document's end at.
//
// It must be exactly what [Stream.isStop] ends a completion on. A grammar given
// no stop ids returns [grammar.ErrNoToken] the instant a document is complete,
// because this language has no trailing whitespace and nothing else is
// admissible -- so every constrained generation would fail at the finish line,
// on the last step, after producing the right answer.
//
// Negative entries are dropped rather than passed on: [resolveSpecials] reports
// a token the tokenizer does not hold as -1, and grammar.Compile refuses an id
// outside the vocabulary, so a checkpoint missing one marker would refuse every
// schema instead of losing one stopping token.
func (m *Model) stopIDs() []int {
	var out []int
	for _, id := range []int{m.special.imEnd, m.special.endOfText} {
		if id >= 0 {
			out = append(out, id)
		}
	}
	return out
}

// CheckSchema reports whether a JSON Schema can be compiled against this
// model's vocabulary, and keeps the result.
//
// It is the same compilation a request carrying [Policy.Schema] performs, done
// early and on its own: a server has to answer a schema it cannot honour with a
// refusal that names the construct, and it must do that before it allocates a
// session, so that a request which will not run does not first take memory from
// one that would.
//
// The error is [grammar.UnsupportedError]'s text where the construct is one
// this compiler refuses -- naming the keyword and the obstruction, 015-D4 --
// and a description of the inconsistency where the schema is malformed.
func (m *Model) CheckSchema(schema []byte) error {
	_, err := m.grammar(schema)
	return err
}

// grammar compiles a schema, or returns the compilation a previous request
// already paid for.
//
// Behind its own mutex and not [Model.mu]: that one is the submission lock, and
// a compilation walking the vocabulary while holding it would stop every
// session in the process from decoding.
func (m *Model) grammar(schema []byte) (*grammar.Grammar, error) {
	if len(schema) == 0 {
		return nil, errors.New("tgo: the schema is empty")
	}
	key := string(schema)
	m.schemaMu.Lock()
	defer m.schemaMu.Unlock()
	if g, ok := m.schemas[key]; ok {
		return g, nil
	}
	g, err := grammar.Compile(schema, vocabulary{tok: m.tok, size: m.cfg.VocabSize},
		grammar.Options{Stop: m.stopIDs()})
	if err != nil {
		return nil, fmt.Errorf("tgo: %w", err)
	}
	if m.schemas == nil || len(m.schemas) >= schemaCacheMax {
		m.schemas = make(map[string]*grammar.Grammar, schemaCacheMax)
	}
	m.schemas[key] = g
	return g, nil
}
