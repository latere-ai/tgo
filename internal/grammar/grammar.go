// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package grammar

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"sync/atomic"
)

// Vocab is the tokenizer's side of the boundary: how many token ids there are,
// and what bytes each one stands for.
//
// An interface rather than a concrete type so that this package depends on no
// tokenizer and the sampler can hand it whatever it already has. Two clauses of
// the contract are load-bearing:
//
//   - Bytes returns the DECODED bytes of a token -- the text the token puts in
//     the output -- not its surface form in a vocabulary file. For a byte-level
//     BPE that means the byte-level alphabet has already been undone.
//   - Bytes returns nil for any token that is not text: a control token, a
//     special token, an unused slot. Handing back the literal characters
//     "<|im_start|>" instead would let the mask admit a control token in the
//     middle of a JSON string, where those characters are perfectly legal.
//
// Size is also the length of the logits row Mask is given.
type Vocab interface {
	Size() int
	Bytes(id int) []byte
}

// Pieces is the simplest Vocab: token id indexes the slice. A nil entry is a
// token with no text.
type Pieces [][]byte

func (p Pieces) Size() int           { return len(p) }
func (p Pieces) Bytes(id int) []byte { return p[id] }

// Options configures a compilation.
type Options struct {
	// Stop is the token ids that may end a generation -- EOS and friends.
	// They are admissible exactly where the document is already complete, and
	// nowhere else. That is the whole guarantee: a grammar that let EOS fire
	// mid-document would emit a truncated document that parses as nothing, and
	// "the output parses by construction" would be false.
	//
	// A stop id is never admissible as text, whatever Bytes says about it.
	Stop []int
}

// UnsupportedError is a schema this package refuses to compile, naming the
// construct that made it refuse.
//
// 015-D4: a keyword silently ignored produces a document that validates against
// a schema the caller did not write. So the front end is an allowlist -- every
// keyword in the schema must be consumed by the compilation, and one that is
// not is reported here rather than dropped.
type UnsupportedError struct {
	Path      string // JSON Pointer to the subschema, "" for the root
	Construct string // the keyword, or the shape, that is not supported
	Why       string
}

func (e *UnsupportedError) Error() string {
	at := e.Path
	if at == "" {
		at = "the root schema"
	}
	return fmt.Sprintf("grammar: %s: %s is not supported: %s", at, e.Construct, e.Why)
}

// SchemaError is a schema this package understands and finds inconsistent --
// malformed JSON, a required property that is not declared, an enum member that
// contradicts the declared type.
type SchemaError struct {
	Path   string
	Reason string
}

func (e *SchemaError) Error() string {
	at := e.Path
	if at == "" {
		at = "the root schema"
	}
	return fmt.Sprintf("grammar: %s: %s", at, e.Reason)
}

// ErrNoToken is returned when no token in the vocabulary can continue the
// document.
//
// It is a hard failure and not an empty mask, because an empty mask is worse
// than useless downstream: a row of -Inf makes every softmax weight NaN, and an
// argmax with a strict comparison then returns token zero without complaining.
// A vocabulary containing every single byte cannot reach this, which is why it
// takes a deliberately holed vocabulary to see it -- and why it must be checked
// rather than assumed.
//
// One reachable case is not a failure at all: a compilation given no Stop ids
// reaches it the moment a closed document is complete, because nothing may
// follow the last brace and there is no stop token to admit. A caller that
// leaves Options.Stop empty must therefore read Accepting before it reads this
// error, or it will report a finished generation as a dead end.
var ErrNoToken = errors.New("grammar: no token in the vocabulary can continue the document")

// ErrDone is returned by Advance once a stop token has been consumed.
var ErrDone = errors.New("grammar: the document is already finished")

// NotAllowedError is a token the grammar does not admit in the current state.
type NotAllowedError struct{ Token int }

func (e *NotAllowedError) Error() string {
	return fmt.Sprintf("grammar: token %d is not admissible here", e.Token)
}

// Grammar is a compiled schema together with the vocabulary it was compiled
// against. It is safe for concurrent use and is meant to be shared: the token
// caches it accumulates are the reason compiling once and reusing is worth
// anything (015-D1).
type Grammar struct {
	v       Vocab
	size    int
	n       *nfa
	accept  int
	stop    []int  // sorted, deduped
	stopSet []bool // indexed by id

	// builds counts the per-state admissible sets computed so far. It is the
	// measurement 015-D1 is a bet on: the ratio of builds to decode steps is
	// how much of the vocabulary walk the cache saved.
	builds atomic.Int64

	mu     sync.Mutex
	states map[string]*dstate
	start  *dstate
}

// Compile turns a JSON Schema document into a grammar over v's vocabulary.
//
// It refuses rather than approximates. See UnsupportedError.
func Compile(schema []byte, v Vocab, opt Options) (*Grammar, error) {
	if v == nil || v.Size() == 0 {
		return nil, &SchemaError{Reason: "the vocabulary is empty"}
	}
	size := v.Size()

	stopSet := make([]bool, size)
	var stop []int
	for _, id := range opt.Stop {
		if id < 0 || id >= size {
			return nil, &SchemaError{Reason: fmt.Sprintf("stop id %d is outside the vocabulary of %d", id, size)}
		}
		if !stopSet[id] {
			stopSet[id] = true
			stop = append(stop, id)
		}
	}
	slices.Sort(stop)

	c := &compiler{n: &nfa{}, seen: map[string]bool{}}
	root, err := c.compile(schema)
	if err != nil {
		return nil, err
	}

	g := &Grammar{
		v:       v,
		size:    size,
		n:       c.n,
		accept:  root.out,
		stop:    stop,
		stopSet: stopSet,
		states:  make(map[string]*dstate),
	}
	g.start = g.intern(g.n.closure([]int{root.in}))
	return g, nil
}

// Start returns a fresh per-request state. States are independent; the caches
// behind them are shared.
func (g *Grammar) Start() *State { return &State{g: g, cur: g.start} }

// States reports how many determinized states the grammar has materialized so
// far.
func (g *Grammar) States() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.states)
}

// Builds reports how many per-state admissible-token sets have been computed.
//
// It is 015-D1 made observable. A decode step over a state that has been seen
// before costs a slice lookup; only a first visit walks the vocabulary. A
// Builds that keeps climbing with the request count means the states are not
// repeating and the cache is not paying for itself.
func (g *Grammar) Builds() int { return int(g.builds.Load()) }

// State is one request's position in the grammar. It is not safe for concurrent
// use, which costs nothing: a State belongs to one sequence, the same way a
// sample.Sampler does.
type State struct {
	g    *Grammar
	cur  *dstate
	done bool
}

// Accepting reports whether the text consumed so far is a complete document.
//
// A state can be accepting and still have continuations: after `1` in a
// top-level number, a digit and a stop token are both admissible.
func (s *State) Accepting() bool { return !s.done && s.cur.acc }

// Done reports whether a stop token has been consumed.
func (s *State) Done() bool { return s.done }

// Allowed returns the admissible token ids, ascending.
//
// The slice is the grammar's own cache and MUST NOT be modified. It is not
// copied because a vocabulary is 152k entries and this is called once per
// decode step.
func (s *State) Allowed() []int {
	if s.done {
		return nil
	}
	s.g.tokens(s.cur)
	return s.cur.allowed
}

// Mask applies the constraint to a row of logits, in place.
//
// specs/015-structured-output.md section 3: an additive negative infinity,
// applied BEFORE the penalties of specs/006-sampling.md section 3. Additive
// rather than a multiply after truncation, so it composes with that order with
// no special case, and so a masked token cannot be brought back by a penalty
// that raises it or a temperature that flattens the row -- both are monotone in
// the logit and -Inf is a fixed point of both.
//
// It returns ErrNoToken when nothing is admissible, rather than writing a row
// of -Inf that would come out of a softmax as NaN.
func (s *State) Mask(logits []float32) error {
	if s.done {
		return ErrDone
	}
	if len(logits) != s.g.size {
		return &SchemaError{Reason: fmt.Sprintf("logits row of %d for a vocabulary of %d", len(logits), s.g.size)}
	}
	allowed := s.Allowed()
	if len(allowed) == 0 {
		return ErrNoToken
	}
	neg := float32(math.Inf(-1))
	j := 0
	for i := range logits {
		if j < len(allowed) && allowed[j] == i {
			j++
			continue
		}
		logits[i] += neg
	}
	return nil
}

// Advance consumes the token that was actually sampled.
//
// It is called once per accepted token, after the draw -- not once per
// candidate. A caller resuming a sequence replays it over the tokens already
// generated to rebuild the position.
func (s *State) Advance(id int) error {
	if s.done {
		return ErrDone
	}
	allowed := s.Allowed()
	i, ok := slices.BinarySearch(allowed, id)
	if !ok {
		return &NotAllowedError{Token: id}
	}
	if next := s.cur.dest[i]; next != nil {
		s.cur = next
	} else {
		s.done = true
	}
	return nil
}

// sortedUnique orders and dedupes, so that a refusal naming one of several
// offending keywords names the same one on every run. A map's iteration order
// would make which keyword a caller sees depend on the process.
func sortedUnique(in []string) []string {
	out := slices.Clone(in)
	slices.Sort(out)
	return slices.Compact(out)
}
