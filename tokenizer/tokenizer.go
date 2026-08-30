// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package tokenizer turns text into the token ids a model was trained on, and
// turns ids back into text.
//
// It implements the byte-level BPE serialised in a Hugging Face
// tokenizer.json: a normalizer, a split pattern that cuts text into pieces, a
// byte-level alphabet, and an ordered merge table. It has no device and no
// network. Its one dependency outside the standard library is
// golang.org/x/text, for NFC (002-D10), which is pure Go and reaches no cgo.
//
// Two things about it are worth knowing before you use it.
//
// A tokenizer whose split pattern is not one this package recognises is
// refused at load, naming the pattern. That is deliberate: a different split
// produces different ids for the same text, silently, and there is nothing for
// a human to read and check. See specs/002-tokenizer.md 002-D7.
//
// NFC normalization is declared by every Qwen checkpoint and is applied here;
// see normalize.go. It ran as an identity seam until 2026-08-24, which meant
// text that was not already in NFC encoded to the ids of its decomposed form.
package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// Tokenizer is an immutable, concurrency-safe byte-level BPE tokenizer.
type Tokenizer struct {
	normalize      normalizer
	pattern        splitPattern
	patternName    string
	addPrefixSpace bool

	tokenID map[string]int // byte-mapped token text to id
	merges  map[pair]int   // merge rank, the pair's index in the merge list

	// piece holds the bytes each id decodes to, indexed by id. A nil entry is
	// an id no table claims. Added tokens hold their literal content: their
	// text is not byte-mapped, so running it back through the alphabet would
	// mangle any added token containing a space.
	piece [][]byte

	added      []addedToken
	addedFirst map[byte][]int // first content byte to indices into added
	addedText  map[string]int
	addedID    map[int]bool // the ids added_tokens claims, for TextBytes
}

// addedToken is one entry of tokenizer.json's added_tokens: a control token
// matched before BPE and never merged into (002-D3).
type addedToken struct {
	id      int
	content string
	special bool
}

// Load reads a tokenizer.json from disk.
func Load(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	t, err := Parse(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return t, nil
}

// Parse reads a tokenizer.json from r.
//
// Parse refuses anything it cannot reproduce exactly rather than approximating
// it. Every refusal below is a setting that changes the ids produced for a
// given string without changing anything a reader could notice.
func Parse(r io.Reader) (*Tokenizer, error) {
	var f tokenizerFile
	dec := json.NewDecoder(r)
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("tokenizer: parse: %w", err)
	}
	t := &Tokenizer{
		tokenID:    make(map[string]int),
		merges:     make(map[pair]int),
		addedFirst: make(map[byte][]int),
		addedText:  make(map[string]int),
		addedID:    make(map[int]bool),
	}
	if err := t.applyNormalizer(f.Normalizer); err != nil {
		return nil, err
	}
	if err := t.applyPreTokenizer(f.PreTokenizer); err != nil {
		return nil, err
	}
	if err := t.applyPostProcessor(f.PostProcessor, f.Decoder); err != nil {
		return nil, err
	}
	if err := t.applyModel(f.Model); err != nil {
		return nil, err
	}
	if err := t.applyAdded(f.AddedTokens); err != nil {
		return nil, err
	}
	return t, nil
}

func (t *Tokenizer) applyNormalizer(n *normalizerJSON) error {
	switch {
	case n == nil:
		return nil
	case n.Type == "NFC":
		t.normalize = nfc
		return nil
	default:
		return fmt.Errorf("tokenizer: normalizer %q is not implemented; only NFC is, and a different normalizer changes ids", n.Type)
	}
}

// applyPreTokenizer walks the pre_tokenizer tree, which is either a single node
// or a Sequence of them, and requires exactly the shape 002-D6 handles: one
// Split on a recognised regex, and one ByteLevel.
func (t *Tokenizer) applyPreTokenizer(p *preTokJSON) error {
	if p == nil {
		return errors.New("tokenizer: no pre_tokenizer")
	}
	var splits, byteLevels []*preTokJSON
	var walk func(*preTokJSON)
	walk = func(n *preTokJSON) {
		switch n.Type {
		case "Sequence":
			for i := range n.PreTokenizers {
				walk(&n.PreTokenizers[i])
			}
		case "Split":
			splits = append(splits, n)
		case "ByteLevel":
			byteLevels = append(byteLevels, n)
		}
	}
	walk(p)

	if len(byteLevels) != 1 {
		return fmt.Errorf("tokenizer: expected exactly one ByteLevel pre_tokenizer, found %d", len(byteLevels))
	}
	t.addPrefixSpace = byteLevels[0].AddPrefixSpace

	if len(splits) != 1 {
		return fmt.Errorf("tokenizer: expected exactly one Split pre_tokenizer, found %d (a ByteLevel with use_regex is GPT-2's built-in pattern and is not implemented)", len(splits))
	}
	s := splits[0]
	if s.Pattern == nil || s.Pattern.Regex == "" {
		return errors.New("tokenizer: Split pre_tokenizer has no Regex pattern")
	}
	if s.Behavior != "Isolated" {
		return fmt.Errorf("tokenizer: Split behavior %q is not implemented; only Isolated is", s.Behavior)
	}
	if s.Invert {
		return errors.New("tokenizer: inverted Split is not implemented")
	}
	digest := patternDigest(s.Pattern.Regex)
	known, ok := knownPatterns[digest]
	if !ok {
		// 002-D7: refused, not warned, and the message carries the pattern so
		// that adding a splitter for it is a mechanical next step.
		return fmt.Errorf("tokenizer: unrecognised split pattern (sha256 %s): %s", digest, s.Pattern.Regex)
	}
	t.pattern, t.patternName = known, known.name
	return nil
}

// applyPostProcessor refuses the two nodes this package reproduces rather than
// interprets, on the same reasoning as applyModel's option list (002-D7).
//
// A ByteLevel post_processor only rewrites character offsets, so ignoring it is
// safe; a TemplateProcessing one *inserts ids* -- Llama-3 files use it to put
// <|begin_of_text|> in front of every sequence -- and a file carrying one would
// otherwise load here and encode every prompt without its BOS, silently. The
// same holds for the decoder: Decode inverts the byte-level alphabet, so a
// Metaspace or Replace decoder would return text this package never checks.
func (t *Tokenizer) applyPostProcessor(post, dec *nodeJSON) error {
	if post != nil && post.Type != "" && post.Type != "ByteLevel" {
		return fmt.Errorf("tokenizer: post_processor %q is not implemented; only ByteLevel is, and a %s post_processor inserts ids Encode would omit", post.Type, post.Type)
	}
	if dec != nil && dec.Type != "" && dec.Type != "ByteLevel" {
		return fmt.Errorf("tokenizer: decoder %q is not implemented; only ByteLevel is, and Decode inverts the byte-level alphabet", dec.Type)
	}
	return nil
}

func (t *Tokenizer) applyModel(m *modelJSON) error {
	if m == nil {
		return errors.New("tokenizer: no model")
	}
	if m.Type != "" && m.Type != "BPE" {
		return fmt.Errorf("tokenizer: model type %q is not implemented; only BPE is", m.Type)
	}
	for _, bad := range []struct {
		set  bool
		name string
	}{
		{m.Dropout != nil && *m.Dropout != 0, "dropout"},
		{m.UnkToken != nil && *m.UnkToken != "", "unk_token"},
		{m.ContinuingSubwordPrefix != "", "continuing_subword_prefix"},
		{m.EndOfWordSuffix != "", "end_of_word_suffix"},
		{m.FuseUnk, "fuse_unk"},
		{m.ByteFallback, "byte_fallback"},
		{m.IgnoreMerges, "ignore_merges"},
	} {
		if bad.set {
			return fmt.Errorf("tokenizer: model option %s is set and is not implemented; it changes the ids produced", bad.name)
		}
	}
	if len(m.Vocab) == 0 {
		return errors.New("tokenizer: empty vocab")
	}

	maxID := 0
	for tok, id := range m.Vocab {
		if id < 0 {
			return fmt.Errorf("tokenizer: vocab entry %q has negative id %d", tok, id)
		}
		if id > maxID {
			maxID = id
		}
	}
	t.piece = make([][]byte, maxID+1)
	for tok, id := range m.Vocab {
		bytes, ok := unmapToken(tok)
		if !ok {
			return fmt.Errorf("tokenizer: vocab entry %q contains a code point outside the byte-level alphabet", tok)
		}
		if t.piece[id] != nil {
			return fmt.Errorf("tokenizer: id %d is claimed by two vocab entries", id)
		}
		t.piece[id] = bytes
		t.tokenID[tok] = id
	}

	// Every one of the 256 single-byte symbols must exist, or an input byte
	// would have no id and Encode would have no total definition. Section 3's
	// bijection is a property of the alphabet; this is the property of the
	// vocabulary that makes it usable.
	for b := range 256 {
		sym := string(byteToRune[b])
		if _, ok := t.tokenID[sym]; !ok {
			return fmt.Errorf("tokenizer: vocab is missing the byte-level symbol for byte 0x%02x", b)
		}
	}

	for i, mg := range m.Merges {
		p := pair{mg[0], mg[1]}
		if p.left == "" || p.right == "" {
			return fmt.Errorf("tokenizer: merge %d is not a pair of two tokens", i)
		}
		if _, dup := t.merges[p]; dup {
			return fmt.Errorf("tokenizer: merge %d repeats the pair (%q, %q); the second rank would be unreachable", i, p.left, p.right)
		}
		// The merge loop only ever produces symbols that are either a single
		// byte symbol or the join of a merge. Requiring the join to be in the
		// vocabulary is therefore what makes "every symbol Encode emits has an
		// id" a property of the file rather than of the inputs tried.
		if _, ok := t.tokenID[p.left+p.right]; !ok {
			return fmt.Errorf("tokenizer: merge %d joins to %q, which is not in the vocab", i, p.left+p.right)
		}
		t.merges[p] = i
	}
	if len(t.merges) == 0 {
		return errors.New("tokenizer: no merges")
	}
	return nil
}

func (t *Tokenizer) applyAdded(list []addedTokenJSON) error {
	for _, a := range list {
		if a.Content == "" {
			return fmt.Errorf("tokenizer: added token %d has empty content", a.ID)
		}
		if a.ID < 0 {
			return fmt.Errorf("tokenizer: added token %q has negative id %d", a.Content, a.ID)
		}
		if a.ID < len(t.piece) && t.piece[a.ID] != nil {
			return fmt.Errorf("tokenizer: added token %q claims id %d, already claimed", a.Content, a.ID)
		}
		if _, dup := t.addedText[a.Content]; dup {
			return fmt.Errorf("tokenizer: added token %q appears twice", a.Content)
		}
		for a.ID >= len(t.piece) {
			t.piece = append(t.piece, nil)
		}
		t.piece[a.ID] = []byte(a.Content)
		t.addedText[a.Content] = a.ID
		t.addedID[a.ID] = true
		idx := len(t.added)
		t.added = append(t.added, addedToken{id: a.ID, content: a.Content, special: a.Special})
		t.addedFirst[a.Content[0]] = append(t.addedFirst[a.Content[0]], idx)
	}
	// Longest first, so that a prefix pair like <think> and <think>\n resolves
	// to the longer one at a given position, which is what the reference's
	// leftmost-longest matcher does.
	for b := range t.addedFirst {
		idxs := t.addedFirst[b]
		sort.Slice(idxs, func(i, j int) bool {
			return len(t.added[idxs[i]].content) > len(t.added[idxs[j]].content)
		})
	}
	return nil
}

// VocabSize returns one past the highest id this tokenizer can produce.
//
// It is the size of the id space, not the model's embedding row count: a
// checkpoint commonly pads its embedding matrix past the last real token, so
// Qwen3 returns 151669 here against an embedding of 151936. Use it to bound an
// id, not to shape a tensor.
func (t *Tokenizer) VocabSize() int { return len(t.piece) }

// TextBytes returns the bytes an id contributes to decoded text, and nil for
// an id that contributes none.
//
// It is the inverse of the byte-level alphabet and not the vocabulary file's
// spelling: the id for " the" comes back as the four bytes of " the", not as
// the five characters of "Ġthe". A caller that reasons about what a token puts
// in the output -- a grammar masking the tokens that cannot continue a document
// -- must read the bytes and not the surface form, because the surface form is
// a different string that happens to be legal in most contexts.
//
// Nil for three cases, which are one case to a caller: an id out of range, an
// id no table claims, and an added token. An added token is nil because its
// piece holds its literal content, so "<|im_end|>" would otherwise read as ten
// characters a caller could believe it is free to emit anywhere those
// characters are legal.
func (t *Tokenizer) TextBytes(id int) []byte {
	if t.addedID[id] {
		return nil
	}
	return t.bytesFor(id)
}

// Special resolves a control token by its literal text.
func (t *Tokenizer) Special(text string) (int, bool) {
	id, ok := t.addedText[text]
	return id, ok
}

// Encode applies the normalizer, the split pattern and BPE, and returns the
// token ids.
//
// Added tokens are matched before BPE and are never merged into (002-D3), but
// only when allowSpecial is set. With allowSpecial false the literal text
// "<|im_start|>" encodes as the characters that spell it, so content can never
// forge a turn boundary; that is the structural half of the injection defence
// in specs/003-chat-template.md section 4, and it is why the caller must ask
// for special matching rather than opt out of it.
func (t *Tokenizer) Encode(s string, allowSpecial bool) []int {
	if t.normalize != nil {
		s = t.normalize(s)
	}
	if t.addPrefixSpace && !strings.HasPrefix(s, " ") {
		s = " " + s
	}
	out := []int{}
	last := 0
	for i := 0; i < len(s); {
		id, n, ok := t.matchAdded(s, i, allowSpecial)
		if !ok {
			i++
			continue
		}
		out = t.encodePlain(s[last:i], out)
		out = append(out, id)
		i += n
		last = i
	}
	return t.encodePlain(s[last:], out)
}

// matchAdded returns the longest added token starting at i.
//
// The caller advances one byte at a time, which cannot produce a match that
// starts inside a character: UTF-8 continuation bytes are 0x80 to 0xBF and no
// text can begin with one, so every candidate position that matches is a
// character boundary.
func (t *Tokenizer) matchAdded(s string, i int, allowSpecial bool) (id, n int, ok bool) {
	if !allowSpecial || len(t.added) == 0 {
		return 0, 0, false
	}
	for _, idx := range t.addedFirst[s[i]] {
		a := t.added[idx]
		if strings.HasPrefix(s[i:], a.content) {
			return a.id, len(a.content), true
		}
	}
	return 0, 0, false
}

// encodePlain encodes a span that contains no added token.
func (t *Tokenizer) encodePlain(s string, out []int) []int {
	for _, piece := range t.pattern.split(s) {
		for _, sym := range t.merge(symbols(mapBytes(piece))) {
			// applyModel proved every reachable symbol has an id.
			out = append(out, t.tokenID[sym])
		}
	}
	return out
}

// Decode is the whole-string inverse of Encode.
//
// It is byte-exact: an id sequence that came from Encode decodes to the exact
// bytes that went in, including bytes that are not valid UTF-8. An id no table
// claims contributes nothing.
//
// Compare Decoder, which holds back a byte sequence that is a valid prefix of a
// code point and not yet a whole one, and replaces one that is still truncated
// at end of stream. That is narrower than "always well-formed UTF-8": a byte
// that cannot begin any code point is emitted at once, because waiting cannot
// make it valid. Concatenating every Push over a stream that ends on a complete
// code point therefore gives byte-for-byte what this gives.
func (t *Tokenizer) Decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		b = append(b, t.bytesFor(id)...)
	}
	return string(b)
}

// bytesFor returns the bytes an id decodes to, or nil for an id out of range.
func (t *Tokenizer) bytesFor(id int) []byte {
	if id < 0 || id >= len(t.piece) {
		return nil
	}
	return t.piece[id]
}
