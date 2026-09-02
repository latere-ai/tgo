// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"encoding/json"
	"fmt"
	"strings"
)

// The subset of the Hugging Face tokenizer.json serialisation this package
// reads. Fields it does not read are fields whose values it refuses to vary
// from, and applyModel names each one in its refusal.

type tokenizerFile struct {
	Normalizer    *normalizerJSON  `json:"normalizer"`
	AddedTokens   []addedTokenJSON `json:"added_tokens"`
	PreTokenizer  *preTokJSON      `json:"pre_tokenizer"`
	PostProcessor *nodeJSON        `json:"post_processor"`
	Decoder       *nodeJSON        `json:"decoder"`
	Model         *modelJSON       `json:"model"`
}

// nodeJSON reads only the discriminator of a serialisation node this package
// reproduces rather than interprets. See applyPostProcessor.
type nodeJSON struct {
	Type string `json:"type"`
}

type normalizerJSON struct {
	Type string `json:"type"`
}

type addedTokenJSON struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type preTokJSON struct {
	Type           string       `json:"type"`
	PreTokenizers  []preTokJSON `json:"pretokenizers"`
	Pattern        *patternJSON `json:"pattern"`
	Behavior       string       `json:"behavior"`
	Invert         bool         `json:"invert"`
	AddPrefixSpace bool         `json:"add_prefix_space"`
	UseRegex       bool         `json:"use_regex"`
}

type patternJSON struct {
	Regex string `json:"Regex"`
}

type modelJSON struct {
	Type                    string         `json:"type"`
	Dropout                 *float64       `json:"dropout"`
	UnkToken                *string        `json:"unk_token"`
	ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
	EndOfWordSuffix         string         `json:"end_of_word_suffix"`
	FuseUnk                 bool           `json:"fuse_unk"`
	ByteFallback            bool           `json:"byte_fallback"`
	IgnoreMerges            bool           `json:"ignore_merges"`
	Vocab                   map[string]int `json:"vocab"`
	Merges                  []mergeJSON    `json:"merges"`
}

// mergeJSON is one merge-list entry. The serialisation changed: tokenizers
// wrote "left right" for years and writes ["left","right"] now, and both forms
// are in circulation on the hub, so both are read. The pair form is
// unambiguous; the string form splits on the first space, which is safe because
// a byte-mapped token never contains one -- U+0020 maps to U+0120.
type mergeJSON [2]string

func (m *mergeJSON) UnmarshalJSON(b []byte) error {
	var asPair [2]string
	if err := json.Unmarshal(b, &asPair); err == nil {
		*m = asPair
		return nil
	}
	var asString string
	if err := json.Unmarshal(b, &asString); err != nil {
		return fmt.Errorf("merge entry is neither a pair nor a string: %s", b)
	}
	left, right, ok := strings.Cut(asString, " ")
	if !ok {
		return fmt.Errorf("merge entry %q has no space separating the pair", asString)
	}
	*m = [2]string{left, right}
	return nil
}
