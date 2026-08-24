// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Command gen writes tokenizer/testdata/synthetic.json, the small checked-in
// tokenizer specs/002-tokenizer.md section 7 asks the suite to run on.
//
// It lives under testdata so the go tool ignores it; run it explicitly:
//
//	go run ./tokenizer/testdata/gen > tokenizer/testdata/synthetic.json
//
// The first thirteen merges are fixed by hand, at the ranks the spec's worked
// example publishes: ("l","o") is 3, ("lo","w") is 7 and ("o","w") is 12, so
// "low" merges the way section 5 traces it. ("Ġ","Ġ") is rank 0 as it is in
// every real Qwen table, and ("e","e") then ("ee","e") with ("e","ee") absent
// is what makes "eee" tell leftmost tie-breaking from rightmost. The rest are
// trained on the corpus below, which is why the fixture behaves like a
// tokenizer rather than like a table of special cases.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

const corpus = `the low tower slowly lowered the yellow flower over the shallow water
they were there when the weather turned and the letters were lost in the tree
an antelope and an anteater entered the tent and then ate the entire dinner
in the interest of interesting internal intervals we intern the interpreter
three trees, three streets, and thirty-three thirsty travellers went there
"quoted text", (parenthesised text), [bracketed text], {braced text}!
numbers 0 1 2 3 4 5 6 7 8 9 10 100 1000 and 2026 and 3.14159 and 42
snake_case camelCase PascalCase kebab-case SCREAMING_CASE path/to/file.go
it's the reader's problem, they'll say, we've seen it, I'm sure they're right
你好世界，这是一个测试。中文字符需要多个字节。世界你好，测试中文。
こんにちは世界、これはテストです。日本語の文字も複数バイトです。
emoji: a family 👨‍👩‍👧‍👦 and a wave 👋🏽 and a heart ❤️ and a rocket 🚀
naive naïve resume résumé cafe café facade façade cooperate coöperate
	tab indented lines and    runs   of    spaces and trailing spaces
eee eeee ee e eeeee and low lower lowest lowering slower flow flowing`

var byteToRune [256]rune

func init() {
	next := 0
	for b := 0; b < 256; b++ {
		if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 {
			byteToRune[b] = rune(b)
			continue
		}
		byteToRune[b] = rune(256 + next)
		next++
	}
}

func mapBytes(s string) string {
	out := make([]rune, 0, len(s))
	for i := 0; i < len(s); i++ {
		out = append(out, byteToRune[s[i]])
	}
	return string(out)
}

// head is the hand-fixed prefix of the merge list. Its ranks are load-bearing;
// tokenizer_test.go asserts them so the fixture cannot drift out from under the
// tests that cite the spec.
var head = [][2]string{
	{"Ġ", "Ġ"},  // rank 0, two spaces, as in every real Qwen table
	{"e", "e"},  // rank 1
	{"ee", "e"}, // rank 2, and ("e","ee") is never emitted
	{"l", "o"},  // rank 3, spec section 5
	{"t", "h"},  // rank 4
	{"Ġ", "t"},  // rank 5
	{"i", "n"},  // rank 6
	{"lo", "w"}, // rank 7, spec section 5
	{"a", "n"},  // rank 8
	{"r", "e"},  // rank 9
	{"o", "n"},  // rank 10
	{"e", "r"},  // rank 11
	{"o", "w"},  // rank 12, spec section 5
}

const wantMerges = 320

func main() {
	merges := train()
	vocab := map[string]int{}
	next := 0
	for b := 0; b < 256; b++ {
		vocab[string(byteToRune[b])] = next
		next++
	}
	for _, m := range merges {
		join := m[0] + m[1]
		if _, ok := vocab[join]; !ok {
			vocab[join] = next
			next++
		}
	}
	added := []map[string]any{}
	for _, a := range []struct {
		content string
		special bool
	}{
		{"<|endoftext|>", true},
		{"<|im_start|>", true},
		{"<|im_end|>", true},
		{"<think>", false},
		{"<think>\n", false},
		{"</think>", false},
	} {
		added = append(added, map[string]any{
			"id": next, "content": a.content, "single_word": false,
			"lstrip": false, "rstrip": false, "normalized": false,
			"special": a.special,
		})
		next++
	}

	pattern := `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	doc := map[string]any{
		"version":      "1.0",
		"added_tokens": added,
		"normalizer":   map[string]any{"type": "NFC"},
		"pre_tokenizer": map[string]any{
			"type": "Sequence",
			"pretokenizers": []any{
				map[string]any{
					"type":     "Split",
					"pattern":  map[string]any{"Regex": pattern},
					"behavior": "Isolated",
					"invert":   false,
				},
				map[string]any{
					"type": "ByteLevel", "add_prefix_space": false,
					"trim_offsets": false, "use_regex": false,
				},
			},
		},
		"decoder": map[string]any{
			"type": "ByteLevel", "add_prefix_space": false,
			"trim_offsets": false, "use_regex": false,
		},
		"model": map[string]any{
			"type": "BPE", "dropout": nil, "unk_token": nil,
			"continuing_subword_prefix": "", "end_of_word_suffix": "",
			"fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
			"vocab": vocab, "merges": merges,
		},
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", " ")
	if err := enc.Encode(doc); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// train runs ordinary BPE training over the corpus, after the fixed head, and
// never emits ("e","ee") -- the pair whose absence is what "eee" tests.
func train() [][2]string {
	pieces := map[string]int{}
	for _, w := range crudeSplit(corpus) {
		pieces[mapBytes(w)]++
	}
	words := make([][]string, 0, len(pieces))
	counts := make([]int, 0, len(pieces))
	keys := make([]string, 0, len(pieces))
	for k := range pieces {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		var syms []string
		for _, r := range k {
			syms = append(syms, string(r))
		}
		words = append(words, syms)
		counts = append(counts, pieces[k])
	}

	merges := append([][2]string(nil), head...)
	seen := map[[2]string]bool{{"e", "ee"}: true}
	for _, m := range merges {
		seen[m] = true
		words = applyAll(words, m)
	}
	for len(merges) < wantMerges {
		freq := map[[2]string]int{}
		for wi, syms := range words {
			for i := 0; i+1 < len(syms); i++ {
				freq[[2]string{syms[i], syms[i+1]}] += counts[wi]
			}
		}
		var bestKeys [][2]string
		for p := range freq {
			if !seen[p] {
				bestKeys = append(bestKeys, p)
			}
		}
		sort.Slice(bestKeys, func(i, j int) bool {
			if freq[bestKeys[i]] != freq[bestKeys[j]] {
				return freq[bestKeys[i]] > freq[bestKeys[j]]
			}
			if bestKeys[i][0] != bestKeys[j][0] {
				return bestKeys[i][0] < bestKeys[j][0]
			}
			return bestKeys[i][1] < bestKeys[j][1]
		})
		if len(bestKeys) == 0 {
			break
		}
		best := bestKeys[0]
		seen[best] = true
		merges = append(merges, best)
		words = applyAll(words, best)
	}
	return merges
}

func applyAll(words [][]string, m [2]string) [][]string {
	for i, syms := range words {
		words[i] = applyOne(syms, m)
	}
	return words
}

func applyOne(syms []string, m [2]string) []string {
	out := syms[:0:0]
	for i := 0; i < len(syms); {
		if i+1 < len(syms) && syms[i] == m[0] && syms[i+1] == m[1] {
			out = append(out, m[0]+m[1])
			i += 2
			continue
		}
		out = append(out, syms[i])
		i++
	}
	return out
}

// crudeSplit approximates the pre-tokenizer well enough to train on. The
// fixture's merges do not have to be the reference's; they have to be an
// ordered list a real BPE would plausibly produce.
func crudeSplit(s string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		switch {
		case r == ' ':
			flush()
			cur.WriteRune(r)
		case unicode.IsLetter(r) || unicode.IsNumber(r):
			cur.WriteRune(r)
		default:
			flush()
			out = append(out, string(r))
		}
	}
	flush()
	return out
}
