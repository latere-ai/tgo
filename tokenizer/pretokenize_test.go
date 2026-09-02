// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

func qwenSplitter() splitPattern   { return knownPatterns[patternDigest(qwenPattern)] }
func cl100kSplitter() splitPattern { return knownPatterns[patternDigest(cl100kPattern)] }

// TestSplitGolden is the hand-checked behaviour of the pattern, alternative by
// alternative. Each case names the alternative it exercises, because a split
// that looks reasonable and is wrong is the failure mode section 4 describes.
func TestSplitGolden(t *testing.T) {
	tests := []struct {
		in   string
		want []string
		why  string
	}{
		{"", nil, "empty input"},
		{"hello", []string{"hello"}, "a bare word"},
		{"hello world", []string{"hello", " world"}, "a word keeps the space in front of it"},
		{"  hello", []string{" ", " hello"}, `\s+(?!\S) gives the last space of a run to the next piece`},
		{"hello   ", []string{"hello", "   "}, "a run at end of input is taken whole"},
		{"hello!", []string{"hello", "!"}, "punctuation after a word splits off"},
		{"hello !", []string{"hello", " !"}, "punctuation keeps its leading space"},
		{"!!!", []string{"!!!"}, "a punctuation run is one piece"},
		{" !!!\n\n", []string{" !!!\n\n"}, `[^\s\p{L}\p{N}]+[\r\n]* swallows the trailing line breaks`},
		{"a\n\nb", []string{"a", "\n\n", "b"}, `\s*[\r\n]+`},
		{"a \n b", []string{"a", " \n", " b"}, "the newline alternative ends at the last line break in the run"},
		{"a  \n\n  b", []string{"a", "  \n\n", " ", " b"}, "whitespace after the last break is left for the next position"},
		{"it's", []string{"it", "'s"}, "a contraction"},
		{"IT'S", []string{"IT", "'S"}, "(?i) on the contraction"},
		{"'sx", []string{"'s", "x"}, "the contraction alternative wins over the word alternative"},
		{"they'll've", []string{"they", "'ll", "'ve"}, "two-letter contractions"},
		{"'zebra", []string{"'zebra"}, "no contraction, so the word alternative takes the quote"},
		{"'", []string{"'"}, "a bare apostrophe at end of input"},
		{"'r", []string{"'r"}, "a two-letter contraction cut short by end of input"},
		{"'ſx", []string{"'ſ", "x"}, "U+017F folds to s under Unicode simple folding, as the reference (?i) does"},
		{"你好世界", []string{"你好世界"}, "CJK is letters"},
		{"café", []string{"café"}, "non-ASCII letters"},
		{"　a", []string{"　a"}, "the word alternative's optional prefix admits any non-letter, U+3000 included"},
		{"　!", []string{"　", "!"}, "U+3000 is whitespace, and a run of one cannot give a character away"},
		{"👋", []string{"👋"}, "an emoji is neither letter nor digit nor space"},
		{"snake_case", []string{"snake", "_case"}, "the underscore is punctuation and leads the next word"},
	}
	for _, tc := range tests {
		t.Run(tc.why, func(t *testing.T) {
			if got := qwenSplitter().split(tc.in); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("split(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestSplitDigits is the one alternative the two registered patterns disagree
// on, and the reason the registry carries two entries rather than one.
func TestSplitDigits(t *testing.T) {
	for _, tc := range []struct {
		in         string
		qwen, gpt4 []string
	}{
		{"7", []string{"7"}, []string{"7"}},
		{"123", []string{"1", "2", "3"}, []string{"123"}},
		{"1234", []string{"1", "2", "3", "4"}, []string{"123", "4"}},
		{"2026 y", []string{"2", "0", "2", "6", " y"}, []string{"202", "6", " y"}},
	} {
		if got := qwenSplitter().split(tc.in); !reflect.DeepEqual(got, tc.qwen) {
			t.Errorf("qwen split(%q) = %q, want %q", tc.in, got, tc.qwen)
		}
		if got := cl100kSplitter().split(tc.in); !reflect.DeepEqual(got, tc.gpt4) {
			t.Errorf("cl100k split(%q) = %q, want %q", tc.in, got, tc.gpt4)
		}
	}
}

func TestSplitCoversInput(t *testing.T) {
	for _, s := range splitCorpus {
		for _, p := range []splitPattern{qwenSplitter(), cl100kSplitter()} {
			pieces := p.split(s)
			if strings.Join(pieces, "") != s {
				t.Fatalf("%s: pieces do not concatenate to the input: %q", p.name, pieces)
			}
			for _, piece := range pieces {
				if piece == "" {
					t.Fatalf("%s: empty piece from %q", p.name, s)
				}
			}
		}
	}
}

// TestSplitBytesSurvive is the property the round trip depends on: pieces are
// substrings, so a byte that is not valid UTF-8 is carried through rather than
// replaced by U+FFFD.
func TestSplitBytesSurvive(t *testing.T) {
	in := "ok" + string([]byte{0xff, 0xfe}) + "ok"
	got := strings.Join(qwenSplitter().split(in), "")
	if got != in {
		t.Fatalf("split lost bytes: %q -> %q", []byte(in), []byte(got))
	}
}

// --- the oracle -------------------------------------------------------------
//
// specs/002-tokenizer.md section 7 asks for the splitter to be checked against
// the reference pattern's behaviour on a corpus. There is no reference engine
// here, so the check is against a second implementation built a different way:
// one anchored regexp per alternative, tried in written order, with the
// lookahead alternative done by explicit backtracking.
//
// Go's regexp gives leftmost-first alternation, the same semantics a
// backtracking engine gives, so the two agree on everything except the two
// places Go's regexp differs from the reference by construction, and both are
// removed rather than papered over:
//
//   - Go's \s is ASCII only and Go has no \p{White_Space}, so the class is
//     generated here from unicode.White_Space, the same table unicode.IsSpace
//     reads. Without this every U+3000 in Chinese text diverges.
//   - Go's regexp decodes invalid bytes to U+FFFD, so the oracle only runs on
//     input that is valid UTF-8. TestSplitBytesSurvive covers the rest.
//
// What stays independent is the structure: the oracle never decides per
// character, and the splitter never builds a regexp.

func whitespaceClassBody() string {
	var b strings.Builder
	for _, r := range unicode.White_Space.R16 {
		for c := rune(r.Lo); c <= rune(r.Hi); c += rune(r.Stride) {
			fmt.Fprintf(&b, `\x{%x}`, c)
		}
	}
	for _, r := range unicode.White_Space.R32 {
		for c := rune(r.Lo); c <= rune(r.Hi); c += rune(r.Stride) {
			fmt.Fprintf(&b, `\x{%x}`, c)
		}
	}
	return b.String()
}

type oracle struct {
	alts []*regexp.Regexp // alternatives 1 to 5, then 7; index 5 is a hole
	ws   *regexp.Regexp
}

const oracleLookaheadAlt = 5 // \s+(?!\S) sits between alternatives 5 and 7

func newOracle(maxDigits int) *oracle {
	ws := whitespaceClassBody()
	digits := `\p{N}`
	if maxDigits > 1 {
		digits = fmt.Sprintf(`\p{N}{1,%d}`, maxDigits)
	}
	src := []string{
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)`,
		`[^\r\n\p{L}\p{N}]?\p{L}+`,
		digits,
		` ?[^` + ws + `\p{L}\p{N}]+[\r\n]*`,
		`[` + ws + `]*[\r\n]+`,
		``, // the lookahead alternative, handled below
		`[` + ws + `]+`,
	}
	o := &oracle{ws: regexp.MustCompile(`^[` + ws + `]+`)}
	for i, s := range src {
		if i == oracleLookaheadAlt {
			o.alts = append(o.alts, nil)
			continue
		}
		o.alts = append(o.alts, regexp.MustCompile(`^(?:`+s+`)`))
	}
	return o
}

func (o *oracle) split(s string) []string {
	var out []string
	for i := 0; i < len(s); {
		n := o.match(s, i)
		if n <= 0 {
			panic("oracle: no alternative matched")
		}
		out = append(out, s[i:i+n])
		i += n
	}
	return out
}

func (o *oracle) match(s string, i int) int {
	for k, re := range o.alts {
		if k == oracleLookaheadAlt {
			if n := o.trailing(s, i); n > 0 {
				return n
			}
			continue
		}
		if loc := re.FindStringIndex(s[i:]); loc != nil && loc[1] > 0 {
			return loc[1]
		}
	}
	return 0
}

// trailing is \s+(?!\S): the greedy run, then one rune of backtracking at a
// time until the position after the match is not a non-space.
func (o *oracle) trailing(s string, i int) int {
	loc := o.ws.FindStringIndex(s[i:])
	if loc == nil {
		return 0
	}
	for n := loc[1]; n >= 1; {
		if i+n >= len(s) {
			return n
		}
		r, _ := utf8.DecodeRuneInString(s[i+n:])
		if unicode.IsSpace(r) {
			return n
		}
		_, w := utf8.DecodeLastRuneInString(s[i : i+n])
		n -= w
	}
	return 0
}

var splitCorpus = []string{
	"", "a", " ", "  ", "   ", "\n", "\r\n", "\t\t", "　", " ",
	"hello world", "  hello", "hello   ", "hello, world! How are you?",
	"it's a dog's life, they'll say, I'm sure we've won and they're right",
	"IT'S A DOG'S LIFE", "'s 'S 't 'RE 've 'M 'll 'D 'x", "'", "'r", "x'", "x'r",
	"snake_case camelCase PascalCase kebab-case path/to/file.go:12:3",
	"numbers 0 1 12 123 1234 12345 3.14159 -42 1e10",
	"你好世界，这是一个测试。", "こんにちは世界、これはテストです。",
	"한국어 텍스트", "Привет мир", "مرحبا بالعالم",
	"👨‍👩‍👧‍👦 👋🏽 ❤️ 🚀",
	"line one\nline two\n\nline four\r\n\r\nline six",
	"  \n\n  indented\n\t\ttabbed  \n   ",
	"<|im_start|>system\nYou are helpful.<|im_end|>\n",
	"def f(x): return x ** 2  # comment",
	"a　b　　c", "  x", "é café",
	"...", "!!!???", "-- --- ----", "$1,000.00 (50%)", "a  \n\n  b",
}

func TestSplitAgainstOracle(t *testing.T) {
	for _, p := range []splitPattern{qwenSplitter(), cl100kSplitter()} {
		o := newOracle(p.maxDigits)
		for _, s := range splitCorpus {
			want := o.split(s)
			got := p.split(s)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("%s: split(%q)\n got %q\nwant %q", p.name, s, got, want)
			}
		}
	}
}

// TestOracleDisagreesWithABrokenSplitter is the negative test CONTRIBUTING
// requires of a checker: an oracle that agrees with everything catches nothing.
func TestOracleDisagreesWithABrokenSplitter(t *testing.T) {
	o := newOracle(1)
	// The classic wrong reading of \s+(?!\S): take the whole run.
	broken := func(s string) []string {
		var out []string
		for i := 0; i < len(s); {
			n := qwenSplitter().match(s, i)
			if run := spaceRun(s, i); run > 0 && run == n+lastRuneLen(s[i:i+run]) {
				n = run
			}
			out = append(out, s[i:i+n])
			i += n
		}
		return out
	}
	found := false
	for _, s := range splitCorpus {
		if !reflect.DeepEqual(broken(s), o.split(s)) {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("the oracle agrees with a splitter that mis-reads the lookahead; it is not testing anything")
	}
}

func FuzzSplitAgainstOracle(f *testing.F) {
	for _, s := range splitCorpus {
		f.Add(s)
	}
	oracles := map[int]*oracle{1: newOracle(1), 3: newOracle(3)}
	f.Fuzz(func(t *testing.T, s string) {
		if !utf8.ValidString(s) {
			t.Skip("Go's regexp decodes invalid bytes to U+FFFD; the oracle cannot speak for them")
		}
		for _, p := range []splitPattern{qwenSplitter(), cl100kSplitter()} {
			want := oracles[p.maxDigits].split(s)
			if got := p.split(s); !reflect.DeepEqual(got, want) {
				t.Fatalf("%s: split(%q)\n got %q\nwant %q", p.name, s, got, want)
			}
		}
	})
}
