// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tokenizer

// GPT-2's byte-level alphabet, specs/002-tokenizer.md section 3.
//
// Every one of the 256 byte values is mapped to a printable, non-space code
// point before BPE runs. The point is not aesthetics: it makes the vocabulary
// a set of ordinary strings while keeping the map a bijection on bytes, so the
// decode side is total over byte sequences including ones that are not valid
// UTF-8. The 188 bytes that are already printable ASCII or printable Latin-1
// map to themselves; the remaining 68 are pushed to U+0100 and up, in
// increasing byte order.

var (
	byteToRune [256]rune
	runeToByte map[rune]byte
)

func init() {
	next := 0
	for b := range 256 {
		if printableByte(byte(b)) {
			byteToRune[b] = rune(b)
			continue
		}
		byteToRune[b] = rune(256 + next)
		next++
	}
	runeToByte = make(map[rune]byte, 256)
	for b, r := range byteToRune {
		runeToByte[r] = byte(b)
	}
}

// printableByte reports whether b is one of the 188 values the alphabet maps to
// itself: printable ASCII, and printable Latin-1 minus the soft hyphen (173),
// which is excluded because it is invisible.
func printableByte(b byte) bool {
	return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174
}

// mapBytes rewrites a raw byte string into alphabet code points. The input is
// treated as bytes, never as runes: a []rune round trip here would replace
// invalid UTF-8 with U+FFFD and break the section 3 bijection.
func mapBytes(s string) string {
	out := make([]rune, 0, len(s))
	for i := 0; i < len(s); i++ {
		out = append(out, byteToRune[s[i]])
	}
	return string(out)
}

// unmapToken inverts mapBytes for one vocabulary token. ok is false when the
// token contains a code point outside the alphabet, which Parse refuses.
func unmapToken(tok string) ([]byte, bool) {
	out := make([]byte, 0, len(tok))
	for _, r := range tok {
		b, ok := runeToByte[r]
		if !ok {
			return nil, false
		}
		out = append(out, b)
	}
	return out, true
}
