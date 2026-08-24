// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tokenizer

import (
	"strings"
	"testing"
)

// byteID returns the id of the single-byte token for b, which the load-time
// alphabet check guarantees exists.
func byteID(t *testing.T, tk *Tokenizer, b byte) int {
	t.Helper()
	return id(t, tk, string(byteToRune[b]))
}

// TestStreamingEqualsBatch is section 7's row: pushing tokens one at a time
// gives the same string as decoding them together. It runs on input that is not
// valid UTF-8 as well, because Push emits raw bytes and only Flush substitutes.
func TestStreamingEqualsBatch(t *testing.T) {
	tk := load(t)
	for _, s := range []string{
		"", "hello world", "你好世界，这是一个测试。",
		"👨‍👩‍👧‍👦 and 👋🏽 and ❤️",
		"<|im_start|>assistant\n<think>\nthinking<|im_end|>",
		"mixed 你 ascii 好 text",
		"raw " + string([]byte{0xff, 0x00}) + " bytes",
		strings.Repeat("é", 40),
	} {
		ids := tk.Encode(s, true)
		var b strings.Builder
		d := tk.NewDecoder()
		for _, i := range ids {
			b.WriteString(d.Push(i))
		}
		b.WriteString(d.Flush())
		if got, want := b.String(), tk.Decode(ids); got != want {
			t.Errorf("streaming %q = %q, batch = %q", s, got, want)
		}
	}
}

// TestDecoderHoldsBackPartialUTF8 is 002-D4. Without the hold-back every CJK
// character renders as two or three U+FFFD before the real character appears.
func TestDecoderHoldsBackPartialUTF8(t *testing.T) {
	tk := load(t)
	d := tk.NewDecoder()
	// U+4F60 is e4 bd a0.
	if got := d.Push(byteID(t, tk, 0xe4)); got != "" {
		t.Errorf("first byte of a three-byte character emitted %q, want nothing held back", got)
	}
	if got := d.Push(byteID(t, tk, 0xbd)); got != "" {
		t.Errorf("second byte emitted %q", got)
	}
	if got := d.Push(byteID(t, tk, 0xa0)); got != "你" {
		t.Errorf("third byte emitted %q, want 你", got)
	}
	if got := d.Flush(); got != "" {
		t.Errorf("Flush after a complete character = %q", got)
	}
}

// TestFlushEmitsReplacement is section 6's terminal edge: a stream that stops
// mid-character reports it rather than dropping the bytes silently.
func TestFlushEmitsReplacement(t *testing.T) {
	tk := load(t)
	d := tk.NewDecoder()
	d.Push(byteID(t, tk, 0xe4))
	d.Push(byteID(t, tk, 0xbd))
	if got := d.Flush(); got != "�" {
		t.Fatalf("Flush after a truncated character = %q, want U+FFFD", got)
	}
	if got := d.Flush(); got != "" {
		t.Fatalf("second Flush = %q, want empty", got)
	}
	// And the decoder keeps working afterwards.
	if got := d.Push(byteID(t, tk, 'a')); got != "a" {
		t.Fatalf("Push after Flush = %q", got)
	}
}

// TestDecoderDoesNotHoldAnImpossibleByte separates "incomplete" from "invalid".
// 0xFF begins no code point, so holding it back could only delay the U+FFFD
// the caller is going to see anyway, and would stall a stream that never
// produces another byte.
func TestDecoderDoesNotHoldAnImpossibleByte(t *testing.T) {
	tk := load(t)
	d := tk.NewDecoder()
	got := d.Push(byteID(t, tk, 0xff))
	if got != string([]byte{0xff}) {
		t.Fatalf("Push(0xff) = %q, want the byte emitted at once", got)
	}
	if len(d.buf) != 0 {
		t.Fatalf("0xff was held back: %v", d.buf)
	}
}

func TestDecoderIgnoresUnclaimedIDs(t *testing.T) {
	tk := load(t)
	d := tk.NewDecoder()
	if got := d.Push(tk.VocabSize() + 1); got != "" {
		t.Errorf("Push of an id out of range = %q", got)
	}
	if got := d.Flush(); got != "" {
		t.Errorf("Flush = %q", got)
	}
}

func TestDecodersAreIndependent(t *testing.T) {
	tk := load(t)
	a, b := tk.NewDecoder(), tk.NewDecoder()
	a.Push(byteID(t, tk, 0xe4))
	if got := b.Push(byteID(t, tk, 'x')); got != "x" {
		t.Fatalf("one decoder's held bytes reached another: %q", got)
	}
}

func TestEmitLen(t *testing.T) {
	for _, tc := range []struct {
		in   []byte
		want int
		why  string
	}{
		{nil, 0, "empty"},
		{[]byte("abc"), 3, "ascii"},
		{[]byte("你"), 3, "a complete three-byte character"},
		{[]byte{0xe4}, 0, "a lead byte alone is completable"},
		{[]byte{0xe4, 0xbd}, 0, "two of three bytes"},
		{[]byte("a\xe4\xbd"), 1, "the complete prefix is emitted and the rest held"},
		{[]byte{0xff}, 1, "0xff begins nothing and is emitted"},
		{[]byte{0xbd}, 1, "a stray continuation byte is emitted"},
		{[]byte("\xe4\xbda"), 3, "a broken sequence followed by more is all emitted"},
		{[]byte{0xf0, 0x9f, 0x91}, 0, "three of a four-byte emoji"},
		{[]byte("👋"), 4, "a complete emoji"},
		// Bytes that can never begin a code point must not be held: a stream
		// that ends on one would otherwise stall until Flush.
		{[]byte{0xc0}, 1, "an overlong two-byte lead"},
		{[]byte{0xe0, 0x80}, 2, "a lead whose continuation is out of range"},
		{[]byte{0xf5}, 1, "a lead above the Unicode maximum"},
	} {
		if got := emitLen(tc.in); got != tc.want {
			t.Errorf("emitLen(% x) = %d, want %d (%s)", tc.in, got, tc.want, tc.why)
		}
	}
}
