// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// The builder below exists so that every row of specs/001-weights.md §6 is a
// one-line mutation of a good file. The header is a map[string]any rather than
// a typed struct on purpose: a struct cannot express "dtype is the string
// XF32", "data_offsets has three elements", or "shape is negative", which are
// exactly the cases the reader must refuse.

// tensor is one tensor to place in a synthesised file.
type tensor struct {
	name  string
	dtype DType
	shape []int
	fill  byte // every byte of the plane gets this value
}

// build lays the tensors out contiguously in order and returns the header map
// and the data region. The caller mutates the header before writing it.
func build(ts ...tensor) (map[string]any, []byte) {
	hdr := map[string]any{}
	var data []byte
	for _, t := range ts {
		n := int64(1)
		for _, d := range t.shape {
			n *= int64(d)
		}
		size := n * dtypeSize[t.dtype]
		begin := int64(len(data))
		for range size {
			data = append(data, t.fill)
		}
		hdr[t.name] = map[string]any{
			"dtype":        string(t.dtype),
			"shape":        t.shape,
			"data_offsets": []int64{begin, begin + size},
		}
	}
	return hdr, data
}

// good is the file every refusal test starts from: two tensors of different
// dtypes and a metadata block, which is not a tensor.
func good() (map[string]any, []byte) {
	hdr, data := build(
		tensor{"a", F32, []int{2, 2}, 0x11},
		tensor{"b", BF16, []int{3}, 0x22},
	)
	hdr[metadataKey] = map[string]string{"format": "pt"}
	return hdr, data
}

// write serialises a header and a data region into dir/name and returns the
// path. A negative lenOverride writes the true header length; any other value
// is written in its place, which is how the truncated-header rows are built.
func write(t *testing.T, dir, name string, hdr map[string]any, data []byte, lenOverride int64) string {
	t.Helper()
	js, err := json.Marshal(hdr)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	return writeRaw(t, dir, name, js, data, lenOverride)
}

// writeRaw is write for a header that is not a JSON object.
func writeRaw(t *testing.T, dir, name string, js, data []byte, lenOverride int64) string {
	t.Helper()
	n := uint64(len(js))
	if lenOverride >= 0 {
		n = uint64(lenOverride)
	}
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, n)
	buf = append(buf, js...)
	buf = append(buf, data...)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, buf, 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return path
}

// writeGood puts a well-formed single-shard file in dir.
func writeGood(t *testing.T, dir, name string) string {
	t.Helper()
	hdr, data := good()
	return write(t, dir, name, hdr, data, -1)
}

func TestOpenReadsHeaderAndPlanes(t *testing.T) {
	path := writeGood(t, t.TempDir(), singleName)
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = f.Close() }()

	if got := f.Names(); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("Names() = %v, want [a b]; __metadata__ must not be a tensor", got)
	}
	if f.Path() != path {
		t.Errorf("Path() = %q, want %q", f.Path(), path)
	}

	e, ok := f.Entry("a")
	if !ok {
		t.Fatal(`Entry("a") missing`)
	}
	if e.DType != F32 || len(e.Shape) != 2 || e.Shape[0] != 2 || e.Shape[1] != 2 {
		t.Errorf("Entry(a) = %+v, want F32 [2 2]", e)
	}
	if e.Begin != 0 || e.End != 16 {
		t.Errorf("Entry(a) offsets = [%d,%d), want [0,16)", e.Begin, e.End)
	}

	b, err := f.Bytes("a")
	if err != nil {
		t.Fatalf("Bytes(a): %v", err)
	}
	if len(b) != 16 {
		t.Fatalf("Bytes(a) = %d bytes, want 16", len(b))
	}
	for i, v := range b {
		if v != 0x11 {
			t.Fatalf("Bytes(a)[%d] = %#x, want 0x11", i, v)
		}
	}
	// The second plane proves the offsets are relative to the end of the
	// header, not to the start of the file.
	b, err = f.Bytes("b")
	if err != nil {
		t.Fatalf("Bytes(b): %v", err)
	}
	if len(b) != 6 || b[0] != 0x22 || b[5] != 0x22 {
		t.Fatalf("Bytes(b) = %#v, want six 0x22 bytes", b)
	}
}

func TestNamesAndShapeAreCopies(t *testing.T) {
	f, err := Open(writeGood(t, t.TempDir(), singleName))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = f.Close() }()

	names := f.Names()
	names[0] = "clobbered"
	if got := f.Names(); got[0] != "a" {
		t.Errorf("Names() = %v after a caller wrote to an earlier result", got)
	}
	e, _ := f.Entry("a")
	e.Shape[0] = 99
	if e2, _ := f.Entry("a"); e2.Shape[0] != 2 {
		t.Errorf("Entry(a).Shape = %v after a caller wrote to an earlier result", e2.Shape)
	}
}

func TestEmptyTensorIsNotAnOverlap(t *testing.T) {
	// A zero-element tensor owns no bytes, and its begin equals the begin of
	// whatever follows it. That is legal, and an overlap check that does not
	// skip empty ranges refuses it.
	hdr, data := build(
		tensor{"empty", F32, []int{0, 4}, 0},
		tensor{"a", F32, []int{2}, 0x33},
	)
	f, err := Open(write(t, t.TempDir(), singleName, hdr, data, -1))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = f.Close() }()
	b, err := f.Bytes("empty")
	if err != nil {
		t.Fatalf("Bytes(empty): %v", err)
	}
	if len(b) != 0 {
		t.Errorf("Bytes(empty) = %d bytes, want 0", len(b))
	}
}

func TestScalarTensor(t *testing.T) {
	// An empty shape is a scalar with one element, not a refusal.
	hdr, data := build(tensor{"s", I64, nil, 0x44})
	f, err := Open(write(t, t.TempDir(), singleName, hdr, data, -1))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = f.Close() }()
	if e, _ := f.Entry("s"); e.End-e.Begin != 8 {
		t.Errorf("scalar I64 spans %d bytes, want 8", e.End-e.Begin)
	}
}

func TestBytesUnknownName(t *testing.T) {
	f, err := Open(writeGood(t, t.TempDir(), singleName))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = f.Close() }()
	if _, ok := f.Entry("nope"); ok {
		t.Error(`Entry("nope") reported a tensor that is not there`)
	}
	_, err = f.Bytes("nope")
	if err == nil || !strings.Contains(err.Error(), "nope") {
		t.Errorf("Bytes(nope) error = %v, want one naming the tensor", err)
	}
}

func TestOpenMissingFile(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "absent.safetensors")); err == nil {
		t.Fatal("Open of a missing file succeeded")
	}
}

func TestOpenDirectory(t *testing.T) {
	// A directory opens but does not stat as a readable header. The check is
	// here because Open reads through the descriptor rather than the path.
	if _, err := Open(t.TempDir()); err == nil {
		t.Fatal("Open of a directory succeeded")
	}
}

func TestCloseIsIdempotent(t *testing.T) {
	f, err := Open(writeGood(t, t.TempDir(), singleName))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestBytesAfterTruncation(t *testing.T) {
	// The header was valid when it was read. Losing the data underneath it
	// afterwards must report a read failure naming the tensor, not a short
	// plane the loader would convert as if it were whole.
	path := writeGood(t, t.TempDir(), singleName)
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = f.Close() }()
	if err := os.Truncate(path, 8); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	if _, err := f.Bytes("a"); err == nil || !strings.Contains(err.Error(), "a:") {
		t.Errorf("Bytes after truncation = %v, want a read error naming the tensor", err)
	}
}

// TestRefusals covers specs/001-weights.md §6. Each case mutates a good file
// and asserts both that the reader refuses and that the message names the
// tensor or the quantity at fault: a refusal a user cannot act on sends them
// to re-download a checkpoint that was never the problem.
func TestRefusals(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(t *testing.T, hdr map[string]any) (data []byte, lenOverride int64)
		want   string
	}{{
		name: "header length exceeds the file",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			return data, 1 << 20
		},
		want: "exceeds the",
	}, {
		name: "header length over the size limit",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			// Lower the cap instead of writing a hundred-megabyte file. The cap
			// exists because the length prefix is read before anything else is
			// known, so it must hold against a file that is mostly a hole.
			old := maxHeaderBytes
			maxHeaderBytes = 4
			t.Cleanup(func() { maxHeaderBytes = old })
			_, data := good()
			return data, -1
		},
		want: "byte limit",
	}, {
		name: "end before begin",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"].(map[string]any)["data_offsets"] = []int64{16, 0}
			return data, -1
		},
		want: "a: data_offsets end 0 is before begin 16",
	}, {
		name: "offsets past the data region",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["b"].(map[string]any)["data_offsets"] = []int64{16, 1024}
			return data, -1
		},
		want: "b: data_offsets [16,1024) lie outside",
	}, {
		name: "negative begin",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"].(map[string]any)["data_offsets"] = []int64{-16, 0}
			return data, -1
		},
		want: "a: data_offsets [-16,0) lie outside",
	}, {
		name: "data_offsets has one element",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"].(map[string]any)["data_offsets"] = []int64{0}
			return data, -1
		},
		want: "a: data_offsets has 1 elements",
	}, {
		name: "overlapping tensors",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			// b claims four of a's sixteen bytes, and its own size still
			// agrees with its shape, so only the overlap check catches it.
			hdr["b"].(map[string]any)["data_offsets"] = []int64{12, 18}
			hdr["b"].(map[string]any)["shape"] = []int{3}
			return data, -1
		},
		want: "b [12,18) overlaps a [0,16)",
	}, {
		name: "shape disagrees with the byte span",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"].(map[string]any)["shape"] = []int{2, 3}
			return data, -1
		},
		want: "a: shape [2 3] of dtype F32 needs 24 bytes, but data_offsets span 16",
	}, {
		name: "unknown dtype",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"].(map[string]any)["dtype"] = "F8_E4M3"
			return data, -1
		},
		want: `a: unknown dtype "F8_E4M3"`,
	}, {
		name: "negative dimension",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"].(map[string]any)["shape"] = []int{2, -2}
			return data, -1
		},
		want: "a: shape dimension 1 is negative (-2)",
	}, {
		name: "shape product overflows",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"].(map[string]any)["shape"] = []int64{1 << 40, 1 << 40}
			return data, -1
		},
		want: "a: shape [1099511627776 1099511627776] is larger than this platform can index",
	}, {
		name: "element count times width overflows",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"].(map[string]any)["dtype"] = string(F64)
			hdr["a"].(map[string]any)["shape"] = []int64{1 << 62}
			return data, -1
		},
		// Pinned to the dtype-bearing message: the loop guard above emits one
		// without a dtype, so the two overflow rows cannot pass for each
		// other's reason. Left loose, both are satisfied by the size-mismatch
		// message that a wrapped product produces.
		want: "a: shape [4611686018427387904] of dtype F64 is larger than this platform can index",
	}, {
		name: "entry is not an object",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr["a"] = 7
			return data, -1
		},
		want: "a: parse entry",
	}, {
		name: "metadata is not a map of strings",
		mutate: func(t *testing.T, hdr map[string]any) ([]byte, int64) {
			_, data := good()
			hdr[metadataKey] = map[string]int{"total_size": 22}
			return data, -1
		},
		want: "__metadata__ is not a map of strings",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hdr, _ := good()
			data, lenOverride := tc.mutate(t, hdr)
			path := write(t, t.TempDir(), singleName, hdr, data, lenOverride)
			f, err := Open(path)
			if err == nil {
				_ = f.Close()
				t.Fatalf("Open accepted %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Open error = %q, want it to contain %q", err, tc.want)
			}
		})
	}
}

func TestRefusesShortAndUnparseableFiles(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name string
		body []byte
		want string
	}{
		{"empty", nil, "shorter than the 8-byte header length"},
		{"seven bytes", []byte("1234567"), "shorter than the 8-byte header length"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name)
			if err := os.WriteFile(path, tc.body, 0o600); err != nil {
				t.Fatal(err)
			}
			_, err := Open(path)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Open error = %v, want it to contain %q", err, tc.want)
			}
		})
	}

	// A header that is valid JSON but not an object, and one that is not JSON.
	_, data := good()
	for _, js := range [][]byte{[]byte(`[1,2,3]`), []byte(`{oops`)} {
		path := writeRaw(t, dir, "j"+string(js[0:1]), js, data, -1)
		if _, err := Open(path); err == nil || !strings.Contains(err.Error(), "parse header") {
			t.Errorf("Open(%s) error = %v, want a header parse failure", js, err)
		}
	}
}

func TestDTypeSize(t *testing.T) {
	want := map[DType]int{
		F64: 8, F32: 4, F16: 2, BF16: 2,
		I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1,
		DType("F8_E4M3"): 0,
	}
	for d, w := range want {
		if got := d.Size(); got != w {
			t.Errorf("DType(%q).Size() = %d, want %d", d, got, w)
		}
	}
}

func TestHeaderPaddingIsAccepted(t *testing.T) {
	// The reference writer pads the JSON with spaces so the data region starts
	// at an 8-byte boundary, and counts the padding inside the header length.
	// Without this test the whole suite agrees with a writer that does not
	// exist: every real checkpoint is padded.
	hdr, data := good()
	js, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	for (8+len(js))%8 != 0 {
		js = append(js, ' ')
	}
	js = append(js, "        "...) // a whole extra block of padding
	path := writeRaw(t, t.TempDir(), singleName, js, data, -1)

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open of a padded header: %v", err)
	}
	defer func() { _ = f.Close() }()
	e, ok := f.Entry("a")
	if !ok || e.Begin != 0 {
		t.Fatalf("Entry(a) = %+v, %v; offsets stay relative to the end of the padded header", e, ok)
	}
	b, err := f.Bytes("a")
	if err != nil {
		t.Fatalf("Bytes(a): %v", err)
	}
	if len(b) != 16 || b[0] != 0x11 || b[15] != 0x11 {
		t.Errorf("Bytes(a) = %#v, want sixteen 0x11 bytes: the padding was not counted", b)
	}
}

func TestConcurrentBytes(t *testing.T) {
	// Bytes reads through ReadAt precisely so that a loader can read planes in
	// parallel. Under -race, this is what makes that claim evidence.
	f, err := Open(writeGood(t, t.TempDir(), singleName))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = f.Close() }()

	want := map[string]byte{"a": 0x11, "b": 0x22}
	var wg sync.WaitGroup
	for range 8 {
		for name, fill := range want {
			wg.Add(1)
			go func(name string, fill byte) {
				defer wg.Done()
				b, err := f.Bytes(name)
				if err != nil {
					t.Errorf("Bytes(%s): %v", name, err)
					return
				}
				for _, v := range b {
					if v != fill {
						t.Errorf("Bytes(%s) has %#x, want %#x", name, v, fill)
						return
					}
				}
			}(name, fill)
		}
	}
	wg.Wait()
}
