// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package safetensors reads the safetensors checkpoint format.
//
// A checkpoint is a file someone downloaded, so this package treats the header
// as hostile input rather than as a description of the file it came with. Every
// row of specs/001-weights.md §6 is checked before Open returns, and each
// refusal names the tensor that caused it: a header that disagrees with itself
// is the only evidence a reader has that the bytes behind it are not the
// weights it was asked for.
//
// The reader returns raw planes. Conversion between the file dtype and the
// device dtype, the transpose that every projection weight needs, and
// quantization all belong to the loader, which knows the target precision
// (001-D1, 001-D3). Keeping them apart is what makes this package testable
// against a synthesised header with no model and no device present.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// DType is a safetensors element type, spelled as the file spells it.
//
// The set is closed, from the grammar in specs/001-weights.md §1.1. An
// unrecognised string is refused rather than guessed at, because a reader that
// picks a plausible width for an unknown name reads the right bytes as the
// wrong numbers and reports nothing (001-D6).
type DType string

// The element types specs/001-weights.md §1.1 admits.
const (
	F64  DType = "F64"
	F32  DType = "F32"
	F16  DType = "F16"
	BF16 DType = "BF16"
	I64  DType = "I64"
	I32  DType = "I32"
	I16  DType = "I16"
	I8   DType = "I8"
	U8   DType = "U8"
	BOOL DType = "BOOL"
)

// dtypeSize is the width in bytes of every dtype this package accepts.
// Membership in this map is the definition of "known dtype".
var dtypeSize = map[DType]int64{
	F64: 8, F32: 4, F16: 2, BF16: 2,
	I64: 8, I32: 4, I16: 2, I8: 1,
	U8: 1, BOOL: 1,
}

// Size is the width of one element in bytes, or 0 if the dtype is not one this
// package accepts. A caller that reached an Entry through this package always
// gets a nonzero width; the zero is for a DType built from an arbitrary string.
func (d DType) Size() int { return int(dtypeSize[d]) }

// Entry is one tensor's header record. Begin and End are byte offsets relative
// to the end of the header, half-open, as the file states them.
type Entry struct {
	DType      DType
	Shape      []int
	Begin, End int64
}

// metadataKey is the reserved key that carries the file's string map. It is not
// a tensor, and a reader that treats it as one fails on every real checkpoint
// (specs/001-weights.md §1.1).
const metadataKey = "__metadata__"

// maxHeaderBytes caps the JSON header. The 8-byte length prefix is read before
// anything else is known about the file, so without a cap a crafted prefix over
// a sparse file turns one 8-byte read into an arbitrary allocation. Real
// headers are a few megabytes at most; this is a variable so the test can lower
// it rather than write a large file.
var maxHeaderBytes int64 = 100 << 20

// maxInt is the largest value an int holds on the building platform. tgo
// cross-compiles to 32-bit targets, where a length that fits in an int64 does
// not fit in a slice index.
const maxInt = int64(^uint(0) >> 1)

// maxInt64 is the largest int64, spelled without importing math for one value.
const maxInt64 = int64(^uint64(0) >> 1)

// File is an open safetensors file with its header parsed. It holds an open
// descriptor and reads tensor bytes on demand; no tensor data is read by Open.
type File struct {
	path      string
	f         *os.File
	dataStart int64
	dataLen   int64
	entries   map[string]Entry
	names     []string
	closed    bool
}

// rawEntry is the on-disk shape of one header record. Unknown fields are
// ignored: writers add keys, and a reader that refuses them refuses files that
// are otherwise correct.
type rawEntry struct {
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

// Open reads and validates the header of the safetensors file at path. It reads
// no tensor data. The returned File must be closed.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	sf, err := parse(f, path, fi.Size())
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	return sf, nil
}

// parse validates the header of an already-open file of the given size.
func parse(f *os.File, path string, size int64) (*File, error) {
	if size < 8 {
		return nil, fmt.Errorf("safetensors: %s: file is %d bytes, shorter than the 8-byte header length", path, size)
	}
	var prefix [8]byte
	if _, err := f.ReadAt(prefix[:], 0); err != nil {
		return nil, fmt.Errorf("safetensors: %s: read header length: %w", path, err)
	}
	// Compare in uint64 before any conversion: on a 32-bit target a crafted
	// length truncates into a small, plausible int.
	n := binary.LittleEndian.Uint64(prefix[:])
	if n > uint64(size-8) {
		return nil, fmt.Errorf("safetensors: %s: header length %d exceeds the %d bytes that follow it", path, n, size-8)
	}
	if n > uint64(maxHeaderBytes) {
		return nil, fmt.Errorf("safetensors: %s: header length %d exceeds the %d-byte limit", path, n, maxHeaderBytes)
	}
	buf := make([]byte, int64(n))
	if _, err := f.ReadAt(buf, 8); err != nil {
		return nil, fmt.Errorf("safetensors: %s: read header: %w", path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: %s: parse header: %w", path, err)
	}

	sf := &File{
		path:      path,
		f:         f,
		dataStart: 8 + int64(n),
		dataLen:   size - 8 - int64(n),
		entries:   make(map[string]Entry, len(raw)),
	}
	// Validate in name order rather than map order. A file with two faults must
	// refuse with the same message every time it is opened, or a user comparing
	// two runs of the same load sees two different problems.
	keys := make([]string, 0, len(raw))
	for name := range raw {
		keys = append(keys, name)
	}
	sort.Strings(keys)
	for _, name := range keys {
		if name == metadataKey {
			var meta map[string]string
			if err := json.Unmarshal(raw[name], &meta); err != nil {
				return nil, fmt.Errorf("safetensors: %s: %s is not a map of strings: %w", path, metadataKey, err)
			}
			continue
		}
		e, err := sf.entry(name, raw[name])
		if err != nil {
			return nil, err
		}
		sf.entries[name] = e
		sf.names = append(sf.names, name) // keys is sorted, so names is
	}
	if err := sf.checkOverlap(); err != nil {
		return nil, err
	}
	return sf, nil
}

// entry validates one header record against the data region.
func (f *File) entry(name string, rec json.RawMessage) (Entry, error) {
	var r rawEntry
	if err := json.Unmarshal(rec, &r); err != nil {
		return Entry{}, fmt.Errorf("safetensors: %s: %s: parse entry: %w", f.path, name, err)
	}
	width, ok := dtypeSize[DType(r.DType)]
	if !ok {
		return Entry{}, fmt.Errorf("safetensors: %s: %s: unknown dtype %q", f.path, name, r.DType)
	}
	if len(r.DataOffsets) != 2 {
		return Entry{}, fmt.Errorf("safetensors: %s: %s: data_offsets has %d elements, want 2", f.path, name, len(r.DataOffsets))
	}
	begin, end := r.DataOffsets[0], r.DataOffsets[1]
	if end < begin {
		return Entry{}, fmt.Errorf("safetensors: %s: %s: data_offsets end %d is before begin %d", f.path, name, end, begin)
	}
	if begin < 0 || end > f.dataLen {
		return Entry{}, fmt.Errorf("safetensors: %s: %s: data_offsets [%d,%d) lie outside the %d-byte data region", f.path, name, begin, end, f.dataLen)
	}

	// The element count is multiplied in int64 with an explicit overflow check.
	// A header states its own shape, so [1<<40, 1<<40] is one edit away from a
	// product that wraps to a small number matching a short byte range, and the
	// same edit truncates on a 32-bit target where int is narrower than the
	// dimension. One condition covers both.
	count := int64(1)
	shape := make([]int, len(r.Shape))
	for i, d := range r.Shape {
		if d < 0 {
			return Entry{}, fmt.Errorf("safetensors: %s: %s: shape dimension %d is negative (%d)", f.path, name, i, d)
		}
		if d > maxInt || (d != 0 && count > maxInt64/d) {
			return Entry{}, fmt.Errorf("safetensors: %s: %s: shape %v is larger than this platform can index", f.path, name, r.Shape)
		}
		shape[i] = int(d)
		count *= d
	}
	if count > maxInt64/width {
		return Entry{}, fmt.Errorf("safetensors: %s: %s: shape %v of dtype %s is larger than this platform can index", f.path, name, r.Shape, r.DType)
	}
	if want := count * width; want != end-begin {
		return Entry{}, fmt.Errorf("safetensors: %s: %s: shape %v of dtype %s needs %d bytes, but data_offsets span %d", f.path, name, r.Shape, r.DType, want, end-begin)
	}
	return Entry{DType: DType(r.DType), Shape: shape, Begin: begin, End: end}, nil
}

// checkOverlap refuses two tensors that share a byte. No writer produces that,
// so it is either corruption or a header crafted to make one tensor's bytes be
// read as another's (specs/001-weights.md §6).
func (f *File) checkOverlap() error {
	type span struct {
		name       string
		begin, end int64
	}
	spans := make([]span, 0, len(f.entries))
	for _, name := range f.names {
		e := f.entries[name]
		// An empty tensor owns no bytes and cannot alias one. Including it
		// would report an overlap between a zero-length range and whatever
		// range starts at the same offset.
		if e.End == e.Begin {
			continue
		}
		spans = append(spans, span{name, e.Begin, e.End})
	}
	// Stable, so that two tensors sharing a begin offset are always reported in
	// the same order: spans is built from f.names, which is sorted.
	sort.SliceStable(spans, func(i, j int) bool { return spans[i].begin < spans[j].begin })
	for i := 1; i < len(spans); i++ {
		if spans[i].begin < spans[i-1].end {
			return fmt.Errorf("safetensors: %s: %s [%d,%d) overlaps %s [%d,%d)", f.path,
				spans[i].name, spans[i].begin, spans[i].end,
				spans[i-1].name, spans[i-1].begin, spans[i-1].end)
		}
	}
	return nil
}

// Path is the file this reader was opened from. It is what the refusals name,
// and what a loader reports when a tensor is not where the index said.
func (f *File) Path() string { return f.path }

// Names lists the tensors in the file, sorted. The result is a fresh slice.
func (f *File) Names() []string {
	out := make([]string, len(f.names))
	copy(out, f.names)
	return out
}

// Entry returns the header record for name. The Shape is a fresh slice.
func (f *File) Entry(name string) (Entry, bool) {
	e, ok := f.entries[name]
	if !ok {
		return Entry{}, false
	}
	shape := make([]int, len(e.Shape))
	copy(shape, e.Shape)
	e.Shape = shape
	return e, true
}

// Bytes reads the raw plane of one tensor. No conversion happens here: the
// bytes are the file's, in the file's dtype and the file's layout.
func (f *File) Bytes(name string) ([]byte, error) {
	e, ok := f.entries[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: %s: no tensor named %q", f.path, name)
	}
	n := e.End - e.Begin
	if n > maxInt {
		return nil, fmt.Errorf("safetensors: %s: %s: %d bytes do not fit in memory on this platform", f.path, name, n)
	}
	b := make([]byte, int(n))
	if n == 0 {
		return b, nil
	}
	// ReadAt rather than Seek plus Read: there is no shared file offset to
	// race on, so a loader may read several tensors concurrently.
	if _, err := f.f.ReadAt(b, f.dataStart+e.Begin); err != nil {
		return nil, fmt.Errorf("safetensors: %s: %s: read plane: %w", f.path, name, err)
	}
	return b, nil
}

// Close releases the descriptor. It is safe to call more than once so that a
// caller closing a partly-built Repo does not have to track which shards it
// already closed.
func (f *File) Close() error {
	if f.closed {
		return nil
	}
	f.closed = true
	return f.f.Close()
}
