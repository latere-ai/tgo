// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package safetensors

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The file names a Hugging Face checkpoint directory uses
// (specs/001-weights.md §1).
const (
	configName = "config.json"
	singleName = "model.safetensors"
	indexName  = "model.safetensors.index.json"
)

// Repo is a model directory: the config and one or more safetensors shards,
// resolved so that a tensor name reaches the shard that holds it.
type Repo struct {
	dir     string
	config  json.RawMessage
	files   map[string]*File // shard file name to the open shard
	shardOf map[string]string
	names   []string
}

// index is the shape of model.safetensors.index.json that matters here. The
// "metadata" block carries a total byte count and nothing this reader needs.
type index struct {
	WeightMap map[string]string `json:"weight_map"`
}

// OpenRepo opens every shard of the model directory at dir and resolves the
// index, if there is one. The returned Repo must be closed.
//
// The directory holds either one model.safetensors or a set of shards with
// model.safetensors.index.json beside them. Where the index is present it is
// authoritative on which shards exist: the shard set is never inferred from
// file names, for the same reason 001-D4 refuses to infer which tensors
// transpose from tensor names.
func OpenRepo(dir string) (*Repo, error) {
	cfg, err := readConfig(dir)
	if err != nil {
		return nil, err
	}
	r := &Repo{
		dir:     dir,
		config:  cfg,
		files:   make(map[string]*File),
		shardOf: make(map[string]string),
	}
	idxPath := filepath.Join(dir, indexName)
	idxRaw, err := os.ReadFile(idxPath)
	switch {
	case err == nil:
		if err := r.openSharded(idxPath, idxRaw); err != nil {
			_ = r.Close()
			return nil, err
		}
	case errors.Is(err, os.ErrNotExist):
		if err := r.openSingle(); err != nil {
			_ = r.Close()
			return nil, err
		}
	default:
		return nil, fmt.Errorf("safetensors: %s: read index: %w", dir, err)
	}
	r.names = make([]string, 0, len(r.shardOf))
	for name := range r.shardOf {
		r.names = append(r.names, name)
	}
	sort.Strings(r.names)
	return r, nil
}

// readConfig reads config.json if it is there. A missing config is not a
// refusal: specs/001-weights.md §6 lists what the reader refuses, and the
// config is not on it. A config that is present but is not JSON is refused,
// because handing a caller bytes that will fail to parse later reports the
// problem in the wrong place.
func readConfig(dir string) (json.RawMessage, error) {
	b, err := os.ReadFile(filepath.Join(dir, configName))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("safetensors: %s: read config: %w", dir, err)
	}
	if !json.Valid(b) {
		return nil, fmt.Errorf("safetensors: %s: %s is not valid JSON", dir, configName)
	}
	return b, nil
}

// openSingle opens the unsharded model.safetensors.
func (r *Repo) openSingle() error {
	f, err := Open(filepath.Join(r.dir, singleName))
	if errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("safetensors: %s: no %s and no %s", r.dir, singleName, indexName)
	}
	if err != nil {
		return err
	}
	r.files[singleName] = f
	for _, name := range f.names {
		r.shardOf[name] = singleName
	}
	return nil
}

// openSharded resolves the weight map and checks the two directions of
// agreement between the index and the shards (specs/001-weights.md §6).
func (r *Repo) openSharded(path string, raw []byte) error {
	var idx index
	if err := json.Unmarshal(raw, &idx); err != nil {
		return fmt.Errorf("safetensors: %s: parse index: %w", path, err)
	}
	if len(idx.WeightMap) == 0 {
		return fmt.Errorf("safetensors: %s: weight_map is empty or absent", path)
	}

	// Walk the weight map in name order. A checkpoint with two faults must
	// refuse with the same message every time it is opened.
	names := make([]string, 0, len(idx.WeightMap))
	for name := range idx.WeightMap {
		names = append(names, name)
	}
	sort.Strings(names)

	// Open each shard the index names. The value is a file name in this
	// directory and nothing else: it comes from a downloaded file, so a value
	// carrying a path separator would let the index read outside the model
	// directory.
	for _, name := range names {
		shard := idx.WeightMap[name]
		if err := checkShardName(path, name, shard); err != nil {
			return err
		}
		if _, ok := r.files[shard]; ok {
			continue
		}
		f, err := Open(filepath.Join(r.dir, shard))
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("safetensors: %s: %s: shard %s is missing", path, name, shard)
		}
		if err != nil {
			return err
		}
		r.files[shard] = f
	}

	// A name the index places in a shard that does not hold it: an incomplete
	// or mismatched download.
	for _, name := range names {
		shard := idx.WeightMap[name]
		if _, ok := r.files[shard].entries[name]; !ok {
			return fmt.Errorf("safetensors: %s: %s: shard %s has no such tensor", path, name, shard)
		}
		r.shardOf[name] = shard
	}

	// And the other direction: a tensor present in a shard that the index does
	// not assign to that shard. Loading it would use weights nothing named.
	shards := make([]string, 0, len(r.files))
	for shard := range r.files {
		shards = append(shards, shard)
	}
	sort.Strings(shards)
	for _, shard := range shards {
		for _, name := range r.files[shard].names {
			if idx.WeightMap[name] != shard {
				return fmt.Errorf("safetensors: %s: %s in shard %s has no entry in the index", path, name, shard)
			}
		}
	}
	return nil
}

// checkShardName refuses a weight_map value that is not a plain file name in
// the model directory.
func checkShardName(path, tensor, shard string) error {
	if shard == "" || shard == "." || shard == ".." ||
		strings.ContainsAny(shard, `/\`) || filepath.Base(shard) != shard {
		return fmt.Errorf("safetensors: %s: %s: shard %q is not a file name in the model directory", path, tensor, shard)
	}
	return nil
}

// Config is the raw config.json, or nil if the directory has none. The result
// is a fresh copy.
func (r *Repo) Config() json.RawMessage {
	if r.config == nil {
		return nil
	}
	out := make(json.RawMessage, len(r.config))
	copy(out, r.config)
	return out
}

// Names lists every tensor in the repo, sorted, across all shards. The result
// is a fresh slice.
func (r *Repo) Names() []string {
	out := make([]string, len(r.names))
	copy(out, r.names)
	return out
}

// Tensor resolves a tensor name through the index to its entry and the shard
// that holds it. The File stays owned by the Repo; closing the Repo closes it.
func (r *Repo) Tensor(name string) (Entry, *File, bool) {
	shard, ok := r.shardOf[name]
	if !ok {
		return Entry{}, nil, false
	}
	// shardOf is built from the shard headers, so the lookup cannot miss.
	f := r.files[shard]
	e, _ := f.Entry(name)
	return e, f, true
}

// Close closes every shard. It reports all failures rather than the first,
// because a descriptor left open is not visible from the caller's side.
func (r *Repo) Close() error {
	var errs []error
	for _, f := range r.files {
		if err := f.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
