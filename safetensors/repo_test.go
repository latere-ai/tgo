// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package safetensors

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// shardedRepo writes a two-shard repo: shard one holds "a", shard two holds
// "b", and the index maps each name to its shard. The returned map is the
// weight map, so a caller can mutate it and rewrite the index.
func shardedRepo(t *testing.T) (dir string, weightMap map[string]string) {
	t.Helper()
	dir = t.TempDir()
	writeConfig(t, dir, `{"architectures":["Qwen3ForCausalLM"]}`)

	hdr, data := build(tensor{"a", F32, []int{2, 2}, 0x11})
	write(t, dir, "model-00001-of-00002.safetensors", hdr, data, -1)
	hdr, data = build(tensor{"b", BF16, []int{3}, 0x22})
	write(t, dir, "model-00002-of-00002.safetensors", hdr, data, -1)

	weightMap = map[string]string{
		"a": "model-00001-of-00002.safetensors",
		"b": "model-00002-of-00002.safetensors",
	}
	writeIndex(t, dir, weightMap)
	return dir, weightMap
}

func writeConfig(t *testing.T, dir, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, configName), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeIndex(t *testing.T, dir string, weightMap map[string]string) {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"metadata":   map[string]any{"total_size": 22},
		"weight_map": weightMap,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, indexName), b, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestOpenRepoSingleShard(t *testing.T) {
	dir := t.TempDir()
	writeConfig(t, dir, `{"hidden_size":64}`)
	writeGood(t, dir, singleName)

	r, err := OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo: %v", err)
	}
	defer r.Close()

	if got := string(r.Config()); got != `{"hidden_size":64}` {
		t.Errorf("Config() = %s", got)
	}
	if got := r.Names(); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("Names() = %v, want [a b]", got)
	}
	e, f, ok := r.Tensor("b")
	if !ok {
		t.Fatal(`Tensor("b") missing`)
	}
	if e.DType != BF16 {
		t.Errorf("Tensor(b).DType = %s, want BF16", e.DType)
	}
	if filepath.Base(f.Path()) != singleName {
		t.Errorf("Tensor(b) resolved to %s, want %s", f.Path(), singleName)
	}
	b, err := f.Bytes("b")
	if err != nil || len(b) != 6 {
		t.Errorf("Bytes(b) = %d bytes, %v", len(b), err)
	}
	if _, _, ok := r.Tensor("absent"); ok {
		t.Error(`Tensor("absent") reported a tensor that is not there`)
	}
}

func TestOpenRepoSharded(t *testing.T) {
	dir, _ := shardedRepo(t)
	r, err := OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo: %v", err)
	}
	defer r.Close()

	if got := r.Names(); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("Names() = %v, want [a b] across both shards", got)
	}
	for name, want := range map[string]string{
		"a": "model-00001-of-00002.safetensors",
		"b": "model-00002-of-00002.safetensors",
	} {
		_, f, ok := r.Tensor(name)
		if !ok {
			t.Fatalf("Tensor(%q) missing", name)
		}
		if got := filepath.Base(f.Path()); got != want {
			t.Errorf("Tensor(%q) resolved to %s, want %s", name, got, want)
		}
	}
}

func TestRepoConfigIsACopy(t *testing.T) {
	dir := t.TempDir()
	writeConfig(t, dir, `{"hidden_size":64}`)
	writeGood(t, dir, singleName)
	r, err := OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo: %v", err)
	}
	defer r.Close()

	cfg := r.Config()
	cfg[0] = 'x'
	if got := string(r.Config()); got[0] != '{' {
		t.Errorf("Config() = %s after a caller wrote to an earlier result", got)
	}
	names := r.Names()
	names[0] = "clobbered"
	if got := r.Names(); got[0] != "a" {
		t.Errorf("Names() = %v after a caller wrote to an earlier result", got)
	}
}

func TestOpenRepoWithoutConfig(t *testing.T) {
	// specs/001-weights.md §6 does not list a missing config.json, and the
	// reader refuses only what that table names.
	dir := t.TempDir()
	writeGood(t, dir, singleName)
	r, err := OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo without config.json: %v", err)
	}
	defer r.Close()
	if r.Config() != nil {
		t.Errorf("Config() = %s, want nil", r.Config())
	}
}

func TestOpenRepoRefusals(t *testing.T) {
	t.Run("no shards at all", func(t *testing.T) {
		dir := t.TempDir()
		writeConfig(t, dir, `{}`)
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "no "+singleName) {
			t.Errorf("OpenRepo of an empty directory = %v", err)
		}
	})

	t.Run("config is not JSON", func(t *testing.T) {
		dir := t.TempDir()
		writeConfig(t, dir, `{not json`)
		writeGood(t, dir, singleName)
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "not valid JSON") {
			t.Errorf("OpenRepo with a broken config = %v", err)
		}
	})

	t.Run("config is unreadable", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.Mkdir(filepath.Join(dir, configName), 0o755); err != nil {
			t.Fatal(err)
		}
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "read config") {
			t.Errorf("OpenRepo with an unreadable config = %v", err)
		}
	})

	t.Run("index is unreadable", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.Mkdir(filepath.Join(dir, indexName), 0o755); err != nil {
			t.Fatal(err)
		}
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "read index") {
			t.Errorf("OpenRepo with an unreadable index = %v", err)
		}
	})

	t.Run("index is not JSON", func(t *testing.T) {
		dir, _ := shardedRepo(t)
		if err := os.WriteFile(filepath.Join(dir, indexName), []byte(`{oops`), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "parse index") {
			t.Errorf("OpenRepo with a broken index = %v", err)
		}
	})

	t.Run("weight_map is absent", func(t *testing.T) {
		dir, _ := shardedRepo(t)
		writeIndex(t, dir, nil)
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "weight_map is empty or absent") {
			t.Errorf("OpenRepo with an empty weight_map = %v", err)
		}
	})

	t.Run("a name in the index has no shard", func(t *testing.T) {
		dir, wm := shardedRepo(t)
		wm["c"] = "model-00003-of-00002.safetensors"
		writeIndex(t, dir, wm)
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "c: shard model-00003-of-00002.safetensors is missing") {
			t.Errorf("OpenRepo with a missing shard = %v, want one naming the tensor", err)
		}
	})

	t.Run("the shard does not hold the name", func(t *testing.T) {
		dir, wm := shardedRepo(t)
		wm["c"] = wm["a"]
		writeIndex(t, dir, wm)
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "c: shard model-00001-of-00002.safetensors has no such tensor") {
			t.Errorf("OpenRepo with an index naming an absent tensor = %v", err)
		}
	})

	t.Run("a shard holds a name the index does not", func(t *testing.T) {
		// The shard is reached because the index names another tensor in it.
		// A weight it also holds and the index never mentions would be loaded
		// as nothing, or silently skipped, which is the incomplete-download
		// case §6 names.
		dir := t.TempDir()
		hdr, data := build(
			tensor{"a", F32, []int{2}, 1},
			tensor{"stowaway", F32, []int{2}, 2},
		)
		write(t, dir, "model-00001-of-00001.safetensors", hdr, data, -1)
		writeIndex(t, dir, map[string]string{"a": "model-00001-of-00001.safetensors"})
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "stowaway in shard model-00001-of-00001.safetensors has no entry in the index") {
			t.Errorf("OpenRepo with an unindexed tensor = %v", err)
		}
	})

	t.Run("the index moves a name to the wrong shard", func(t *testing.T) {
		// Both shards are opened and both hold their tensor, but the index
		// assigns "a" to the shard holding "b". The two directions of the check
		// disagree, and the first to fire reports it.
		dir, wm := shardedRepo(t)
		wm["a"] = wm["b"]
		writeIndex(t, dir, wm)
		_, err := OpenRepo(dir)
		want := "a: shard model-00002-of-00002.safetensors has no such tensor"
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("OpenRepo with a misplaced name = %v, want it to contain %q", err, want)
		}
	})

	t.Run("a shard is malformed", func(t *testing.T) {
		dir, wm := shardedRepo(t)
		if err := os.WriteFile(filepath.Join(dir, wm["a"]), []byte("short"), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "shorter than the 8-byte header length") {
			t.Errorf("OpenRepo with a malformed shard = %v", err)
		}
	})

	t.Run("the single shard is malformed", func(t *testing.T) {
		dir := t.TempDir()
		hdr, data := good()
		hdr["a"].(map[string]any)["dtype"] = "F8_E4M3"
		write(t, dir, singleName, hdr, data, -1)
		_, err := OpenRepo(dir)
		if err == nil || !strings.Contains(err.Error(), "unknown dtype") {
			t.Errorf("OpenRepo with a malformed shard = %v", err)
		}
	})
}

// TestRefusesShardPathEscape covers a case specs/001-weights.md §6 does not
// list. The weight map comes from a downloaded file and its values become file
// paths, so a value carrying a separator reads outside the model directory.
func TestRefusesShardPathEscape(t *testing.T) {
	for _, shard := range []string{
		"../secrets.safetensors",
		"sub/model.safetensors",
		`..\secrets.safetensors`,
		"..",
		".",
		"",
	} {
		t.Run(shard, func(t *testing.T) {
			dir, wm := shardedRepo(t)
			wm["a"] = shard
			writeIndex(t, dir, wm)
			r, err := OpenRepo(dir)
			if err == nil {
				r.Close()
				t.Fatalf("OpenRepo accepted shard %q", shard)
			}
			if !strings.Contains(err.Error(), "is not a file name in the model directory") {
				t.Errorf("OpenRepo error = %v, want a refusal of the shard name", err)
			}
			if !strings.Contains(err.Error(), "a:") {
				t.Errorf("OpenRepo error = %v, want it to name the tensor", err)
			}
		})
	}
}

func TestRepoCloseClosesEveryShard(t *testing.T) {
	dir, _ := shardedRepo(t)
	r, err := OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	for _, f := range r.files {
		if !f.closed {
			t.Errorf("%s stayed open after Repo.Close", f.Path())
		}
		if _, err := f.Bytes("a"); err == nil && f.entries["a"].End > 0 {
			t.Errorf("%s still reads after Repo.Close", f.Path())
		}
	}
	// Close twice: the shards are already closed and must not report it.
	if err := r.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestRepoSharesOneFilePerShard(t *testing.T) {
	// Two tensors in one shard must resolve to the same *File; opening the
	// shard once per tensor would use one descriptor per weight.
	dir := t.TempDir()
	hdr, data := build(
		tensor{"a", F32, []int{2}, 1},
		tensor{"b", F32, []int{2}, 2},
	)
	write(t, dir, "model-00001-of-00001.safetensors", hdr, data, -1)
	writeIndex(t, dir, map[string]string{
		"a": "model-00001-of-00001.safetensors",
		"b": "model-00001-of-00001.safetensors",
	})
	r, err := OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo: %v", err)
	}
	defer r.Close()
	_, fa, _ := r.Tensor("a")
	_, fb, _ := r.Tensor("b")
	if fa != fb {
		t.Errorf("Tensor(a) and Tensor(b) returned different readers for one shard")
	}
	if len(r.files) != 1 {
		t.Errorf("repo holds %d open shards, want 1", len(r.files))
	}
}
