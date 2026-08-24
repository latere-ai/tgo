// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package model holds the registry, the config, and the weight map that stand
// between a checkpoint directory and a graph.
//
// Three things live here, and the reason they live together is that all three
// are decided by the same line of config.json:
//
//   - [Config] is specs/004-model-graph.md §5 parsed. Every shape the graph
//     uses comes from the file; nothing depends on a number written in a spec
//     (004-D8).
//   - [WeightSpec] is §4's table as data. For each checkpoint tensor it names
//     the port it binds, the shape the file must hold, whether it transposes,
//     and whether its output channels permute for the rotary convention
//     (004-D9).
//   - [Register] and [Open] are §6's registry, keyed on architectures[0]. An
//     unknown architecture is refused with the list of known ones rather than
//     run through a generic path, because a model run with the wrong
//     architecture produces fluent wrong text and nobody finds out (004-D2).
//
// The forward pass is not here. specs/004-model-graph.md §6 states it as a
// fourth [Builder] method, Forward(g *nn.Graph, in Inputs) *tensor.Tensor, and
// it is the one row of §6 this package does not yet carry. The blocks it would
// call now exist in tgo/nn; adding it extends [Builder] with one method and
// changes nothing else here.
package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/latere-ai/tgo/chat"
)

// configName is the file Open reads. A checkpoint directory always has one; a
// directory without it is not a model directory (specs/001-weights.md §1).
const configName = "config.json"

// Builder is what one architecture contributes: the parsed config, the weight
// map that config implies, and the prompt format the model was tuned on.
//
// specs/004-model-graph.md §6 states a fourth method,
// Forward(g *nn.Graph, in Inputs) *tensor.Tensor, which this interface does not
// yet declare. Declaring it would make model import nn, and nn imports accel's
// tensor layer, so the registry — which a caller uses to ask what a directory
// claims to be — would pull in a device-facing dependency to answer a question
// about config.json. It is added with the forward pass itself.
type Builder interface {
	// Config is the parsed config.json. Fields a specific architecture adds
	// beyond §5's table are reachable through the concrete builder type.
	Config() *Config

	// Weights is the weight map for this config: specs/004-model-graph.md §4,
	// with the layer templating expanded and the shapes filled in.
	Weights() []WeightSpec

	// Template renders a conversation into the exact prompt bytes this model
	// was trained on (specs/003-chat-template.md).
	Template() chat.Renderer
}

// registry maps an architecture string to its constructor.
//
// Guarded because Register runs from init in one goroutine and Open runs from
// wherever the caller is. The lock costs nothing next to reading a checkpoint
// and it keeps the race detector honest about a map two goroutines touch.
var registry struct {
	mu sync.RWMutex
	m  map[string]func(json.RawMessage) (Builder, error)
}

// Register makes an architecture available to Open and New. It is meant to be
// called from an init function: adding a model is one file and one init
// (specs/004-model-graph.md §6).
//
// It panics on an empty name, a nil constructor, or a second registration of
// the same architecture. A duplicate is a build-time mistake — two packages
// claiming one architecture string — and silently keeping either one of them
// picks the weight map of a model the caller did not ask for.
func Register(architecture string, new func(json.RawMessage) (Builder, error)) {
	if architecture == "" {
		panic("model: Register with an empty architecture")
	}
	if new == nil {
		panic("model: Register(" + architecture + ") with a nil constructor")
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.m == nil {
		registry.m = make(map[string]func(json.RawMessage) (Builder, error))
	}
	if _, dup := registry.m[architecture]; dup {
		panic("model: Register called twice for architecture " + architecture)
	}
	registry.m[architecture] = new
}

// Architectures lists every registered architecture, sorted.
func Architectures() []string {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	out := make([]string, 0, len(registry.m))
	for a := range registry.m {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}

// Open reads config.json from a checkpoint directory and builds the model its
// architectures[0] names.
//
// It reads the config and nothing else: the weights are the loader's, and a
// caller that wants to know what a directory claims to be should not have to
// map a gigabyte of tensors to find out.
func Open(dir string) (Builder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, configName))
	if err != nil {
		return nil, fmt.Errorf("model: read config: %w", err)
	}
	b, err := New(raw)
	if err != nil {
		return nil, fmt.Errorf("model: %s: %w", dir, err)
	}
	return b, nil
}

// New builds the model a config.json describes. It is Open with the file read
// already, which is what a caller holding an open safetensors.Repo has.
func New(config json.RawMessage) (Builder, error) {
	arch, err := architecture(config)
	if err != nil {
		return nil, err
	}
	registry.mu.RLock()
	new, ok := registry.m[arch]
	registry.mu.RUnlock()
	if !ok {
		return nil, unknown(arch, Architectures())
	}
	return new(config)
}

// unknown builds §6's refusal. Naming the known set is the whole value of it:
// the next question after "unknown architecture" is always "then which ones do
// you know", and the answer is a build-time property of this binary rather than
// something the caller can look up.
//
// The alternative -- falling back to a generic Llama path -- is what this
// refuses. A model run under the wrong architecture produces fluent text and
// nobody finds out (004-D2).
func unknown(arch string, known []string) error {
	if len(known) == 0 {
		return fmt.Errorf("model: unknown architecture %q: no architecture is registered", arch)
	}
	return fmt.Errorf("model: unknown architecture %q: known architectures are %s",
		arch, strings.Join(known, ", "))
}

// architecture pulls architectures[0] out of a config without parsing the rest,
// so that an unknown architecture is refused by name rather than by whichever
// field of §5's table its config happens to be missing.
func architecture(config json.RawMessage) (string, error) {
	var head struct {
		Architectures []string `json:"architectures"`
	}
	if err := json.Unmarshal(config, &head); err != nil {
		return "", fmt.Errorf("model: parse config: %w", err)
	}
	if len(head.Architectures) == 0 {
		return "", fmt.Errorf("model: config: architectures[0] is required; it is the registry key")
	}
	if head.Architectures[0] == "" {
		return "", fmt.Errorf("model: config: architectures[0] is empty; it is the registry key")
	}
	return head.Architectures[0], nil
}
