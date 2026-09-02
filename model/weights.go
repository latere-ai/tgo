// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package model

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// ErrTiedHeadShipped is the contradiction §4 names: a checkpoint whose config
// sets tie_word_embeddings and which also ships an LM head of its own.
//
// It is a sentinel because the header cannot settle it. The two planes may hold
// the same weights, in which case both readings of the checkpoint agree, or
// different ones, in which case nothing says which the model was trained with —
// and the difference is in bytes this package does not read. A caller holding
// both planes may prove them identical and re-run [Check] against have with the
// alias tensor deleted; a caller that has not proven it must refuse.
var ErrTiedHeadShipped = errors.New("model: checkpoint is tied and also ships an LM head")

// Kind is what a weight is for. The loader reads it to choose a precision: an
// embedding table and an LM head are the largest tensors in a small model and
// the most sensitive to quantization, and holding those two at f16 while
// everything else is int8 is a defensible point that needs the distinction
// (specs/001-weights.md §5).
type Kind int

// The kinds of weight the map names.
const (
	// KindGain is an RMSNorm gain: one value per feature, never transposed,
	// never quantized.
	KindGain Kind = iota

	// KindProjection is a matrix a MatMul contracts against.
	KindProjection

	// KindEmbedding is the token embedding table, which GatherRows reads by
	// row rather than contracting against.
	KindEmbedding

	// KindLMHead is the output projection to vocabulary logits. It is a
	// projection that is separated out because it is the other tensor a
	// precision override usually names, and because it is the port a tied
	// checkpoint feeds from the embedding table.
	KindLMHead
)

// String names a Kind for an error message.
func (k Kind) String() string {
	switch k {
	case KindGain:
		return "gain"
	case KindProjection:
		return "projection"
	case KindEmbedding:
		return "embedding"
	case KindLMHead:
		return "lm_head"
	}
	return fmt.Sprintf("Kind(%d)", int(k))
}

// WeightSpec is one row of specs/004-model-graph.md §4: a checkpoint tensor, the
// graph port it binds, and the two layout changes the loader applies between
// them.
//
// It is data rather than code because 001-D4 puts the transpose decision with
// the model rather than with a name heuristic in the loader. A new architecture
// states its own table and the loader does not change.
type WeightSpec struct {
	// Tensor is the name in the checkpoint. Two specs may name one tensor: a
	// tied checkpoint feeds both the embedding port and the LM head port from
	// model.embed_tokens.weight, in two different layouts (004-D7).
	Tensor string

	// Port is the graph port this plane binds, "ℓ.wq" spelled "0.wq".
	Port string

	// Shape is the shape the checkpoint must hold, as the file stores it —
	// before the transpose, in Hugging Face's [out_features, in_features]
	// order for a projection.
	Shape []int

	// Layer is the layer index, or -1 for a tensor outside the layer stack.
	Layer int

	// Kind is what the weight is for.
	Kind Kind

	// Transpose reports whether the loader transposes the plane. Hugging Face
	// stores a linear weight as [out, in] and computes xWᵀ; accel's MatMul
	// contracts x[M,K] against w[K,N], so every projection transposes on the
	// host at load (001-D3).
	Transpose bool

	// Permute reports whether the loader permutes the plane's output channels
	// into accel's interleaved rotary pairing.
	//
	// accel's RoPE rotates (x₀,x₁), (x₂,x₃), … and Qwen3 was trained on
	// (x₀,x_{d_h/2}), (x₁,x_{d_h/2+1}), …. Nothing refuses the mismatch: every
	// shape checks and the model produces fluent text with degraded long-range
	// coherence. The fix is y[2i] = x[i], y[2i+1] = x[i+d_h/2], applied per
	// head after the transpose and before quantization (004-D9,
	// specs/004-model-graph.md §2.5.2).
	Permute bool

	// Heads is how many heads the permutation splits the port's output channels
	// into: H for q_proj, H_kv for k_proj, and 1 for the QK-norm gains, whose
	// d_h values are one head's channels shared across heads. It is 0 when
	// Permute is false.
	Heads int

	// Alias is the tensor name this port would have carried had the checkpoint
	// not tied it to another one, and is empty otherwise.
	//
	// It exists so that a checkpoint which sets tie_word_embeddings and also
	// ships lm_head.weight is refused as the contradiction it is, rather than
	// reported as a stray tensor (004-D7).
	Alias string
}

// Check reports whether a checkpoint holds exactly the tensors a weight map
// names, in exactly the shapes it names.
//
// have maps a tensor name to its shape, which is what a safetensors header
// gives without reading a byte of tensor data.
//
// A tensor the map does not mention is refused rather than ignored. It means
// the architecture string matched and the weights did not: the checkpoint is a
// variant the map was not written for, and loading the intersection produces a
// model that runs (specs/004-model-graph.md §4).
// CheckOption adjusts what [Check] can decide.
type CheckOption func(*checkOpts)

type checkOpts struct {
	identical func(a, b string) (bool, error)
}

// WithPlaneComparator lets [Check] resolve a tied checkpoint that also ships an
// LM head, by asking whether the two tensors hold identical bytes.
//
// specs/004-model-graph.md §4 and 004-D10: redundancy is not a contradiction.
// An exporter writing lm_head.weight out beside a tied embedding is common --
// Qwen3-0.6B does it, and its two planes hash identically -- so refusing on the
// config-versus-tensor mismatch alone refuses the model tgo exists to run.
// Shapes cannot tell the two cases apart, which is why this takes a comparator
// and why the comparison belongs to whoever holds the file.
//
// Without this option a tied-and-shipped head stays a refusal, because the
// alternative is to guess which plane the model was trained with.
func WithPlaneComparator(f func(a, b string) (bool, error)) CheckOption {
	return func(o *checkOpts) { o.identical = f }
}

func Check(specs []WeightSpec, have map[string][]int, opts ...CheckOption) error {
	var o checkOpts
	for _, fn := range opts {
		fn(&o)
	}

	// redundant collects alias tensors a comparator proved identical to the
	// port they duplicate. They are named by the map after all -- as the same
	// weights under a second name -- so the extras check below must not report
	// them as tensors the architecture does not know about.
	redundant := map[string]bool{}
	// The tie contradiction is checked first because it explains a file that
	// would otherwise be reported as carrying one extra tensor, which is the
	// true statement that sends a reader looking in the wrong place.
	for _, s := range specs {
		if s.Alias == "" {
			continue
		}
		if _, ok := have[s.Alias]; !ok {
			continue
		}
		// Redundancy is not a contradiction. Ask, when a caller gave us the
		// means to ask; refuse when it did not, rather than guessing which of
		// two disagreeing planes the model was trained with.
		if o.identical != nil {
			same, err := o.identical(s.Tensor, s.Alias)
			if err != nil {
				return fmt.Errorf("model: comparing %s against %s: %w", s.Tensor, s.Alias, err)
			}
			if same {
				redundant[s.Alias] = true
				continue
			}
			return fmt.Errorf("%w: tie_word_embeddings says port %q is %s transposed, "+
				"and the checkpoint holds a %s whose bytes differ from it; the config "+
				"and the weights disagree about which the head has",
				ErrTiedHeadShipped, s.Port, s.Tensor, s.Alias)
		}
		return fmt.Errorf("%w: tie_word_embeddings says port %q is %s transposed, "+
			"and the checkpoint also holds %s; pass WithPlaneComparator so the two "+
			"can be compared, since identical planes are redundancy rather than a "+
			"contradiction (004-D10)", ErrTiedHeadShipped, s.Port, s.Tensor, s.Alias)
	}

	want := make(map[string][]int, len(specs))
	for _, s := range specs {
		want[s.Tensor] = s.Shape
	}
	for _, s := range specs {
		if s.Alias != "" && redundant[s.Alias] {
			want[s.Alias] = s.Shape
		}
	}

	var missing []string
	for name := range want {
		if _, ok := have[name]; !ok {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("model: checkpoint is missing %d tensor(s) the weight map names: %s",
			len(missing), list(missing))
	}

	var extra []string
	for name := range have {
		if _, ok := want[name]; !ok {
			extra = append(extra, name)
		}
	}
	if len(extra) > 0 {
		sort.Strings(extra)
		return fmt.Errorf("model: checkpoint has %d tensor(s) the weight map does not name: %s; "+
			"the architecture string matched and the weights did not", len(extra), list(extra))
	}

	// Shapes are compared in the file's order, before any transpose: the map
	// states what the file must hold, so a mismatch is reported in the terms
	// the header is written in.
	names := make([]string, 0, len(specs))
	byName := make(map[string]WeightSpec, len(specs))
	for _, s := range specs {
		prev, seen := byName[s.Tensor]
		if !seen {
			names = append(names, s.Tensor)
			byName[s.Tensor] = s
			continue
		}
		// Two ports on one tensor is the tied case. The embedding row is the
		// one to keep, because its row count is vocab_size and the message
		// below names that field; keeping whichever came first in the slice
		// would make the field-naming depend on the order the map was built in.
		if prev.Kind != KindEmbedding && s.Kind == KindEmbedding {
			byName[s.Tensor] = s
		}
	}
	sort.Strings(names)
	for _, name := range names {
		s := byName[name]
		got := have[name]
		if sameShape(s.Shape, got) {
			continue
		}
		// The embedding table's row count is vocab_size. Reporting it as a
		// shape mismatch would be true and useless: the field a caller has to
		// change is in config.json, so the message names it.
		if s.Kind == KindEmbedding && len(got) == len(s.Shape) && len(got) == 2 &&
			got[1] == s.Shape[1] {
			return fmt.Errorf("model: config: vocab_size is %d and %s has %d rows; "+
				"the config and the weights are from different models",
				s.Shape[0], name, got[0])
		}
		return fmt.Errorf("model: %s: the weight map expects shape %v and the checkpoint "+
			"holds %v", name, s.Shape, got)
	}
	return nil
}

// sameShape compares two shapes.
func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// list renders names for an error message, truncated so that a checkpoint of
// the wrong architecture does not print three hundred lines at a caller who
// needs the first one.
func list(names []string) string {
	const max = 8
	if len(names) <= max {
		return strings.Join(names, ", ")
	}
	return fmt.Sprintf("%s, and %d more", strings.Join(names[:max], ", "), len(names)-max)
}
