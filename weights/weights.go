// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Package weights turns a checkpoint tensor into device memory.
//
// A safetensors plane and an accel buffer are four conversions apart, and
// specs/001-weights.md §2 names all four in the order they must happen:
//
//	dtype  ->  transpose  ->  permute  ->  precision
//
// The dtype conversion widens bf16 or f16 to f32. The transpose turns Hugging
// Face's [out, in] Linear weight into the [in, out] that accel's MatMul
// contracts against. The permutation rewrites RoPE's channel pairs from the
// half-split convention the checkpoint uses into the interleaved one accel's
// kernel rotates. The precision step narrows to f16 or to int8 quants with one
// f16 scale per block.
//
// # The order is not a preference
//
// The permutation runs after the transpose and before quantization.
// quant.Int8Quantize blocks over the flattened matrix in runs of
// quant.Int8Block, so permuting afterwards would scatter each weight away from
// the scale computed for it. Nothing in accel refuses a mistake here — every
// shape still checks — and the model produces fluent text that loses coherence
// over a few sentences (specs/004-model-graph.md §2.5.2, 004-D9).
//
// # What is declared, not guessed
//
// Which tensors transpose, and which carry a head dimension to permute, is a
// property of the operator that consumes them. The model states it; this
// package never infers it from a name (001-D4). [Tensor] is that declaration.
//
// # Where the bytes go
//
// The converted plane is written straight into device memory through
// accel.Buffer.Access, so it never exists as a host allocation (001-D8). Access
// needs a host-visible pool and refuses one that is not, so on a device with no
// unified memory the loader stages through Queue.WriteBuffer instead (001-D9).
// [Report.Mapped] says which happened.
package weights

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"

	"github.com/latere-ai/tgo/safetensors"
)

// Precision is the form a weight takes on the device.
type Precision int

const (
	// Inherit is the zero value. On a [Tensor] it means "use the load policy";
	// as a policy it means [Auto].
	Inherit Precision = iota

	// Auto picks between F16 and Int8 by whether the f16 footprint fits the
	// budget, and prints which it chose. It is a policy, not a per-tensor
	// choice: a model whose tensors each decided independently would have no
	// footprint to decide against.
	Auto

	// F16 stores one 16-bit float per weight.
	F16

	// Int8 stores one int8 quant per weight plus one f16 scale per
	// quant.Int8Block of them: half of F16's bytes, at a bounded accuracy cost
	// that quant.Int8ErrorBound states.
	Int8

	// Int4 stores eight 4-bit codes per u32 word, with an f16 scale and an f16
	// zero per quant.Int4Group.
	//
	// 0.53125 bytes per weight against Int8's 1.0625, which is what decides
	// whether a 27B-class model fits hardware people own (specs/001-weights.md
	// §2.1).
	//
	// **Never chosen as a preference.** accel's own tests show int4 beating
	// int8 on a group of weights clustered away from zero and losing on one
	// centred on it, so it is not uniformly better, and a budget rule that
	// reached for it early would quietly degrade a model that fits at int8.
	Int4
)

func (p Precision) String() string {
	switch p {
	case Inherit:
		return "inherit"
	case Auto:
		return "auto"
	case F16:
		return "f16"
	case Int8:
		return "int8"
	case Int4:
		return "int4"
	}
	return fmt.Sprintf("Precision(%d)", int(p))
}

// Tensor declares one checkpoint tensor and what the operator consuming it
// needs done to it.
//
// It is the model's statement, not a heuristic over names (001-D4). A model
// that gets HeadDim wrong, or omits it on q_proj, produces a running model with
// no error anywhere, so the declaration is explicit and per tensor.
type Tensor struct {
	// Name is the tensor's name in the checkpoint.
	Name string

	// As is the name the loaded value is filed under, and defaults to Name.
	//
	// It exists because one checkpoint tensor can serve two ports in two
	// layouts. A checkpoint with tie_word_embeddings ships no lm_head.weight,
	// so the LM head is model.embed_tokens.weight transposed, while the
	// embedding table is the same plane untransposed: two declarations, one
	// name in the file, two values on the device (004-D7).
	As string

	// Transpose turns a [out, in] Hugging Face Linear weight into the [in, out]
	// accel's MatMul contracts. True for every projection, false for norm gains
	// and the embedding table.
	Transpose bool

	// HeadDim is the rotary head width, and zero means no permutation. Set it on
	// q_proj, k_proj, and identically on the q_norm and k_norm gains, because
	// QK-norm is applied per channel before RoPE and its gain must follow its
	// channels. Leave it zero everywhere else: v_proj and o_proj are not rotated,
	// and attention's q·kᵀ is invariant under a permutation applied to both q
	// and k, so nothing downstream undoes it.
	//
	// It comes from the config's head_dim. Qwen3-0.6B stores head_dim 128 while
	// hidden_size/num_attention_heads is 64, so a loader that derives it is
	// wrong on a real checkpoint (specs/004-model-graph.md §5).
	HeadDim int

	// Gathered says this tensor is read a row at a time rather than contracted
	// against, which is what an embedding table is.
	//
	// It caps the tensor at [Int8] however narrow the policy: accel registers
	// no int4 gather -- QuantGatherRows reads a quant plane and a scale plane
	// and has no three-plane form -- so a gathered tensor at int4 is a refusal
	// at record time rather than a load that works.
	//
	// It is a property of the tensor and not a policy, which is why it is
	// declared here rather than pinned by a caller: two callers pinning it
	// separately is two rules that have to agree, and the footprint `tgo info`
	// prints is computed by one of them and the load by the other.
	Gathered bool

	// Precision overrides the load policy for this tensor. The case it exists
	// for is holding the embedding table and the LM head at F16 — the largest
	// tensors in a small model and the most sensitive to quantization — while
	// everything else is Int8.
	Precision Precision
}

// key is what the loaded value is filed under.
func (d Tensor) key() string {
	if d.As != "" {
		return d.As
	}
	return d.Name
}

// Options configures a load.
type Options struct {
	// Policy is F16, Int8 or Auto. The zero value is Auto.
	Policy Precision

	// Budget is the device bytes the weights may occupy, which Auto compares the
	// f16 footprint against. Zero takes the device's MaxPoolBytes, which is a
	// cap on one allocation rather than a report of free memory; accel exposes
	// no such report, so a caller who knows the machine should say so here.
	Budget int64

	// MaxSaturation fails the load when any tensor saturates more than this
	// fraction of its weights on the way to f16. Zero takes
	// DefaultMaxSaturation and any negative value takes RefuseAnySaturation. It
	// is a fraction and not a count because a 155-million-element embedding
	// table and a 128-element norm gain cannot share an absolute one.
	MaxSaturation float64

	// Log receives the precision choice and the per-tensor saturation counts.
	// Zero writes to os.Stderr: specs/001-weights.md §5 requires the choice to
	// be printed, never silent. Use io.Discard to silence it deliberately.
	Log io.Writer
}

// DefaultMaxSaturation is the fraction of a tensor that may saturate to ±65504
// before the load fails.
//
// Trained transformer weights are almost entirely within [-1, 1], so any
// saturation at all is a signal that the checkpoint is not what it claims and
// the int8 path is not a fix for it either. specs/001-weights.md §3 requires a
// fraction and names no number; this one is small enough that a real checkpoint
// clears it with a factor of thousands to spare and large enough that a single
// stray value in a large tensor is reported rather than fatal.
const DefaultMaxSaturation = 1e-5

// RefuseAnySaturation is the Options.MaxSaturation that admits no saturation at
// all.
//
// It is a negative number rather than zero because zero is the field's unset
// value and the default has to stay reachable without every caller writing it
// out. The strict end of the range needs to be expressible: §3 argues that a
// nonzero count is a signal and not routine, and a caller who agrees has no
// other way to say so — the fraction that means "not one element" depends on
// the size of a tensor they have not read yet.
const RefuseAnySaturation = -1.0

// Value is one converted weight, resident on the device.
type Value struct {
	// Name is what this value is filed under: the declaration's As, or its
	// Name where As is empty. It is what Set.Get takes and Set.Names lists.
	Name string

	// Source is the checkpoint tensor the plane was read from. It differs from
	// Name only where a declaration set As, which is how a tied checkpoint
	// gives one plane to two ports.
	Source string

	// Shape is what accel sees: the file's shape, reversed if Transpose was set.
	Shape []int

	// Precision is F16, Int8 or Int4, resolved. Never Inherit or Auto.
	Precision Precision

	// Data is the f16 plane, the i8 quants at Int8, or the packed u32 codes at
	// Int4.
	Data *accel.Buffer

	// Scales is the f16 block scales, and is nil at F16.
	Scales *accel.Buffer

	// Zeros is the f16 zero point per group, and is nil unless Precision is
	// Int4: int8 is symmetric and needs none.
	Zeros *accel.Buffer

	// Saturated is how many elements hit ±65504 on the way to f16, and is zero
	// on the Int8 path, which has no f16 range to overflow. A nonzero value is
	// the first thing to check when a converted model produces noise.
	Saturated int

	// Elements is the number of weights, the product of Shape.
	Elements int
}

// Bytes is what this value occupies on the device.
func (v Value) Bytes() int64 {
	switch v.Precision {
	case Int8:
		return int8Bytes(v.Elements)
	case Int4:
		return int4Bytes(v.Elements)
	}
	return f16Bytes(v.Elements)
}

// f16Bytes and int8Bytes are specs/000-decisions.md decision 5's footprint
// arithmetic for one tensor of n weights: two bytes each at f16, and one byte
// each plus one f16 scale per quant.Int8Block at int8.
func f16Bytes(n int) int64  { return int64(n) * 2 }
func int8Bytes(n int) int64 { return int64(n) + int64(blocks(n))*2 }

// int4Bytes is the same arithmetic one width down: eight codes to a u32 word,
// and an f16 scale *and* an f16 zero per accel's group.
//
// 0.53125 bytes per weight at a group of 128, against int8's 1.0625. The
// metadata's share is what the group size buys: halving the payload doubles
// what a scale per 32 would cost as a fraction of it, so int4 groups 128 and
// pays 6.2% rather than 12.5% (accel specs/048-int4.md §2).
func int4Bytes(n int) int64 {
	words := (n + 7) / 8
	groups := (n + quant.Int4Group - 1) / quant.Int4Group
	return int64(words)*4 + int64(groups)*2*2
}

// Report describes the load as a whole.
type Report struct {
	// Chosen is the precision the policy resolved to, before any per-tensor
	// override.
	Chosen Precision

	// Int4Bytes is what the declared tensors would occupy at int4, which
	// [Auto] reaches for only when int8 does not fit.
	Int4Bytes int64

	// F16Bytes and Int8Bytes are what the declared tensors would occupy at each
	// precision, honouring per-tensor overrides. They are what Auto compared.
	F16Bytes, Int8Bytes int64

	// Budget is what they were compared against.
	Budget int64

	// Bytes is the sum of every loaded Value.Bytes(): the weights themselves,
	// plus the int8 block scales. It is not the pool footprint, which is larger
	// by each buffer's rounding up to the device's suballocation granularity.
	Bytes int64

	// Saturated is the total count across every tensor.
	Saturated int

	// Mapped reports whether the conversion wrote into device memory directly.
	// False means the device has no unified memory and every tensor was staged
	// through the queue, which is a different peak-host-memory promise.
	Mapped bool
}

// Set is the loaded weights. It owns its device memory.
type Set struct {
	values map[string]Value
	order  []string
	report Report
	arena  *arena
}

// Get returns one converted weight. The Shape is the caller's to keep; the
// buffers belong to the Set.
func (s *Set) Get(name string) (Value, bool) {
	v, ok := s.values[name]
	if !ok {
		return Value{}, false
	}
	shape := make([]int, len(v.Shape))
	copy(shape, v.Shape)
	v.Shape = shape
	return v, true
}

// Names lists the loaded tensors in declaration order.
func (s *Set) Names() []string {
	out := make([]string, len(s.order))
	copy(out, s.order)
	return out
}

// Report describes what the load chose and what it cost.
func (s *Set) Report() Report { return s.report }

// Close releases every buffer and pool. It reports all failures rather than the
// first, because memory left behind is not visible from the caller's side.
func (s *Set) Close() error {
	var errs []error
	for _, name := range s.order {
		v := s.values[name]
		// Every plane the value carries, walked rather than named one at a
		// time. Naming them is how the third one came to be allocated by the
		// int4 path and closed by nothing: accel counts live children and
		// refuses to close a device under them, so the leak surfaced as a
		// close error on an unrelated test rather than as anything about
		// int4.
		for _, b := range []*accel.Buffer{v.Data, v.Scales, v.Zeros} {
			if b != nil {
				errs = append(errs, b.Close())
			}
		}
	}
	s.values, s.order = nil, nil
	if s.arena != nil {
		errs = append(errs, s.arena.close())
		s.arena = nil
	}
	return errors.Join(errs...)
}

// Load converts every declared tensor and uploads it.
//
// Tensors are converted one at a time and the host scratch for each is released
// before the next, so peak host memory is one plane plus its f32 working copies
// rather than the whole model (001-D7).
func Load(dev *accel.Device, repo *safetensors.Repo, decls []Tensor, opts Options) (*Set, error) {
	if dev == nil || repo == nil {
		return nil, errors.New("weights: Load needs a device and a repo")
	}
	if len(decls) == 0 {
		return nil, errors.New("weights: Load was given no tensors to load; the model " +
			"declares which tensors it needs and how (001-D4)")
	}
	log := opts.Log
	if log == nil {
		log = os.Stderr
	}
	maxSat := opts.MaxSaturation
	switch {
	case maxSat == 0:
		maxSat = DefaultMaxSaturation
	case maxSat < 0:
		maxSat = 0
	}
	budget := opts.Budget
	if budget == 0 {
		budget = int64(dev.Limits().MaxPoolBytes)
	}

	plan, err := planLoad(repo, decls, opts.Policy, budget)
	if err != nil {
		return nil, err
	}

	set := &Set{values: make(map[string]Value, len(decls)), arena: newArena(dev)}
	set.report = plan.report
	set.report.Mapped = set.arena.mapped()
	_, _ = fmt.Fprintf(log, "weights: %s, %d tensors, f16 %s, int8 %s, budget %s, %s\n",
		plan.report.Chosen, len(decls),
		humanBytes(plan.report.F16Bytes), humanBytes(plan.report.Int8Bytes),
		humanBytes(plan.report.Budget), mappedNote(set.arena.mapped()))

	for i, d := range decls {
		v, err := convertOne(set.arena, repo, d, plan.precision[i], maxSat)
		if err != nil {
			_ = set.Close()
			return nil, err
		}
		set.values[d.key()] = v
		set.order = append(set.order, d.key())
		set.report.Bytes += v.Bytes()
		set.report.Saturated += v.Saturated
		if v.Saturated > 0 {
			_, _ = fmt.Fprintf(log, "weights: %s saturated %d of %d elements to ±65504\n",
				v.Source, v.Saturated, v.Elements)
		}
	}
	if err := set.arena.flush(); err != nil {
		_ = set.Close()
		return nil, fmt.Errorf("weights: staging the uploads: %w", err)
	}
	return set, nil
}

// loadPlan is what the policy decided, per tensor, before any bytes moved.
type loadPlan struct {
	precision []Precision
	report    Report
}

// planLoad resolves the precision of every tensor and the footprint each choice
// implies. It runs before the first byte is read so that a model too large for
// its budget is refused rather than discovered part way through.
func planLoad(repo *safetensors.Repo, decls []Tensor, policy Precision, budget int64) (loadPlan, error) {
	var fixedF16, fixedInt8, fixedInt4, flexF16, flexInt8, flexInt4 int64
	seen := make(map[string]bool, len(decls))
	for _, d := range decls {
		if seen[d.key()] {
			return loadPlan{}, fmt.Errorf("weights: %q is declared twice", d.key())
		}
		seen[d.key()] = true
		e, _, ok := repo.Tensor(d.Name)
		if !ok {
			return loadPlan{}, fmt.Errorf("weights: the model declares %q, which this "+
				"checkpoint does not contain", d.Name)
		}
		n := 1
		for _, s := range e.Shape {
			n *= s
		}
		// Footprints are summed per tensor because the scale array is per
		// tensor: a trailing partial block gets its own scale, so blocks over a
		// summed element count is not the number of scales the load allocates.
		switch d.Precision {
		case F16:
			fixedF16 += f16Bytes(n)
			fixedInt8 += f16Bytes(n)
			fixedInt4 += f16Bytes(n)
		case Int8:
			fixedF16 += int8Bytes(n)
			fixedInt8 += int8Bytes(n)
			fixedInt4 += int8Bytes(n)
		case Int4:
			fixedF16 += int4Bytes(n)
			fixedInt8 += int4Bytes(n)
			fixedInt4 += int4Bytes(n)
		case Inherit, Auto:
			flexF16 += f16Bytes(n)
			flexInt8 += int8Bytes(n)
			// A gathered tensor cannot pack, so at int4 it costs what int8
			// costs. Counted here rather than corrected later, because this
			// sum is what the choice is made against.
			if d.Gathered {
				flexInt4 += int8Bytes(n)
			} else {
				flexInt4 += int4Bytes(n)
			}
		default:
			return loadPlan{}, fmt.Errorf("weights: %q declares precision %v, which is not "+
				"F16, Int8, Int4 or the zero value", d.Name, d.Precision)
		}
	}

	rep := Report{
		Budget:    budget,
		F16Bytes:  fixedF16 + flexF16,
		Int8Bytes: fixedInt8 + flexInt8,
		Int4Bytes: fixedInt4 + flexInt4,
	}

	switch policy {
	case F16, Int8, Int4:
		rep.Chosen = policy
	case Inherit, Auto:
		// Decision 5 of specs/000-decisions.md: the widest form that fits.
		// Above roughly 8 GB of weights int8 is not an optimisation, it is the
		// only way the model loads, and int4 is the same statement one width
		// down for a 27B-class model on hardware people own.
		//
		// **Narrowing is a last resort at every step and int4 especially.**
		// accel's own tests show int4 beating int8 on a group of weights
		// clustered away from zero and losing on one centred on it, so it is
		// not uniformly better: a rule that preferred it would trade accuracy
		// for memory nobody asked to save. Auto reaches for it only when int8
		// does not fit, which is the case where the alternative is not loading.
		rep.Chosen = F16
		if rep.F16Bytes > budget {
			rep.Chosen = Int8
		}
		if rep.Chosen == Int8 && rep.Int8Bytes > budget {
			rep.Chosen = Int4
		}
	default:
		return loadPlan{}, fmt.Errorf("weights: policy %v is not F16, Int8, Int4 or Auto",
			policy)
	}
	if want := rep.Chosen; want != F16 {
		if got := map[Precision]int64{Int8: rep.Int8Bytes, Int4: rep.Int4Bytes}[want]; got > budget {
			return loadPlan{}, fmt.Errorf("weights: the model needs %s at %v, which is more "+
				"than the %s budget; no supported precision fits",
				humanBytes(got), want, humanBytes(budget))
		}
	}

	plan := loadPlan{precision: make([]Precision, len(decls)), report: rep}
	for i, d := range decls {
		plan.precision[i] = d.Precision
		if d.Precision == Inherit || d.Precision == Auto {
			plan.precision[i] = rep.Chosen
		}
		if d.Gathered && plan.precision[i] == Int4 {
			plan.precision[i] = Int8
		}
	}
	return plan, nil
}

// convertOne runs specs/001-weights.md §2's four conversions over one tensor
// and uploads the result.
func convertOne(a *arena, repo *safetensors.Repo, d Tensor, p Precision, maxSat float64) (Value, error) {
	e, file, ok := repo.Tensor(d.Name)
	if !ok {
		return Value{}, fmt.Errorf("weights: the model declares %q, which this checkpoint "+
			"does not contain", d.Name)
	}
	shape, err := targetShape(d, e.Shape)
	if err != nil {
		return Value{}, err
	}
	n := 1
	for _, s := range shape {
		n *= s
	}
	if n == 0 {
		return Value{}, fmt.Errorf("weights: %q has shape %v, which holds no elements", d.Name, e.Shape)
	}

	plane, err := readPlane(file, d.Name, e.DType, n)
	if err != nil {
		return Value{}, err
	}

	if d.Transpose {
		// The only step that cannot reuse its input: a non-square transpose is
		// not an in-place permutation. §7 counts this as the loader's remaining
		// transient.
		out := make([]float32, n)
		transpose(plane, out, e.Shape[0], e.Shape[1])
		plane = out
	}
	if d.HeadDim > 0 {
		if err := permuteHeads(plane, shape[len(shape)-1], d.HeadDim); err != nil {
			return Value{}, fmt.Errorf("weights: %q: %w", d.Name, err)
		}
	}

	v := Value{Name: d.key(), Source: d.Name, Shape: shape, Precision: p, Elements: n}
	switch p {
	case Int8:
		if err := uploadInt8(a, &v, plane); err != nil {
			return Value{}, err
		}
		return v, nil
	case Int4:
		if err := uploadInt4(a, &v, plane); err != nil {
			return Value{}, err
		}
		return v, nil
	}
	if err := uploadF16(a, &v, plane); err != nil {
		return Value{}, err
	}
	if frac := float64(v.Saturated) / float64(n); frac > maxSat {
		v.close()
		return Value{}, fmt.Errorf("weights: %q saturated %d of %d elements (%.3g) to ±65504, "+
			"over the %.3g threshold: this checkpoint holds values f16 cannot represent, and "+
			"int8 is not a fix for it either", d.Name, v.Saturated, n, frac, maxSat)
	}
	return v, nil
}

// readPlane reads one tensor's bytes and widens them to f32. The raw plane goes
// out of scope with the call, so the file bytes and the f32 scratch are not
// both live while the next tensor loads (001-D7).
func readPlane(file *safetensors.File, name string, dt safetensors.DType, n int) ([]float32, error) {
	raw, err := file.Bytes(name)
	if err != nil {
		return nil, fmt.Errorf("weights: %q: %w", name, err)
	}
	plane := make([]float32, n)
	if err := decodeF32(dt, raw, plane); err != nil {
		return nil, fmt.Errorf("weights: %q: %w", name, err)
	}
	return plane, nil
}

func uploadF16(a *arena, v *Value, plane []float32) error {
	buf, err := a.alloc(accel.F16, v.Elements, v.Name)

	if err != nil {
		return err
	}
	v.Data = buf
	if err := a.fill(buf, func(dst []byte) error {
		v.Saturated = toF16(plane, dst)
		return nil
	}); err != nil {
		v.close()
		return fmt.Errorf("weights: %q: writing the f16 plane: %w", v.Name, err)
	}
	return nil
}

func uploadInt8(a *arena, v *Value, plane []float32) error {
	quants, err := a.alloc(accel.I8, v.Elements, v.Name)
	if err != nil {
		return err
	}
	v.Data = quants
	scales, err := a.alloc(accel.F16, blocks(v.Elements), v.Name+".scales")
	if err != nil {
		v.close()
		return err
	}
	v.Scales = scales

	// Two buffers, so two writes into device memory. The quants pass carries the
	// scales for its own blocks in a host slice of one block at a time, and the
	// scales pass recomputes them: recomputation is cheaper than the second
	// full-tensor host allocation it would take to carry them across.
	scaleHost := make([]byte, blocks(v.Elements)*2)
	if err := a.fill(quants, func(dst []byte) error {
		quantizeInto(plane, dst, scaleHost)
		return nil
	}); err != nil {
		v.close()
		return fmt.Errorf("weights: %q: writing the int8 quants: %w", v.Name, err)
	}
	if err := a.fill(scales, func(dst []byte) error {
		copy(dst, scaleHost)
		return nil
	}); err != nil {
		v.close()
		return fmt.Errorf("weights: %q: writing the block scales: %w", v.Name, err)
	}
	return nil
}

// uploadInt4 packs a plane into codes, scales and zeros and writes all three.
//
// Three buffers where int8 has two, and the third is not optional: at eight bits
// the codes reach far enough that a scale alone spends them well, and at four
// they have to be spent where the weights actually are. A code plane bound
// against another matrix's scales compiles and produces noise, which is why
// accel bundles the triple and why this writes all of it or none.
//
// Unlike [uploadInt8], the packing is done once into host slices rather than
// recomputed per pass. quant.Int4Quantize returns the three planes together and
// there is no way to ask it for one of them, so the choice is between holding
// all three and calling it three times; the codes are an eighth of the input
// and the metadata is 6.2% of that, so holding them is the cheap side.
func uploadInt4(a *arena, v *Value, plane []float32) error {
	codes, scales, zeros := quant.Int4Quantize(plane)

	for _, p := range []struct {
		dst   **accel.Buffer
		dt    accel.DType
		n     int
		label string
	}{
		{&v.Data, accel.U32, len(codes), v.Name},
		{&v.Scales, accel.F16, len(scales), v.Name + ".scales"},
		{&v.Zeros, accel.F16, len(zeros), v.Name + ".zeros"},
	} {
		buf, err := a.alloc(p.dt, p.n, p.label)
		if err != nil {
			v.close()
			return err
		}
		*p.dst = buf
	}

	if err := a.fill(v.Data, func(dst []byte) error {
		for i, w := range codes {
			binary.LittleEndian.PutUint32(dst[i*4:], w)
		}
		return nil
	}); err != nil {
		v.close()
		return fmt.Errorf("weights: %q: writing the int4 codes: %w", v.Name, err)
	}
	for _, p := range []struct {
		buf  *accel.Buffer
		from []accel.Float16
		what string
	}{
		{v.Scales, scales, "group scales"},
		{v.Zeros, zeros, "group zero points"},
	} {
		if err := a.fill(p.buf, func(dst []byte) error {
			for i, h := range p.from {
				binary.LittleEndian.PutUint16(dst[i*2:], h.Bits())
			}
			return nil
		}); err != nil {
			v.close()
			return fmt.Errorf("weights: %q: writing the %s: %w", v.Name, p.what, err)
		}
	}
	return nil
}

func (v *Value) close() {
	for _, b := range []**accel.Buffer{&v.Data, &v.Scales, &v.Zeros} {
		if *b != nil {
			_ = (*b).Close()
			*b = nil
		}
	}
}

// targetShape is the shape accel sees, and the place every rank rule is
// checked.
func targetShape(d Tensor, fileShape []int) ([]int, error) {
	switch len(fileShape) {
	case 1, 2:
	default:
		return nil, fmt.Errorf("weights: %q has rank %d; a weight this loader converts is a "+
			"matrix or a gain vector", d.Name, len(fileShape))
	}
	if d.Transpose && len(fileShape) != 2 {
		return nil, fmt.Errorf("weights: %q is declared transposed but has shape %v; only a "+
			"matrix has axes to exchange", d.Name, fileShape)
	}
	out := make([]int, len(fileShape))
	copy(out, fileShape)
	if d.Transpose {
		out[0], out[1] = out[1], out[0]
	}
	return out, nil
}

func mappedNote(mapped bool) string {
	if mapped {
		return "written into device memory"
	}
	return "staged through the queue"
}

// humanBytes formats a byte count for the load report.
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit && exp < 3; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %ciB", float64(n)/float64(div), "KMGT"[exp])
}
