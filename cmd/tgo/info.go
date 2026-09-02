// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"io"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/weights"
)

// defaultContext is the KV capacity every command prices and runs at.
//
// 4096 rather than the model's max_position_embeddings, which for Qwen3 is
// 40960: specs/005-kv-cache.md §4 makes capacity a session parameter precisely
// so that a default does not reserve the whole trained context before the first
// token. What that costs is printed, at the moment the user asks for it.
const defaultContext = 4096

// modelFacts is the model as a report carries it: what config.json says, plus
// the parameter count implied by the weight map.
type modelFacts struct {
	Dir          string `json:"dir"`
	Architecture string `json:"architecture"`
	HiddenSize   int    `json:"hidden_size"`
	Layers       int    `json:"layers"`

	// CachedLayers is how many of them hold a key/value cache, which for a
	// hybrid is one in four. It is what the per-position arithmetic divides
	// by (specs/023-cache-kinds.md section 8); zero means it was not read and
	// Layers is used.
	CachedLayers int `json:"cached_layers,omitempty"`

	Heads            int   `json:"heads"`
	KVHeads          int   `json:"kv_heads"`
	HeadDim          int   `json:"head_dim"`
	IntermediateSize int   `json:"intermediate_size"`
	VocabSize        int   `json:"vocab_size"`
	Parameters       int64 `json:"parameters"`
	Planes           int64 `json:"device_elements"`
	TiedEmbeddings   bool  `json:"tied_embeddings"`
	TrainedContext   int   `json:"trained_context"`
}

// precisionFacts is specs/001-weights.md §5's choice with the evidence that
// produced it. The two footprints and the budget are carried with the answer
// because §5 requires the choice to be printed and a bare word is not a
// printed choice: a reader has to be able to redo the comparison.
type precisionFacts struct {
	Requested string `json:"requested"`
	Chosen    string `json:"chosen"`
	F16Bytes  int64  `json:"f16_bytes"`
	Int8Bytes int64  `json:"int8_bytes"`
	Int4Bytes int64  `json:"int4_bytes"`
	Budget    int64  `json:"budget"`
	Why       string `json:"why"`
}

// memoryFacts is what the model costs at one capacity: the weights at the
// chosen precision, and specs/005-kv-cache.md §3's cache arithmetic.
type memoryFacts struct {
	Context            int    `json:"context"`
	CacheDType         string `json:"cache_dtype"`
	CacheElementBytes  int64  `json:"cache_element_bytes"`
	WeightBytes        int64  `json:"weight_bytes"`
	KVBytesPerPosition int64  `json:"kv_bytes_per_position"`
	KVBytes            int64  `json:"kv_bytes"`
	ResidentBytes      int64  `json:"resident_bytes"`
}

// modelReport is everything a command knows about a model before it runs one
// token: the model, the precision decision, the memory it implies, and the
// machine and build that will produce the numbers.
//
// It is one struct because `tgo info` prints it and `tgo bench` embeds it as
// the conditions of every measurement (017-D4). Two structs would let the two
// drift, and a benchmark whose conditions disagree with `info` on the same
// directory is worse than one with no conditions at all.
type modelReport struct {
	Model       modelFacts     `json:"model"`
	Precision   precisionFacts `json:"precision"`
	Memory      memoryFacts    `json:"memory"`
	Hardware    hardware       `json:"hardware"`
	Environment environment    `json:"environment"`
}

// describeOptions is what a description needs beyond the model directory.
type describeOptions struct {
	// Policy is the --precision flag.
	Policy weights.Precision

	// Context is C, the KV capacity to price.
	Context int

	// Budget is the device bytes the weights may occupy. Zero takes the
	// device's MaxPoolBytes, which is what weights.Load does with a zero
	// Options.Budget -- a cap on one allocation rather than a report of free
	// memory, because accel exposes no such report. The flag exists so that a
	// caller who knows the machine can say so.
	Budget int64

	// Cache is the dtype the key and value states hold. The zero value is
	// accel.F32, which is what model.GraphSpec records today.
	Cache accel.DType

	// Device is the accelerator to describe. The zero value is
	// tgo.AutoDevice, which is what a command with no --device asks for.
	Device tgo.Device
}

// describe computes everything a report says about a model.
//
// It takes a model.Builder rather than a directory so that every number below
// is reachable from a config in memory: specs/000-decisions.md decision 8 keeps
// the real checkpoint out of the default test run, and arithmetic that could
// only be checked against a 1.4 GiB download is arithmetic nobody checks.
func describe(dir string, b model.Builder, o describeOptions, hw hardware, env environment) (modelReport, error) {
	if o.Context < 1 {
		return modelReport{}, fmt.Errorf("%w: --context is %d; a cache holds at least one position", errUsage, o.Context)
	}
	c := b.Config()
	specs := b.Weights()

	budget := o.Budget
	if budget == 0 {
		budget = hw.MaxPoolBytes
	}
	params, planes, f16, int8, int4 := footprint(specs)
	choice, err := choosePrecision(o.Policy, f16, int8, int4, budget)
	if err != nil {
		return modelReport{}, err
	}

	weightBytes := f16
	if choice.Chosen == weights.Int8.String() {
		weightBytes = int8
	}
	perPosition := kvBytesPerPosition(c, o.Cache)
	kv := perPosition * int64(o.Context)

	return modelReport{
		Model: modelFacts{
			Dir: dir, Architecture: c.Architecture, HiddenSize: c.HiddenSize,
			Layers: c.NumLayers, CachedLayers: cachedLayerCount(c),
			Heads: c.NumHeads, KVHeads: c.NumKVHeads,
			HeadDim: c.HeadDim, IntermediateSize: c.IntermediateSize,
			VocabSize: c.VocabSize, Parameters: params, Planes: planes,
			TiedEmbeddings: c.TieWordEmbeddings, TrainedContext: c.MaxPositionEmbeddings,
		},
		Precision: choice,
		Memory: memoryFacts{
			Context: o.Context, CacheDType: o.Cache.String(),
			CacheElementBytes: int64(o.Cache.Size()),
			WeightBytes:       weightBytes, KVBytesPerPosition: perPosition,
			KVBytes: kv, ResidentBytes: weightBytes + kv,
		},
		Hardware:    hw,
		Environment: env,
	}, nil
}

// footprint sums the weight map two ways.
//
// params counts each checkpoint tensor once and is the number a model card
// calls its size. planes counts every declared port and is what the device
// holds: a tied checkpoint binds one file tensor to two ports in two layouts,
// so the embedding table is uploaded twice (004-D7) and the device pays for
// both. Reporting only the first would understate a tied model's footprint by
// the largest tensor it has.
//
// The two byte counts are specs/001-weights.md §5's, per tensor rather than
// over a summed element count: a trailing partial block gets its own scale, so
// the number of int8 scales is a sum of per-tensor block counts.
func footprint(specs []model.WeightSpec) (params, planes, f16Bytes, int8Bytes, int4Bytes int64) {
	seen := make(map[string]bool, len(specs))
	for _, s := range specs {
		n := int64(1)
		for _, d := range s.Shape {
			n *= int64(d)
		}
		planes += n
		f16Bytes += planeBytes(s.Kind, n, weights.F16)
		int8Bytes += planeBytes(s.Kind, n, weights.Int8)
		int4Bytes += planeBytes(s.Kind, n, weights.Int4)
		if !seen[s.Tensor] {
			seen[s.Tensor] = true
			params += n
		}
	}
	return params, planes, f16Bytes, int8Bytes, int4Bytes
}

// planeBytes is what one plane of n elements occupies on the device, at the
// narrow precision or the wide one.
//
// A norm gain is f32 at both, and that is not a rounding detail.
// specs/004-model-graph.md §3 declares the gain ports f32 — nn.Graph.Gain takes
// no policy — and accel binds by exact dtype, so a gain cannot come through
// specs/001-weights.md §2's pipeline at all: it is uploaded wide and the
// quantized path never sees it. Pricing one at f16 understates Qwen3-0.6B by
// 128 KiB, which is small and is not zero, and a `tgo info` that prints a
// footprint the model does not have is the silent number §5 exists to prevent.
func planeBytes(kind model.Kind, n int64, p weights.Precision) int64 {
	if kind == model.KindGain {
		return n * int64(accel.F32.Size())
	}
	// The embedding table is gathered rather than multiplied and accel
	// registers no int4 gather, so a load at int4 pins it to int8. Pricing it
	// at int4 here would print a footprint the model does not have, which is
	// the silent number specs/001-weights.md §5 exists to prevent -- and it is
	// the same argument the gain note above makes, at a different tensor.
	if p == weights.Int4 && kind == model.KindEmbedding {
		p = weights.Int8
	}
	switch p {
	case weights.Int8:
		return n + (n+quant.Int8Block-1)/quant.Int8Block*2
	case weights.Int4:
		return (n+7)/8*4 + (n+int64(quant.Int4Group)-1)/int64(quant.Int4Group)*2*2
	}
	return n * int64(accel.F16.Size())
}

// choosePrecision is specs/001-weights.md §5 and decision 5 of
// specs/000-decisions.md: int8 when the f16 footprint exceeds the budget, f16
// otherwise, and the choice is printed with what it compared.
//
// It restates weights.planLoad's rule, which is unexported and reachable only
// through a load that needs a device and moves every byte of the checkpoint.
// `tgo info` has to print the choice without doing that, so the rule is here
// too and the two are pinned against each other by TestChoosePrecisionMatches-
// TheLoader below. See the discrepancy note in the package's report.
func choosePrecision(policy weights.Precision, f16Bytes, int8Bytes, int4Bytes, budget int64) (precisionFacts, error) {
	p := precisionFacts{
		Requested: policy.String(), F16Bytes: f16Bytes,
		Int8Bytes: int8Bytes, Int4Bytes: int4Bytes, Budget: budget,
	}
	if policy == weights.Inherit {
		// The zero value means "use the load policy", which as a policy is
		// Auto (weights.Inherit). A record that said "inherit" would name an
		// answer nobody asked for.
		p.Requested = weights.Auto.String()
	}
	switch policy {
	case weights.F16, weights.Int8, weights.Int4:
		p.Chosen = policy.String()
		p.Why = fmt.Sprintf("asked for: --precision %s", policy)
	case weights.Inherit, weights.Auto:
		switch {
		case f16Bytes <= budget:
			p.Chosen = weights.F16.String()
			p.Why = fmt.Sprintf("auto: f16 needs %s, which fits the %s budget",
				weights.HumanBytes(f16Bytes), weights.HumanBytes(budget))
		case int8Bytes <= budget:
			p.Chosen = weights.Int8.String()
			p.Why = fmt.Sprintf("auto: f16 needs %s, which is more than the %s budget, so int8 at %s",
				weights.HumanBytes(f16Bytes), weights.HumanBytes(budget), weights.HumanBytes(int8Bytes))
		default:
			p.Chosen = weights.Int4.String()
			p.Why = fmt.Sprintf("auto: int8 needs %s, which is more than the %s budget, so int4 at %s",
				weights.HumanBytes(int8Bytes), weights.HumanBytes(budget), weights.HumanBytes(int4Bytes))
		}
	default:
		return precisionFacts{}, fmt.Errorf("%w: precision policy %v is not f16, int8, int4 or auto",
			errUsage, policy)
	}
	for _, c := range []struct {
		name  string
		bytes int64
	}{{weights.Int8.String(), int8Bytes}, {weights.Int4.String(), int4Bytes}} {
		if p.Chosen == c.name && c.bytes > budget {
			return precisionFacts{}, fmt.Errorf("the model needs %s at %s, which is more than "+
				"the %s budget; no supported precision fits",
				weights.HumanBytes(c.bytes), c.name, weights.HumanBytes(budget))
		}
	}
	return p, nil
}

// kvBytesPerPosition is specs/005-kv-cache.md §3's arithmetic for one position:
//
//	M_kv = 2 · L · C · H_kv · d_h · w
//
// divided by C, so that a caller can multiply by whatever capacity it is
// pricing and a reader can see the per-position cost that makes a long context
// expensive. The leading 2 is the key state and the value state.
//
// L is the layers that **have** a key/value cache, which for a hybrid is one in
// four (specs/023-cache-kinds.md §8). Without that, cacheWidth reports
// `unknown` for every hybrid: it divides the engine's reported bytes by
// 2·L·C·H_kv·d_h and a byte count computed over 16 layers against an L of 64
// leaves a remainder.
func kvBytesPerPosition(c *model.Config, dt accel.DType) int64 {
	return 2 * int64(cachedLayerCount(c)) * int64(c.NumKVHeads) * int64(c.HeadDim) *
		int64(dt.Size())
}

// cachedLayerCount is how many layers of a stack hold a key/value cache.
func cachedLayerCount(c *model.Config) int {
	if c.LayerTypes.Hybrid() {
		return c.LayerTypes.Count(model.LayerFullAttention)
	}
	return c.NumLayers
}

// resolvedInto folds what the engine loaded into the description this process
// predicted from config.json and a device limit.
//
// The predicted footprints stay, because specs/001-weights.md §5 requires the
// comparison to be printed and not only its answer. What the engine resolved
// replaces the answer and the memory that follows from it: `tgo run` and
// `tgo bench` print a precision beside text the model actually produced, and a
// header naming the precision this process guessed while the loader chose
// another is worse under §5 than printing none. A disagreement is stated in the
// reason rather than overwritten.
func resolvedInto(rep modelReport, in engineInfo) modelReport {
	if in.Precision != "" && in.Precision != rep.Precision.Chosen {
		rep.Precision.Why = fmt.Sprintf("%s. The loader resolved %s instead, and %s is what ran",
			rep.Precision.Why, in.Precision, in.Precision)
		rep.Precision.Chosen = in.Precision
	}
	if in.Context > 0 {
		rep.Memory.Context = in.Context
	}
	if in.WeightBytes > 0 {
		rep.Memory.WeightBytes = in.WeightBytes
	}
	if in.CacheBytesPerSession > 0 {
		rep.Memory.KVBytes = in.CacheBytesPerSession
		rep.Memory.KVBytesPerPosition, rep.Memory.CacheElementBytes, rep.Memory.CacheDType =
			cacheWidth(rep.Model, in.CacheBytesPerSession, rep.Memory.Context, rep.Memory.CacheDType)
	}
	rep.Memory.ResidentBytes = rep.Memory.WeightBytes + rep.Memory.KVBytes
	return rep
}

// cacheWidth divides specs/005-kv-cache.md §3's element count back out of the
// cache size the engine reported:
//
//	M_kv = 2 · L · C · H_kv · d_h · w   ⇒   w = M_kv / (2 · L · C · H_kv · d_h)
//
// The width is measured rather than assumed because it is going to move. §3
// states that the design tgo builds is f16 and that the f32 constraint which
// forced the wider store is closed upstream, so the day the key and value
// states narrow, a table that kept printing f32 would overstate every cache it
// prices by a factor of two while the engine reported the truth.
//
// A width that does not divide evenly keeps the total and loses the label: a
// remainder means the engine's cache is not the shape this formula describes,
// and a dtype printed beside a number it does not explain is worse than none.
func cacheWidth(m modelFacts, cacheBytes int64, context int, predicted string) (perPosition, width int64, dtype string) {
	// The layers that hold a cache, not the layers of the stack: a hybrid's
	// three gated-delta layers in four write no key or value, so dividing by
	// the stack leaves a remainder and loses the label
	// (specs/023-cache-kinds.md §8).
	cached := m.CachedLayers
	if cached == 0 {
		cached = m.Layers
	}
	elements := 2 * int64(cached) * int64(m.KVHeads) * int64(m.HeadDim) * int64(context)
	if elements <= 0 || cacheBytes%elements != 0 {
		return cacheBytes / max(int64(context), 1), 0,
			fmt.Sprintf("unknown: %s over %d positions is not 2 · %d · C · %d · %d elements of a whole number of bytes",
				weights.HumanBytes(cacheBytes), context, cached, m.KVHeads, m.HeadDim)
	}
	width = cacheBytes / elements
	dtype = predicted
	if int64(dtypeSize(predicted)) != width {
		dtype = fmt.Sprintf("%d bytes per element (the engine's; this build predicted %s)", width, predicted)
	}
	return cacheBytes / int64(context), width, dtype
}

// dtypeSize is the width of a dtype named by [accel.DType.String].
//
// A reverse lookup over the three widths a cache can hold, because accel names
// a dtype and does not parse one. An unknown name is zero, which never matches
// a measured width and so is reported as a disagreement.
func dtypeSize(name string) int {
	for _, dt := range []accel.DType{accel.F32, accel.F16, accel.I8} {
		if dt.String() == name {
			return dt.Size()
		}
	}
	return 0
}

// infoFlagSet declares what `tgo info` accepts. See [runFlagSet] for why
// declaring is separate from parsing.
func infoFlagSet() (*flag.FlagSet, *infoFlags) {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	return fs, &infoFlags{
		precision: fs.String("precision", "auto", "f16, int8, int4 or auto"),
		context:   fs.Int("context", defaultContext, "KV cache capacity in positions"),
		budget:    fs.Int64("budget", 0, "device bytes the weights may occupy; 0 asks the device"),
		device:    fs.String("device", "auto", "auto, cpu or metal"),
	}
}

// infoFlags holds `tgo info`'s flag values.
type infoFlags struct {
	precision, device *string
	context           *int
	budget            *int64
}

// cmdInfo prints what a model is, what precision it will run at and why, and
// what it costs in memory at the capacity it will run at.
func cmdInfo(args []string, stdout, stderr io.Writer) error {
	fs, f := infoFlagSet()
	dir, err := modelDir(fs, args)
	if err != nil {
		return err
	}
	policy, err := parsePrecision(*f.precision)
	if err != nil {
		return err
	}
	dev, err := parseDevice(*f.device)
	if err != nil {
		return err
	}
	rep, err := openAndDescribe(dir, describeOptions{
		Policy: policy, Context: *f.context, Budget: *f.budget, Device: dev})
	if err != nil {
		return err
	}
	renderInfo(stdout, rep)
	return nil
}

// openAndDescribe reads the model directory and the device, and describes the
// two together. It is the wiring `info`, `run` and `bench` share.
func openAndDescribe(dir string, o describeOptions) (modelReport, error) {
	b, err := model.Open(dir)
	if err != nil {
		return modelReport{}, err
	}
	dev, err := openDevice(o.Device)
	if err != nil {
		return modelReport{}, err
	}
	defer func() { _ = dev.Close() }()
	return describe(dir, b, o, stampHardware(dev), stampEnvironment())
}

// renderInfo writes a description for a person to read.
func renderInfo(w io.Writer, r modelReport) {
	m, p, mem := r.Model, r.Precision, r.Memory
	_, _ = fmt.Fprintf(w, "model      %s\n", m.Dir)
	_, _ = fmt.Fprintf(w, "  architecture      %s\n", m.Architecture)
	_, _ = fmt.Fprintf(w, "  parameters        %s (%d distinct tensors' elements)\n", humanCount(m.Parameters), m.Parameters)
	_, _ = fmt.Fprintf(w, "  hidden            %d over %d layers\n", m.HiddenSize, m.Layers)
	_, _ = fmt.Fprintf(w, "  heads             %d query, %d key/value, head_dim %d\n", m.Heads, m.KVHeads, m.HeadDim)
	_, _ = fmt.Fprintf(w, "  mlp               %d\n", m.IntermediateSize)
	_, _ = fmt.Fprintf(w, "  vocabulary        %d\n", m.VocabSize)
	_, _ = fmt.Fprintf(w, "  tied embeddings   %t\n", m.TiedEmbeddings)
	_, _ = fmt.Fprintf(w, "  trained context   %d positions (advisory; capacity is a session parameter)\n", m.TrainedContext)

	_, _ = fmt.Fprintf(w, "\nprecision  %s\n", p.Chosen)
	_, _ = fmt.Fprintf(w, "  why               %s\n", p.Why)
	_, _ = fmt.Fprintf(w, "  f16 footprint     %s\n", weights.HumanBytes(p.F16Bytes))
	_, _ = fmt.Fprintf(w, "  int8 footprint    %s\n", weights.HumanBytes(p.Int8Bytes))
	_, _ = fmt.Fprintf(w, "  int4 footprint    %s\n", weights.HumanBytes(p.Int4Bytes))
	_, _ = fmt.Fprintf(w, "  budget            %s\n", weights.HumanBytes(p.Budget))
	if m.TiedEmbeddings {
		_, _ = fmt.Fprintf(w, "  note              the footprints cover %s device elements: a tied checkpoint\n"+
			"                    uploads the embedding table twice, once per layout (004-D7)\n", humanCount(m.Planes))
	}

	_, _ = fmt.Fprintf(w, "\nmemory     at %d positions of context\n", mem.Context)
	_, _ = fmt.Fprintf(w, "  weights           %s at %s\n", weights.HumanBytes(mem.WeightBytes), p.Chosen)
	_, _ = fmt.Fprintf(w, "  kv cache          %s = 2 · %d layers · %d positions · %d kv heads · %d head_dim · %d bytes (%s)\n",
		weights.HumanBytes(mem.KVBytes), m.Layers, mem.Context, m.KVHeads, m.HeadDim, mem.CacheElementBytes, mem.CacheDType)
	_, _ = fmt.Fprintf(w, "  per position      %s\n", weights.HumanBytes(mem.KVBytesPerPosition))
	_, _ = fmt.Fprintf(w, "  resident          %s (weights plus cache; excludes activations and the host heap)\n",
		weights.HumanBytes(mem.ResidentBytes))

	_, _ = fmt.Fprintf(w, "\ndevice     %s\n", r.Hardware.Backend)
	_, _ = fmt.Fprintf(w, "  name              %s (%s)\n", r.Hardware.Device, r.Hardware.Vendor)
	_, _ = fmt.Fprintf(w, "  software          %t\n", r.Hardware.Software)
	_, _ = fmt.Fprintf(w, "  unified memory    %t\n", r.Hardware.UnifiedMemory)
	_, _ = fmt.Fprintf(w, "  max pool          %s\n", weights.HumanBytes(r.Hardware.MaxPoolBytes))
	_, _ = fmt.Fprintf(w, "  build             %s %s/%s, accel %s\n",
		r.Environment.Go, r.Environment.GOOS, r.Environment.GOARCH, r.Environment.Accel)
}
