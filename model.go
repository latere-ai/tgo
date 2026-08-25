// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"

	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/safetensors"
	"github.com/latere-ai/tgo/tokenizer"
	"github.com/latere-ai/tgo/weights"
)

// Info describes an open model: what it is, how big it is, and what [Open]
// resolved every automatic choice to.
//
// Every number is read from the checkpoint rather than written down here
// (004-D8), and the three resolved fields are what a caller who passed
// [AutoDevice] or [AutoPrecision] has no other way to learn.
type Info struct {
	// Architecture is config.json's architectures[0].
	Architecture string

	// Layers, HiddenSize, Heads, KVHeads, HeadDim, IntermediateSize and
	// VocabSize are specs/004-model-graph.md §3's L, d, H, H_kv, d_h, f and V.
	Layers           int
	HiddenSize       int
	Heads            int
	KVHeads          int
	HeadDim          int
	IntermediateSize int
	VocabSize        int

	// TrainedContext is max_position_embeddings, which is advisory: capacity
	// is a session parameter (005-D2). It is reported so a caller choosing
	// [WithContext] knows what the model was trained for.
	TrainedContext int

	// Context is the KV capacity a new session gets by default.
	Context int

	// Device is the accelerator that was opened, resolved from [WithDevice].
	Device Device

	// Precision is what the weights are stored as, resolved from
	// [WithPrecision].
	Precision Precision

	// WeightBytes is what the weights occupy on the device, block scales
	// included. It is not the pool footprint, which is larger by each buffer's
	// rounding up to the device's suballocation granularity.
	WeightBytes int64

	// CacheBytesPerSession is what one session's key and value states cost at
	// Context positions: specs/005-kv-cache.md §3's M_kv.
	CacheBytesPerSession int64
}

// Model is a loaded model: a device, the weights on it, and the compiled plans
// that read them. One per process per model.
//
// A Model is safe for concurrent use, and that needs a lock rather than only a
// claim (007-D9). [tensor.PlanCache] returns the same *[tensor.Plan] for an
// identical graph, and a plan refuses a second submission while one is in
// flight, so two sessions decoding at once share one decode plan and the second
// would get a failed fence — not a data race, which means a -race test stays
// green while a server returns errors under load. The lock is held across
// binding, submission, the fence and the readback: every one of those touches
// the queue, and [accel.Queue.ReadBuffer] flushes it.
//
// A [Session] is deliberately not concurrency-safe (007-D1). The two are not in
// tension: this lock protects a resource accel shares between sessions, and the
// Session's absence of one reports a mistake only the caller can make.
type Model struct {
	builder model.Builder
	cfg     *model.Config
	tok     *tokenizer.Tokenizer
	special specials

	dev   *accel.Device
	rt    *tensor.Runtime
	cache *tensor.PlanCache

	set   *weights.Set
	gains map[string]*accel.Buffer

	// weightBind is every weight port's binding, built once. A submission
	// clones it and adds the step's own ports, so the hot path never rebuilds
	// three hundred entries that did not change.
	weightBind map[string]accel.BufferView

	// stored answers nn.Graph.Stored: how the loader stored one weight port.
	stored func(string) accel.DType

	info    Info
	context int

	// mu is 007-D9's submission lock.
	mu sync.Mutex

	closeOnce sync.Once
	closeErr  error
}

// specials are the token ids the decode loop reads structure from.
//
// By id and never by matching decoded text: a stop that is found in text has
// the boundary problem specs/003-chat-template.md 003-D6 rejects for turn
// markers, and it fails the same way — a user who asks the model to describe
// "</think>" would have the description cut short.
type specials struct {
	imEnd     int
	endOfText int
	think     []int
	thinkEnd  int
	toolCall  int
	toolEnd   int
}

// Open loads a model directory and prepares a device.
//
// The directory holds config.json, tokenizer.json, and either one
// model.safetensors or a set of shards with an index beside them. Nothing about
// it is fetched: a checkpoint is a file someone already has.
func Open(dir string, opts ...Option) (*Model, error) {
	o := defaults()
	for _, fn := range opts {
		fn(&o)
	}
	if o.context <= 0 {
		return nil, fmt.Errorf("tgo: WithContext is %d; a cache holds at least one position",
			o.context)
	}

	b, err := model.Open(dir)
	if err != nil {
		return nil, err
	}
	cfg := b.Config()

	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("tgo: %w", err)
	}

	repo, err := safetensors.OpenRepo(dir)
	if err != nil {
		return nil, err
	}
	defer func() { _ = repo.Close() }()

	specs := b.Weights()
	if err := checkAgainst(repo, specs); err != nil {
		return nil, err
	}

	m := &Model{
		builder: b,
		cfg:     cfg,
		tok:     tok,
		special: resolveSpecials(tok),
		context: o.context,
		gains:   map[string]*accel.Buffer{},
	}

	// 005-D3: the number before the allocation, not after the failure. A
	// session's cache is the largest thing a request costs, and a caller who
	// raised the capacity is the one who wants to know.
	perSession := cacheBytes(cfg, o.context)
	if o.context > DefaultContext {
		fmt.Fprintf(os.Stderr, "tgo: a %d-position context costs %s of key/value cache "+
			"per session (%d layers x %d positions x %d kv heads x %d head dim x 2 states "+
			"x 4 bytes)\n", o.context, bytesText(perSession), cfg.NumLayers, o.context,
			cfg.NumKVHeads, cfg.HeadDim)
	}

	dev, resolved, err := openDevice(o.device)
	if err != nil {
		return nil, err
	}
	m.dev = dev

	if err := m.load(repo, specs, o.precision); err != nil {
		_ = m.Close()
		return nil, err
	}

	rt, err := tensor.NewRuntime(dev)
	if err != nil {
		_ = m.Close()
		return nil, fmt.Errorf("tgo: %w", err)
	}
	m.rt = rt
	m.cache = tensor.NewPlanCache(rt)

	rep := m.set.Report()
	m.info = Info{
		Architecture:         cfg.Architecture,
		Layers:               cfg.NumLayers,
		HiddenSize:           cfg.HiddenSize,
		Heads:                cfg.NumHeads,
		KVHeads:              cfg.NumKVHeads,
		HeadDim:              cfg.HeadDim,
		IntermediateSize:     cfg.IntermediateSize,
		VocabSize:            cfg.VocabSize,
		TrainedContext:       cfg.MaxPositionEmbeddings,
		Context:              o.context,
		Device:               resolved,
		Precision:            fromLoader(rep.Chosen),
		WeightBytes:          rep.Bytes + gainBytes(specs, cfg),
		CacheBytesPerSession: perSession,
	}
	return m, nil
}

// Info reports what this model is and what every automatic choice resolved to.
func (m *Model) Info() Info { return m.info }

// Close releases the plans, the weights and the device. It is safe to call more
// than once, and reports every failure rather than the first: device memory
// left behind is not visible from the caller's side.
//
// Close every [Session] first. accel closes in order rather than recursively,
// so a device with live buffers refuses to close and says how many it has.
func (m *Model) Close() error {
	m.closeOnce.Do(func() {
		var errs []error
		if m.dev != nil {
			errs = append(errs, m.dev.Queue().Flush().Wait())
		}
		if m.cache != nil {
			errs = append(errs, m.cache.Close())
		}
		if m.rt != nil {
			errs = append(errs, m.rt.Close())
		}
		for _, b := range m.gains {
			errs = append(errs, b.Close())
		}
		if m.set != nil {
			errs = append(errs, m.set.Close())
		}
		if m.dev != nil {
			errs = append(errs, m.dev.Close())
		}
		m.closeErr = errors.Join(errs...)
	})
	return m.closeErr
}

// openDevice opens the accelerator an option names, and reports which it was.
func openDevice(want Device) (*accel.Device, Device, error) {
	switch want {
	case CPU:
		d, err := accel.OpenCPU(accel.CPUOptions{})
		return d, CPU, wrapOpen(err, CPU)
	case Metal:
		d, err := accel.OpenBest(accel.Policy{Prefer: []accel.Backend{accel.BackendMetal}})
		return d, Metal, wrapOpen(err, Metal)
	case AutoDevice:
		d, err := accel.OpenBest(accel.Policy{AllowCPU: true})
		if err != nil {
			return nil, AutoDevice, wrapOpen(err, AutoDevice)
		}
		if d.Info().Backend == accel.BackendCPU {
			return d, CPU, nil
		}
		return d, Metal, nil
	}
	return nil, want, fmt.Errorf("tgo: WithDevice is %v; it is one of auto, cpu or metal", want)
}

func wrapOpen(err error, d Device) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("tgo: opening the %v device: %w", d, err)
}

// checkAgainst holds specs/004-model-graph.md §4's map against the
// checkpoint's own header, and resolves the one case a header cannot settle.
//
// A checkpoint that sets tie_word_embeddings and also ships lm_head.weight is a
// contradiction on the evidence a header carries, and Qwen3-0.6B is exactly
// that file with two byte-identical planes (004-D10). The comparator is the
// engine's because the engine is what holds the file open.
func checkAgainst(repo *safetensors.Repo, specs []model.WeightSpec) error {
	have := map[string][]int{}
	for _, name := range repo.Names() {
		e, _, ok := repo.Tensor(name)
		if !ok {
			return fmt.Errorf("tgo: %q is named by the checkpoint index and is in no shard",
				name)
		}
		have[name] = e.Shape
	}
	return model.Check(specs, have, model.WithPlaneComparator(func(a, b string) (bool, error) {
		// One plane at a time, hashed and released. Comparing the two slices
		// directly would hold 622 MB of a 0.6B checkpoint live at once, to
		// answer a question a digest answers.
		ha, err := planeDigest(repo, a)
		if err != nil {
			return false, err
		}
		hb, err := planeDigest(repo, b)
		if err != nil {
			return false, err
		}
		return ha == hb, nil
	}))
}

func planeDigest(repo *safetensors.Repo, name string) ([32]byte, error) {
	_, file, ok := repo.Tensor(name)
	if !ok {
		return [32]byte{}, fmt.Errorf("tgo: the checkpoint has no %q", name)
	}
	raw, err := file.Bytes(name)
	if err != nil {
		return [32]byte{}, fmt.Errorf("tgo: %q: %w", name, err)
	}
	return sha256.Sum256(raw), nil
}

// load converts every weight and uploads it.
//
// The norm gains do not go through the loader, and that is not a shortcut: see
// [loadGains].
func (m *Model) load(repo *safetensors.Repo, specs []model.WeightSpec, p Precision) error {
	decls := make([]weights.Tensor, 0, len(specs))
	for _, s := range specs {
		if s.Kind == model.KindGain {
			continue
		}
		d := weights.Tensor{Name: s.Tensor, As: s.Port, Transpose: s.Transpose}
		if s.Permute {
			d.HeadDim = m.cfg.HeadDim
		}
		decls = append(decls, d)
	}
	// The log is stderr when the policy is Auto and silent when it is not.
	// specs/001-weights.md §5 requires the precision *choice* to be printed and
	// never silent, and a caller who wrote WithPrecision(F16) made no choice
	// for the loader to announce back to them.
	log := io.Writer(io.Discard)
	if p == AutoPrecision {
		log = os.Stderr
	}
	set, err := weights.Load(m.dev, repo, decls, weights.Options{
		Policy: toLoader(p),
		Log:    log,
	})
	if err != nil {
		return err
	}
	m.set = set

	if err := m.loadGains(repo, specs); err != nil {
		return err
	}

	m.stored = func(name string) accel.DType {
		v, ok := set.Get(name)
		if !ok {
			return accel.F16
		}
		if v.Precision == weights.Int8 {
			return accel.I8
		}
		return accel.F16
	}

	// Every weight port, bound once. specs/007-engine.md §6: every model
	// tensor is a tensor.Weight port, and declaring one as an Input does not
	// give a wrong answer, it gives a plan cache that misses on every step.
	m.weightBind = map[string]accel.BufferView{}
	for _, name := range set.Names() {
		v, _ := set.Get(name)
		dt := accel.F16
		if v.Precision == weights.Int8 {
			dt = accel.I8
		}
		if err := bindBuffer(m.weightBind, name, v.Data, dt, v.Elements); err != nil {
			return err
		}
		if v.Scales != nil {
			n := v.Scales.Count()
			if err := bindBuffer(m.weightBind, name+scaleSuffix, v.Scales, accel.F16, n); err != nil {
				return err
			}
		}
	}
	for name, buf := range m.gains {
		n := buf.Count()
		if err := bindBuffer(m.weightBind, name, buf, accel.F32, n); err != nil {
			return err
		}
	}
	// The gain uploads are queue writes and a queue write is batched. They
	// finish here, before the first plan reads them and before anything can
	// close a buffer that still has one outstanding.
	if err := m.dev.Queue().Flush().Wait(); err != nil {
		return fmt.Errorf("tgo: completing the weight uploads: %w", err)
	}
	return nil
}

func bindBuffer(into map[string]accel.BufferView, name string, buf *accel.Buffer,
	dt accel.DType, count int) error {

	v, err := buf.View(0, count)
	if err != nil {
		return fmt.Errorf("tgo: binding %q: %w", name, err)
	}
	v.DType = dt
	into[name] = v
	return nil
}

// cacheBytes is specs/005-kv-cache.md §3's M_kv for one session, at f32.
//
// A function rather than a comment, because a memory model nobody executes is a
// comment (005 §7).
func cacheBytes(c *model.Config, capacity int) int64 {
	const f32 = 4
	return 2 * int64(c.NumLayers) * int64(capacity) * int64(c.NumKVHeads) *
		int64(c.HeadDim) * f32
}

// bytesText renders a byte count the way a person reads one.
func bytesText(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.2f MiB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.2f KiB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

func toLoader(p Precision) weights.Precision {
	switch p {
	case F16:
		return weights.F16
	case Int8:
		return weights.Int8
	}
	return weights.Auto
}

func fromLoader(p weights.Precision) Precision {
	switch p {
	case weights.F16:
		return F16
	case weights.Int8:
		return Int8
	}
	return AutoPrecision
}

// resolveSpecials looks up the ids the decode loop reads structure from.
//
// A token the tokenizer does not hold resolves to -1, which no id equals, so a
// checkpoint whose vocabulary has no thinking markers streams as plain text
// instead of failing to open.
func resolveSpecials(t *tokenizer.Tokenizer) specials {
	id := func(s string) int {
		if v, ok := t.Special(s); ok {
			return v
		}
		return -1
	}
	// Two spellings of the thinking opener, because a checkpoint's added
	// tokens hold "<think>" and "<think>\n" as separate entries and the model
	// may emit either.
	open := []int{id("<think>"), id("<think>\n")}
	return specials{
		imEnd:     id("<|im_end|>"),
		endOfText: id("<|endoftext|>"),
		think:     open,
		thinkEnd:  id("</think>"),
		toolCall:  id("<tool_call>"),
		toolEnd:   id("</tool_call>"),
	}
}

// scaleSuffix is the name nn gives a quantized weight's scale plane. It is
// spelled here rather than imported so that this package does not depend on nn
// for one string; a test asserts the two agree.
const scaleSuffix = ".scales"

// renderer is the model's chat template.
func (m *Model) renderer() chat.Renderer { return m.builder.Template() }
