// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.design/x/accel"

	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/safetensors"
	"github.com/latere-ai/tgo/weights"
)

// TestKVCacheArithmetic checks specs/005-kv-cache.md §3 against the worked
// example the spec states: a Qwen3-4B-shaped model, L=36, H_kv=8, d_h=128, is
// 2·36·8·128 = 73728 elements per position, which is 288 KB in f32 and 144 KB
// in f16.
//
// The numbers are the spec's own, so this is the one test that could catch the
// command line pricing a cache by a formula nobody wrote down.
func TestKVCacheArithmetic(t *testing.T) {
	c := &model.Config{NumLayers: 36, NumKVHeads: 8, HeadDim: 128}
	for _, tc := range []struct {
		dt   accel.DType
		want int64
	}{
		{accel.F32, 288 * 1024},
		{accel.F16, 144 * 1024},
	} {
		if got := kvBytesPerPosition(c, tc.dt); got != tc.want {
			t.Errorf("kvBytesPerPosition(%v) = %d, want %d", tc.dt, got, tc.want)
		}
	}
	// And §3's table: 4096 positions of f32 is 1.21 GB, which is 1.13 GiB.
	total := kvBytesPerPosition(c, accel.F32) * 4096
	if got, want := float64(total)/1e9, 1.21; math.Abs(got-want) > 0.005 {
		t.Errorf("4096 positions cost %.2f GB, and §3's table says %.2f GB", got, want)
	}
}

// TestFootprintCountsTheTiedPlaneTwice pins the distinction between what a
// model card calls its size and what the device holds. A tied checkpoint binds
// one file tensor to two ports in two layouts (004-D7), and a footprint that
// counted it once would understate a small model by its largest tensor.
func TestFootprintCountsTheTiedPlaneTwice(t *testing.T) {
	b := syntheticBuilder(t)
	specs := b.Weights()
	params, planes, f16, int8, int4 := footprint(specs)
	_ = int4

	// The tied plane is the one checkpoint tensor two specs name.
	count := make(map[string]int, len(specs))
	var shared int64
	for _, s := range specs {
		count[s.Tensor]++
		if count[s.Tensor] == 2 {
			shared = 1
			for _, d := range s.Shape {
				shared *= int64(d)
			}
		}
	}
	if shared == 0 {
		t.Fatal("no checkpoint tensor is bound to two ports; the synthetic model is tied and should have one")
	}
	embed := shared
	if !b.Config().TieWordEmbeddings {
		t.Fatal("the synthetic model is expected to be tied")
	}
	if planes-params != embed {
		t.Errorf("planes - params = %d, want the embedding table's %d elements: the tied head "+
			"is a second plane on the device", planes-params, embed)
	}
	// Two bytes for every element the loader narrows and four for every gain,
	// which is what [planeBytes] prices and what the device holds.
	var want int64
	for _, sp := range specs {
		n := int64(1)
		for _, d := range sp.Shape {
			n *= int64(d)
		}
		want += planeBytes(sp.Kind, n, weights.F16)
	}
	if f16 != want {
		t.Errorf("f16 footprint = %d, want %d", f16, want)
	}
	if f16 <= planes*2 {
		t.Error("the f16 footprint is two bytes per element throughout; the gains are f32")
	}
	if int8 <= planes || int8 >= f16 {
		t.Errorf("int8 footprint = %d, want it between the %d quants and the %d f16 bytes",
			int8, planes, f16)
	}
}

// TestGainsArePricedAtF32AtEveryPrecision is the rule the real checkpoint
// taught this package.
//
// specs/004-model-graph.md §3 declares a norm gain as an f32 port and accel
// binds by exact dtype, so a gain never enters specs/001-weights.md §2's
// precision pipeline: the engine uploads it wide whatever --precision says.
// A footprint that narrowed one would print a number the device never has, and
// asking for int8 would appear to shrink weights that do not shrink.
func TestGainsArePricedAtF32AtEveryPrecision(t *testing.T) {
	const n = 1024
	for _, p := range []weights.Precision{weights.F16, weights.Int8, weights.Int4} {
		if got := planeBytes(model.KindGain, n, p); got != n*4 {
			t.Errorf("a %d-element gain costs %d bytes at %v, want %d", n, got, p, n*4)
		}
	}
	if got := planeBytes(model.KindProjection, n, weights.F16); got != n*2 {
		t.Errorf("an f16 projection costs %d bytes, want %d", got, n*2)
	}
	if got := planeBytes(model.KindProjection, n, weights.Int8); got >= n*2 || got <= n {
		t.Errorf("an int8 projection costs %d bytes, want a quant plus its scales", got)
	}
	// int4 is under half of int8 and over the codes alone: (n+7)/8 words of
	// four bytes is n/2, plus a scale and a zero per group.
	i8 := planeBytes(model.KindProjection, n, weights.Int8)
	if got := planeBytes(model.KindProjection, n, weights.Int4); got >= i8 || got <= n/2 {
		t.Errorf("an int4 projection costs %d bytes against int8's %d, want it between "+
			"the %d bytes of codes and int8", got, i8, n/2)
	}
	// The embedding is gathered and accel registers no int4 gather, so at int4
	// it costs what int8 costs. A footprint that priced it at int4 would print
	// a number the device never has.
	if got, want := planeBytes(model.KindEmbedding, n, weights.Int4),
		planeBytes(model.KindEmbedding, n, weights.Int8); got != want {
		t.Errorf("a gathered plane costs %d bytes at int4 and %d at int8; it cannot "+
			"pack, so the two are the same number", got, want)
	}
	// A partial block gets its own scale, which is why the count is per plane
	// and not over a summed element count.
	if got, want := planeBytes(model.KindEmbedding, 1, weights.Int8), int64(1+2); got != want {
		t.Errorf("a one-element int8 plane costs %d bytes, want %d", got, want)
	}
}

// TestFootprintMatchesTheLoader is the pin.
//
// choosePrecision and footprint restate weights.planLoad, which is unexported
// and reachable only through a load that needs a device and moves every byte of
// the checkpoint. `tgo info` prints the choice without doing that, so the rule
// exists twice and the two must not drift. This test runs a real load over a
// synthesised checkpoint and compares the loader's own Report against what this
// package computed from the declared shapes.
//
// The checkpoint is four tensors of a few hundred bytes, so it costs nothing:
// the shapes are chosen so that one of them ends in a partial quant block,
// which is where a footprint summed over a total element count goes wrong.
func TestFootprintMatchesTheLoader(t *testing.T) {
	planes := []struct {
		name  string
		shape []int
	}{
		{"a", []int{3, 5}},   // 15 elements: one partial block
		{"b", []int{64, 2}},  // 128 elements: exactly four blocks
		{"c", []int{7, 7}},   // 49 elements: one full block and a partial
		{"d", []int{16, 16}}, // 256 elements: eight blocks
	}
	dir := t.TempDir()
	entries := make(map[string][]float32, len(planes))
	for _, p := range planes {
		n := 1
		for _, d := range p.shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(i%7) / 8 // exact in f16, so nothing saturates
		}
		entries[p.name] = v
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), planes, entries)

	repo, err := safetensors.OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo: %v", err)
	}
	defer func() { _ = repo.Close() }()

	dev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open the CPU device: %v", err)
	}
	defer func() { _ = dev.Close() }()

	decls := make([]weights.Tensor, 0, len(planes))
	specs := make([]model.WeightSpec, 0, len(planes))
	for _, p := range planes {
		decls = append(decls, weights.Tensor{Name: p.name})
		// KindProjection explicitly: the zero Kind is KindGain, which
		// [planeBytes] prices at f32 because the engine uploads it there
		// (specs/004-model-graph.md §3). These planes go through the loader,
		// so they are what the loader narrows.
		specs = append(specs, model.WeightSpec{
			Tensor: p.name, Port: p.name, Shape: p.shape, Kind: model.KindProjection,
		})
	}

	// A budget the f16 footprint fits, and one it does not, so that both arms
	// of decision 5 are compared against the loader rather than only the arm
	// this machine happens to take.
	_, _, f16, int8, int4 := footprint(specs)
	for _, budget := range []int64{f16 * 2, f16 - 1} {
		set, err := weights.Load(dev, repo, decls, weights.Options{Budget: budget, Log: io.Discard})
		if err != nil {
			t.Fatalf("weights.Load with a %d budget: %v", budget, err)
		}
		got := set.Report()
		_ = set.Close()

		if got.F16Bytes != f16 || got.Int8Bytes != int8 || got.Int4Bytes != int4 {
			t.Errorf("the loader reports f16=%d int8=%d int4=%d and this package "+
				"computed f16=%d int8=%d int4=%d", got.F16Bytes, got.Int8Bytes,
				got.Int4Bytes, f16, int8, int4)
		}
		mine, err := choosePrecision(weights.Auto, f16, int8, int4, budget)
		if err != nil {
			t.Fatalf("choosePrecision: %v", err)
		}
		if mine.Chosen != got.Chosen.String() {
			t.Errorf("with a %d budget the loader chose %v and this package chose %s",
				budget, got.Chosen, mine.Chosen)
		}
	}
}

// writeSafetensors writes a minimal checkpoint: an 8-byte little-endian header
// length, a JSON header, then the planes in declaration order.
func writeSafetensors(t *testing.T, path string, planes []struct {
	name  string
	shape []int
}, data map[string][]float32) {
	t.Helper()
	header := make(map[string]any, len(planes))
	var body []byte
	for _, p := range planes {
		v := data[p.name]
		begin := int64(len(body))
		for _, f := range v {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(f))
			body = append(body, b[:]...)
		}
		header[p.name] = map[string]any{
			"dtype":        "F32",
			"shape":        p.shape,
			"data_offsets": []int64{begin, int64(len(body))},
		}
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal the header: %v", err)
	}
	out := make([]byte, 8, 8+len(raw)+len(body))
	binary.LittleEndian.PutUint64(out, uint64(len(raw)))
	out = append(out, raw...)
	out = append(out, body...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestChoosePrecision(t *testing.T) {
	const f16, int8 = 1000, 600
	for _, tc := range []struct {
		name       string
		policy     weights.Precision
		budget     int64
		wantChosen string
		wantWhy    string
	}{
		{"auto fits", weights.Auto, 2000, "f16", "which fits"},
		{"auto does not fit", weights.Auto, 800, "int8", "which is more than"},
		// The boundary weights.planLoad draws, both sides of it. It refuses
		// f16 on "more than the budget" and not on "as much as", so a model
		// that exactly fills the pool loads at f16 and one byte more of it
		// does not. A comparison off by one here silently halves the precision
		// of every model whose weights are the size of the device's pool.
		{"auto at exactly the budget", weights.Auto, 1000, "f16", "which fits"},
		{"auto one byte over the budget", weights.Auto, 999, "int8", "which is more than"},
		{"the zero value is auto", weights.Inherit, 2000, "f16", "which fits"},
		{"asked for f16", weights.F16, 10, "f16", "asked for"},
		{"asked for int8", weights.Int8, 10, "int8", "asked for"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := choosePrecision(tc.policy, f16, int8, int8/2, tc.budget)
			// Asking for f16 or int8 above the budget is the caller's
			// statement and not auto's decision, so only int8 above the budget
			// is refused; f16 above it is a choice the loader will attempt.
			if tc.policy == weights.Int8 && int8 > tc.budget {
				if err == nil {
					t.Fatal("int8 above the budget was accepted")
				}
				return
			}
			if err != nil {
				t.Fatalf("choosePrecision: %v", err)
			}
			if got.Chosen != tc.wantChosen {
				t.Errorf("chose %s, want %s", got.Chosen, tc.wantChosen)
			}
			if !strings.Contains(got.Why, tc.wantWhy) {
				t.Errorf("why = %q, want it to contain %q", got.Why, tc.wantWhy)
			}
			// specs/001-weights.md §5: the choice is printed with what it
			// compared, never as a bare word.
			if got.F16Bytes != f16 || got.Int8Bytes != int8 || got.Budget != tc.budget {
				t.Errorf("the evidence is %+v, want f16=%d int8=%d budget=%d",
					got, f16, int8, tc.budget)
			}
		})
	}
}

// TestChoosePrecisionRefusesAModelThatCannotFit walks decision 5 to its end: a
// model too large at the narrowest form has no precision left, and the refusal
// names both numbers rather than loading and failing part way through.
func TestChoosePrecisionRefusesAModelThatCannotFit(t *testing.T) {
	// int4 is now the last resort, so it is int4 that has to not fit.
	_, err := choosePrecision(weights.Auto, 4000, 2200, 2100, 2000)
	if err == nil {
		t.Fatal("a model larger than the budget at int4 was accepted")
	}
	for _, want := range []string{"no supported precision fits", "int4", "2.05 KiB", "1.95 KiB"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %v, want it to contain %q", err, want)
		}
	}

	// And int8 not fitting is no longer the end: auto narrows again, and only
	// refuses when the narrowest does not fit either.
	got, err := choosePrecision(weights.Auto, 4000, 2200, 1900, 2000)
	if err != nil {
		t.Fatalf("a model that fits at int4 was refused: %v", err)
	}
	if got.Chosen != weights.Int4.String() {
		t.Errorf("auto chose %s for a model that fits only at int4", got.Chosen)
	}
}

func TestChoosePrecisionRefusesAnUnknownPolicy(t *testing.T) {
	if _, err := choosePrecision(weights.Precision(99), 1, 1, 1, 1); err == nil {
		t.Fatal("an unknown precision policy was accepted")
	}
}

func TestDescribe(t *testing.T) {
	b := syntheticBuilder(t)
	hw := hardware{Backend: "cpu", Device: "test", MaxPoolBytes: 1 << 30}
	rep, err := describe("models/synthetic", b, describeOptions{Context: 2048}, hw, stampEnvironment())
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	c := b.Config()
	if rep.Model.Layers != c.NumLayers || rep.Model.HeadDim != c.HeadDim || rep.Model.KVHeads != c.NumKVHeads {
		t.Errorf("the model facts do not match the config: %+v", rep.Model)
	}
	// The default cache dtype is f32, which is what model.GraphSpec records.
	if rep.Memory.CacheDType != accel.F32.String() || rep.Memory.CacheElementBytes != 4 {
		t.Errorf("cache dtype = %s at %d bytes, want f32 at 4",
			rep.Memory.CacheDType, rep.Memory.CacheElementBytes)
	}
	if want := kvBytesPerPosition(c, accel.F32) * 2048; rep.Memory.KVBytes != want {
		t.Errorf("kv bytes = %d, want %d", rep.Memory.KVBytes, want)
	}
	if rep.Memory.ResidentBytes != rep.Memory.WeightBytes+rep.Memory.KVBytes {
		t.Error("the resident footprint is not the weights plus the cache")
	}
	// The budget defaults to the device's, which is what weights.Load does
	// with a zero Options.Budget.
	if rep.Precision.Budget != hw.MaxPoolBytes {
		t.Errorf("budget = %d, want the device's %d", rep.Precision.Budget, hw.MaxPoolBytes)
	}
	// A budget the caller states wins over the device's, and a budget between
	// the two footprints takes decision 5's other arm.
	tight := rep.Precision.Int8Bytes + 1
	stated, err := describe("d", b, describeOptions{Context: 16, Budget: tight}, hw, stampEnvironment())
	if err != nil {
		t.Fatalf("describe with a stated budget: %v", err)
	}
	if stated.Precision.Budget != tight {
		t.Errorf("budget = %d, want the stated %d", stated.Precision.Budget, tight)
	}
	if stated.Precision.Chosen != "int8" {
		t.Errorf("with a budget of %d the choice is %s, want int8", tight, stated.Precision.Chosen)
	}
}

func TestDescribeRefusesAnEmptyCache(t *testing.T) {
	_, err := describe("d", syntheticBuilder(t), describeOptions{Context: 0}, hardware{MaxPoolBytes: 1 << 30}, environment{})
	if err == nil || !strings.Contains(err.Error(), "at least one position") {
		t.Fatalf("describe with a zero context = %v, want a refusal", err)
	}
}

// TestRenderInfoPrintsTheChoiceAndItsEvidence is specs/001-weights.md §5 as a
// test: the precision is printed, and it is printed with the comparison that
// produced it, so a reader can redo the arithmetic.
func TestRenderInfoPrintsTheChoiceAndItsEvidence(t *testing.T) {
	b := syntheticBuilder(t)
	rep, err := describe("models/synthetic", b, describeOptions{Context: 2048},
		hardware{Backend: "cpu", Device: "test", Vendor: "go", MaxPoolBytes: 1 << 30}, stampEnvironment())
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	var sb strings.Builder
	renderInfo(&sb, rep)
	out := sb.String()
	for _, want := range []string{
		model.Qwen3Architecture,
		"precision  f16",
		rep.Precision.Why,
		weights.HumanBytes(rep.Precision.F16Bytes),
		weights.HumanBytes(rep.Precision.Int8Bytes),
		weights.HumanBytes(rep.Precision.Budget),
		"kv cache",
		fmt.Sprintf("%d layers", rep.Model.Layers),
		weights.HumanBytes(rep.Memory.ResidentBytes),
		"tied checkpoint",
		"cpu",
		rep.Environment.Go,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the description does not mention %q:\n%s", want, out)
		}
	}
}

// TestResolvedIntoPrefersWhatTheLoaderDid is specs/001-weights.md §5 at the
// seam where a prediction and a decision can differ. describe compares the f16
// footprint against a device limit this process read; weights.Load compares it
// against the device the model holds. §5 requires the printed choice to be the
// one that ran, and the prediction stays in the sentence as the evidence rather
// than being overwritten.
func TestResolvedIntoPrefersWhatTheLoaderDid(t *testing.T) {
	base, err := describe("d", syntheticBuilder(t), describeOptions{Context: 128},
		hardware{Backend: "cpu", MaxPoolBytes: 1 << 30}, environment{})
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	if base.Precision.Chosen != "f16" {
		t.Fatalf("the prediction is %s; this test needs it to differ from the resolved int8",
			base.Precision.Chosen)
	}
	predicted := base.Precision.Why

	got := resolvedInto(base, engineInfo{
		Precision: "int8", WeightBytes: 1_000_000, CacheBytesPerSession: 4096, Context: 128,
	})
	if got.Precision.Chosen != "int8" {
		t.Errorf("chosen = %s, want the loader's int8", got.Precision.Chosen)
	}
	if !strings.Contains(got.Precision.Why, predicted) {
		t.Error("the reason dropped the comparison that produced the prediction")
	}
	if !strings.Contains(got.Precision.Why, "resolved int8") {
		t.Errorf("the reason does not say the loader disagreed: %q", got.Precision.Why)
	}
	if got.Memory.WeightBytes != 1_000_000 || got.Memory.KVBytes != 4096 {
		t.Errorf("memory = %+v, want the engine's numbers", got.Memory)
	}
	if got.Memory.ResidentBytes != 1_000_000+4096 {
		t.Errorf("resident = %d, want the resolved weights plus the resolved cache", got.Memory.ResidentBytes)
	}
	// The footprints that produced the prediction are evidence and survive.
	if got.Precision.F16Bytes != base.Precision.F16Bytes || got.Precision.Int8Bytes != base.Precision.Int8Bytes {
		t.Error("the evidence for the choice was overwritten by the choice")
	}
}

// TestResolvedIntoLeavesAnAgreeingReportAlone: when the loader chose what this
// process predicted, nothing is said about a disagreement that did not happen.
func TestResolvedIntoLeavesAnAgreeingReportAlone(t *testing.T) {
	base, err := describe("d", syntheticBuilder(t), describeOptions{Context: 128},
		hardware{MaxPoolBytes: 1 << 30}, environment{})
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	got := resolvedInto(base, engineInfo{
		Precision: base.Precision.Chosen, WeightBytes: base.Memory.WeightBytes,
		CacheBytesPerSession: base.Memory.KVBytes, Context: base.Memory.Context,
	})
	if strings.Contains(got.Precision.Why, "resolved") {
		t.Errorf("an agreeing loader was reported as a disagreement: %q", got.Precision.Why)
	}
	if got.Memory != base.Memory {
		t.Errorf("memory changed with no disagreement:\n got %+v\nwant %+v", got.Memory, base.Memory)
	}
	// An engine that reports nothing leaves the prediction standing, which is
	// what a fake with no resolved facts does.
	if bare := resolvedInto(base, engineInfo{}); bare.Memory != base.Memory ||
		bare.Precision.Chosen != base.Precision.Chosen {
		t.Errorf("an engine that resolved nothing changed the report: %+v", bare)
	}
}

// TestCacheWidthIsMeasuredNotAssumed is the drift specs/005-kv-cache.md §3
// promises: the f32 constraint that forced the wide store is closed upstream
// and the design tgo builds is f16, so the day the key and value states narrow,
// a table that kept printing f32 would overstate every cache it prices by two.
func TestCacheWidthIsMeasuredNotAssumed(t *testing.T) {
	m := modelFacts{Layers: 36, KVHeads: 8, HeadDim: 128}
	const context = 4096
	elements := int64(2 * 36 * 8 * 128 * context)

	// The agreeing case keeps the label the build predicted.
	per, width, dtype := cacheWidth(m, elements*4, context, accel.F32.String())
	if width != 4 || dtype != accel.F32.String() || per != 288*1024 {
		t.Errorf("f32 cache = %d bytes per position, width %d, %q", per, width, dtype)
	}
	// A narrower cache than the build predicted is named as the engine's.
	per, width, dtype = cacheWidth(m, elements*2, context, accel.F32.String())
	if width != 2 || per != 144*1024 {
		t.Errorf("f16 cache = %d bytes per position, width %d", per, width)
	}
	if !strings.Contains(dtype, "2 bytes per element") || !strings.Contains(dtype, "predicted f32") {
		t.Errorf("dtype = %q, want it to name the engine's width and the prediction it broke", dtype)
	}
	// A size the formula does not explain loses the label and keeps the total.
	per, width, dtype = cacheWidth(m, elements*4+1, context, accel.F32.String())
	if width != 0 || !strings.Contains(dtype, "unknown") {
		t.Errorf("an unexplained cache size was labelled %q at width %d", dtype, width)
	}
	if per == 0 {
		t.Error("the per-position cost was dropped along with the label")
	}
	// A zero context divides by nothing rather than panicking.
	if _, _, d := cacheWidth(m, 1024, 0, accel.F32.String()); !strings.Contains(d, "unknown") {
		t.Errorf("a zero context produced %q", d)
	}
}

func TestDTypeSize(t *testing.T) {
	for _, dt := range []accel.DType{accel.F32, accel.F16, accel.I8} {
		if got := dtypeSize(dt.String()); got != dt.Size() {
			t.Errorf("dtypeSize(%q) = %d, want %d", dt, got, dt.Size())
		}
	}
	if got := dtypeSize("bf16"); got != 0 {
		t.Errorf("dtypeSize of a name this build does not price = %d, want 0", got)
	}
}

// TestRenderInfoNamesEachExtentOnItsOwnLine is the check a Contains cannot
// make.
//
// Two lines of the description print a layer count and two print a head count,
// so a renderer that put the key/value head count where the layer count belongs
// still leaves "2 layers" somewhere in the output and passes a substring test.
// Each labelled line is read out and pinned whole, against the config the
// builder parsed rather than against numbers written here.
func TestRenderInfoNamesEachExtentOnItsOwnLine(t *testing.T) {
	b := syntheticBuilder(t)
	rep, err := describe("models/synthetic", b, describeOptions{Context: 2048},
		hardware{Backend: "cpu", Device: "test", MaxPoolBytes: 1 << 30}, environment{})
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	var sb strings.Builder
	renderInfo(&sb, rep)

	c := b.Config()
	for label, want := range map[string]string{
		"hidden":     fmt.Sprintf("%d over %d layers", c.HiddenSize, c.NumLayers),
		"heads":      fmt.Sprintf("%d query, %d key/value, head_dim %d", c.NumHeads, c.NumKVHeads, c.HeadDim),
		"mlp":        fmt.Sprint(c.IntermediateSize),
		"vocabulary": fmt.Sprint(c.VocabSize),
		"kv cache": fmt.Sprintf("2 · %d layers · %d positions · %d kv heads · %d head_dim",
			c.NumLayers, rep.Memory.Context, c.NumKVHeads, c.HeadDim),
	} {
		got := lineWith(t, sb.String(), label)
		if !strings.Contains(got, want) {
			t.Errorf("the %q line reads %q, want it to name %q", label, got, want)
		}
	}
}

// lineWith returns the one line of a description whose label is the given one.
// Two lines with the same label, or none, is itself the failure: a description
// a test cannot address by line is one a substring check would have to read.
func lineWith(t *testing.T, out, label string) string {
	t.Helper()
	var found []string
	for line := range strings.SplitSeq(out, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), label+" ") {
			found = append(found, strings.TrimSpace(line))
		}
	}
	if len(found) != 1 {
		t.Fatalf("the description has %d lines labelled %q, want one:\n%s", len(found), label, out)
	}
	return found[0]
}

// TestCacheWidthKnowsAHybridsLayerCount is specs/023-cache-kinds.md §8's third
// change, and it is verifiable without a hybrid checkpoint: hand cacheWidth a
// byte count computed over the layers that cache and a stack that is four times
// longer, and see whether it recovers the width.
//
// Without the cached-layer count it reports `unknown` for every hybrid — it
// divides by 2·L·C·H_kv·d_h and a count over one layer in four leaves a
// remainder. `unknown` is the branch cacheWidth's own comment calls worse than
// printing nothing, because it prints a total and loses the label.
func TestCacheWidthKnowsAHybridsLayerCount(t *testing.T) {
	const context, kvHeads, headDim = 1024, 4, 256
	// Sixty-four layers, sixteen of which cache: `full_attention_interval: 4`.
	hybrid := modelFacts{Layers: 64, CachedLayers: 16, KVHeads: kvHeads,
		HeadDim: headDim}
	bytes := int64(2*16*context*kvHeads*headDim) * 2 // f16

	per, width, dtype := cacheWidth(hybrid, bytes, context, "f16")
	if width != 2 {
		t.Errorf("width = %d, want 2: the bytes were computed over the layers that "+
			"cache, and so is the division", width)
	}
	if strings.HasPrefix(dtype, "unknown") {
		t.Errorf("dtype = %q; a hybrid's cache has a width and this reports none", dtype)
	}
	if want := bytes / context; per != want {
		t.Errorf("per position = %d, want %d", per, want)
	}

	// The same bytes divided by the whole stack is the failure this fixes, and
	// asserting it is what stops the row above passing for another reason.
	dense := modelFacts{Layers: 64, KVHeads: kvHeads, HeadDim: headDim}
	if _, w, d := cacheWidth(dense, bytes, context, "f16"); w != 0 ||
		!strings.HasPrefix(d, "unknown") {
		t.Errorf("dividing a hybrid's bytes by 64 layers gave width %d and %q; the "+
			"fixture must leave a remainder for the row above to mean anything", w, d)
	}
}
