// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package weights

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"

	"github.com/latere-ai/tgo/safetensors"
)

// tensorSpec is one tensor to put in a synthesised checkpoint.
type tensorSpec struct {
	name  string
	shape []int
	vals  []float32
}

// writeRepo builds a one-shard checkpoint directory holding bf16 planes. Every
// device test below runs against synthesised weights: no test that runs by
// default reads a real checkpoint (specs/000-decisions.md decision 8).
func writeRepo(t *testing.T, specs ...tensorSpec) *safetensors.Repo {
	t.Helper()
	dir := t.TempDir()

	header := map[string]any{}
	var data []byte
	for _, s := range specs {
		n := 1
		for _, d := range s.shape {
			n *= d
		}
		if len(s.vals) != n {
			t.Fatalf("%s: %d values for shape %v", s.name, len(s.vals), s.shape)
		}
		begin := len(data)
		for _, v := range s.vals {
			data = binary.LittleEndian.AppendUint16(data, accel.ToBFloat16(v).Bits())
		}
		header[s.name] = map[string]any{
			"dtype":        "BF16",
			"shape":        s.shape,
			"data_offsets": []int{begin, len(data)},
		}
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var file []byte
	file = binary.LittleEndian.AppendUint64(file, uint64(len(raw)))
	file = append(file, raw...)
	file = append(file, data...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), file, 0o600); err != nil {
		t.Fatal(err)
	}

	repo, err := safetensors.OpenRepo(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repo.Close() })
	return repo
}

func openCPU(t *testing.T) *accel.Device {
	t.Helper()
	dev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Skipf("no CPU device: %v", err)
	}
	t.Cleanup(func() { _ = dev.Close() })
	return dev
}

// ramp is a deterministic, f16-exact spread of values: every one is a small
// multiple of 1/16, so the bf16 store, the f32 widen and the f16 narrow are all
// exact and a content assertion tests the layout rather than the rounding.
func ramp(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i%97-48) / 16
	}
	return out
}

func readF16(t *testing.T, dev *accel.Device, b *accel.Buffer) []float32 {
	t.Helper()
	bits := make([]uint16, b.Count())
	if err := dev.Queue().ReadBuffer(b, 0, bits); err != nil {
		t.Fatalf("ReadBuffer: %v", err)
	}
	if err := dev.Queue().Flush().Wait(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	out := make([]float32, len(bits))
	for i, v := range bits {
		out[i] = accel.Float16FromBits(v).F32()
	}
	return out
}

// wantConverted runs the four conversions on the host, independently of the
// loader, so a content assertion checks the pipeline rather than restating it.
func wantConverted(vals []float32, shape []int, transpose bool, headDim int) []float32 {
	out := make([]float32, len(vals))
	copy(out, vals)
	if transpose {
		rows, cols := shape[0], shape[1]
		for r := range rows {
			for c := range cols {
				out[c*rows+r] = vals[r*cols+c]
			}
		}
	}
	if headDim > 0 {
		half := headDim / 2
		for off := 0; off < len(out); off += headDim {
			seg := append([]float32(nil), out[off:off+headDim]...)
			for i := range half {
				out[off+2*i] = seg[i]
				out[off+2*i+1] = seg[i+half]
			}
		}
	}
	return out
}

func TestLoadF16ConvertsTransposesAndPermutes(t *testing.T) {
	dev := openCPU(t)
	// q_proj is [out, in] in the file, so [16, 4] transposes to [4, 16]: two
	// heads of eight channels along the output axis.
	//
	// Eight and not two. A head of two channels makes permuteHeads the identity
	// — y[0]=x[0], y[1]=x[1] — so a loader that never permuted at all would pass
	// this test, and the f16 path's permutation would ship with no end-to-end
	// cover. Eight is the smallest width here that is neither the identity nor
	// its own inverse.
	repo := writeRepo(t,
		tensorSpec{"embed", []int{5, 4}, ramp(20)},
		tensorSpec{"q_proj", []int{16, 4}, ramp(64)},
		tensorSpec{"q_norm", []int{8}, ramp(8)},
	)
	decls := []Tensor{
		{Name: "embed"},
		{Name: "q_proj", Transpose: true, HeadDim: 8},
		{Name: "q_norm", HeadDim: 8},
	}
	var log bytes.Buffer
	set, err := Load(dev, repo, decls, Options{Policy: F16, Log: &log})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = set.Close() }()

	if got := set.Names(); len(got) != 3 || got[0] != "embed" {
		t.Errorf("Names = %v", got)
	}
	rep := set.Report()
	if rep.Chosen != F16 || rep.Saturated != 0 {
		t.Errorf("report = %+v", rep)
	}
	if rep.Bytes != int64(20+64+8)*2 {
		t.Errorf("Bytes = %d", rep.Bytes)
	}
	if !strings.Contains(log.String(), "f16") {
		t.Errorf("the precision choice was not printed: %q", log.String())
	}

	for _, c := range []struct {
		name      string
		shape     []int
		transpose bool
		headDim   int
		vals      []float32
	}{
		{"embed", []int{5, 4}, false, 0, ramp(20)},
		{"q_proj", []int{4, 16}, true, 8, ramp(64)},
		{"q_norm", []int{8}, false, 8, ramp(8)},
	} {
		v, ok := set.Get(c.name)
		if !ok {
			t.Fatalf("%s missing", c.name)
		}
		if fmt.Sprint(v.Shape) != fmt.Sprint(c.shape) {
			t.Errorf("%s shape = %v, want %v", c.name, v.Shape, c.shape)
		}
		if v.Precision != F16 || v.Scales != nil || v.Data.DType() != accel.F16 {
			t.Errorf("%s = %+v", c.name, v)
		}
		fileShape := c.shape
		if c.transpose {
			fileShape = []int{c.shape[1], c.shape[0]}
		}
		want := wantConverted(c.vals, fileShape, c.transpose, c.headDim)
		// A degeneracy floor on the permutation, for the same reason
		// TestF16AgreesWithADeviceCastEverywhereItCan has one: at headDim 2 the
		// permutation is the identity, and an expectation that does not move
		// when the permutation is removed proves nothing about it.
		if c.headDim > 0 {
			unpermuted := wantConverted(c.vals, fileShape, c.transpose, 0)
			same := true
			for i := range want {
				if want[i] != unpermuted[i] {
					same = false
					break
				}
			}
			if same {
				t.Fatalf("%s: the permutation is the identity at headDim %d, so this "+
					"assertion would pass with no permutation at all", c.name, c.headDim)
			}
		}
		got := readF16(t, dev, v.Data)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s element %d = %v, want %v", c.name, i, got[i], want[i])
			}
		}
	}
}

func TestLoadInt8QuantizesAfterThePermutation(t *testing.T) {
	dev := openCPU(t)
	// 64 output channels over 3 inputs, so the flattened plane spans several
	// quant blocks and a permutation applied after quantizing would scatter
	// weights away from their scales (004-D9's forced order).
	const in, out, headDim = 3, 64, 8
	repo := writeRepo(t, tensorSpec{"q_proj", []int{out, in}, ramp(in * out)})
	set, err := Load(dev, repo, []Tensor{{Name: "q_proj", Transpose: true, HeadDim: headDim}},
		Options{Policy: Int8, Log: io.Discard})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = set.Close() }()

	v, _ := set.Get("q_proj")
	if v.Precision != Int8 || v.Data.DType() != accel.I8 || v.Scales == nil {
		t.Fatalf("value = %+v", v)
	}
	if v.Scales.Count() != blocks(in*out) {
		t.Errorf("%d scales for %d weights", v.Scales.Count(), in*out)
	}
	if v.Bytes() != int64(in*out)+int64(blocks(in*out))*2 {
		t.Errorf("Bytes = %d", v.Bytes())
	}

	quants := make([]int8, v.Data.Count())
	if err := dev.Queue().ReadBuffer(v.Data, 0, quants); err != nil {
		t.Fatal(err)
	}
	scaleBits := make([]uint16, v.Scales.Count())
	if err := dev.Queue().ReadBuffer(v.Scales, 0, scaleBits); err != nil {
		t.Fatal(err)
	}
	if err := dev.Queue().Flush().Wait(); err != nil {
		t.Fatal(err)
	}
	scales := make([]accel.Float16, len(scaleBits))
	for i, b := range scaleBits {
		scales[i] = accel.Float16FromBits(b)
	}
	back := quant.Int8Dequantize(quants, scales)

	want := wantConverted(ramp(in*out), []int{out, in}, true, headDim)
	// One weight reconstructs within half a scale step, and the scale is the
	// block's peak over Int8Max. specs/010-conformance.md §5.1's int8 row, per
	// element rather than per dot product.
	for i := range want {
		s := float64(scales[i/quant.Int8Block].F32())
		if diff := math.Abs(float64(back[i] - want[i])); diff > s/2+1e-6 {
			t.Fatalf("element %d reconstructed as %v, want %v (block scale %v)", i, back[i], want[i], s)
		}
	}
}

func TestLoadAutoChoosesAndPrints(t *testing.T) {
	dev := openCPU(t)
	repo := writeRepo(t, tensorSpec{"w", []int{16, 16}, ramp(256)})
	decl := []Tensor{{Name: "w"}}

	f16Bytes := int64(256 * 2)
	int8Bytes := int64(256) + int64(blocks(256))*2

	for _, c := range []struct {
		name   string
		budget int64
		want   Precision
	}{
		{"fits at f16", f16Bytes, F16},
		{"only int8 fits", f16Bytes - 1, Int8},
	} {
		t.Run(c.name, func(t *testing.T) {
			var log bytes.Buffer
			set, err := Load(dev, repo, decl, Options{Policy: Auto, Budget: c.budget, Log: &log})
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer func() { _ = set.Close() }()
			rep := set.Report()
			if rep.Chosen != c.want {
				t.Errorf("chose %v, want %v", rep.Chosen, c.want)
			}
			if rep.F16Bytes != f16Bytes || rep.Int8Bytes != int8Bytes {
				t.Errorf("footprints = %d f16, %d int8", rep.F16Bytes, rep.Int8Bytes)
			}
			// §5: the choice is printed, never silent.
			if !strings.Contains(log.String(), c.want.String()) {
				t.Errorf("choice %v not printed: %q", c.want, log.String())
			}
		})
	}

	// Neither fits, so the load refuses rather than quantizing into a device
	// that cannot hold the result either.
	if _, err := Load(dev, repo, decl, Options{Policy: Auto, Budget: 8, Log: io.Discard}); err == nil {
		t.Error("Load accepted a model larger than its budget at every precision")
	}
}

func TestLoadPerTensorOverrideSurvivesTheAutoChoice(t *testing.T) {
	dev := openCPU(t)
	// The case §5 names: hold the embedding table at f16 while everything else
	// is int8.
	repo := writeRepo(t,
		tensorSpec{"embed", []int{16, 16}, ramp(256)},
		tensorSpec{"w", []int{16, 16}, ramp(256)},
	)
	decls := []Tensor{{Name: "embed", Precision: F16}, {Name: "w"}}
	set, err := Load(dev, repo, decls, Options{Policy: Auto, Budget: 800, Log: io.Discard})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = set.Close() }()
	if got := set.Report().Chosen; got != Int8 {
		t.Fatalf("policy chose %v, want int8", got)
	}
	e, _ := set.Get("embed")
	w, _ := set.Get("w")
	if e.Precision != F16 || e.Scales != nil {
		t.Errorf("the override did not hold: %+v", e)
	}
	if w.Precision != Int8 || w.Scales == nil {
		t.Errorf("the policy did not reach w: %+v", w)
	}
	if got, want := set.Report().Bytes, e.Bytes()+w.Bytes(); got != want {
		t.Errorf("Bytes = %d, want %d", got, want)
	}
}

func TestLoadRefusesTooMuchSaturation(t *testing.T) {
	dev := openCPU(t)
	// Half the tensor is beyond f16's range, which is what a checkpoint that is
	// not in the range f16 can hold looks like. int8 is not a fix for it either,
	// so the load fails rather than reporting and continuing (001-D2).
	vals := []float32{1, 1e30, 2, -1e30}
	repo := writeRepo(t, tensorSpec{"w", []int{2, 2}, vals})
	_, err := Load(dev, repo, []Tensor{{Name: "w"}}, Options{Policy: F16, Log: io.Discard})
	if err == nil {
		t.Fatal("Load accepted a tensor that half saturated")
	}
	if !strings.Contains(err.Error(), "65504") {
		t.Errorf("the refusal does not name the saturation value: %v", err)
	}

	// Under the threshold it loads, reports the count, and prints it.
	var log bytes.Buffer
	set, err := Load(dev, repo, []Tensor{{Name: "w"}},
		Options{Policy: F16, MaxSaturation: 0.75, Log: &log})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = set.Close() }()
	v, _ := set.Get("w")
	if v.Saturated != 2 || set.Report().Saturated != 2 {
		t.Errorf("saturated = %d, report = %d", v.Saturated, set.Report().Saturated)
	}
	if !strings.Contains(log.String(), "saturated 2 of 4") {
		t.Errorf("the count was not printed: %q", log.String())
	}
	got := readF16(t, dev, v.Data)
	if got[1] != maxF16 || got[3] != -maxF16 {
		t.Errorf("saturated elements are %v and %v, want ±65504", got[1], got[3])
	}
}

func TestRefuseAnySaturationAdmitsNone(t *testing.T) {
	// Zero is the field's unset value, so the strict end of the range needs a
	// name of its own. A caller who agrees with §3 that a nonzero count is a
	// signal has no other way to say so: the fraction that means "not one
	// element" depends on the size of a tensor they have not read yet.
	dev := openCPU(t)
	clean := writeRepo(t, tensorSpec{"w", []int{2, 2}, ramp(4)})
	set, err := Load(dev, clean, []Tensor{{Name: "w"}},
		Options{Policy: F16, MaxSaturation: RefuseAnySaturation, Log: io.Discard})
	if err != nil {
		t.Fatalf("Load refused a tensor that saturated nothing: %v", err)
	}
	_ = set.Close()

	// One element in four is past f16's range, which the default fraction would
	// also refuse; what this checks is that the strict setting is expressible
	// and that it is stricter, not that it refuses this particular tensor.
	one := writeRepo(t, tensorSpec{"w", []int{2, 2}, []float32{1, 2, 3, 1e30}})
	if _, err := Load(dev, one, []Tensor{{Name: "w"}},
		Options{Policy: F16, MaxSaturation: RefuseAnySaturation, Log: io.Discard}); err == nil {
		t.Error("RefuseAnySaturation admitted a saturated element")
	}
	// The same tensor loads where the caller allows a quarter of it.
	set, err = Load(dev, one, []Tensor{{Name: "w"}},
		Options{Policy: F16, MaxSaturation: 0.25, Log: io.Discard})
	if err != nil {
		t.Fatalf("a threshold of 0.25 refused one saturation in four: %v", err)
	}
	defer func() { _ = set.Close() }()
	if v, _ := set.Get("w"); v.Saturated != 1 {
		t.Errorf("saturated = %d, want 1", v.Saturated)
	}
}

func TestLoadRefusals(t *testing.T) {
	dev := openCPU(t)
	repo := writeRepo(t,
		tensorSpec{"w", []int{4, 4}, ramp(16)},
		tensorSpec{"g", []int{4}, ramp(4)},
		tensorSpec{"cube", []int{2, 2, 2}, ramp(8)},
	)
	opts := Options{Policy: F16, Log: io.Discard}
	for _, c := range []struct {
		name  string
		decls []Tensor
		opts  Options
		want  string
	}{
		{"missing tensor", []Tensor{{Name: "nope"}}, opts, "does not contain"},
		{"declared twice", []Tensor{{Name: "w"}, {Name: "w"}}, opts, "declared twice"},
		{"rank three", []Tensor{{Name: "cube"}}, opts, "rank 3"},
		{"transposing a vector", []Tensor{{Name: "g", Transpose: true}}, opts, "axes to exchange"},
		{"head dim does not divide", []Tensor{{Name: "w", HeadDim: 3}}, opts, "head dim"},
		{"unknown per-tensor precision", []Tensor{{Name: "w", Precision: Precision(9)}}, opts, "precision"},
		{"unknown policy", []Tensor{{Name: "w"}}, Options{Policy: Precision(9), Log: io.Discard}, "policy"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := Load(dev, repo, c.decls, c.opts)
			if err == nil {
				t.Fatalf("Load accepted %s", c.name)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not mention %q", err, c.want)
			}
		})
	}

	if _, err := Load(nil, repo, []Tensor{{Name: "w"}}, opts); err == nil {
		t.Error("Load accepted a nil device")
	}
	if _, err := Load(dev, nil, []Tensor{{Name: "w"}}, opts); err == nil {
		t.Error("Load accepted a nil repo")
	}
	if _, err := Load(dev, repo, nil, opts); err == nil {
		t.Error("Load accepted no declarations")
	}
}

func TestLoadRefusesAnEmptyTensor(t *testing.T) {
	dev := openCPU(t)
	repo := writeRepo(t, tensorSpec{"w", []int{0, 4}, nil})
	if _, err := Load(dev, repo, []Tensor{{Name: "w"}}, Options{Policy: F16, Log: io.Discard}); err == nil {
		t.Error("Load accepted a tensor with no elements")
	}
}

func TestLoadStagesWhereAccessIsRefused(t *testing.T) {
	// 001-D9: Buffer.Access refuses a pool with no host mapping, so on a
	// discrete device the loader stages through Queue.WriteBuffer. This forces
	// that path on a machine whose memory is unified, because otherwise the
	// branch that every discrete GPU takes would never run here.
	forceStaging = true
	defer func() { forceStaging = false }()

	dev := openCPU(t)
	repo := writeRepo(t,
		// [16, 6] transposes to [6, 16]: two heads of eight. Eight rather than
		// two, because a two-channel head permutes to itself and the staged
		// bytes would then be the same with or without the permutation.
		tensorSpec{"w", []int{16, 6}, ramp(96)},
		tensorSpec{"q", []int{8, 4}, ramp(32)},
	)
	set, err := Load(dev, repo, []Tensor{
		{Name: "w", Transpose: true, HeadDim: 8},
		{Name: "q", Precision: Int8},
	}, Options{Policy: F16, Log: io.Discard})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = set.Close() }()
	if set.Report().Mapped {
		t.Error("Report says the conversion was mapped, but staging was forced")
	}

	v, _ := set.Get("w")
	want := wantConverted(ramp(96), []int{16, 6}, true, 8)
	got := readF16(t, dev, v.Data)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("staged element %d = %v, want %v", i, got[i], want[i])
		}
	}
	q, _ := set.Get("q")
	if q.Data.DType() != accel.I8 || q.Scales.DType() != accel.F16 {
		t.Errorf("staged int8 value = %+v", q)
	}
}

func TestLoadStagesInt4CodesTheDeviceCannotBeHandedDirectly(t *testing.T) {
	// The staging path is per dtype, and int4's code plane is the one width no
	// other precision uses: f16 and int8 both write through it, and u32 did not
	// until this test. A load at Int4 on a discrete device died with "no staging
	// path for U32" — every unified-memory machine takes the mapped branch, so
	// the whole int4 precision was untested on the hardware it exists for.
	forceStaging = true
	defer func() { forceStaging = false }()

	dev := openCPU(t)
	// One group's worth and a bit, so the plane spans two groups and the second
	// is short: a packing that only ever sees whole groups can be wrong at the
	// tail and still pass.
	const n = quant.Int4Group + 8
	vals := ramp(n)
	repo := writeRepo(t, tensorSpec{"w", []int{n / 8, 8}, vals})

	set, err := Load(dev, repo, []Tensor{{Name: "w", Precision: Int4}},
		Options{Policy: Int4, Log: io.Discard})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = set.Close() }()

	v, _ := set.Get("w")
	if v.Data.DType() != accel.U32 || v.Scales.DType() != accel.F16 ||
		v.Zeros.DType() != accel.F16 {
		t.Fatalf("staged int4 value = %+v", v)
	}

	// The bytes, not just the shapes. Staging re-reads the plane at the buffer's
	// dtype, so a width read wrongly is a buffer full of plausible codes.
	wantCodes, wantScales, wantZeros := quant.Int4Quantize(vals)
	gotCodes := make([]uint32, v.Data.Count())
	if err := dev.Queue().ReadBuffer(v.Data, 0, gotCodes); err != nil {
		t.Fatalf("ReadBuffer: %v", err)
	}
	if err := dev.Queue().Flush().Wait(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if !slices.Equal(gotCodes, wantCodes) {
		t.Errorf("staged codes = %v, want %v", gotCodes, wantCodes)
	}
	for _, p := range []struct {
		what string
		buf  *accel.Buffer
		want []accel.Float16
	}{
		{"scales", v.Scales, wantScales},
		{"zeros", v.Zeros, wantZeros},
	} {
		got := readF16(t, dev, p.buf)
		for i, h := range p.want {
			if got[i] != h.F32() {
				t.Errorf("staged %s[%d] = %v, want %v", p.what, i, got[i], h.F32())
			}
		}
	}
}

func TestArenaStagesEveryWidthAWeightIsStoredIn(t *testing.T) {
	// The complement of TestArenaHasNoStagingPathForAnUnexpectedDType: that one
	// says an unknown width is refused, and this says the known ones are not.
	// Without it, "refuse what you do not know" is satisfied by refusing
	// everything, which is how U32 came to be missing.
	dev := openCPU(t)
	for _, dt := range []accel.DType{accel.F16, accel.I8, accel.U32} {
		t.Run(dt.String(), func(t *testing.T) {
			a := &arena{dev: dev, kind: accel.MemoryDevice, max: 1 << 20}
			defer func() { _ = a.close() }()
			b, err := a.alloc(dt, 8, "w")
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = b.Close() }()
			if err := a.fill(b, func(dst []byte) error { return nil }); err != nil {
				t.Fatalf("fill: %v", err)
			}
			if err := a.flush(); err != nil {
				t.Fatalf("flush: %v", err)
			}
		})
	}
}

func TestArenaOpensMorePoolsAndRefusesWhatCannotFit(t *testing.T) {
	dev := openCPU(t)
	// A pool never grows, so the arena has to open another one when the current
	// pool fills. MaxPoolBytes is 2 GB on this device, so the cap is forced here
	// instead of allocating that much.
	a := &arena{dev: dev, kind: accel.MemoryShared, max: 4096}
	defer func() { _ = a.close() }()
	for i := range 8 {
		b, err := a.alloc(accel.F16, 512, fmt.Sprintf("t%d", i))
		if err != nil {
			t.Fatalf("alloc %d: %v", i, err)
		}
		defer func() { _ = b.Close() }()
	}
	if len(a.pools) < 2 {
		t.Errorf("%d pools for 8 KiB of buffers in 4 KiB pools", len(a.pools))
	}
	if _, err := a.alloc(accel.F16, 1<<20, "huge"); err == nil {
		t.Error("the arena allocated a buffer larger than MaxPoolBytes")
	}

	// A buffer of exactly MaxPoolBytes is refused too, because the pool holding
	// it has to be larger than it is: accel's suballocator rounds a request up
	// to a size class before searching, so a pool sized to its exact contents
	// refuses its own single allocation. The message has to say that, not
	// "needs 4096 bytes, more than MaxPoolBytes of 4096".
	_, err := a.alloc(accel.F16, a.max/2, "exact")
	if err == nil {
		t.Fatal("a buffer of exactly MaxPoolBytes was allocated")
	}
	if !strings.Contains(err.Error(), "needs a pool of") {
		t.Errorf("the refusal does not distinguish the buffer from its pool: %v", err)
	}
}

func TestArenaHasNoStagingPathForAnUnexpectedDType(t *testing.T) {
	dev := openCPU(t)
	a := &arena{dev: dev, kind: accel.MemoryDevice, max: 1 << 20}
	defer func() { _ = a.close() }()
	b, err := a.alloc(accel.F32, 4, "f32")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()
	// Only the two dtypes a weight is stored in have a staging path. A third
	// arriving here means a precision was added without one.
	if err := a.fill(b, func(dst []byte) error { return nil }); err == nil {
		t.Error("stage accepted an f32 buffer")
	}
	if err := a.flush(); err != nil {
		t.Errorf("flush: %v", err)
	}
	// A conversion that fails is reported rather than leaving a half-written
	// buffer resident.
	want := fmt.Errorf("conversion failed")
	if err := a.fill(b, func(dst []byte) error { return want }); !errors.Is(err, want) {
		t.Errorf("fill returned %v, want the conversion's own error", err)
	}
}

func TestSetGetIsAMissAndCloseIsSafeTwice(t *testing.T) {
	dev := openCPU(t)
	repo := writeRepo(t, tensorSpec{"w", []int{2, 2}, ramp(4)})
	set, err := Load(dev, repo, []Tensor{{Name: "w"}}, Options{Policy: F16, Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := set.Get("absent"); ok {
		t.Error("Get found a tensor that was never declared")
	}
	// The returned Shape is the caller's: mutating it must not reach the Set.
	v, _ := set.Get("w")
	v.Shape[0] = 99
	again, _ := set.Get("w")
	if again.Shape[0] != 2 {
		t.Errorf("Get handed out the Set's own shape slice")
	}
	if err := set.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if err := set.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

func TestLoadDefaultsPrintToStderr(t *testing.T) {
	// §5 requires the choice to be printed. A zero Log must not be a silent one:
	// the default is stderr, and io.Discard is how a caller silences it on
	// purpose.
	dev := openCPU(t)
	repo := writeRepo(t, tensorSpec{"w", []int{2, 2}, ramp(4)})

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved := os.Stderr
	os.Stderr = w
	set, loadErr := Load(dev, repo, []Tensor{{Name: "w"}}, Options{Policy: F16})
	os.Stderr = saved
	_ = w.Close()
	out, _ := io.ReadAll(r)
	_ = r.Close()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	defer func() { _ = set.Close() }()
	if !strings.Contains(string(out), "weights: f16") {
		t.Errorf("nothing was printed to stderr: %q", out)
	}
}

func TestPrecisionStringAndHumanBytes(t *testing.T) {
	for p, want := range map[Precision]string{
		Inherit: "inherit", Auto: "auto", F16: "f16", Int8: "int8", Precision(9): "Precision(9)",
	} {
		if got := p.String(); got != want {
			t.Errorf("Precision(%d).String() = %q, want %q", int(p), got, want)
		}
	}
	for n, want := range map[int64]string{
		512: "512 B", 2048: "2.00 KiB", 3 << 20: "3.00 MiB", 5 << 30: "5.00 GiB", 2 << 40: "2.00 TiB",
	} {
		if got := HumanBytes(n); got != want {
			t.Errorf("HumanBytes(%d) = %q, want %q", n, got, want)
		}
	}
}

func TestLoadReportsAReadThatFails(t *testing.T) {
	// The header is parsed at open and the planes are read at load, so a shard
	// that goes away between the two is a read error rather than a refusal the
	// reader could have made earlier.
	dev := openCPU(t)
	repo := writeRepo(t, tensorSpec{"w", []int{2, 2}, ramp(4)})
	if err := repo.Close(); err != nil {
		t.Fatal(err)
	}
	_, err := Load(dev, repo, []Tensor{{Name: "w"}}, Options{Policy: F16, Log: io.Discard})
	if err == nil {
		t.Fatal("Load read a tensor out of a closed shard")
	}
	if !strings.Contains(err.Error(), "w") {
		t.Errorf("the error does not name the tensor: %v", err)
	}
}

func TestRoundShiftByZeroIsTheIdentity(t *testing.T) {
	// The guard exists because a shift of a full word is undefined in C-shaped
	// languages and because a caller narrowing to the same width should get its
	// value back rather than a rounded one.
	if got := roundShift(0x5a5a, 0); got != 0x5a5a {
		t.Errorf("roundShift(v, 0) = %#x, want the value unchanged", got)
	}
}

// openRepoAt opens a checkpoint directory the caller names. Only the
// TGO_MODEL-gated test uses it.
func openRepoAt(t *testing.T, dir string) *safetensors.Repo {
	t.Helper()
	repo, err := safetensors.OpenRepo(dir)
	if err != nil {
		t.Fatalf("OpenRepo(%s): %v", dir, err)
	}
	t.Cleanup(func() { _ = repo.Close() })
	return repo
}

func TestLoadGivesOnePlaneToTwoPorts(t *testing.T) {
	// A checkpoint with tie_word_embeddings ships no lm_head.weight: the LM
	// head is the embedding table transposed, and the embedding table is the
	// same plane as it lies in the file. Two declarations, one tensor, two
	// values on the device (004-D7).
	dev := openCPU(t)
	repo := writeRepo(t, tensorSpec{"model.embed_tokens.weight", []int{6, 4}, ramp(24)})
	set, err := Load(dev, repo, []Tensor{
		{Name: "model.embed_tokens.weight"},
		{Name: "model.embed_tokens.weight", As: "lm_head", Transpose: true},
	}, Options{Policy: F16, Log: io.Discard})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = set.Close() }()

	table, ok := set.Get("model.embed_tokens.weight")
	if !ok {
		t.Fatal("the embedding table was not loaded")
	}
	head, ok := set.Get("lm_head")
	if !ok {
		t.Fatal("the tied LM head was not loaded")
	}
	if head.Source != "model.embed_tokens.weight" || head.Name != "lm_head" {
		t.Errorf("head = %q from %q", head.Name, head.Source)
	}
	// Two ports really are two device buffers, so the footprint counts the
	// plane twice. A tied checkpoint is not smaller than an untied one on the
	// device; it is smaller on disk.
	if rep := set.Report(); rep.Bytes != 24*2*2 || rep.F16Bytes != 24*2*2 {
		t.Errorf("report = %+v, want 96 bytes at f16 for two copies of a 24-element plane", rep)
	}
	if fmt.Sprint(table.Shape) != "[6 4]" || fmt.Sprint(head.Shape) != "[4 6]" {
		t.Errorf("shapes are %v and %v, want [6 4] and [4 6]", table.Shape, head.Shape)
	}
	if table.Data == head.Data {
		t.Error("both ports got the same buffer; they are the same plane in two layouts")
	}
	gotHead := readF16(t, dev, head.Data)
	wantHead := wantConverted(ramp(24), []int{6, 4}, true, 0)
	for i := range wantHead {
		if gotHead[i] != wantHead[i] {
			t.Fatalf("head element %d = %v, want %v", i, gotHead[i], wantHead[i])
		}
	}

	// Two declarations that file under the same name are still a mistake.
	_, err = Load(dev, repo, []Tensor{
		{Name: "model.embed_tokens.weight"},
		{Name: "model.embed_tokens.weight"},
	}, Options{Policy: F16, Log: io.Discard})
	if err == nil {
		t.Error("Load accepted two declarations filing under one name")
	}
}

// TestAutoWalksDownToInt4AndLoads is the ladder's last rung, taken.
//
// Auto is f16 → int8 → int4, and only the *refusals* were covered: that auto
// never prefers int4 to int8 (int4_test.go), and that a budget below every
// width fails by name. Nothing set a budget between the int8 and int4
// footprints and let the load run, so the one case int4 exists for — a model
// that does not fit at int8 and does at int4 — was never exercised end to end.
//
// It is the case with the least margin for a mistake to hide in: the plan picks
// a width, and three planes with two different dtypes have to reach the device
// for it.
func TestAutoWalksDownToInt4AndLoads(t *testing.T) {
	dev := openCPU(t)
	// A tensor large enough that the three footprints are far apart: f16 is
	// 2 bytes a weight, int8 about 1.06, int4 about 0.53.
	const rows, cols = 256, 128
	repo := writeRepo(t, tensorSpec{"w", []int{rows, cols}, ramp(rows * cols)})
	decls := []Tensor{{Name: "w"}}

	// Price the widths through the loader itself rather than restating the
	// arithmetic: a budget derived from a formula this test wrote would pass
	// against a loader that priced them differently.
	sizes := map[Precision]int64{}
	for _, p := range []Precision{F16, Int8, Int4} {
		set, err := Load(dev, repo, decls, Options{Policy: p, Log: io.Discard})
		if err != nil {
			t.Fatalf("Load at %v: %v", p, err)
		}
		sizes[p] = set.Report().Bytes
		if err := set.Close(); err != nil {
			t.Fatalf("close the %v set: %v", p, err)
		}
	}
	if sizes[Int4] >= sizes[Int8] || sizes[Int8] >= sizes[F16] {
		t.Fatalf("the widths do not order: int4 %d, int8 %d, f16 %d",
			sizes[Int4], sizes[Int8], sizes[F16])
	}

	// A budget that int8 misses and int4 clears. The report counts payload
	// bytes and the arena rounds each buffer up, so the budget is the int8
	// footprint less one byte: below int8, at or above int4.
	budget := sizes[Int8] - 1
	if budget < sizes[Int4] {
		t.Fatalf("int8 is %d and int4 is %d; there is no budget between them",
			sizes[Int8], sizes[Int4])
	}

	set, err := Load(dev, repo, decls, Options{Policy: Auto, Budget: budget, Log: io.Discard})
	if err != nil {
		t.Fatalf("Load at auto with a budget between int8 and int4: %v", err)
	}
	defer func() { _ = set.Close() }()

	rep := set.Report()
	if rep.Chosen != Int4 {
		t.Fatalf("auto chose %v for a budget of %d that int8 (%d) misses and int4 "+
			"(%d) clears", rep.Chosen, budget, sizes[Int8], sizes[Int4])
	}

	// And it loaded: three planes, the codes packed eight to a u32 word.
	v, ok := set.Get("w")
	if !ok {
		t.Fatal("the loaded set has no tensor w")
	}
	if v.Data.DType() != accel.U32 || v.Scales.DType() != accel.F16 ||
		v.Zeros.DType() != accel.F16 {
		t.Fatalf("int4 value = %+v; want u32 codes with f16 scales and zeros", v)
	}
	if want := (rows*cols + 7) / 8; v.Data.Count() != want {
		t.Errorf("the code plane holds %d words, want %d for %d weights",
			v.Data.Count(), want, rows*cols)
	}
}
