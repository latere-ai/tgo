// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// specs/023-cache-kinds.md §3.1's probe, and it is the condition §12 puts the
// whole spec's shape on.
//
// A hybrid holds a depthwise convolution window per gated-delta layer.
// [023-D2] makes that window **flat** — one row per position rather than a
// slots axis — because tensor.ScatterRows computes a row's width as
// elements/shape[0], so on a [slots, K-1+T, C] state a "row" is a whole slot
// and writing one token's C values is not expressible. The slot axis becomes
// arithmetic in the u32 index ports, exactly as paging already is, and the
// per-tap read becomes GatherRows instead of Slice + Contiguous.
//
// That rests on one thing nothing has proven: **GatherRows over a LayerState
// view**. [C12](../../specs/010-conformance.md) closed on Attention and
// ScatterRows binding a layer view, and the gather in nn.DepthwiseCausalConv
// runs over a whole state rather than a slice of one. A gather that silently
// read layer 0's rows for every layer would give a hybrid 48 convolution layers
// that all convolve the first one's window, which is fluent and wrong.
//
// So this is a **value** probe and not a records-without-error one: the state is
// preloaded with values that name their own layer, and the rows that come back
// have to say which layer they came from (010-D7).

// gatherShape is the probe's geometry. No two extents are equal and none is a
// multiple of another where the confusion would be silent: a layer count equal
// to a row count is the identity for every confusion between the two.
var gatherShape = struct{ layers, rows, width int }{5, 6, 3}

// gatherCell is what row r of layer l holds at column c. Every cell names its
// own layer, so a gather that read the wrong one cannot produce a right answer.
func gatherCell(layer, row, col int) float32 {
	return float32(layer*100 + row*10 + col)
}

// newGatherProbe builds a [layers, rows, width] state preloaded so that every
// cell names its own coordinates.
func newGatherProbe(t *testing.T, label string) (*Rig, *tensor.State) {
	t.Helper()
	r := New(t, Tier1, Options{Eps: 1e-6, Label: label})
	s := tensor.NewState(r.G.B, tensor.StateDesc{
		Name: "w", DType: accel.F32,
		Shape: tensor.Shape{gatherShape.layers, gatherShape.rows, gatherShape.width},
	})
	if s == nil {
		t.Fatal("NewState returned nothing")
	}
	data := make([]float32, 0, gatherShape.layers*gatherShape.rows*gatherShape.width)
	for l := range gatherShape.layers {
		for row := range gatherShape.rows {
			for c := range gatherShape.width {
				data = append(data, gatherCell(l, row, c))
			}
		}
	}
	r.F32("w", data)
	return r, s
}

// TestAGatherOverALayerViewReadsThatLayer is the probe.
func TestAGatherOverALayerViewReadsThatLayer(t *testing.T) {
	const layer = 3
	// Not in order, and not starting at zero: an implementation that ignored
	// the indices and returned the first rows of the table would agree with
	// {0,1,2} and disagree with this.
	want := []int{4, 0, 2}

	r, s := newGatherProbe(t, "layer-gather")
	view := tensor.LayerState(r.G.B, s, layer)
	if view == nil {
		t.Fatal("this accel does not slice a state per layer, and C12 closed on " +
			"the claim that it does")
	}
	table := tensor.ReadState(r.G.B, view)
	if table == nil {
		t.Fatal("reading a layer view returned nothing")
	}
	ids := r.Input("ids", accel.U32, tensor.Shape{len(want)})
	r.U32("ids", []uint32{4, 0, 2})

	out := tensor.GatherRows(r.G.B, table, ids)
	if out == nil {
		t.Fatalf("GatherRows over a layer view was refused: %v", r.G.Err())
	}
	got, _ := r.Run(out)

	ref := make([]float64, 0, len(want)*gatherShape.width)
	for _, row := range want {
		for c := range gatherShape.width {
			ref = append(ref, float64(gatherCell(layer, row, c)))
		}
	}
	Compare(t, got, ref, RoundF32(0), "a gather over layer 3's view")
}

// TestAGatherOverALayerViewIsNotAlwaysLayerZero is the control the row above
// needs. Without it the probe passes on an implementation that reads layer 0
// for every view, which is precisely the failure a hybrid could not see: 48
// convolution layers all convolving the first one's window.
func TestAGatherOverALayerViewIsNotAlwaysLayerZero(t *testing.T) {
	for _, layer := range []int{0, 2, gatherShape.layers - 1} {
		t.Run(fmt.Sprintf("layer%d", layer), func(t *testing.T) {
			r, s := newGatherProbe(t, fmt.Sprintf("layer-gather-%d", layer))
			view := tensor.LayerState(r.G.B, s, layer)
			if view == nil {
				t.Fatal("this accel does not slice a state per layer")
			}
			ids := r.Input("ids", accel.U32, tensor.Shape{1})
			r.U32("ids", []uint32{1})

			out := tensor.GatherRows(r.G.B, tensor.ReadState(r.G.B, view), ids)
			if out == nil {
				t.Fatalf("GatherRows over layer %d was refused: %v", layer, r.G.Err())
			}
			got, _ := r.Run(out)

			ref := make([]float64, 0, gatherShape.width)
			for c := range gatherShape.width {
				ref = append(ref, float64(gatherCell(layer, 1, c)))
			}
			Compare(t, got, ref, RoundF32(0),
				fmt.Sprintf("row 1 of layer %d", layer))
		})
	}
}

// TestAnOutOfRangeGatherOverALayerViewReadsZeros is the second half of §3's
// "two properties this layout inherits for free": a pad row's tap index is R,
// so it reads zeros and needs no mask — the same shape
// [C23](../../specs/010-conformance.md) gave the ragged step.
//
// The interesting part is *which* R. Out of range for a layer view has to mean
// past that layer's rows, not past the whole buffer: a view whose bound were
// the parent's would let a pad row read the next layer's first rows, which is
// the one wrong answer that looks like data.
func TestAnOutOfRangeGatherOverALayerViewReadsZeros(t *testing.T) {
	const layer = 2
	r, s := newGatherProbe(t, "layer-gather-oob")
	view := tensor.LayerState(r.G.B, s, layer)
	if view == nil {
		t.Fatal("this accel does not slice a state per layer")
	}
	ids := r.Input("ids", accel.U32, tensor.Shape{2})
	// The layer's own row count, and one past it. Both are inside the parent
	// buffer and outside this view.
	r.U32("ids", []uint32{uint32(gatherShape.rows), uint32(gatherShape.rows) + 1})

	out := tensor.GatherRows(r.G.B, tensor.ReadState(r.G.B, view), ids)
	if out == nil {
		t.Fatalf("GatherRows over a layer view was refused: %v", r.G.Err())
	}
	got, _ := r.Run(out)

	ref := make([]float64, len(got))
	Compare(t, got, ref, RoundF32(0),
		"a gather past a layer view's rows")
}
