// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// The C16 probe: a step in which one sequence contributes a prefill chunk and
// another contributes a single decode token, in one dispatch.
//
// specs/010-conformance.md §2 "How a row's state is decided" is what this file
// obeys, and the reason it is long. Reading tensor/attention.go and finding
// AttentionOptions.QueryExtents there settles nothing: §2.2.1 records four rows
// that closed upstream, read as closed, and were not. So this binds real
// buffers, asserts values against a reference computed beside it, checks
// Selections() names the ragged kernel rather than a fallback, runs the same
// tokens as separate dispatches and requires the two to agree, and varies the
// extents to prove the option is read at all.

// raggedShape is the probe's geometry, small enough to reason about by hand and
// wide enough that the grouped-query fan-out and the page indirection are both
// exercised.
type raggedShape struct {
	qHeads, kvHeads, headDim int
	block, maxPages          int
}

var probeShape = raggedShape{qHeads: 4, kvHeads: 2, headDim: 4, block: 2, maxPages: 3}

// seq is one member of a ragged step.
type seq struct {
	// extent is how many query tokens it contributes this step.
	extent int
	// length is how many cached positions it holds *after* this step, which is
	// what AttentionOptions.Lengths means: token i of the extent sits at
	// length-extent+i (accel specs/046-segmented-extents.md §2.2).
	length int
	// pages is its page-table row: pages[i] is the physical block holding its
	// i-th logical block.
	pages []int
}

// raggedCase is a whole step.
type raggedCase struct {
	seqs  []seq
	pool  int // physical blocks in the pool
	scale float32
}

// probeCase is the mixed step the row is about: a three-token prefill chunk
// beside a decode, and a third sequence admitted with nothing to contribute.
//
// The page rows are deliberately not the identity and not in order. A table
// that happened to be [0,1,2] for every sequence would pass whether the kernel
// read it or read the cache contiguously, which is how accel issue 10 stayed
// invisible: a paged prefill that dropped its table still produced a fluent
// answer (§2's C13 note).
var probeCase = raggedCase{
	pool:  9,
	scale: 0.5,
	seqs: []seq{
		{extent: 3, length: 3, pages: []int{5, 1, 8}},
		{extent: 1, length: 5, pages: []int{2, 7, 0}},
		{extent: 0, length: 2, pages: []int{4, 3, 6}},
	},
}

func (c raggedCase) tokens() int {
	n := 0
	for _, s := range c.seqs {
		n += s.extent
	}
	return n
}

// cacheRows is how many rows the physical pool holds.
func (c raggedCase) cacheRows(sh raggedShape) int { return c.pool * sh.block }

// row maps a sequence's logical position to its physical cache row.
func (s seq) row(sh raggedShape, pos int) int {
	return s.pages[pos/sh.block]*sh.block + pos%sh.block
}

// raggedInputs is one probe's bound data, generated rather than written out so
// that the reference and the device read the same numbers by construction.
type raggedInputs struct {
	q, k, v        []float32
	pages, lengths []uint32
	extents        []uint32
	shape          raggedShape
	c              raggedCase
}

func newRaggedInputs(sh raggedShape, c raggedCase) raggedInputs {
	in := raggedInputs{shape: sh, c: c,
		q: spread(c.tokens()*sh.qHeads*sh.headDim, 7),
		k: spread(c.cacheRows(sh)*sh.kvHeads*sh.headDim, 13),
		v: spread(c.cacheRows(sh)*sh.kvHeads*sh.headDim, 29)}
	for _, s := range c.seqs {
		in.extents = append(in.extents, uint32(s.extent))
		in.lengths = append(in.lengths, uint32(s.length))
		for i := 0; i < sh.maxPages; i++ {
			in.pages = append(in.pages, uint32(s.pages[i]))
		}
	}
	return in
}

// spread produces values that vary in sign and magnitude.
//
// Attention over near-identical scores is nearly a mean, and a mean is the one
// thing a wrong page table still gets close to. Values that differ per row are
// what make an index error show up as a value error.
func spread(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		x := float64((i*seed)%97) / 97.0
		out[i] = float32(math.Sin(6.0*x) * (0.5 + x))
	}
	return out
}

// raggedOracle is specs/010-conformance.md §5's reference for this step: pure
// Go, float64, written from the definition of attention and from what
// specs/046-segmented-extents.md §2.2 says a token's position is.
//
// It walks the page table itself. A reference that indexed the cache
// contiguously would agree with a kernel that ignored Pages, which is the
// failure this probe exists to be able to see.
// budget is what the reference observed while computing, so the tolerance is
// derived from the numbers that were actually summed rather than from the
// answer they cancelled into.
type budget struct {
	// value is max|v| over the rows any token attended to. The output is a
	// convex combination of those rows, so it is the magnitude the relative
	// terms apply to.
	value float64
	// score is the largest |score| any row exponentiated.
	score float64
	// positions is the most rows any single token summed over.
	positions int
}

func (in raggedInputs) oracle() ([]float64, budget) {
	sh, c := in.shape, in.c
	out := make([]float64, c.tokens()*sh.qHeads*sh.headDim)
	var b budget
	group := sh.qHeads / sh.kvHeads
	tok := 0
	for _, s := range c.seqs {
		for i := 0; i < s.extent; i++ {
			// The token's position in its own sequence, not its index in the
			// flat query buffer: this step's tokens occupy the last extent of
			// the length, so token i sits at length-extent+i and is causal
			// against everything at or before it.
			limit := s.length - s.extent + i
			for h := 0; h < sh.qHeads; h++ {
				kvHead := h / group
				scores := make([]float64, limit+1)
				max := math.Inf(-1)
				for p := 0; p <= limit; p++ {
					dot := 0.0
					for d := 0; d < sh.headDim; d++ {
						qv := float64(in.q[(tok*sh.qHeads+h)*sh.headDim+d])
						kv := float64(in.k[(s.row(sh, p)*sh.kvHeads+kvHead)*sh.headDim+d])
						dot += qv * kv
					}
					scores[p] = dot * float64(c.scale)
					if scores[p] > max {
						max = scores[p]
					}
					if a := math.Abs(scores[p]); a > b.score {
						b.score = a
					}
				}
				if limit+1 > b.positions {
					b.positions = limit + 1
				}
				sum := 0.0
				for p := range scores {
					scores[p] = math.Exp(scores[p] - max)
					sum += scores[p]
				}
				for d := 0; d < sh.headDim; d++ {
					acc := 0.0
					for p := 0; p <= limit; p++ {
						val := float64(
							in.v[(s.row(sh, p)*sh.kvHeads+kvHead)*sh.headDim+d])
						acc += scores[p] * val
						if a := math.Abs(val); a > b.value {
							b.value = a
						}
					}
					out[(tok*sh.qHeads+h)*sh.headDim+d] = acc / sum
				}
			}
			tok++
		}
	}
	return out, b
}

// raggedTerms is the budget, every term named.
//
// A dot product over headDim; the softmax weight that dot feeds, whose
// exponential turns the score's absolute error into a relative one; the
// weighted sum over the positions a token attended to; and the exponential and
// division themselves. The magnitude is max|v| and not the answer, because the
// output is a convex combination of value rows and a combination that cancels
// to near zero still carries the error of the rows that cancelled -- which is
// the term the first version of this probe was missing, and which its own
// failure message named.
func raggedTerms(sh raggedShape, b budget) Terms {
	return AccumF32(sh.headDim).
		And(SoftmaxWeight(b.score, sh.headDim)).
		And(AccumF32(b.positions)).
		And(RoundF32(3)).
		And(Magnitude(b.value))
}

// runRagged records and submits the step, returning the result and the plan.
func runRagged(t *testing.T, in raggedInputs) ([]float32, *tensor.Plan) {
	t.Helper()
	sh, c := in.shape, in.c
	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c16-ragged"})

	k := tensor.NewState(r.G.B, tensor.StateDesc{Name: "k", DType: accel.F32,
		Shape: tensor.Shape{c.cacheRows(sh), sh.kvHeads, sh.headDim}})
	v := tensor.NewState(r.G.B, tensor.StateDesc{Name: "v", DType: accel.F32,
		Shape: tensor.Shape{c.cacheRows(sh), sh.kvHeads, sh.headDim}})
	r.F32("k", in.k)
	r.F32("v", in.v)

	q := r.Input("q", accel.F32, tensor.Shape{c.tokens(), sh.qHeads, sh.headDim})
	r.F32("q", in.q)
	pages := r.Input("pages", accel.U32, tensor.Shape{len(c.seqs), sh.maxPages})
	r.U32("pages", in.pages)
	lengths := r.Input("lengths", accel.U32, tensor.Shape{len(c.seqs)})
	r.U32("lengths", in.lengths)
	extents := r.Input("extents", accel.U32, tensor.Shape{len(c.seqs)})
	r.U32("extents", in.extents)
	r.ScalarF32("scale", c.scale)

	out := tensor.Attention(r.G.B, q, k, v, tensor.AttentionOptions{
		Lengths:      lengths,
		Pages:        pages,
		Block:        sh.block,
		ScaleName:    "scale",
		QueryExtents: extents,
	})
	return r.Run(out)
}

// TestC16RaggedStepMatchesItsOracle is the value half of the probe.
func TestC16RaggedStepMatchesItsOracle(t *testing.T) {
	in := newRaggedInputs(probeShape, probeCase)
	got, plan := runRagged(t, in)

	want, b := in.oracle()
	Compare(t, got, want, raggedTerms(probeShape, b), "the ragged step")

	// Which kernel ran, not which one the shape implies. §2's rule: a row's
	// state is what Selections() reports, because an operator that accepts a
	// binding and runs something else is the class that hides.
	var names []string
	for _, s := range plan.Selections() {
		if s.Op == "Attention" {
			names = append(names, s.Kernel)
		}
	}
	if len(names) != 1 || !strings.Contains(names[0], "Ragged") {
		t.Fatalf("the step selected %v; a ragged step runs the ragged kernel, and "+
			"anything else means the extents were read as something they are not",
			names)
	}
	t.Logf("C16: %d tokens over %d sequences in one dispatch, kernel %s",
		probeCase.tokens(), len(probeCase.seqs), names[0])
}

// TestC16MixedStepEqualsTheSeparateSteps is the claim the register row is
// actually about, and it is not the same claim as the one above.
//
// An oracle says the kernel computes attention. This says that putting a
// prefill chunk and a decode in *one* dispatch computes what running them
// apart computes -- which is what a scheduler is going to rely on, and which an
// oracle written from the same segmented reading would not catch if the
// reading itself were wrong.
func TestC16MixedStepEqualsTheSeparateSteps(t *testing.T) {
	in := newRaggedInputs(probeShape, probeCase)
	together, _ := runRagged(t, in)

	// The same tokens, one sequence per step, with each step's own extent.
	// Same q, same caches, same page rows: only the batching differs.
	at, sh := 0, probeShape
	for si, s := range probeCase.seqs {
		if s.extent == 0 {
			continue
		}
		alone := raggedCase{pool: probeCase.pool, scale: probeCase.scale,
			seqs: []seq{s}}
		one := raggedInputs{shape: sh, c: alone,
			k: in.k, v: in.v,
			q:       in.q[at*sh.qHeads*sh.headDim : (at+s.extent)*sh.qHeads*sh.headDim],
			lengths: []uint32{uint32(s.length)},
			extents: []uint32{uint32(s.extent)},
		}
		for i := 0; i < sh.maxPages; i++ {
			one.pages = append(one.pages, uint32(s.pages[i]))
		}
		apart, _ := runRagged(t, one)

		for i, want := range apart {
			got := together[at*sh.qHeads*sh.headDim+i]
			// Exact. The two runs perform the same reductions over the same
			// values in the same order, so any difference is the batching
			// changing what a token attends to rather than f32 reassociation.
			if got != want {
				t.Fatalf("sequence %d element %d is %v in a mixed step of %d "+
					"sequences and %v on its own; a ragged step has to be the "+
					"steps it batches", si, i, got, len(probeCase.seqs), want)
			}
		}
		at += s.extent
	}
	t.Logf("C16: a %d-token chunk and a decode in one dispatch are bit-identical "+
		"to the two steps run apart", probeCase.seqs[0].extent)
}

// TestC16TheExtentsAreRead varies the option and requires the output to move.
//
// specs/010-conformance.md §2: "where an option is optional, vary it and check
// the output moves. An option that changes nothing is either honoured and
// irrelevant, or ignored." The extents are the whole row, so a step that
// produced the same answer under a different split would be reading the flat
// buffer as one sequence and the probe would be green for the wrong reason.
func TestC16TheExtentsAreRead(t *testing.T) {
	in := newRaggedInputs(probeShape, probeCase)
	base, _ := runRagged(t, in)

	// The same four tokens, split 2/1/1 instead of 3/1/0. Every sequence's
	// length is unchanged, so this moves only which token belongs to whom and
	// therefore what each is causal against.
	split := raggedCase{pool: probeCase.pool, scale: probeCase.scale,
		seqs: []seq{
			{extent: 2, length: probeCase.seqs[0].length, pages: probeCase.seqs[0].pages},
			{extent: 1, length: probeCase.seqs[1].length, pages: probeCase.seqs[1].pages},
			{extent: 1, length: probeCase.seqs[2].length, pages: probeCase.seqs[2].pages},
		}}
	moved := newRaggedInputs(probeShape, split)
	moved.q, moved.k, moved.v = in.q, in.k, in.v
	got, _ := runRagged(t, moved)

	if len(got) != len(base) {
		t.Fatalf("the two splits produced %d and %d values", len(got), len(base))
	}
	same := true
	for i := range got {
		if got[i] != base[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("splitting the same tokens 2/1/1 instead of 3/1/0 produced the " +
			"same output; QueryExtents decides which sequence a token belongs to " +
			"and what it is causal against, so a step blind to it is reading the " +
			"flat buffer as one sequence")
	}
	// And the moved split is right, not merely different.
	want, b := moved.oracle()
	Compare(t, got, want, raggedTerms(probeShape, b), "the 2/1/1 split")
}

// TestC22ARaggedStepOverAnF16CacheMatchesTheF32One closes C22.
//
// The row was open because the ragged kernel read f32 only, so the operator
// that makes batching possible gave back the halving [C5] closed for. accel
// shipped the narrow variant against tgo's report
// ([#23](https://github.com/golang-design/accel/issues/23)), and this is the
// probe that says so by value rather than by the issue being closed -- §2.2.1
// records four rows that closed upstream and stayed open here.
func TestC22ARaggedStepOverAnF16CacheMatchesTheF32One(t *testing.T) {
	sh, c := probeShape, probeCase
	in := newRaggedInputs(sh, c)

	// The cache as f16, and the values widened back: what the kernel reads is
	// what the reference must compute from, or the comparison is charged for
	// the storage error twice.
	half := func(v []float32) ([]uint16, []float32) {
		bits := make([]uint16, len(v))
		back := make([]float32, len(v))
		for i, x := range v {
			h := accel.ToFloat16(x)
			bits[i], back[i] = h.Bits(), h.F32()
		}
		return bits, back
	}
	kBits, kBack := half(in.k)
	vBits, vBack := half(in.v)
	narrow := in
	narrow.k, narrow.v = kBack, vBack

	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c22-ragged-f16"})
	shape := tensor.Shape{c.cacheRows(sh), sh.kvHeads, sh.headDim}
	k := tensor.NewState(r.G.B, tensor.StateDesc{Name: "k", DType: accel.F16, Shape: shape})
	v := tensor.NewState(r.G.B, tensor.StateDesc{Name: "v", DType: accel.F16, Shape: shape})
	r.bind("k", accel.F16, len(kBits), kBits)
	r.bind("v", accel.F16, len(vBits), vBits)

	q := r.Input("q", accel.F32, tensor.Shape{c.tokens(), sh.qHeads, sh.headDim})
	r.F32("q", in.q)
	pages := r.Input("pages", accel.U32, tensor.Shape{len(c.seqs), sh.maxPages})
	r.U32("pages", in.pages)
	lengths := r.Input("lengths", accel.U32, tensor.Shape{len(c.seqs)})
	r.U32("lengths", in.lengths)
	extents := r.Input("extents", accel.U32, tensor.Shape{len(c.seqs)})
	r.U32("extents", in.extents)
	r.ScalarF32("scale", c.scale)

	out := tensor.Attention(r.G.B, q, k, v, tensor.AttentionOptions{
		Lengths: lengths, Pages: pages, Block: sh.block,
		ScaleName: "scale", QueryExtents: extents,
	})
	got, plan := r.Run(out)

	want, b := narrow.oracle()
	Compare(t, got, want, raggedTerms(probeShape, b), "the ragged step over an f16 cache")

	var kernel string
	for _, sel := range plan.Selections() {
		if sel.Op == "Attention" {
			kernel = sel.Kernel
		}
	}
	if !strings.Contains(kernel, "Ragged") {
		t.Fatalf("the step selected %q", kernel)
	}
	t.Logf("C22: %s over an f16 cache, half the largest allocation a serving "+
		"process has after the weights", kernel)
}

// TestC23AQueryRowPastTheExtentsIsInert closes C23.
//
// The row was open because a token past the last segment counted every row,
// indexed one past the offsets array, and on a GPU read another sequence's
// cache back as a fluent answer. accel took the shape tgo's report recommended
// ([#24](https://github.com/golang-design/accel/issues/24)): such a row
// contributes nothing rather than being clamped into the last sequence, which
// is the difference between padding and a wrong answer.
//
// What it buys tgo is the padding rule. A batched step can pad q to a plan
// shape and let the extra rows fall off the end, instead of charging them to a
// real sequence's extent.
func TestC23AQueryRowPastTheExtentsIsInert(t *testing.T) {
	sh, c := probeShape, probeCase
	in := newRaggedInputs(sh, c)
	claimed, _ := runRagged(t, in)

	// The same step with two more query rows that no extent claims. The rows
	// carry real values, so a kernel that scored them would produce something
	// rather than zeros.
	padded := in
	padded.q = append(append([]float32(nil), in.q...),
		spread(2*sh.qHeads*sh.headDim, 61)...)

	r := New(t, Tier1, Options{Eps: 1e-6, Label: "c23-padded"})
	shape := tensor.Shape{c.cacheRows(sh), sh.kvHeads, sh.headDim}
	k := tensor.NewState(r.G.B, tensor.StateDesc{Name: "k", DType: accel.F32, Shape: shape})
	v := tensor.NewState(r.G.B, tensor.StateDesc{Name: "v", DType: accel.F32, Shape: shape})
	r.F32("k", in.k)
	r.F32("v", in.v)
	q := r.Input("q", accel.F32, tensor.Shape{c.tokens() + 2, sh.qHeads, sh.headDim})
	r.F32("q", padded.q)
	pages := r.Input("pages", accel.U32, tensor.Shape{len(c.seqs), sh.maxPages})
	r.U32("pages", in.pages)
	lengths := r.Input("lengths", accel.U32, tensor.Shape{len(c.seqs)})
	r.U32("lengths", in.lengths)
	extents := r.Input("extents", accel.U32, tensor.Shape{len(c.seqs)})
	r.U32("extents", in.extents)
	r.ScalarF32("scale", c.scale)

	out := tensor.Attention(r.G.B, q, k, v, tensor.AttentionOptions{
		Lengths: lengths, Pages: pages, Block: sh.block,
		ScaleName: "scale", QueryExtents: extents,
	})
	if err := r.G.Err(); err != nil {
		t.Fatalf("a step with unclaimed query rows was refused: %v", err)
	}
	got, _ := r.Run(out)

	// Every claimed row is untouched by the rows past it.
	if len(got) < len(claimed) {
		t.Fatalf("the padded step produced %d values and the claimed one %d",
			len(got), len(claimed))
	}
	for i := range claimed {
		if got[i] != claimed[i] {
			t.Fatalf("element %d is %v with two unclaimed rows after it and %v "+
				"without; a row belonging to no sequence has to reach nothing",
				i, got[i], claimed[i])
		}
	}
	t.Logf("C23: %d query rows past the extents left every claimed row identical",
		2)
}
