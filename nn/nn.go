// Package nn is the library of composites a transformer is made of.
//
// A block takes tensors and returns tensors. It holds no state, no device and
// no weights: weights arrive as [tensor.Weight] ports declared by name, which
// is specs/004-model-graph.md 004-D1. A model is a function that calls these
// blocks; the registry owns naming and the loader owns bytes.
//
// # Errors
//
// A block that cannot be built records the problem on its [Graph] and returns
// nil, which accel reads as a poisoned operand. Model code therefore has no
// error branch per line, exactly as accel's builder intends, and [Graph.Err]
// reports what nn and accel between them found.
package nn

import (
	"errors"
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"
)

// ScaleSuffix names the scale plane of a quantized weight port.
//
// A quantized matrix is two device buffers -- i8 quants and f16 scales -- and
// nothing above accel fixed a name for the second. nn fixes it here so the
// loader and the graph agree by construction rather than by convention: the
// quants keep the weight's own name and the scales append this.
const ScaleSuffix = ".scales"

// Graph is what every block records into.
//
// It is [tensor.Builder] plus the per-model constants a block would otherwise
// take as six arguments.
type Graph struct {
	// B is the builder the blocks record into.
	B *tensor.Builder

	// Eps is rms_norm_eps. It must be positive: it is what keeps a row of
	// zeros from dividing by zero, and accel refuses a zero value rather than
	// treating an unset field as a default.
	Eps float32

	// Prefix is the weight-name prefix for the block being recorded, joined to
	// a block's name with no separator. The caller owns the separator because
	// nn holds no naming policy -- the checkpoint does.
	Prefix string

	// Stored reports how the loader stored the weight of this full name: it
	// answers [accel.F16] for a dense plane and [accel.I8] for quants plus
	// scales. A nil func means every weight is f16.
	//
	// A function rather than a map so a model with 28 layers does not build one
	// entry per port to say the same thing about all of them, and per name
	// rather than per graph because a checkpoint may quantize the projections
	// and leave the embedding dense.
	Stored func(name string) accel.DType

	errs []error
}

// Err reports what nn and accel between them found.
//
// Both, because a block that refuses records here and accel's operators record
// there, and a caller who checked only one would compile a graph that had
// already failed.
func (g *Graph) Err() error {
	if len(g.errs) == 0 {
		if g.B == nil {
			return nil
		}
		return g.B.Err()
	}
	errs := g.errs
	if g.B != nil {
		if err := g.B.Err(); err != nil {
			errs = append(append([]error{}, errs...), err)
		}
	}
	return errors.Join(errs...)
}

// fail records a refusal and returns nil, which accel's operators read as a
// poisoned operand: one diagnostic per mistake, and no error branch per line.
//
// The prefix is part of the message because a block is called once per layer,
// and "Attention: head_dim disagrees" without one does not say which layer's
// weights were declared wrong.
func (g *Graph) fail(op, format string, args ...any) *tensor.Tensor {
	where := g.Prefix
	if where == "" {
		where = "<no prefix>"
	}
	g.errs = append(g.errs, fmt.Errorf("tgo/nn: %s: %s: %s", op, where,
		fmt.Sprintf(format, args...)))
	return nil
}

// Operand is a weight that is either f16 or int8 quants with f16 scales.
//
// It exists so a block writes one call rather than branching on precision at
// every projection (004-D6): precision is a load-time decision and not a graph
// one.
type Operand struct {
	// Dense is an f16 matrix, [K, N].
	Dense *tensor.Tensor

	// Quant is the same matrix as i8 quants and f16 scales.
	Quant tensor.Quantized
}

// IsQuant reports whether this operand carries the quantized form.
func (o Operand) IsQuant() bool { return o.Quant.Quants != nil || o.Quant.Scales != nil }

// ok reports whether exactly one of the two forms is present.
func (o Operand) ok() bool {
	if o.IsQuant() {
		return o.Dense == nil && o.Quant.Quants != nil && o.Quant.Scales != nil
	}
	return o.Dense != nil
}

// Weight declares a f16 or quantized weight port by name, resolving which of
// the two from how the loader stored it.
//
// shape is the matrix as the graph multiplies it, [K, N] -- already transposed,
// which is specs/001-weights.md's job and not this one's. A quantized port
// declares two: the quants under name, and the scales under name+[ScaleSuffix]
// with one f16 per [quant.Int8Block] weights of the flattened matrix.
func (g *Graph) Weight(name string, shape tensor.Shape) Operand {
	full := g.Prefix + name
	if len(shape) == 0 || shape.Elements() <= 0 {
		g.fail("Weight", "%q is %v; a weight port has a positive extent", full, shape)
		return Operand{}
	}
	dtype := accel.F16
	if g.Stored != nil {
		dtype = g.Stored(full)
	}
	switch dtype {
	case accel.F16:
		return Operand{Dense: tensor.Weight(g.B, tensor.ValueDesc{
			Name: full, DType: accel.F16, Shape: shape,
		})}
	case accel.I8:
		blocks := (shape.Elements() + quant.Int8Block - 1) / quant.Int8Block
		return Operand{Quant: tensor.Quantized{
			Quants: tensor.Weight(g.B, tensor.ValueDesc{
				Name: full, DType: accel.I8, Shape: shape,
			}),
			Scales: tensor.Weight(g.B, tensor.ValueDesc{
				Name: full + ScaleSuffix, DType: accel.F16, Shape: tensor.Shape{blocks},
			}),
		}}
	default:
		g.fail("Weight", "%q is stored as %v; a weight is f16 or int8 quants with f16 "+
			"scales (specs/001-weights.md section 2)", full, dtype)
		return Operand{}
	}
}

// Gain declares an f32 norm gain port of width values by name.
//
// f32 and never quantized: a gain is one value per feature, so it is the
// smallest thing in the checkpoint and the one whose rounding a normalization
// would carry into every row it scales.
func (g *Graph) Gain(name string, width int) *tensor.Tensor {
	full := g.Prefix + name
	if width <= 0 {
		return g.fail("Gain", "%q is %d wide; a gain is one value per feature", full, width)
	}
	return tensor.Weight(g.B, tensor.ValueDesc{
		Name: full, DType: accel.F32, Shape: tensor.Shape{width},
	})
}
