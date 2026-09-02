// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tgo

import (
	"encoding/binary"
	"fmt"
	"math"

	"golang.design/x/accel"

	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/safetensors"
)

// loadGains uploads every RMSNorm gain as f32, which the loader cannot do.
//
// # Why this is here and not in tgo/weights
//
// specs/004-model-graph.md §3 declares a norm gain as an f32 port —
// nn.Graph.Gain writes accel.F32 and takes no policy — and
// specs/001-weights.md §2's pipeline ends at "precision → f16 or int8", so
// weights.Load has no f32 output at all. accel binds by exact dtype
// (accel/tensor's plan refuses a view whose dtype is not the port's), so a gain
// that came through the loader fails at the first submission with "declared f32
// and the bound view is f16".
//
// The two shipped packages therefore cannot be composed without a widening step
// between them, and the engine is the first place they meet. See this package's
// reported discrepancies.
//
// It is small: a gain is one value per feature, so all 113 of Qwen3-0.6B's are
// 28·(1024 + 128 + 128 + 1024) + 1024 = 65536 floats — 256 KB against 1.4 GB of
// projections, and 198 tensors reach the loader rather than 311. What it must
// not skip is
// the rotary permutation on the q_norm and k_norm gains, which follow the
// channels they scale (004-D9): getting that wrong scales the wrong channels
// and produces fluent text that loses coherence, with nothing to catch it.
func (m *Model) loadGains(repo *safetensors.Repo, specs []model.WeightSpec) error {
	for _, s := range specs {
		if s.Kind != model.KindGain {
			continue
		}
		plane, err := gainPlane(repo, s)
		if err != nil {
			return err
		}
		if s.Permute {
			if err := permuteHeads(plane, s.Heads); err != nil {
				return fmt.Errorf("tgo: %q: %w", s.Tensor, err)
			}
		}
		buf, err := m.dev.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: len(plane), Label: s.Port,
			Usage: accel.BufferStorage | accel.BufferCopyDst | accel.BufferCopySrc,
		})
		if err != nil {
			return fmt.Errorf("tgo: allocating the %q gain: %w", s.Port, err)
		}
		if err := m.dev.Queue().WriteBuffer(buf, 0, plane); err != nil {
			_ = buf.Close()
			return fmt.Errorf("tgo: uploading the %q gain: %w", s.Port, err)
		}
		m.gains[s.Port] = buf
	}
	return nil
}

// gainPlane reads one gain and widens it to f32.
func gainPlane(repo *safetensors.Repo, s model.WeightSpec) ([]float32, error) {
	e, file, ok := repo.Tensor(s.Tensor)
	if !ok {
		return nil, fmt.Errorf("tgo: the checkpoint has no %q", s.Tensor)
	}
	if len(e.Shape) != 1 {
		return nil, fmt.Errorf("tgo: %q has shape %v; a norm gain is one value per feature",
			s.Tensor, e.Shape)
	}
	raw, err := file.Bytes(s.Tensor)
	if err != nil {
		return nil, fmt.Errorf("tgo: %q: %w", s.Tensor, err)
	}
	out := make([]float32, e.Shape[0])
	if err := widen(e.DType, raw, out); err != nil {
		return nil, fmt.Errorf("tgo: %q: %w", s.Tensor, err)
	}
	return out, nil
}

// widen decodes a checkpoint plane into f32.
//
// The three float widths a checkpoint holds and nothing else: a reader that
// picked a plausible width for an unknown dtype would read the right bytes as
// the wrong numbers and report nothing (001-D6).
func widen(dt safetensors.DType, src []byte, dst []float32) error {
	if want := len(dst) * dt.Size(); dt.Size() == 0 || len(src) != want {
		return fmt.Errorf("a %v plane of %d values is %d bytes and the file holds %d",
			dt, len(dst), want, len(src))
	}
	switch dt {
	case safetensors.BF16:
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[2*i:])) << 16)
		}
	case safetensors.F16:
		for i := range dst {
			dst[i] = accel.Float16FromBits(binary.LittleEndian.Uint16(src[2*i:])).F32()
		}
	case safetensors.F32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[4*i:]))
		}
	default:
		return fmt.Errorf("dtype is %v; a norm gain is BF16, F16 or F32", dt)
	}
	return nil
}

// permuteHeads rewrites a plane's channels from Qwen3's half-split rotary
// pairing into accel's interleaved one, per head:
//
//	y[2i] = x[i], y[2i+1] = x[i + d_h/2], 0 <= i < d_h/2
//
// specs/004-model-graph.md §2.5.2. A QK-norm gain carries one head's channels,
// shared across heads, so heads is 1 and the whole vector is one head.
func permuteHeads(plane []float32, heads int) error {
	if heads <= 0 || len(plane)%heads != 0 {
		return fmt.Errorf("a %d-value plane does not split into %d heads", len(plane), heads)
	}
	dh := len(plane) / heads
	if dh%2 != 0 {
		return fmt.Errorf("head_dim is %d; RoPE rotates pairs, so it is even", dh)
	}
	half := dh / 2
	tmp := make([]float32, dh)
	for h := range heads {
		head := plane[h*dh : (h+1)*dh]
		for i := range half {
			tmp[2*i] = head[i]
			tmp[2*i+1] = head[i+half]
		}
		copy(head, tmp)
	}
	return nil
}

// gainBytes is what the f32 gains occupy on the device, which the loader's
// report does not count because the loader did not upload them.
func gainBytes(specs []model.WeightSpec, c *model.Config) int64 {
	var n int64
	for _, s := range specs {
		if s.Kind != model.KindGain {
			continue
		}
		w := int64(c.HiddenSize)
		if len(s.Shape) == 1 {
			w = int64(s.Shape[0])
		}
		n += 4 * w
	}
	return n
}
