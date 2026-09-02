// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package tgo

import (
	"fmt"
	"math"
	"strings"

	"github.com/latere-ai/tgo/chat"
	"github.com/latere-ai/tgo/internal/grammar"
	"github.com/latere-ai/tgo/sample"
	"github.com/latere-ai/tgo/tokenizer"
)

// decoder is what turns one row of logits into events and a stopping decision.
//
// It is separate from what *produces* the row, and that split is the seam
// specs/022-batched-serving.md §5 names: a single session runs its own forward
// pass and a batched step runs one pass for every slot, and both then do
// exactly this. Building the second one's decode loop beside the first would
// make a sampling bug and a batching bug indistinguishable, which is
// [008-D8](specs/008-scheduler.md)'s argument applied one layer up.
//
// It holds no session and no slot. The token history is a parameter of
// [decoder.consume] rather than a field, because under a scheduler the history
// belongs to the slot and is appended to by Scheduler.Feed: two owners of one
// slice is the aliasing Scheduler.Admit's copy exists to prevent.
type decoder struct {
	m   *Model
	pol Policy
	sp  sample.Policy

	sampler *sample.Sampler
	dec     *tokenizer.Decoder

	// gram is this request's position in the grammar [Policy.Schema] compiled
	// to, and is nil for an unconstrained request. Its state belongs to this
	// request; the caches behind it are the Model's and are shared (015-D1).
	gram *grammar.State

	// maxTokens is the completion budget: [Policy.MaxTokens], or what the
	// capacity left after the prompt allowed.
	maxTokens int

	queue []Event
	head  int
	cur   Event

	done  bool
	err   error
	usage Usage

	// feed is the token the next decode step scores: the one just drawn.
	feed int

	// pending is the decoded text held back because it could still begin a
	// stop string. It is empty whenever Policy.Stop is.
	pending string
	stopped bool

	// stopSeq is the stop string that matched, and reason is why generation
	// ended. Both are set once, by the branch that ends the decode.
	stopSeq string
	reason  StopReason

	// probs is the last step's token probabilities, reused across steps, and
	// is nil unless Policy.LogProbs is set (030-D1).
	probs []TokenProb

	openBlock chat.BlockType
}

// newDecoder prepares the host half of one request.
func newDecoder(m *Model, p Policy, max int, g *grammar.Grammar) *decoder {
	var st *grammar.State
	if g != nil {
		st = g.Start()
	}
	return &decoder{
		m: m, pol: p, sp: p.sampling(), sampler: sample.New(p.Seed),
		dec: m.tok.NewDecoder(), gram: st, maxTokens: max, feed: -1,
	}
}

// consume turns one row of logits into the next token and its events, and
// reports whether the request is over.
//
// history is the tokens already scored, which the penalties read. The drawn
// token is **not** in it: the caller appends afterwards, which is the order
// both [Stream.advance] and a batched step follow.
//
// The row is rewritten in place -- the mask writes over it and the sampler's
// stages rewrite it -- so it is not the distribution the token was drawn from
// once this returns. That is why the logprobs copy is taken before the draw.
func (d *decoder) consume(logits []float32, history []int) (bool, error) {
	// The mask goes on before the draw, which is where 015-D2 puts it: the
	// penalties and the temperature live inside Next, and both are monotone in
	// the logit with -Inf as a fixed point, so a token masked here cannot be
	// brought back by either.
	if d.gram != nil {
		if err := d.gram.Mask(logits); err != nil {
			return true, fmt.Errorf("tgo: masking a constrained step: %w", err)
		}
	}
	// Before the draw, and on a copy: Next rewrites the row in place, so
	// afterwards these are not the logits the token was drawn from. After the
	// mask, because a grammar-forbidden token must report -Inf rather than the
	// chance it had before the grammar cut it (030 §5).
	var dist []float32
	if d.pol.LogProbs {
		dist = d.sampler.Probs(logits, history, d.sp)
	}

	tok := d.sampler.Next(logits, history, d.sp)
	d.feed = tok
	d.recordProbs(dist, tok)

	if d.isStop(tok) {
		// An end-of-turn id. It is the model saying it is done, which is a
		// different answer from a caller's budget running out and from a stop
		// string the caller wrote -- and /v1/messages renders all three
		// differently.
		d.reason = StopEndTurn
		return true, nil
	}
	// Advance consumes the token that was drawn, and only a token that is part
	// of the document. A stop id is the branch above: the request does not emit
	// it and does not count it, and the grammar admits it exactly where the
	// document is already complete, so advancing over it would mutate a state
	// nothing reads again.
	if d.gram != nil {
		if err := d.gram.Advance(tok); err != nil {
			return true, fmt.Errorf("tgo: advancing a constrained step: %w", err)
		}
	}
	d.usage.CompletionTokens++
	d.emit(tok)
	switch {
	case d.stopped:
		d.reason = StopSequence
		return true, nil
	case d.usage.CompletionTokens >= d.maxTokens:
		d.reason = StopMaxTokens
		return true, nil
	}
	return false, nil
}

// isStop reports whether a token ends the completion without being part of it.
func (d *decoder) isStop(tok int) bool {
	sp := d.m.special
	return tok == sp.imEnd || tok == sp.endOfText
}

// emit turns one token into events.
//
// The structural markers are matched by id and never by looking at decoded
// text. A stop found in text has the boundary problem 003-D6 rejects for turn
// markers and fails the same way: a user who asks the model to explain
// "</think>" would have the explanation cut in half.
func (d *decoder) emit(tok int) {
	sp := d.m.special
	switch tok {
	case sp.think[0], sp.think[1]:
		d.startBlock(chat.BlockThinking)
		return
	case sp.toolCall:
		d.startBlock(chat.BlockToolUse)
		return
	case sp.thinkEnd, sp.toolEnd:
		d.endBlock()
		return
	}
	if text := d.dec.Push(tok); text != "" {
		d.delta(text)
	}
}

// delta queues a piece of text, opening a text block if none is open.
func (d *decoder) delta(text string) {
	if d.openBlock == "" {
		d.openBlock = chat.BlockText
		d.queue = append(d.queue, Event{Kind: BlockStart, Block: chat.BlockText})
	}
	d.pending += text
	d.drain(false)
}

// startBlock closes whatever block is open and opens this one.
func (d *decoder) startBlock(bt chat.BlockType) {
	d.endBlock()
	d.openBlock = bt
	d.queue = append(d.queue, Event{Kind: BlockStart, Block: bt})
}

// endBlock flushes what is held back and closes the open block.
func (d *decoder) endBlock() {
	if d.openBlock == "" {
		return
	}
	d.drain(true)
	d.queue = append(d.queue, Event{Kind: BlockStop, Block: d.openBlock})
	d.openBlock = ""
}

// drain releases as much held-back text as is safe.
//
// While a stop string is set, the longest suffix of the output that could still
// begin one is held: a stop string need not align to a token boundary, so
// matching it means matching decoded text, and text already handed to the
// caller cannot be taken back (006-D4).
func (d *decoder) drain(final bool) {
	if d.pending == "" {
		return
	}
	if i, which := firstStop(d.pending, d.pol.Stop); i >= 0 {
		d.emitText(d.pending[:i])
		d.pending = ""
		d.stopped = true
		d.stopSeq = which
		return
	}
	keep := 0
	if !final {
		keep = holdBack(d.pending, d.pol.Stop)
	}
	out := d.pending[:len(d.pending)-keep]
	d.pending = d.pending[len(d.pending)-keep:]
	d.emitText(out)
}

// emitText queues one delta of the open block's kind.
func (d *decoder) emitText(s string) {
	if s == "" {
		return
	}
	d.queue = append(d.queue, Event{Kind: deltaKind(d.openBlock), Block: d.openBlock,
		Text: s})
}

// end closes the decode, flushing the detokenizer and closing any open block.
//
// The flush happens even on a failure: text the model produced before the
// device failed is text the caller already paid for, and dropping it would make
// the error harder to read rather than easier.
func (d *decoder) end(err error) {
	if d.done {
		return
	}
	d.done = true
	d.err = err
	// Through delta rather than straight onto pending: what the detokenizer
	// holds at the end is a truncated code point, and if no block was ever
	// opened -- a completion that is one partial character -- appending to
	// pending would leave it in a buffer endBlock does not drain.
	if rest := d.dec.Flush(); rest != "" && !d.stopped {
		d.delta(rest)
	}
	d.endBlock()
}

// recordProbs fills the step's [Stream.LogProbs], reusing the backing array.
//
// dist is nil when [Policy.LogProbs] is off, and then the slice is emptied
// rather than left holding the previous step's answer -- a stale value here
// would be read as this token's.
func (d *decoder) recordProbs(dist []float32, tok int) {
	d.probs = d.probs[:0]
	if dist == nil {
		return
	}
	d.probs = append(d.probs, d.tokenProb(dist, tok))
	if d.pol.TopLogProbs > 0 {
		d.probs[0].Top = d.topProbs(dist)
	}
}

// tokenProb is one entry: the id, what it contributes to the output, and ln of
// its probability.
func (d *decoder) tokenProb(dist []float32, id int) TokenProb {
	p := 0.0
	if id >= 0 && id < len(dist) {
		p = float64(dist[id])
	}
	// math.Log(0) is -Inf already, which is the value 030-D3 wants, so this is
	// not a special case -- it is here because Log of a negative would be NaN
	// and a distribution cannot hold one.
	return TokenProb{ID: id, Text: string(d.m.tok.TextBytes(id)), LogProb: math.Log(p)}
}

// topProbs is the Policy.TopLogProbs most likely tokens, descending.
//
// Ties go to the lower id, which is accel's rule and 006's: two tokens with the
// same weight are ordinary at the tail of a 151936-entry distribution, and an
// unstated order would let two runs of one seed report different alternatives.
func (d *decoder) topProbs(dist []float32) []TokenProb {
	n := min(d.pol.TopLogProbs, len(dist))
	idx := make([]int, 0, n)
	for id, w := range dist {
		if len(idx) == n && !(w > dist[idx[n-1]] || (w == dist[idx[n-1]] && id < idx[n-1])) {
			continue
		}
		if len(idx) < n {
			idx = append(idx, 0)
		}
		j := len(idx) - 1
		for j > 0 && (w > dist[idx[j-1]] || (w == dist[idx[j-1]] && id < idx[j-1])) {
			idx[j] = idx[j-1]
			j--
		}
		idx[j] = id
	}
	out := make([]TokenProb, len(idx))
	for i, id := range idx {
		out[i] = d.tokenProb(dist, id)
	}
	return out
}

// Event is the current event.
func (d *decoder) Event() Event { return d.cur }

// Text is the current event's text delta, and is empty for an event that is not
// one.
func (d *decoder) Text() string { return d.cur.Text }

// Usage is the prompt and completion token counts so far.
func (d *decoder) Usage() Usage { return d.usage }

// LogProbs is the tokens the last [Stream.Next] produced, with their
// probabilities, and is empty unless [Policy.LogProbs] is set.
//
// A slice and not one value because a step can produce no token -- a prefill --
// and a batched step may one day produce more than one, and a caller reading a
// length does not have to change when either happens.
//
// **Valid until the next Next.** The backing array is reused: a per-token
// allocation in the decode loop is the cost specs/017-benchmarks.md 017-D3
// warns an instrument must not impose on what it measures. A caller keeping
// them appends.
func (d *decoder) LogProbs() []TokenProb { return d.probs }

// StopReason is why the stream ended, and is [StopRunning] until it has.
//
// It is not set when the stream ended in an error or a cancellation: those are
// [Stream.Err]'s, and reporting a reason for a completion that did not complete
// would let a caller answer a failed request as a finished one.
func (d *decoder) StopReason() StopReason { return d.reason }

// StopSequence is the stop string that ended the stream, and is empty unless
// [Stream.StopReason] is [StopSequence].
func (d *decoder) StopSequence() string { return d.stopSeq }

// deltaKind is the delta event a block's text belongs to.
func deltaKind(bt chat.BlockType) EventKind {
	switch bt {
	case chat.BlockThinking:
		return ThinkingDelta
	case chat.BlockToolUse:
		return ToolArgsDelta
	}
	return TextDelta
}

// firstStop is the index of the earliest stop string in s, or -1.
func firstStop(s string, stops []string) (int, string) {
	best, which := -1, ""
	for _, stop := range stops {
		if i := strings.Index(s, stop); i >= 0 && (best < 0 || i < best) {
			best, which = i, stop
		}
	}
	return best, which
}

// holdBack is how many trailing bytes of s could still begin a stop string.
//
// A proper prefix only: a suffix that is a whole stop string is a stop, which
// [firstStop] has already found, and holding it back would be waiting for a
// match that has happened.
func holdBack(s string, stops []string) int {
	keep := 0
	for _, stop := range stops {
		n := min(len(stop)-1, len(s))
		for k := n; k > 0; k-- {
			if strings.HasSuffix(s, stop[:k]) {
				keep = max(keep, k)
				break
			}
		}
	}
	return keep
}
