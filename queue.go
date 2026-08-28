// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/latere-ai/tgo/internal/prefix"
)

// The queue in front of admission.
//
// specs/021-admission-queue.md. [Scheduler.Admit] refuses when every slot is
// live or when the block pool cannot hold the prompt and its reserve. A refusal
// is the right answer to a caller who can retry and the wrong answer to an HTTP
// request, so this file is the one place a request waits on the conjunction
// specs/008-scheduler.md §3 states:
//
//	admissible <=> (a free slot) and ceil((T_prompt + R) / B_block) <= blocks free
//
// Only the scheduler can evaluate it, because only it holds both the slot table
// and the block pool. So the queue drives the scheduler through the
// non-blocking surface it already has, and does the waiting itself.

const (
	// DefaultQueueDepth is how many waiters the queue holds per slot, so the
	// bound scales with a deployment that raises its slot count rather than
	// staying at a constant somebody chose for four slots.
	//
	// specs/021-admission-queue.md §4: 8 x 4 slots is 32, which is the number
	// server.DefaultQueue already is.
	DefaultQueueDepth = 8

	// DefaultQueueWait is how long a waiter waits before it is refused. It is
	// server.DefaultQueueWait, and it is the only interval the queue can
	// promise a client: it is the longest a request that *was* queued would
	// have waited.
	DefaultQueueWait = 30 * time.Second
)

var (
	// ErrQueueFull is what [Queue.Admit] refuses with past its depth bound. An
	// unbounded queue converts a refusal into an unbounded latency (009-D3).
	ErrQueueFull = errors.New("tgo: the admission queue is full")

	// ErrQueueTimeout is what [Queue.Admit] refuses with past its wait budget.
	// It is deliberately not the caller's context error: the two answer 429 and
	// 499, and a context derived from the budget would collapse them into one
	// (021-D7).
	ErrQueueTimeout = errors.New("tgo: the admission queue's wait budget elapsed")

	// ErrQueueClosed is what a waiter gets when the queue is closed under it.
	ErrQueueClosed = errors.New("tgo: the admission queue is closed")
)

// Admitter is what a [Queue] needs of a scheduler. [Scheduler] satisfies it.
//
// The interface is what makes the queue testable without a device.
// specs/008-scheduler.md §8 records that the scheduler's own policy is pure
// functions over integers, tested in microseconds; the queue is the same class
// of policy, and a fake admitter that refuses on demand exercises every
// ordering, bound and cancellation case with no weights loaded.
type Admitter interface {
	// Slots is the batch width, which is what the queue's defaults scale with.
	Slots() int

	// Feasible reports whether a prompt of this length could ever be admitted.
	Feasible(prompt int) error

	// Admit takes a slot or refuses. It never waits, which is the point.
	Admit(prompt []int, salt string) (int, error)

	// Finish gives a slot back. The queue calls it for a slot it won for a
	// caller who had already left (021-D10).
	Finish(slot int) error

	// Capacity fires when a slot or a block was released, so a waiter wakes on
	// an event rather than on a timer.
	Capacity() <-chan struct{}
}

// QueueOptions are the three numbers §3 and §4 leave to a deployment.
type QueueOptions struct {
	// Depth is how many requests may wait. Zero takes [DefaultQueueDepth]
	// times the admitter's slot count.
	Depth int

	// Wait is how long one of them waits before it is refused. Zero takes
	// [DefaultQueueWait].
	Wait time.Duration

	// Overtake is K: how many times the waiter at the head may be passed by a
	// later one before it becomes reserving and nothing is admitted before it.
	//
	// It is a pointer because zero is a meaningful value -- it is strict FIFO
	// -- so the usual "zero takes the default" convention cannot express both.
	// Nil takes the admitter's slot count.
	Overtake *int
}

// Ticket is a request's place in arrival order.
//
// A caller takes one when the request arrives and passes it to every
// [Queue.Admit] for that request. A sequence the scheduler evicts is
// resubmitted with the ticket it already holds, so it re-enters ahead of the
// waiters that arrived after it and an eviction cannot livelock against the
// queue (specs/021-admission-queue.md §3, 021-D5).
type Ticket uint64

// QueueStats is what the queue measures, drained.
//
// The queue is in package tgo and the Prometheus exposition is server's, whose
// metrics type is package-private (021-D9). So the queue reports and server
// names the series.
//
// Every field but Depth is **since the last read**: [Queue.Stats] drains them,
// so a caller adds the deltas to its own counters and two callers do not each
// see the whole history. Depth is instantaneous.
type QueueStats struct {
	// Depth is how many requests are waiting right now.
	Depth int

	// Waits is one observation per admission. A request admitted without ever
	// being deferred observes zero rather than nothing, so the histogram's
	// count is admissions and its shape says what share of them waited at all.
	Waits []time.Duration

	// Deferred counts admissions that were refused and retried, by reason:
	// "no_slot" or "block_pool". specs/008-scheduler.md §3 requires the two to
	// be distinguishable, because a server that reports one number for both is
	// indistinguishable from a slow one.
	Deferred map[string]int

	// Refused counts requests that left without a slot, by reason:
	// "infeasible", "queue_full", "queue_timeout" or "client_gone".
	Refused map[string]int
}

// Queue is FIFO admission over an [Admitter], with a bounded overtake.
//
// # Where the waiting happens
//
// One goroutine drives the whole waiter list. specs/021-admission-queue.md §3's
// rules -- arrival order, the head tried first, the overtake count, the
// monotone reserving flag -- are global state, and a waiter looping on
// Capacity() by itself can see none of them. A capacity channel with N readers
// also loses wakeups: the reader that wins the signal may be the one rule §3
// forbids admitting, and the rest sleep until the next release.
//
// # Where the lock is not
//
// The driver never holds the queue's lock across [Admitter.Admit].
// [Scheduler.Step] holds the scheduler's own lock for a whole device dispatch,
// so a queue that held its lock through an admission would stall every arrival
// and every cancellation for the length of a forward pass (021-D2).
type Queue struct {
	a        Admitter
	depth    int
	wait     time.Duration
	overtake int

	mu      sync.Mutex
	waiters []*waiter
	closed  bool

	waits     []time.Duration
	deferrals map[string]int
	refusals  map[string]int

	// tickets stamps arrival. It is separate from the lock because a ticket is
	// taken when a request arrives, which is before the queue is involved.
	tickets atomic.Uint64

	wake    chan struct{}
	done    chan struct{}
	stopped chan struct{}
}

// Admitter is satisfied by the scheduler, which is the only thing that can
// evaluate §1's conjunction.
var _ Admitter = (*Scheduler)(nil)

// The three states a waiter can be in. Exactly one transition out of waiting
// happens, under the queue's lock, which is what makes cancel-after-admit a
// decision rather than a race (§5).
const (
	waiting = iota
	resolved
	gone
)

type outcome struct {
	slot int
	err  error
}

type waiter struct {
	prompt []int
	salt   string
	ticket Ticket
	at     time.Time

	// passed is how many times a later waiter was admitted ahead of this one,
	// and reserving is set on the K-th. It is monotone: once set it stays set
	// until the waiter leaves, so a head cannot be starved by a stream of small
	// prompts behind it.
	passed    int
	reserving bool

	// deferred is how many admissions were refused for this waiter. Zero is
	// what "did not wait" means here: a fact about the queue rather than a
	// clock comparison that can never be exactly zero.
	deferred int

	state int
	out   outcome
	res   chan outcome
}

// NewQueue builds a queue over an admitter and starts its driver.
//
// The caller closes it with [Queue.Close].
func NewQueue(a Admitter, o QueueOptions) (*Queue, error) {
	if a == nil {
		return nil, errors.New("tgo: a queue needs an admitter to admit into")
	}
	n := a.Slots()
	if n < 1 {
		return nil, fmt.Errorf("tgo: the admitter has %d slots; a queue in front of "+
			"nothing would refuse every request it accepted", n)
	}
	if o.Depth == 0 {
		o.Depth = DefaultQueueDepth * n
	}
	if o.Depth < 1 {
		return nil, fmt.Errorf("tgo: the queue's depth is %d; a queue that holds "+
			"nobody is a refusal with extra steps", o.Depth)
	}
	if o.Wait == 0 {
		o.Wait = DefaultQueueWait
	}
	if o.Wait < 0 {
		return nil, fmt.Errorf("tgo: the queue's wait budget is %s; it is how long a "+
			"request waits before it is refused", o.Wait)
	}
	k := n
	if o.Overtake != nil {
		k = *o.Overtake
	}
	if k < 0 {
		return nil, fmt.Errorf("tgo: the queue's overtake bound is %d; zero is strict "+
			"arrival order and larger is how many times a head may be passed", k)
	}
	q := &Queue{
		a: a, depth: o.Depth, wait: o.Wait, overtake: k,
		deferrals: map[string]int{}, refusals: map[string]int{},
		wake: make(chan struct{}, 1), done: make(chan struct{}),
		stopped: make(chan struct{}),
	}
	go q.run()
	return q, nil
}

// NewTicket stamps a request's arrival.
//
// Take one per request, not one per admission: the second [Queue.Admit] for a
// sequence the scheduler evicted must carry the first one's ticket, which is
// what keeps an eviction from pushing a request behind the waiters that arrived
// after it (021-D5).
func (q *Queue) NewTicket() Ticket { return Ticket(q.tickets.Add(1)) }

// Wait is the queue's budget: the only interval it can promise, and what a
// Retry-After is derived from (021-D6).
func (q *Queue) Wait() time.Duration { return q.wait }

// MaxDepth is the bound on how many requests may wait at once.
func (q *Queue) MaxDepth() int { return q.depth }

// Depth is how many requests are waiting right now. It is [Queue.Stats]'s
// Depth without the drain, so a caller may read it as often as it likes.
func (q *Queue) Depth() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.waiters)
}

// Admit places a prompt in a slot, waiting until one is free or until the
// request runs out of budget, context or queue.
//
// t orders the request. Take one ticket per request with [Queue.NewTicket] and
// pass the same one to every Admit for it, so a sequence the scheduler evicted
// re-enters where it was rather than behind everything that arrived since.
func (q *Queue) Admit(ctx context.Context, t Ticket, prompt []int, salt string) (int, error) {
	// The door. An unsatisfiable request is refused here rather than queued,
	// which is what makes §3's head-of-line bound finite: every waiter at the
	// head is admissible eventually, because blocks come back when a sequence
	// finishes (021-D3).
	if err := q.a.Feasible(len(prompt)); err != nil {
		q.refuse("infeasible")
		return -1, err
	}
	w := &waiter{
		prompt: prompt, salt: salt, ticket: t, at: time.Now(),
		res: make(chan outcome, 1),
	}
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return -1, ErrQueueClosed
	}
	if len(q.waiters) >= q.depth {
		q.refusals["queue_full"]++
		d := q.depth
		q.mu.Unlock()
		return -1, fmt.Errorf("%w: %d requests are waiting", ErrQueueFull, d)
	}
	q.insert(w)
	q.mu.Unlock()
	q.nudge()

	// The budget is a timer beside the context and not a context derived from
	// it, because the two produce different answers -- 429 for the budget, 499
	// for the client -- and context.WithTimeout would report DeadlineExceeded
	// for both (021-D7).
	timer := time.NewTimer(q.wait)
	defer timer.Stop()
	select {
	case out := <-w.res:
		return out.slot, out.err
	case <-timer.C:
		return q.leave(w, "queue_timeout",
			fmt.Errorf("%w: waited %s for a slot", ErrQueueTimeout, q.wait))
	case <-ctx.Done():
		return q.leave(w, "client_gone", ctx.Err())
	}
}

// Stats reports what the queue measured and drains everything but the depth.
func (q *Queue) Stats() QueueStats {
	q.mu.Lock()
	defer q.mu.Unlock()
	s := QueueStats{
		Depth: len(q.waiters), Waits: q.waits,
		Deferred: q.deferrals, Refused: q.refusals,
	}
	q.waits, q.deferrals, q.refusals = nil, map[string]int{}, map[string]int{}
	return s
}

// Close stops the driver and drains every waiter with [ErrQueueClosed].
//
// A waiter left hanging on a closed queue is a request that never answers, so
// closing refuses them rather than waiting for their budgets to elapse.
func (q *Queue) Close() error {
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return nil
	}
	q.closed = true
	ws := q.waiters
	q.waiters = nil
	for _, w := range ws {
		if w.state == waiting {
			w.state, w.out = resolved, outcome{slot: -1, err: ErrQueueClosed}
			w.res <- w.out
		}
	}
	q.mu.Unlock()
	close(q.done)
	<-q.stopped
	return nil
}

// run is the driver. It re-evaluates the whole waiter list on every event that
// could change the answer, and on nothing else.
func (q *Queue) run() {
	defer close(q.stopped)
	capacity := q.a.Capacity()
	for {
		select {
		case <-q.done:
			return
		case <-capacity:
		case <-q.wake:
		}
		q.drive()
	}
}

// drive walks the waiters in arrival order and applies §3.
func (q *Queue) drive() {
	for {
		q.mu.Lock()
		if q.closed || len(q.waiters) == 0 {
			q.mu.Unlock()
			return
		}
		head := q.waiters[0]
		q.mu.Unlock()

		if q.attempt(head) {
			// The head left with a slot, so the list has a new head and the
			// same question is asked of it.
			continue
		}
		q.mu.Lock()
		if q.closed || len(q.waiters) == 0 || q.waiters[0] != head {
			q.mu.Unlock()
			continue
		}
		if head.passed >= q.overtake {
			head.reserving = true
		}
		if head.reserving {
			q.mu.Unlock()
			return
		}
		followers := append([]*waiter(nil), q.waiters[1:]...)
		q.mu.Unlock()

		progress := false
		for _, f := range followers {
			if !q.attempt(f) {
				continue
			}
			progress = true
			q.mu.Lock()
			head.passed++
			if head.passed >= q.overtake {
				head.reserving = true
			}
			stop := head.reserving
			q.mu.Unlock()
			if stop {
				break
			}
		}
		if !progress {
			return
		}
	}
}

// attempt tries one waiter against the admitter and reports whether it left.
//
// The admitter is called off the queue's lock, so a waiter can cancel or time
// out while the call is in flight. That is the race §5 is about, and the answer
// is here: the state check and the release are one critical section, so a slot
// won for a caller who has already gone is given back rather than held for the
// life of the process (021-D10).
func (q *Queue) attempt(w *waiter) bool {
	q.mu.Lock()
	if q.closed || w.state != waiting {
		q.mu.Unlock()
		return false
	}
	q.mu.Unlock()

	slot, err := q.a.Admit(w.prompt, w.salt)

	q.mu.Lock()
	if err != nil {
		if deferrable(err) {
			w.deferred++
			q.deferrals[deferral(err)]++
			q.mu.Unlock()
			return false
		}
		// Not a wait: the request is wrong, or the batch is gone. The waiter
		// leaves with the error rather than retrying it forever.
		q.settle(w, outcome{slot: -1, err: err})
		q.mu.Unlock()
		return false
	}
	if w.state != waiting {
		q.mu.Unlock()
		_ = q.a.Finish(slot)
		return true
	}
	q.settle(w, outcome{slot: slot})
	q.mu.Unlock()
	return true
}

// settle resolves a waiter and takes it out of the list. The caller holds q.mu
// and has checked that the waiter is still waiting.
func (q *Queue) settle(w *waiter, out outcome) {
	w.state, w.out = resolved, out
	w.res <- out
	q.remove(w)
	if out.err != nil {
		return
	}
	d := time.Duration(0)
	if w.deferred > 0 {
		d = time.Since(w.at)
	}
	q.waits = append(q.waits, d)
}

// leave is a waiter giving up: past its budget, or because its caller did.
func (q *Queue) leave(w *waiter, reason string, err error) (int, error) {
	q.mu.Lock()
	q.refusals[reason]++
	if w.state == resolved {
		// The admitting side won the race and this side is returning an error,
		// so nothing else will ever release the slot.
		slot := w.out.slot
		q.mu.Unlock()
		if slot >= 0 {
			_ = q.a.Finish(slot)
		}
		return -1, err
	}
	w.state = gone
	q.remove(w)
	q.mu.Unlock()
	return -1, err
}

// insert places a waiter by its ticket, so a readmitted sequence lands where it
// arrived rather than at the back.
func (q *Queue) insert(w *waiter) {
	i := sort.Search(len(q.waiters), func(i int) bool {
		return q.waiters[i].ticket > w.ticket
	})
	q.waiters = append(q.waiters, nil)
	copy(q.waiters[i+1:], q.waiters[i:])
	q.waiters[i] = w
}

func (q *Queue) remove(w *waiter) {
	for i, x := range q.waiters {
		if x == w {
			q.waiters = append(q.waiters[:i], q.waiters[i+1:]...)
			return
		}
	}
}

func (q *Queue) refuse(reason string) {
	q.mu.Lock()
	q.refusals[reason]++
	q.mu.Unlock()
}

func (q *Queue) nudge() {
	select {
	case q.wake <- struct{}{}:
	default:
	}
}

// deferrable says whether a refusal is one that waiting can fix.
func deferrable(err error) bool {
	return errors.Is(err, ErrNoSlot) || errors.Is(err, prefix.ErrExhausted)
}

// deferral names which of §3's two conditions refused this admission. They are
// counted apart because a server that reports one number for both is
// indistinguishable from a slow one (specs/008-scheduler.md §3).
func deferral(err error) string {
	if errors.Is(err, ErrNoSlot) {
		return "no_slot"
	}
	return "block_pool"
}
