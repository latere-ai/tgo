// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package tgo

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/latere-ai/tgo/internal/prefix"
)

// The queue's suite runs against a fake admitter, which is 021-D2's point:
// every ordering, bound and cancellation case is decided from integers, so none
// of it needs a device. The one row that needs a real Scheduler is the last.

// fakeAdmitter is a slot table and a block count, and nothing else.
type fakeAdmitter struct {
	mu      sync.Mutex
	block   int
	blocks  int
	reserve int
	live    []bool
	holds   []int
	used    int
	admits  int
	tries   int

	// gate, when non-nil, is received from at the top of Admit. Closing it
	// releases every attempt, which is how a test gets the whole waiter list in
	// place before the driver may admit anybody.
	gate chan struct{}

	// fail, when non-nil, is what Admit answers instead of taking a slot. It is
	// a refusal no amount of waiting fixes.
	fail error

	// before, when non-nil, runs after a successful Admit has taken the slot
	// and before it hands it back. It is the window §5's race lives in.
	before func()

	capacity chan struct{}
}

func newFake(slots, blocks, block, reserve int) *fakeAdmitter {
	return &fakeAdmitter{
		block: block, blocks: blocks, reserve: reserve,
		live: make([]bool, slots), holds: make([]int, slots),
		capacity: make(chan struct{}, 1),
	}
}

func (f *fakeAdmitter) Slots() int { return len(f.live) }

// need is the pool's arithmetic, with the request's own reserve where it named
// one and the fake's deployment default where it did not (022-D7).
func (f *fakeAdmitter) need(prompt, reserve int) int {
	if reserve <= 0 {
		reserve = f.reserve
	}
	return (prompt + reserve + f.block - 1) / f.block
}

func (f *fakeAdmitter) Feasible(prompt, reserve int) error {
	if prompt <= 0 {
		return errors.New("fake: the prompt is empty")
	}
	if n := f.need(prompt, reserve); n > f.blocks {
		return fmt.Errorf("fake: %d blocks for %d: %w", n, f.blocks, prefix.ErrExhausted)
	}
	return nil
}

func (f *fakeAdmitter) Admit(prompt []int, salt string, reserve int) (int, error) {
	if f.gate != nil {
		<-f.gate
	}
	f.mu.Lock()
	f.tries++
	f.mu.Unlock()
	if f.fail != nil {
		return -1, f.fail
	}
	slot, err := f.take(len(prompt), reserve)
	if err != nil {
		return -1, err
	}
	if f.before != nil {
		f.before()
	}
	return slot, nil
}

// take is Admit without the gate or the hook, so a test can set the fake up
// from a state the queue did not produce.
func (f *fakeAdmitter) take(prompt, reserve int) (int, error) {
	f.mu.Lock()
	slot := -1
	for i, l := range f.live {
		if !l {
			slot = i
			break
		}
	}
	if slot < 0 {
		f.mu.Unlock()
		return -1, ErrNoSlot
	}
	n := f.need(prompt, reserve)
	if f.used+n > f.blocks {
		f.mu.Unlock()
		return -1, fmt.Errorf("fake: %d of %d blocks are held: %w",
			f.used, f.blocks, prefix.ErrExhausted)
	}
	f.live[slot], f.holds[slot] = true, n
	f.used += n
	f.admits++
	f.mu.Unlock()
	return slot, nil
}

func (f *fakeAdmitter) Finish(slot int) error {
	f.mu.Lock()
	if slot < 0 || slot >= len(f.live) {
		f.mu.Unlock()
		return fmt.Errorf("fake: slot %d is outside %d", slot, len(f.live))
	}
	if !f.live[slot] {
		f.mu.Unlock()
		return fmt.Errorf("fake: slot %d is already free", slot)
	}
	f.live[slot] = false
	f.used -= f.holds[slot]
	f.holds[slot] = 0
	f.mu.Unlock()
	f.signal()
	return nil
}

func (f *fakeAdmitter) Capacity() <-chan struct{} { return f.capacity }

func (f *fakeAdmitter) signal() {
	select {
	case f.capacity <- struct{}{}:
	default:
	}
}

// occupy takes a slot outside the queue, so a test can start from a busy
// admitter without a queue having put it there.
func (f *fakeAdmitter) occupy(t *testing.T, prompt int) int {
	t.Helper()
	slot, err := f.take(prompt, 0)
	if err != nil {
		t.Fatalf("occupying with a %d-token prompt: %v", prompt, err)
	}
	return slot
}

func (f *fakeAdmitter) freeSlots() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, l := range f.live {
		if !l {
			n++
		}
	}
	return n
}

// tried is how many times the driver has asked the admitter, which is what a
// test waits on when it needs the waiter to have been *refused* rather than
// merely enqueued. Depth reaching one says the request arrived; it does not say
// the driver has looked at it yet.
func (f *fakeAdmitter) tried() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.tries
}

func (f *fakeAdmitter) inUse() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.used
}

func newQueue(t *testing.T, a Admitter, o QueueOptions) *Queue {
	t.Helper()
	q, err := NewQueue(a, o)
	if err != nil {
		t.Fatalf("NewQueue(%+v): %v", o, err)
	}
	t.Cleanup(func() {
		if err := q.Close(); err != nil {
			t.Errorf("Queue.Close: %v", err)
		}
	})
	return q
}

// waitFor polls a predicate rather than sleeping a guess, so a slow machine
// takes longer and a fast one does not race.
//
// The deadline is generous because it is a deadline and not a measurement:
// these cases are goroutine handoffs that take microseconds, and the budget has
// to survive a -race run beside the model-backed tests in this package, where a
// runnable goroutine can wait a long time for a core.
func waitFor(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(handoff)
	for time.Now().Before(deadline) {
		if ok() {
			return
		}
		time.Sleep(200 * time.Microsecond)
	}
	t.Fatalf("waited %s for %s and it did not happen", handoff, what)
}

// handoff is how long a queue test waits for something that takes
// microseconds. See [waitFor].
const handoff = 30 * time.Second

// admission is one background Admit, so a test can hold several at once.
type admission struct {
	slot chan int
	err  chan error
}

func admit(q *Queue, ctx context.Context, t Ticket, prompt int) *admission {
	a := &admission{slot: make(chan int, 1), err: make(chan error, 1)}
	go func() {
		s, err := q.Admit(ctx, t, make([]int, prompt), "", 0)
		a.slot <- s
		a.err <- err
	}()
	return a
}

func (a *admission) got(t *testing.T, what string) int {
	t.Helper()
	select {
	case err := <-a.err:
		if err != nil {
			t.Fatalf("%s: Admit: %v", what, err)
		}
		return <-a.slot
	case <-time.After(handoff):
		t.Fatalf("%s: Admit did not return", what)
		return -1
	}
}

func (a *admission) refused(t *testing.T, what string) error {
	t.Helper()
	select {
	case err := <-a.err:
		<-a.slot
		if err == nil {
			t.Fatalf("%s: Admit succeeded and the test needs it to be refused", what)
		}
		return err
	case <-time.After(handoff):
		t.Fatalf("%s: Admit did not return", what)
		return nil
	}
}

func (a *admission) pending() bool {
	select {
	case err := <-a.err:
		a.err <- err
		return false
	default:
		return true
	}
}

// TestQueueAdmitsImmediatelyWhenASlotIsFree is §9's first row, and the
// interesting half is the observation: an admission that never waited records a
// zero rather than nothing, so the histogram's count is admissions.
func TestQueueAdmitsImmediatelyWhenASlotIsFree(t *testing.T) {
	t.Parallel()
	f := newFake(2, 16, 4, 0)
	q := newQueue(t, f, QueueOptions{})

	slot, err := q.Admit(context.Background(), q.NewTicket(), make([]int, 4), "", 0)
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	if slot != 0 {
		t.Errorf("Admit took slot %d, want the first free one, 0", slot)
	}
	s := q.Stats()
	if len(s.Waits) != 1 || s.Waits[0] != 0 {
		t.Errorf("Waits = %v, want exactly one observation of 0", s.Waits)
	}
	if len(s.Deferred) != 0 {
		t.Errorf("Deferred = %v, want nothing deferred", s.Deferred)
	}
	if s.Depth != 0 {
		t.Errorf("Depth = %d after the admission, want 0", s.Depth)
	}
}

// TestQueueWaitsRatherThanRefusing is the whole point of the spec: what
// Scheduler.Admit answers with ErrNoSlot, the queue answers by waiting.
func TestQueueWaitsRatherThanRefusing(t *testing.T) {
	t.Parallel()
	f := newFake(1, 16, 4, 0)
	q := newQueue(t, f, QueueOptions{})
	held := f.occupy(t, 4)

	a := admit(q, context.Background(), q.NewTicket(), 4)
	waitFor(t, "the request to be refused once", func() bool { return f.tried() > 0 })
	if !a.pending() {
		t.Fatal("Admit returned while every slot was live; it must wait instead")
	}
	if err := f.Finish(held); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if slot := a.got(t, "the waiter"); slot != held {
		t.Errorf("the waiter took slot %d, want the freed %d", slot, held)
	}
	s := q.Stats()
	if s.Deferred["no_slot"] == 0 {
		t.Errorf("Deferred = %v, want at least one no_slot", s.Deferred)
	}
	if len(s.Waits) != 1 || s.Waits[0] == 0 {
		t.Errorf("Waits = %v, want one non-zero observation", s.Waits)
	}
}

// TestQueueIsFIFO is arrival order under capacity arriving one slot at a time.
func TestQueueIsFIFO(t *testing.T) {
	t.Parallel()
	f := newFake(1, 64, 4, 0)
	q := newQueue(t, f, QueueOptions{})
	held := f.occupy(t, 4)

	// Tickets are taken in order here rather than inside the goroutines,
	// because what is under test is arrival order and a goroutine's start is
	// not one.
	as := make([]*admission, 3)
	for i := range as {
		as[i] = admit(q, context.Background(), q.NewTicket(), 4)
		waitFor(t, fmt.Sprintf("waiter %d to be queued", i), func() bool {
			return q.Depth() == i+1
		})
	}
	prev := held
	for i, a := range as {
		if err := f.Finish(prev); err != nil {
			t.Fatalf("Finish before waiter %d: %v", i, err)
		}
		prev = a.got(t, fmt.Sprintf("waiter %d", i))
		for j := i + 1; j < len(as); j++ {
			if !as[j].pending() {
				t.Fatalf("waiter %d was admitted before waiter %d", j, i)
			}
		}
	}
}

// TestQueueOvertakesTheHeadAtMostKTimes is 021-D4, and it needs both halves:
// the overtake happens, and then it stops while a slot is still free -- so the
// bound is the rule rather than the capacity.
func TestQueueOvertakesTheHeadAtMostKTimes(t *testing.T) {
	t.Parallel()
	k := 2
	f := newFake(4, 8, 1, 0)
	f.gate = make(chan struct{})
	q := newQueue(t, f, QueueOptions{Overtake: &k})

	// Five of eight blocks held, and one of four slots. A six-block head cannot
	// fit; a one-block follower can.
	held := f.occupy(t, 5)

	// One at a time, and the head first. The driver cannot apply an ordering
	// rule to a request that has not arrived, so a test that let the followers
	// race the head would be asserting the goroutine scheduler.
	head := admit(q, context.Background(), q.NewTicket(), 6)
	waitFor(t, "the head to be queued", func() bool { return q.Depth() == 1 })
	follow := make([]*admission, 3)
	for i := range follow {
		follow[i] = admit(q, context.Background(), q.NewTicket(), 1)
		waitFor(t, fmt.Sprintf("follower %d to be queued", i), func() bool {
			return q.Depth() == i+2
		})
	}
	close(f.gate)

	for i := 0; i < k; i++ {
		follow[i].got(t, fmt.Sprintf("follower %d", i))
	}
	waitFor(t, "the queue to settle at the head and the last follower",
		func() bool { return q.Depth() == 2 })
	if !head.pending() {
		t.Fatal("the head was admitted; the fixture needs it not to fit")
	}
	if !follow[k].pending() {
		t.Fatalf("follower %d overtook the head a %d(st/nd/rd) time past K=%d",
			k, k+1, k)
	}
	// The rule and not the capacity: a slot is free and blocks are free, and
	// the follower is still not admitted.
	if got := f.freeSlots(); got != 1 {
		t.Fatalf("%d slots are free, want 1: the fixture must leave room so the "+
			"refusal is the reserving head rather than a full batch", got)
	}
	if got, want := f.inUse(), 7; got != want {
		t.Fatalf("%d of 8 blocks held, want %d, so a one-block follower would fit",
			got, want)
	}

	// The head is reserving, so it is admitted the moment it fits.
	if err := f.Finish(held); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	head.got(t, "the head")
	if !follow[k].pending() {
		t.Error("the last follower was admitted before the reserving head")
	}
}

// TestQueueStrictFIFOAdmitsNobodyBeforeTheHead is K=0, which the pointer in
// QueueOptions exists to express: zero is a policy, not an unset field.
func TestQueueStrictFIFOAdmitsNobodyBeforeTheHead(t *testing.T) {
	t.Parallel()
	k := 0
	f := newFake(4, 8, 1, 0)
	f.gate = make(chan struct{})
	q := newQueue(t, f, QueueOptions{Overtake: &k})
	f.occupy(t, 5)

	head := admit(q, context.Background(), q.NewTicket(), 6)
	waitFor(t, "the head to be queued", func() bool { return q.Depth() == 1 })
	small := admit(q, context.Background(), q.NewTicket(), 1)
	waitFor(t, "the follower to be queued", func() bool { return q.Depth() == 2 })
	close(f.gate)

	// Nothing more will happen, so wait on the fake rather than on a clock: one
	// deferral of the head is the whole pass.
	waitFor(t, "the head to be tried", func() bool {
		return q.Stats().Deferred["block_pool"] > 0
	})
	if !head.pending() || !small.pending() {
		t.Fatalf("head pending = %v, follower pending = %v; strict FIFO admits "+
			"neither while the head does not fit", head.pending(), small.pending())
	}
}

// TestQueueRefusesAnEmptyPrompt is a negative: the refusal happens at the door
// and the queue never holds the request.
func TestQueueRefusesAnEmptyPrompt(t *testing.T) {
	t.Parallel()
	f := newFake(2, 16, 4, 0)
	q := newQueue(t, f, QueueOptions{})

	if _, err := q.Admit(context.Background(), q.NewTicket(), nil, "", 0); err == nil {
		t.Fatal("an empty prompt was admitted; there is nothing to condition on")
	}
	if d := q.Depth(); d != 0 {
		t.Errorf("Depth = %d, want 0: the refusal is before the enqueue", d)
	}
	if got := q.Stats().Refused["infeasible"]; got != 1 {
		t.Errorf("Refused[infeasible] = %d, want 1", got)
	}
}

// TestQueueRefusesAnInfeasiblePromptAtTheDoor is 021-D3. A prompt that needs
// more blocks than the pool holds never resolves, so queueing it would make the
// head-of-line bound infinite.
func TestQueueRefusesAnInfeasiblePromptAtTheDoor(t *testing.T) {
	t.Parallel()
	f := newFake(2, 4, 4, 0)
	q := newQueue(t, f, QueueOptions{})

	// 20 tokens is five blocks and the pool holds four.
	_, err := q.Admit(context.Background(), q.NewTicket(), make([]int, 20), "", 0)
	if !errors.Is(err, prefix.ErrExhausted) {
		t.Fatalf("Admit refused with %v, want a wrapped prefix.ErrExhausted", err)
	}
	if d := q.Depth(); d != 0 {
		t.Errorf("Depth = %d, want 0: an infeasible request is never enqueued", d)
	}
	// The same length, admitted once the reserve is out of the way, proves the
	// refusal was the arithmetic and not the length.
	if _, err := q.Admit(context.Background(), q.NewTicket(), make([]int, 16), "", 0); err != nil {
		t.Fatalf("a 16-token prompt fills the pool exactly and was refused: %v", err)
	}
}

// TestQueueFullIsRefusedWithRetryAfter is the depth bound, and the Retry-After
// a caller derives from the budget rather than from a guessed service time.
func TestQueueFullIsRefusedWithRetryAfter(t *testing.T) {
	t.Parallel()
	f := newFake(1, 64, 4, 0)
	q := newQueue(t, f, QueueOptions{Depth: 2, Wait: 90 * time.Second})
	f.occupy(t, 4)

	for i := range 2 {
		admit(q, context.Background(), q.NewTicket(), 4)
		waitFor(t, fmt.Sprintf("waiter %d to be queued", i), func() bool {
			return q.Depth() == i+1
		})
	}
	_, err := q.Admit(context.Background(), q.NewTicket(), make([]int, 4), "", 0)
	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf("the third request past a depth of 2 got %v, want ErrQueueFull", err)
	}
	if got := q.Stats().Refused["queue_full"]; got != 1 {
		t.Errorf("Refused[queue_full] = %d, want 1", got)
	}
	if got, want := q.Wait(), 90*time.Second; got != want {
		t.Errorf("Wait = %s, want %s: Retry-After is derived from the budget and "+
			"from nothing else (021-D6)", got, want)
	}
	if got, want := q.MaxDepth(), 2; got != want {
		t.Errorf("MaxDepth = %d, want %d", got, want)
	}
}

// TestQueueDepthDefaultsToEightPerSlot is §4's D = 8N, stated as a multiple so
// it scales with a deployment that raises its slot count.
func TestQueueDepthDefaultsToEightPerSlot(t *testing.T) {
	t.Parallel()
	for _, slots := range []int{1, 4, 16} {
		q := newQueue(t, newFake(slots, 64, 4, 0), QueueOptions{})
		if got, want := q.MaxDepth(), 8*slots; got != want {
			t.Errorf("%d slots: MaxDepth = %d, want %d", slots, got, want)
		}
	}
}

// TestQueueTimeoutIsNotClientGone is 021-D7: the budget and the caller's
// context answer two different statuses, so they must not be one error.
func TestQueueTimeoutIsNotClientGone(t *testing.T) {
	t.Parallel()
	f := newFake(1, 64, 4, 0)
	q := newQueue(t, f, QueueOptions{Wait: 20 * time.Millisecond})
	f.occupy(t, 4)

	ctx := t.Context()
	err := admit(q, ctx, q.NewTicket(), 4).refused(t, "the waiter")
	if !errors.Is(err, ErrQueueTimeout) {
		t.Fatalf("past the budget the waiter got %v, want ErrQueueTimeout", err)
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("the budget reported a context error (%v); 429 and 499 would "+
			"become one status", err)
	}
	if ctx.Err() != nil {
		t.Errorf("the caller's context was disturbed: %v", ctx.Err())
	}
	s := q.Stats()
	if s.Refused["queue_timeout"] != 1 || s.Refused["client_gone"] != 0 {
		t.Errorf("Refused = %v, want one queue_timeout and no client_gone", s.Refused)
	}
	if s.Depth != 0 {
		t.Errorf("Depth = %d after the refusal, want 0", s.Depth)
	}
}

// TestQueueCancelledWaiterLeaves is the second transition of §5's state
// machine: the entry is unlinked, the depth decremented, and no slot taken.
func TestQueueCancelledWaiterLeaves(t *testing.T) {
	t.Parallel()
	f := newFake(1, 64, 4, 0)
	q := newQueue(t, f, QueueOptions{})
	f.occupy(t, 4)

	ctx, cancel := context.WithCancel(context.Background())
	a := admit(q, ctx, q.NewTicket(), 4)
	waitFor(t, "the waiter to be queued", func() bool { return q.Depth() == 1 })
	cancel()

	err := a.refused(t, "the cancelled waiter")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("the cancelled waiter got %v, want context.Canceled", err)
	}
	if d := q.Depth(); d != 0 {
		t.Errorf("Depth = %d, want 0: the entry is unlinked when its caller goes", d)
	}
	if got := f.freeSlots(); got != 0 {
		t.Errorf("%d slots free, want 0: a departing waiter takes no slot", got)
	}
	if got := q.Stats().Refused["client_gone"]; got != 1 {
		t.Errorf("Refused[client_gone] = %d, want 1", got)
	}
}

// TestQueueCancelAfterAdmitReleasesTheSlot is 021-D10, and it is the failure
// that has no other owner: the caller is returning an error and holds no slot
// index, so nothing else would ever call Finish and the slot -- with its blocks
// -- would be out of the batch for the life of the process.
//
// The race is made deterministic rather than hoped for: the fake takes the slot
// and then holds, the test cancels and waits until the waiter has left, and
// only then does the admission return.
func TestQueueCancelAfterAdmitReleasesTheSlot(t *testing.T) {
	t.Parallel()
	f := newFake(1, 64, 4, 0)
	q := newQueue(t, f, QueueOptions{})

	ctx, cancel := context.WithCancel(context.Background())
	var once sync.Once
	f.before = func() {
		once.Do(func() {
			cancel()
			waitFor(t, "the cancelled waiter to leave the queue", func() bool {
				return q.Depth() == 0
			})
		})
	}
	a := admit(q, ctx, q.NewTicket(), 4)
	if err := a.refused(t, "the cancelled waiter"); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	waitFor(t, "the slot the queue won for a caller who had gone", func() bool {
		return f.freeSlots() == 1 && f.inUse() == 0
	})
	// The slot really is usable again, which is the assertion a free-count
	// alone would not make.
	if slot := admit(q, context.Background(), q.NewTicket(), 4).got(t, "the next request"); slot != 0 {
		t.Errorf("the next request took slot %d, want the released 0", slot)
	}
}

// TestQueueClosedWithWaiters: closing refuses the waiters rather than leaving
// them to their budgets, because a request that never answers is worse than one
// refused.
func TestQueueClosedWithWaiters(t *testing.T) {
	t.Parallel()
	f := newFake(1, 64, 4, 0)
	q, err := NewQueue(f, QueueOptions{Wait: time.Hour})
	if err != nil {
		t.Fatalf("NewQueue: %v", err)
	}
	f.occupy(t, 4)

	as := make([]*admission, 3)
	for i := range as {
		as[i] = admit(q, context.Background(), q.NewTicket(), 4)
		waitFor(t, "the waiters to queue", func() bool { return q.Depth() == i+1 })
	}
	if err := q.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	for i, a := range as {
		if err := a.refused(t, fmt.Sprintf("waiter %d", i)); !errors.Is(err, ErrQueueClosed) {
			t.Errorf("waiter %d got %v, want ErrQueueClosed", i, err)
		}
	}
	if _, err := q.Admit(context.Background(), q.NewTicket(), make([]int, 4), "", 0); !errors.Is(err, ErrQueueClosed) {
		t.Errorf("admitting into a closed queue got %v, want ErrQueueClosed", err)
	}
	if err := q.Close(); err != nil {
		t.Errorf("the second Close: %v", err)
	}
}

// TestQueueEvictedSequenceKeepsItsArrivalStamp is 021-D5's other half: a
// sequence the scheduler evicted re-enters where it arrived, so an eviction
// cannot push a request behind the waiters that arrived after it.
func TestQueueEvictedSequenceKeepsItsArrivalStamp(t *testing.T) {
	t.Parallel()
	f := newFake(1, 64, 4, 0)
	q := newQueue(t, f, QueueOptions{})

	evicted := q.NewTicket()
	slot, err := q.Admit(context.Background(), evicted, make([]int, 4), "", 0)
	if err != nil {
		t.Fatalf("the first admission: %v", err)
	}
	next := admit(q, context.Background(), q.NewTicket(), 4)
	last := admit(q, context.Background(), q.NewTicket(), 4)
	waitFor(t, "both later requests to queue", func() bool { return q.Depth() == 2 })

	// The scheduler evicts, and its caller resubmits with the ticket it holds.
	if err := f.Finish(slot); err != nil {
		t.Fatalf("evicting: %v", err)
	}
	held := next.got(t, "the request that arrived next")
	back := admit(q, context.Background(), evicted, 4)
	waitFor(t, "the resubmission to queue", func() bool { return q.Depth() == 2 })

	if err := f.Finish(held); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	back.got(t, "the resubmitted request")
	if !last.pending() {
		t.Error("the request that arrived last was admitted before the resubmitted " +
			"one, so an eviction moved a request to the back of the queue")
	}
}

// TestQueueStatsCountBothDeferralReasons is 008 §3's requirement: a rejection
// for "no slot" and one for "the pool cannot hold this" are different states of
// a deployment, and one number for both is indistinguishable from slowness.
func TestQueueStatsCountBothDeferralReasons(t *testing.T) {
	t.Parallel()
	f := newFake(2, 4, 1, 0)
	q := newQueue(t, f, QueueOptions{})
	a := f.occupy(t, 1)
	b := f.occupy(t, 1)

	ctx, cancel := context.WithCancel(context.Background())
	first := admit(q, ctx, q.NewTicket(), 1)
	waitFor(t, "the no-slot deferral", func() bool {
		return q.Stats().Deferred["no_slot"] > 0
	})
	cancel()
	first.refused(t, "the first waiter")

	// A slot is free now and the blocks are not: one of four is held and the
	// waiter needs all four.
	if err := f.Finish(a); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	_ = b
	ctx2, cancel2 := context.WithCancel(context.Background())
	second := admit(q, ctx2, q.NewTicket(), 4)
	waitFor(t, "the block-pool deferral", func() bool {
		return q.Stats().Deferred["block_pool"] > 0
	})
	if got := f.freeSlots(); got != 1 {
		t.Errorf("%d slots free, want 1: the second deferral must be the pool and "+
			"not the slot table", got)
	}
	cancel2()
	second.refused(t, "the second waiter")
}

// TestQueueUnderRace is the invariant the rest of the suite cannot state: under
// -race, with every ordering and cancellation happening at once, no slot is
// admitted twice and none is leaked.
func TestQueueUnderRace(t *testing.T) {
	t.Parallel()
	const slots, callers = 4, 64
	f := newFake(slots, 4*slots, 4, 0)
	q := newQueue(t, f, QueueOptions{Wait: handoff})

	var wg sync.WaitGroup
	for i := range callers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			// A third of the callers hang up, and they do it at whatever moment
			// the scheduler of the day picks -- which is the point.
			if i%3 == 0 {
				go func() {
					time.Sleep(time.Duration(i%7) * 50 * time.Microsecond)
					cancel()
				}()
			}
			slot, err := q.Admit(ctx, q.NewTicket(), make([]int, 4), "", 0)
			if err != nil {
				return
			}
			if err := f.Finish(slot); err != nil {
				t.Errorf("caller %d finishing slot %d: %v", i, slot, err)
			}
		}(i)
	}
	wg.Wait()
	if got := f.freeSlots(); got != slots {
		t.Errorf("%d of %d slots are free at the end; the rest are leaked", got, slots)
	}
	if got := f.inUse(); got != 0 {
		t.Errorf("%d blocks are still held at the end", got)
	}
	if d := q.Depth(); d != 0 {
		t.Errorf("Depth = %d at the end, want 0", d)
	}
}

// TestNewQueueRefusesWhatItCannotHonour is the constructor's negatives: an
// option that cannot be met is refused at build time rather than surprising a
// request.
func TestNewQueueRefusesWhatItCannotHonour(t *testing.T) {
	t.Parallel()
	negative := -1
	for _, c := range []struct {
		name string
		a    Admitter
		o    QueueOptions
	}{
		{"no admitter", nil, QueueOptions{}},
		{"no slots", newFake(0, 8, 4, 0), QueueOptions{}},
		{"a negative depth", newFake(2, 8, 4, 0), QueueOptions{Depth: -1}},
		{"a negative wait", newFake(2, 8, 4, 0), QueueOptions{Wait: -time.Second}},
		{"a negative overtake", newFake(2, 8, 4, 0), QueueOptions{Overtake: &negative}},
	} {
		q, err := NewQueue(c.a, c.o)
		if err == nil {
			_ = q.Close()
			t.Errorf("NewQueue with %s was accepted", c.name)
		}
	}
}

// TestSchedulerCapacityFiresOnFinish is the one row that needs a real
// scheduler: the interface the queue waits on is the one the scheduler
// implements, and a full channel must not block the release that filled it.
func TestSchedulerCapacityFiresOnFinish(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	s := newScheduler(t, m, 2, SchedulerOptions{Chunk: 8, Reserve: CacheBlock})

	drain := func() {
		select {
		case <-s.Capacity():
		default:
		}
	}
	for _, c := range []struct {
		name string
		act  func(t *testing.T)
	}{
		{"Finish", func(t *testing.T) {
			slot, err := s.Admit(promptIDs(1, 6), "", 0)
			if err != nil {
				t.Fatalf("Admit: %v", err)
			}
			drain()
			if err := s.Finish(slot); err != nil {
				t.Fatalf("Finish: %v", err)
			}
		}},
		{"Evict", func(t *testing.T) {
			if _, err := s.Admit(promptIDs(2, 6), "", 0); err != nil {
				t.Fatalf("Admit: %v", err)
			}
			drain()
			if _, err := s.Evict(); err != nil {
				t.Fatalf("Evict: %v", err)
			}
		}},
		{"Step", func(t *testing.T) {
			slot, err := s.Admit(promptIDs(3, 6), "", 0)
			if err != nil {
				t.Fatalf("Admit: %v", err)
			}
			defer func() {
				if err := s.Finish(slot); err != nil {
					t.Fatalf("Finish: %v", err)
				}
			}()
			drain()
			if _, err := s.Step(); err != nil {
				t.Fatalf("Step: %v", err)
			}
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			c.act(t)
			select {
			case <-s.Capacity():
			case <-time.After(time.Second):
				t.Fatalf("%s did not make Capacity readable", c.name)
			}
		})
	}

	// A full channel does not block the release that would have filled it,
	// which is what lets Finish be called from inside a driver's own loop.
	slot, err := s.Admit(promptIDs(4, 6), "", 0)
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	for i := range 8 {
		if _, err := s.Step(); err != nil {
			t.Fatalf("Step %d: %v", i, err)
		}
	}
	if err := s.Finish(slot); err != nil {
		t.Fatalf("Finish with a full capacity channel: %v", err)
	}
}

// TestSchedulerFeasibleIsThePoolsArithmetic is 021-D3 against the real pool:
// the door's answer and Admit's answer are the same arithmetic, so they cannot
// drift.
func TestSchedulerFeasibleIsThePoolsArithmetic(t *testing.T) {
	t.Parallel()
	m := batchModel(t)
	s := newScheduler(t, m, 2, SchedulerOptions{Chunk: 8, Reserve: CacheBlock})
	blocks := m.blocks.pool.Blocks()

	if err := s.Feasible(0, 0); err == nil {
		t.Error("an empty prompt was called feasible")
	}
	// One block short of the pool, minus the reserve's block, fits.
	if err := s.Feasible((blocks-1)*CacheBlock, 0); err != nil {
		t.Errorf("a prompt that fits with its reserve was refused: %v", err)
	}
	// The whole pool plus a reserve does not, and Admit agrees.
	tooBig := blocks * CacheBlock
	if err := s.Feasible(tooBig, 0); !errors.Is(err, prefix.ErrExhausted) {
		t.Errorf("Feasible(%d) = %v, want a wrapped prefix.ErrExhausted", tooBig, err)
	}
	if _, err := s.Admit(promptIDs(1, tooBig), "", 0); !errors.Is(err, prefix.ErrExhausted) {
		t.Errorf("Admit of the same prompt = %v, want the same refusal", err)
	}
}

// TestQueueLeavingAWaiterTheDriverAlreadyWonReleasesItsSlot is 021-D10's other
// side, and it is written against the queue's own state rather than against a
// race, because the window is between a waiter's select returning and its next
// lock acquisition -- narrow enough that a test racing for it would pass by not
// reaching it.
//
// Both winners of §5's single-winner state machine must release: the driver
// releases a slot won for a waiter that has already left, and a waiter that
// leaves after the driver won releases the slot it is refusing to return.
func TestQueueLeavingAWaiterTheDriverAlreadyWonReleasesItsSlot(t *testing.T) {
	t.Parallel()
	f := newFake(1, 64, 4, 0)
	q := newQueue(t, f, QueueOptions{})

	// A slot the driver won, for a waiter whose caller is on its way out.
	slot := f.occupy(t, 4)
	w := &waiter{res: make(chan outcome, 1), state: resolved, out: outcome{slot: slot}}

	got, err := q.leave(w, "client_gone", context.Canceled)
	if got != -1 || !errors.Is(err, context.Canceled) {
		t.Fatalf("leave = (%d, %v), want (-1, context.Canceled)", got, err)
	}
	if free := f.freeSlots(); free != 1 {
		t.Errorf("%d slots free, want 1: the caller is returning an error and holds "+
			"no slot index, so nothing else would ever release it", free)
	}
	if got := q.Stats().Refused["client_gone"]; got != 1 {
		t.Errorf("Refused[client_gone] = %d, want 1", got)
	}
}

// TestQueueReturnsARefusalWaitingCannotFix is the third thing Admit can answer.
// ErrNoSlot and an exhausted pool are waits; anything else -- a closed batch, a
// prompt the batch itself refuses -- is the request's answer, and a queue that
// retried it would hold the caller until its budget elapsed and then report a
// timeout for a request that was never going to be admitted.
func TestQueueReturnsARefusalWaitingCannotFix(t *testing.T) {
	t.Parallel()
	f := newFake(2, 64, 4, 0)
	f.fail = errors.New("fake: the batch is closed")
	q := newQueue(t, f, QueueOptions{Wait: time.Hour})

	_, err := q.Admit(context.Background(), q.NewTicket(), make([]int, 4), "", 0)
	if err == nil || !strings.Contains(err.Error(), "the batch is closed") {
		t.Fatalf("Admit = %v, want the admitter's own refusal", err)
	}
	if d := q.Depth(); d != 0 {
		t.Errorf("Depth = %d, want 0: the waiter left with the error", d)
	}
	s := q.Stats()
	if len(s.Waits) != 0 {
		t.Errorf("Waits = %v, want nothing: a refusal is not an admission", s.Waits)
	}
	if len(s.Deferred) != 0 {
		t.Errorf("Deferred = %v, want nothing: this refusal is not a deferral", s.Deferred)
	}
}
