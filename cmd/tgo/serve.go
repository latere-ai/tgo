// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/server"
)

// shutdownGrace is how long a graceful stop waits for in-flight requests.
//
// It is the same 30 seconds as [server.DefaultQueueWait], for one reason: a
// request that was admitted after waiting the full queue wait is the last one
// still generating when the signal arrives, and a grace shorter than the wait
// would cut off exactly the request that waited longest. A stream that is still
// open when the grace runs out is closed, and the operator is told so rather
// than left with a process that will not exit.
const shutdownGrace = 30 * time.Second

// serveRoutes is what `tgo serve` prints, one line per route the handler
// answers on.
//
// It is written here rather than asked of the server because
// [server.Server] serves a fixed [http.ServeMux] and exports no way to
// enumerate it. TestServeRoutesAreTheRoutesTheServerAnswers probes every line
// below against a real server, so a route that is renamed upstream fails a test
// here rather than being printed at an operator who then cannot reach it.
var serveRoutes = []struct{ Method, Path, What string }{
	{"POST", "/v1/chat/completions", "OpenAI Chat Completions"},
	{"POST", "/v1/messages", "Anthropic Messages"},
	{"POST", "/v1/responses", "OpenAI Responses"},
	{"POST", "/v1/completions", "OpenAI legacy completions: raw text, no chat template"},
	{"GET", "/v1/models", "the one model id this process serves"},
	{"GET", "/health", "liveness"},
	{"GET", "/metrics", "Prometheus text exposition"},
}

// defaultSessions is the pool `tgo serve` builds when nothing says otherwise.
//
// It is two numbers at once (specs/019-session-affinity.md §4): how many
// requests may generate at the same time, and how many conversations keep their
// key/value cache between turns. The second is what decides a hit -- a turn
// reuses its own prefix when fewer than N other conversations were served since
// its last one (§3.1) -- so a pool of one serves one conversation well and
// alternating conversations not at all.
//
// Four rather than the device's whole capacity, because every one of them is
// allocated at startup and held for the process's life. An operator who wants
// more asks for it with --sessions and reads what it costs in the report; one
// whose device cannot hold four gets what it can hold, and never less than one.
const defaultSessions = 4

// serveOptions is `tgo serve`'s command line, parsed.
type serveOptions struct {
	Dir    string
	Addr   string
	Name   string
	Public bool

	// Slots is how many requests generate at once: the pool's size under a
	// pooled engine and the batch width under a batched one. Zero takes the
	// default for whichever it is.
	Slots int

	// SlotsFlag is the name the operator used for Slots, so a refusal names the
	// flag they typed.
	SlotsFlag string

	// Notes are lines the report prints about the command line itself, such as
	// a flag that has been renamed.
	Notes []string

	Engine engineOptions
}

// serveFlagSet declares what `tgo serve` accepts. See [runFlagSet] for why
// declaring is separate from parsing.
func serveFlagSet() (*flag.FlagSet, *serveFlags) {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	return fs, &serveFlags{
		addr:      fs.String("addr", server.DefaultAddr, "host:port to listen on"),
		public:    fs.Bool("public", false, "allow a bind that is not loopback, which has no authentication"),
		precision: fs.String("precision", "auto", "f16, int8, int4 or auto"),
		context:   fs.Int("context", defaultContext, "KV cache capacity per session, in positions"),
		device:    fs.String("device", "auto", "auto, cpu or metal"),
		sessions:  fs.Int("sessions", 0, "deprecated alias for --slots"),
		slots: fs.Int("slots", 0,
			"how many requests generate at once (0 takes 4 pooled, or 8 batched, "+
				"or fewer if the device holds fewer)"),
		kv: fs.Int("kv", 0,
			"shared block pool, in positions; needs --prefix-cache process "+
				"(0 takes slots x context)"),
		prefixCache: registerScope(fs),
		batched: fs.Bool("batched", false,
			"put every in-flight request in one forward pass; implies "+
				"--prefix-cache process"),
	}
}

// registerScope declares --prefix-cache on fs and returns the value it writes.
func registerScope(fs *flag.FlagSet) *scopeFlag {
	v := &scopeFlag{}
	fs.Var(v, "prefix-cache",
		"off, session or process: reuse the key/value state a conversation already "+
			"paid for, which changes what an answer costs and, slightly, what it says "+
			"(bare --prefix-cache is session)")
	return v
}

// serveFlags holds `tgo serve`'s flag values.
type serveFlags struct {
	addr, precision, device *string
	public                  *bool
	prefixCache             *scopeFlag
	batched                 *bool
	context, sessions       *int
	slots, kv               *int
}

// parseServe parses and checks `tgo serve`'s arguments.
func parseServe(args []string) (serveOptions, error) {
	fs, f := serveFlagSet()
	dir, err := modelDir(fs, args)
	if err != nil {
		return serveOptions{}, err
	}
	policy, err := parsePrecision(*f.precision)
	if err != nil {
		return serveOptions{}, err
	}
	dev, err := parseDevice(*f.device)
	if err != nil {
		return serveOptions{}, err
	}
	if err := positive("context", *f.context); err != nil {
		return serveOptions{}, err
	}
	// --slots and --sessions are one number, and --sessions is the old name.
	//
	// Under a scheduler the flag stops describing a pool: what it says is how
	// many requests generate at once, which is the batch width, and a batched
	// slot costs a page table rather than a context of key/value cache
	// (specs/022-batched-serving.md §8). Renaming it and keeping the old name
	// working is what does not break every existing command line for a rename.
	set := map[string]bool{}
	fs.Visit(func(fl *flag.Flag) { set[fl.Name] = true })
	slots, slotsFlag, notes := *f.slots, "--slots", []string(nil)
	if set["sessions"] {
		if set["slots"] {
			return serveOptions{}, fmt.Errorf("%w: --slots and --sessions are one "+
				"number and both were given (%d and %d); --sessions is the old name",
				errUsage, *f.slots, *f.sessions)
		}
		slots, slotsFlag = *f.sessions, "--sessions"
		notes = append(notes, "--sessions is the old name for --slots and still works. "+
			"It named a pool, and\n                    under --batched what it names is "+
			"the batch width.")
	}
	// Zero is "as many as the device holds, up to the default", which is what
	// the admission arithmetic answers once the weights are on the device. A
	// negative one is a deployment that runs no request.
	if slots < 0 {
		return serveOptions{}, fmt.Errorf("%w: %s is %d; a deployment runs at "+
			"least one request at a time", errUsage, slotsFlag, slots)
	}
	if *f.kv < 0 {
		return serveOptions{}, fmt.Errorf("%w: --kv is %d; it is the shared block "+
			"pool's size in positions", errUsage, *f.kv)
	}
	// 022-D1: the batched path and the process scope are one configuration,
	// because NewBatch refuses a model with no shared block pool. So --batched
	// implies the scope rather than failing later with an error about a pool
	// the operator never mentioned -- and an operator who asked for a different
	// scope is told the two cannot both hold rather than having theirs
	// overwritten.
	scope := f.prefixCache.scope
	if *f.batched {
		if f.prefixCache.set && scope != tgo.CacheProcess {
			return serveOptions{}, fmt.Errorf("%w: --batched needs --prefix-cache "+
				"process and this is %s; sequences that step together have different "+
				"lengths, so a contiguous per-session cache would pad every one of "+
				"them to the longest (specs/022-batched-serving.md 022-D1)",
				errUsage, scope)
		}
		scope = tgo.CacheProcess
	}
	// --kv sizes the one thing a process-scoped cache has and the other two
	// scopes do not. Refused rather than ignored: a number an operator passed
	// and nothing read is the shape of a deployment that thinks it configured
	// something.
	if set["kv"] && scope != tgo.CacheProcess {
		return serveOptions{}, fmt.Errorf("%w: --kv sizes the shared block pool and "+
			"--prefix-cache is %s, which gives every session its own cache; --kv "+
			"needs --prefix-cache process or --batched", errUsage, scope)
	}
	addr := strings.TrimSpace(*f.addr)
	if addr == "" {
		addr = server.DefaultAddr
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return serveOptions{}, fmt.Errorf("%w: --addr %q is not a host:port address: %w", errUsage, addr, err)
	}
	if !*f.public && isPublicAddr(host) {
		return serveOptions{}, publicBindRefusal(addr)
	}
	return serveOptions{
		Dir: dir, Addr: addr, Name: modelID(dir), Public: *f.public, Slots: slots,
		SlotsFlag: slotsFlag, Notes: notes,
		Engine: engineOptions{Precision: policy, Context: *f.context, Device: dev,
			PrefixCache: scope, Slots: slots, KV: *f.kv, Batched: *f.batched},
	}, nil
}

// isPublicAddr reports whether an address would be reachable from off the
// machine, from its host alone.
//
// It restates [server.Server.Listen]'s rule, which is unexported and reachable
// only by binding -- and binding happens after the weights are on the device,
// which on a 1.4 GiB checkpoint is minutes after the user typed the flag they
// got wrong. The rule is therefore checked twice: here, so that a command line
// which cannot work is refused before anything is loaded, and in Listen, which
// is the enforcement and stays the authority. TestPublicAddressAgreesWithTheServer
// holds the two against each other over the same addresses, which is what keeps
// the restatement from drifting into a second rule.
//
// An empty host is the wildcard: ":8080" binds every interface and reads like
// it binds none. A name that is not an IP literal and is not localhost is
// treated as public, because guessing the other way would be a hostname away
// from an unauthenticated server on the network.
func isPublicAddr(host string) bool {
	switch {
	case host == "":
		return true
	case strings.EqualFold(host, "localhost"):
		return false
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	switch {
	case ip == nil, ip.IsUnspecified():
		return true
	}
	return !ip.IsLoopback()
}

// publicBindRefusal is 009-D8 stated at the command line: an unauthenticated
// server is not exposed by omission, and the flag that would allow it is named.
func publicBindRefusal(addr string) error {
	return fmt.Errorf("%w: --addr %s is reachable from the network and this server has no "+
		"authentication (specs/009-server.md 009-D8); pass --public to bind it anyway", errUsage, addr)
}

// modelID is the id a request must name, taken from the directory the model was
// loaded from.
//
// There is no flag for it, because this process serves one model (009-D5) and a
// name it invented would be a second thing to get right. The id is printed at
// startup and answered by GET /v1/models, so it is discoverable rather than
// guessed. Trailing separators are stripped first: `tgo serve ./models/qwen/`
// and `tgo serve ./models/qwen` must serve the same id.
func modelID(dir string) string {
	if name := filepath.Base(filepath.Clean(dir)); name != "." && name != string(filepath.Separator) {
		return name
	}
	return "model"
}

// servable is a loaded model as `tgo serve` needs it: a way to build the
// engine once its size is known, what the loader resolved, and the release.
//
// The engine is not built here because it cannot be. specs/019-session-affinity.md
// 019-D2 reserves N sessions' key/value cache at startup, and N comes from the
// admission arithmetic, which needs what the loader resolved -- auto precision
// can land on int8, and a budget computed from a footprint this process
// predicted would subtract weights that are not on the device.
type servable struct {
	// Pool builds the server's engine over a pool of the given size.
	Pool func(sessions int) (server.Engine, error)

	Info engineInfo

	// Close releases the pool and the model, in that order, which is the order
	// accel requires: it closes in order rather than recursively, so a buffer
	// outliving its device is a leak the runtime reports.
	Close func() error
}

// openServable loads a model and adapts it to the server's interface.
//
// It is a variable for the same reason [openEngine] is: everything this file
// does around the model -- the bind rule, the admission arithmetic, the report
// and the graceful stop -- is then reachable from a test with no device, no
// weights and no checkpoint.
var openServable = func(dir, name string, o engineOptions) (servable, error) {
	opts := []tgo.Option{
		tgo.WithPrecision(livePrecision(o.Precision)),
		tgo.WithContext(o.Context),
		tgo.WithDevice(o.Device),
	}
	switch o.PrefixCache {
	case tgo.CacheSession:
		// The budget is the whole context, because a pooled session's history
		// is its own and there is nothing else to spend it on.
		opts = append(opts, tgo.WithPrefixCache(tgo.CacheSession, o.Context))
	case tgo.CacheProcess:
		// One pool, and --kv is what sizes it. Its default is exactly what the
		// per-session caches would have cost -- slots x context positions, held
		// once instead of once each -- so a deployment that does not set it
		// holds the same bytes it held before, and they become fungible: a long
		// conversation and seven short ones fit where that many fixed caches
		// would have refused the first (specs/022-batched-serving.md §9).
		//
		// The slot count is the operator's, or the default, and is resolved
		// here rather than from the admission below because the pool is
		// allocated while the model loads and the admission needs the loaded
		// weights. Where the two disagree the admission is the one that
		// reports it, in --slots and --context terms.
		opts = append(opts, tgo.WithPrefixCache(tgo.CacheProcess, poolPositions(o)))
	}
	m, err := tgo.Open(dir, opts...)
	if err != nil {
		return servable{}, err
	}
	i := m.Info()
	var pool *server.PoolEngine
	var batch *server.RunnerEngine
	return servable{
		Pool: func(sessions int) (server.Engine, error) {
			if o.Batched {
				e, err := server.WrapRunner(m, name, tgo.RunnerOptions{
					Slots: sessions, Reserve: serveReserve(o.Context),
				})
				if err != nil {
					return nil, err
				}
				batch = e
				return e, nil
			}
			p, err := server.WrapPool(m, name, sessions)
			if err != nil {
				return nil, err
			}
			pool = p
			return p, nil
		},
		Info: engineInfo{
			Precision: i.Precision.String(), WeightBytes: i.WeightBytes,
			CacheBytesPerSession: i.CacheBytesPerSession, Context: i.Context,
		},
		Close: func() error {
			var errs []error
			if pool != nil {
				errs = append(errs, pool.Close())
			}
			if batch != nil {
				errs = append(errs, batch.Close())
			}
			return errors.Join(append(errs, m.Close())...)
		},
	}, nil
}

// poolPositions is how large the shared block pool is, in positions.
//
// --kv where the operator set it, and slots x context otherwise, which is the
// bytes a pool of that many sessions reserves today (§8).
func poolPositions(o engineOptions) int {
	if o.KV > 0 {
		return o.KV
	}
	slots := o.Slots
	if slots == 0 {
		slots = defaultSlots(o.Batched)
	}
	return slots * o.Context
}

// defaultSlots is how many requests generate at once when nothing says.
//
// Eight batched and four pooled, because a slot costs a different thing under
// each: a pooled session reserves a whole context of key/value cache (019-D2)
// and a batched slot reserves a page table and a row of the per-step ports,
// which are O(rows + B x V) rather than O(B x context) (specs/022-batched-serving.md §8).
// The number that is generous for one is wasteful for the other, and the
// startup report prints which one this deployment got.
func defaultSlots(batched bool) int {
	if batched {
		return defaultBatchSlots
	}
	return defaultSessions
}

// defaultBatchSlots is §8's default batch width.
const defaultBatchSlots = 8

// serveReserve is §3's R for a batched deployment: how many positions beyond
// its prompt an admitted sequence holds blocks for.
//
// [tgo.DefaultReserve], capped at half the context. The cap is what keeps the
// promise payable on a short context: a reserve larger than the context admits
// nobody, because ceil((T+R)/B) is then more blocks than one sequence's share
// of the pool. It is one number for the whole deployment, which
// [022-D7](../../specs/022-batched-serving.md) makes per request in a later
// pass.
func serveReserve(context int) int {
	r := tgo.DefaultReserve
	if half := context / 2; r > half {
		r = half
	}
	if r < tgo.CacheBlock {
		r = tgo.CacheBlock
	}
	return r
}

// admission is specs/009-server.md §6's limit with the terms that produced it.
//
// The three numbers are carried with the answer for the reason
// specs/001-weights.md §5 gives for the precision choice: a bare concurrency is
// a number an operator cannot check, and the one that surprises them -- a
// 16 GiB machine admitting one session -- is explained entirely by the two
// subtrahends.
type admission struct {
	Pool       int64
	Weights    int64
	Budget     int64
	PerSession int64

	// Fits is how many sessions' cache the budget holds, which is §6's N_max.
	Fits int

	// Sessions is how many the pool actually reserves, which is at most Fits.
	// The two differ because 019-D2 allocates every one of them at startup and
	// holds it for the process's life, so the device's capacity is a ceiling
	// rather than a target.
	//
	// Under a batched engine it is the batch width instead, and it is not
	// capped by Fits: a slot there reserves a page table rather than a context
	// of key/value cache, and what the budget bounds is Positions.
	Sessions int

	// Positions is the shared block pool's size, and is zero for a deployment
	// that reserves one cache per session.
	Positions int

	// Reserved is what the deployment actually holds for its life: N sessions'
	// caches, or one pool of Positions.
	Reserved int64
}

// admissionShape says what a deployment reserves, which is not the same
// quantity under the two engines (specs/022-batched-serving.md §8).
type admissionShape struct {
	// Positions is the shared pool's size. Zero is a deployment that reserves
	// one cache per session, which is what every scope but a batched one does.
	Positions int

	// Context is what one session's cache is priced at, so a pool of Positions
	// can be priced from the same number.
	Context int

	// Default is how many requests generate at once when the operator named no
	// number.
	Default int

	// Flag is the name the operator used for the slot count, so a refusal
	// names the flag they typed rather than the one it was renamed to.
	Flag string
}

// shapeOf is what this deployment reserves, in the terms [kvAdmission] prices.
func shapeOf(o serveOptions, context int) admissionShape {
	if !o.Engine.Batched {
		return admissionShape{Default: defaultSessions, Flag: o.SlotsFlag}
	}
	return admissionShape{
		Positions: poolPositions(o.Engine), Context: context,
		Default: defaultBatchSlots, Flag: o.SlotsFlag,
	}
}

// kvAdmission is §6's arithmetic:
//
//	N_max = floor((M_available - M_weights) / M_kv(C))
//
// M_available is the device's MaxPoolBytes, which is the same quantity
// `tgo info` prints as the budget and is a cap on one allocation rather than a
// report of free memory -- accel exposes no such report. It therefore
// overstates what is free on a machine running anything else, which is why the
// derivation is printed rather than the answer alone. See this package's
// reported discrepancies.
// want is how many sessions the operator asked for, and zero is "as many as
// fit, up to [defaultSessions]". A pool larger than the device holds is refused
// here rather than at the allocation that would fail part way through it: every
// session's cache is reserved at startup (019-D2), so the ceiling is a number
// this arithmetic already has.
func kvAdmission(pool, weightBytes, perSession int64, want int,
	shape admissionShape) (admission, error) {

	a := admission{Pool: pool, Weights: weightBytes, Budget: pool - weightBytes, PerSession: perSession}
	if shape.Default == 0 {
		shape.Default = defaultSessions
	}
	if shape.Flag == "" {
		shape.Flag = "--slots"
	}
	switch {
	case perSession <= 0:
		return admission{}, fmt.Errorf("the model reports a key/value cache of %s per session, "+
			"so no admission limit can be computed from memory", humanBytes(perSession))
	case a.Budget <= 0:
		return admission{}, fmt.Errorf("the weights take %s of the device's %s, leaving nothing "+
			"for a key/value cache; load at a narrower --precision or on another --device",
			humanBytes(weightBytes), humanBytes(pool))
	case a.Budget < perSession:
		return admission{}, fmt.Errorf("%s is left after the weights and one session's cache is %s "+
			"at the requested context; lower --context, which is what a session reserves",
			humanBytes(a.Budget), humanBytes(perSession))
	}
	a.Fits = int(a.Budget / perSession)
	if shape.Positions > 0 {
		// One pool, priced from the same per-position cost a session's cache
		// is: what the budget bounds is the pool, and the slot count costs a
		// page table and a row of the per-step ports rather than a cache.
		a.Positions = shape.Positions
		a.Reserved = perSession * int64(shape.Positions) / int64(shape.Context)
		if a.Reserved > a.Budget {
			return admission{}, fmt.Errorf("--kv %d positions reserves %s of key/value "+
				"cache and %s is left after the weights, which holds %d positions at "+
				"this context; lower --kv or --context", shape.Positions,
				humanBytes(a.Reserved), humanBytes(a.Budget),
				a.Budget*int64(shape.Context)/perSession)
		}
		a.Sessions = want
		if want == 0 {
			a.Sessions = shape.Default
		}
		return a, nil
	}
	switch {
	case want == 0:
		a.Sessions = min(shape.Default, a.Fits)
	case want > a.Fits:
		return admission{}, fmt.Errorf("%s %d reserves %s of key/value cache and %s "+
			"is left after the weights, which holds %d session(s) at %s each; lower "+
			"%s or --context", shape.Flag, want, humanBytes(int64(want)*perSession),
			humanBytes(a.Budget), a.Fits, humanBytes(perSession), shape.Flag)
	default:
		a.Sessions = want
	}
	a.Reserved = int64(a.Sessions) * perSession
	return a, nil
}

// interrupts is the context a long-running command stops on.
//
// A variable rather than a call to [signal.NotifyContext], so that the wiring
// in [cmdServe] and [cmdPull] -- the load, the bind, the report, the graceful
// stop -- is reachable from a test. Delivering a real SIGINT to the test binary
// is the alternative, and it is a race against the handler's registration on
// every GOOS that has one.
var interrupts = func() (context.Context, func()) {
	return signal.NotifyContext(context.Background(), os.Interrupt)
}

// serving is a server that is listening and has not been served yet: the
// listener, the handler, and the release of what loading the model took.
type serving struct {
	ln      net.Listener
	srv     *server.Server
	release func() error
}

// cmdServe loads a model and serves it over HTTP until it is interrupted.
func cmdServe(args []string, stdout, stderr io.Writer) error {
	sv, err := startServe(args, stdout, stderr)
	if err != nil {
		return err
	}
	defer func() { _ = sv.release() }()

	ctx, stop := interrupts()
	defer stop()
	return serveUntil(ctx, stop, sv.ln, sv.srv, shutdownGrace, stderr)
}

// startServe does everything `tgo serve` does before it blocks: it parses the
// command line, loads the model, computes the admission limit, binds the
// address and prints the report.
//
// It is separate from [cmdServe] because everything above is what can be wrong
// and nothing above blocks. A test drives it to a bound listener on
// 127.0.0.1:0, reads the report and closes it; what is left in cmdServe is the
// signal handler and [serveUntil], which is tested on its own.
func startServe(args []string, stdout, stderr io.Writer) (*serving, error) {
	o, err := parseServe(args)
	if err != nil {
		return nil, err
	}
	rep, err := openAndDescribe(o.Dir, describeOptions{
		Policy: o.Engine.Precision, Context: o.Engine.Context, Device: o.Engine.Device})
	if err != nil {
		return nil, err
	}
	sv, err := openServable(o.Dir, o.Name, o.Engine)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			_ = sv.Close()
		}
	}()

	// After the model opens, not before: auto can resolve to int8, and a
	// budget computed from the f16 footprint this process predicted would
	// subtract weights that are not on the device (specs/001-weights.md §5).
	rep = resolvedInto(rep, sv.Info)
	adm, err := kvAdmission(rep.Hardware.MaxPoolBytes, rep.Memory.WeightBytes, rep.Memory.KVBytes,
		o.Slots, shapeOf(o, rep.Memory.Context))
	if err != nil {
		return nil, err
	}

	// The pool is built before the server, and the server's concurrency is the
	// pool's size rather than the same division done twice: two numbers that
	// have to agree is the shape this bug takes (019 §4).
	eng, err := sv.Pool(adm.Sessions)
	if err != nil {
		return nil, err
	}
	opts := []server.Option{server.WithConcurrency(adm.Sessions), server.WithNotice(stderr)}
	if o.Public {
		opts = append(opts, server.WithPublicBind())
	}
	srv, err := server.New(eng, opts...)
	if err != nil {
		return nil, err
	}

	// Listen before the report, so that the address printed is the one bound:
	// --addr 127.0.0.1:0 asks the kernel for a port, and a report that echoed
	// the flag would print a port nothing is listening on. The bind is also
	// where 009-D8's rule lives -- a non-loopback address without --public is
	// refused here, and taken with it prints the server's own line saying it
	// has no authentication. [parseServe] refuses the same address before a
	// byte of the checkpoint is read, so what is left for Listen is the
	// enforcement and the addresses the kernel refuses.
	ln, err := srv.Listen(o.Addr)
	if err != nil {
		return nil, err
	}
	renderServe(stdout, rep, o, srv, adm, ln.Addr())
	ok = true
	return &serving{ln: ln, srv: srv, release: sv.Close}, nil
}

// serveUntil serves ln until ctx is done, then stops gracefully.
//
// stop releases the signal handler. It is called the moment the context is
// done and before the grace begins, so that a second Ctrl-C during a 30-second
// shutdown reaches the default disposition and kills the process: an operator
// who interrupts twice has said they are not waiting, and a handler still
// registered would swallow the second signal.
func serveUntil(ctx context.Context, stop func(), ln net.Listener, h http.Handler,
	grace time.Duration, stderr io.Writer) error {

	// ReadHeaderTimeout and nothing else. A request that opens a connection
	// and sends its headers one byte a minute costs a goroutine for as long as
	// it likes, and --public makes that reachable from the network; bounding
	// the headers closes it without touching the body. A write or an idle
	// timeout would be wrong here: a streamed completion is a long response on
	// an open connection, which is exactly what those two cut.
	hs := &http.Server{Handler: h, ReadHeaderTimeout: 30 * time.Second}
	errs := make(chan error, 1)
	go func() { errs <- hs.Serve(ln) }()

	select {
	case err := <-errs:
		// The listener failed on its own. ErrServerClosed cannot appear here,
		// because nothing has asked for a shutdown yet.
		return err
	case <-ctx.Done():
	}
	stop()

	_, _ = fmt.Fprintf(stderr, "\ntgo: stopping; in-flight requests have %s to finish\n", humanDuration(grace))
	// ctx is already cancelled -- that is how we got here -- so the grace
	// period hangs off a copy with the cancellation stripped and the values
	// kept. context.Background() would drop the values too, and a shutdown
	// that inherited the cancellation would expire before it began.
	sctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), grace)
	defer cancel()
	if err := hs.Shutdown(sctx); err != nil {
		// A stream still open when the grace expires is cut, and the reason is
		// stated: the alternative is a process that does not exit, and an
		// operator who cannot tell a hung shutdown from a slow one.
		_, _ = fmt.Fprintf(stderr, "tgo: %s passed with requests still in flight; closing them\n", humanDuration(grace))
		_ = hs.Close()
	}
	if err := <-errs; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// renderServe reports what an operator got: the model, where it is, the routes
// it answers, and the memory the admission limit was divided out of.
func renderServe(w io.Writer, rep modelReport, o serveOptions, srv *server.Server,
	adm admission, addr net.Addr) {

	_, _ = fmt.Fprintf(w, "model      %s\n", o.Name)
	_, _ = fmt.Fprintf(w, "  directory         %s\n", o.Dir)
	_, _ = fmt.Fprintf(w, "  architecture      %s, %s\n", rep.Model.Architecture, rep.Precision.Why)
	_, _ = fmt.Fprintf(w, "  device            %s, %s (%s)\n",
		rep.Hardware.Backend, rep.Hardware.Device, rep.Hardware.Vendor)

	_, _ = fmt.Fprintf(w, "\nlistening  http://%s\n", addr)
	for _, r := range serveRoutes {
		_, _ = fmt.Fprintf(w, "  %-4s %-22s %s\n", r.Method, r.Path, r.What)
	}

	if o.Engine.Batched {
		_, _ = fmt.Fprintf(w, "\nslots      %d, and every in-flight request is in one forward pass\n",
			srv.Concurrency())
		_, _ = fmt.Fprintf(w, "  reserved          %s = one shared pool of %d positions at %d "+
			"positions of context\n", humanBytes(adm.Reserved), adm.Positions,
			rep.Memory.Context)
		_, _ = fmt.Fprintf(w, "  chunk and reserve %d prompt tokens a step, and %d positions held "+
			"beyond a prompt at admission\n", tgo.DefaultChunk,
			serveReserve(rep.Memory.Context))
	} else {
		_, _ = fmt.Fprintf(w, "\nslots      %d pooled sessions, reserved now and held until this "+
			"process exits\n", srv.Concurrency())
		_, _ = fmt.Fprintf(w, "  reserved          %s = %d x %s at %d positions of context\n",
			humanBytes(adm.Reserved), adm.Sessions, humanBytes(adm.PerSession),
			rep.Memory.Context)
	}
	_, _ = fmt.Fprintf(w, "  kv budget         %s = %s device pool - %s weights, which holds %d\n",
		humanBytes(adm.Budget), humanBytes(adm.Pool), humanBytes(adm.Weights), adm.Fits)
	_, _ = fmt.Fprintf(w, "  admission         %d generating at once, %d waiting, refused with 429 after %s\n",
		srv.Concurrency(), server.DefaultQueue, humanDuration(server.DefaultQueueWait))
	_, _ = fmt.Fprintf(w, "  prefix cache      %s\n", prefixCacheLine(o.Engine.PrefixCache))
	_, _ = fmt.Fprintf(w, "  note              %s\n", batchingNote(o.Engine.Batched))
	for _, n := range o.Notes {
		_, _ = fmt.Fprintf(w, "  deprecated        %s\n", n)
	}
	_, _ = fmt.Fprintf(w, "\nCtrl-C stops it, letting in-flight requests finish.\n")
}

// batchingNote is what the concurrency buys, which is a different thing under
// each engine and is the sentence an operator reads before believing a number.
func batchingNote(batched bool) string {
	if batched {
		return "the device pool is a cap on one allocation, not free memory;\n" +
			"                    concurrent requests share one forward pass, so the weights are\n" +
			"                    read once for all of them, and their key/value state is one\n" +
			"                    shared pool rather than a reservation each"
	}
	return "the device pool is a cap on one allocation, not free memory;\n" +
		"                    without batching, concurrent requests interleave rather than\n" +
		"                    go faster, and a session's cache is not returned between requests"
}

// prefixCacheLine says what --prefix-cache did, in the terms an operator
// decides on.
//
// Both halves are stated because both are real. On, a conversation's next turn
// pays for its new tokens alone, and the answer it gets is the one a cold
// request gets in distribution rather than bit for bit (016-D6). Off, every
// request prefills its whole prompt and two identical requests give identical
// answers.
func prefixCacheLine(scope tgo.CacheScope) string {
	switch scope {
	case tgo.CacheSession:
		return "on, scoped to one pooled session; a turn prefills only what is new, and a " +
			"warm\n                    answer matches a cold one in distribution rather than " +
			"bit for bit"
	case tgo.CacheProcess:
		return "on, shared across every session; two conversations with the same system\n" +
			"                    prompt prefill it once between them, a request's " +
			"cache_salt is what\n                    keeps tenants apart, and --sessions " +
			"is concurrency rather than\n                    reuse depth"
	}
	return "off; every request prefills its whole prompt. --prefix-cache reuses what a\n" +
		"                    conversation already paid for, which is the reason to pool " +
		"sessions"
}
