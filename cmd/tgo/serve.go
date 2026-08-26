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
	Dir      string
	Addr     string
	Name     string
	Public   bool
	Sessions int
	Engine   engineOptions
}

// serveFlagSet declares what `tgo serve` accepts. See [runFlagSet] for why
// declaring is separate from parsing.
func serveFlagSet() (*flag.FlagSet, *serveFlags) {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	return fs, &serveFlags{
		addr:      fs.String("addr", server.DefaultAddr, "host:port to listen on"),
		public:    fs.Bool("public", false, "allow a bind that is not loopback, which has no authentication"),
		precision: fs.String("precision", "auto", "f16, int8 or auto"),
		context:   fs.Int("context", defaultContext, "KV cache capacity per session, in positions"),
		device:    fs.String("device", "auto", "auto, cpu or metal"),
		sessions: fs.Int("sessions", 0,
			"pooled sessions, which is the concurrency and the reuse depth "+
				"(0 takes 4, or fewer if the device holds fewer)"),
		prefixCache: fs.Bool("prefix-cache", false,
			"reuse the key/value state a conversation already paid for, which changes what an answer costs and, slightly, what it says"),
	}
}

// serveFlags holds `tgo serve`'s flag values.
type serveFlags struct {
	addr, precision, device *string
	public, prefixCache     *bool
	context, sessions       *int
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
	// Zero is "as many as the device holds, up to the default", which is what
	// the admission arithmetic answers once the weights are on the device. A
	// negative one is a pool that holds no conversation.
	if *f.sessions < 0 {
		return serveOptions{}, fmt.Errorf("%w: --sessions is %d; a pool holds at least one "+
			"conversation", errUsage, *f.sessions)
	}
	addr := strings.TrimSpace(*f.addr)
	if addr == "" {
		addr = server.DefaultAddr
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return serveOptions{}, fmt.Errorf("%w: --addr %q is not a host:port address: %v", errUsage, addr, err)
	}
	if !*f.public && isPublicAddr(host) {
		return serveOptions{}, publicBindRefusal(addr)
	}
	return serveOptions{
		Dir: dir, Addr: addr, Name: modelID(dir), Public: *f.public, Sessions: *f.sessions,
		Engine: engineOptions{Precision: policy, Context: *f.context, Device: dev,
			PrefixCache: *f.prefixCache},
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
	if o.PrefixCache {
		// Scoped to the session, which is the only scope tgo can honour: the
		// process scope needs a page table specs/004-model-graph.md §3 declares
		// no port for. The budget is the whole context, because a pooled
		// session's history is its own and there is nothing else to spend it
		// on.
		opts = append(opts, tgo.WithPrefixCache(tgo.CacheSession, o.Context))
	}
	m, err := tgo.Open(dir, opts...)
	if err != nil {
		return servable{}, err
	}
	i := m.Info()
	var pool *server.PoolEngine
	return servable{
		Pool: func(sessions int) (server.Engine, error) {
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
			return errors.Join(append(errs, m.Close())...)
		},
	}, nil
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
	Sessions int
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
func kvAdmission(pool, weightBytes, perSession int64, want int) (admission, error) {
	a := admission{Pool: pool, Weights: weightBytes, Budget: pool - weightBytes, PerSession: perSession}
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
	switch {
	case want == 0:
		a.Sessions = min(defaultSessions, a.Fits)
	case want > a.Fits:
		return admission{}, fmt.Errorf("--sessions %d reserves %s of key/value cache and %s "+
			"is left after the weights, which holds %d session(s) at %s each; lower "+
			"--sessions or --context", want, humanBytes(int64(want)*perSession),
			humanBytes(a.Budget), a.Fits, humanBytes(perSession))
	default:
		a.Sessions = want
	}
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
	defer sv.release()

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
			sv.Close()
		}
	}()

	// After the model opens, not before: auto can resolve to int8, and a
	// budget computed from the f16 footprint this process predicted would
	// subtract weights that are not on the device (specs/001-weights.md §5).
	rep = resolvedInto(rep, sv.Info)
	adm, err := kvAdmission(rep.Hardware.MaxPoolBytes, rep.Memory.WeightBytes, rep.Memory.KVBytes,
		o.Sessions)
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

	fmt.Fprintf(stderr, "\ntgo: stopping; in-flight requests have %s to finish\n", humanDuration(grace))
	sctx, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()
	if err := hs.Shutdown(sctx); err != nil {
		// A stream still open when the grace expires is cut, and the reason is
		// stated: the alternative is a process that does not exit, and an
		// operator who cannot tell a hung shutdown from a slow one.
		fmt.Fprintf(stderr, "tgo: %s passed with requests still in flight; closing them\n", humanDuration(grace))
		hs.Close()
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

	fmt.Fprintf(w, "model      %s\n", o.Name)
	fmt.Fprintf(w, "  directory         %s\n", o.Dir)
	fmt.Fprintf(w, "  architecture      %s, %s\n", rep.Model.Architecture, rep.Precision.Why)
	fmt.Fprintf(w, "  device            %s, %s (%s)\n",
		rep.Hardware.Backend, rep.Hardware.Device, rep.Hardware.Vendor)

	fmt.Fprintf(w, "\nlistening  http://%s\n", addr)
	for _, r := range serveRoutes {
		fmt.Fprintf(w, "  %-4s %-22s %s\n", r.Method, r.Path, r.What)
	}

	fmt.Fprintf(w, "\nsessions   %d pooled, reserved now and held until this process exits\n",
		srv.Concurrency())
	fmt.Fprintf(w, "  reserved          %s = %d x %s at %d positions of context\n",
		humanBytes(int64(adm.Sessions)*adm.PerSession), adm.Sessions,
		humanBytes(adm.PerSession), rep.Memory.Context)
	fmt.Fprintf(w, "  kv budget         %s = %s device pool - %s weights, which holds %d\n",
		humanBytes(adm.Budget), humanBytes(adm.Pool), humanBytes(adm.Weights), adm.Fits)
	fmt.Fprintf(w, "  admission         %d generating at once, %d waiting, refused with 429 after %s\n",
		srv.Concurrency(), server.DefaultQueue, humanDuration(server.DefaultQueueWait))
	fmt.Fprintf(w, "  prefix cache      %s\n", prefixCacheLine(o.Engine.PrefixCache))
	fmt.Fprintf(w, "  note              the device pool is a cap on one allocation, not free memory;\n"+
		"                    without batching, concurrent requests interleave rather than\n"+
		"                    go faster, and a session's cache is not returned between requests\n")
	fmt.Fprintf(w, "\nCtrl-C stops it, letting in-flight requests finish.\n")
}

// prefixCacheLine says what --prefix-cache did, in the terms an operator
// decides on.
//
// Both halves are stated because both are real. On, a conversation's next turn
// pays for its new tokens alone, and the answer it gets is the one a cold
// request gets in distribution rather than bit for bit (016-D6). Off, every
// request prefills its whole prompt and two identical requests give identical
// answers.
func prefixCacheLine(on bool) string {
	if on {
		return "on, scoped to one pooled session; a turn prefills only what is new, and a " +
			"warm\n                    answer matches a cold one in distribution rather than " +
			"bit for bit"
	}
	return "off; every request prefills its whole prompt. --prefix-cache reuses what a\n" +
		"                    conversation already paid for, which is the reason to pool " +
		"sessions"
}
