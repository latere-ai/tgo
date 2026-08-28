// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/server"
)

// fakeServerEngine is a model as [server.Engine] sees one.
//
// Five of its seven methods are constants, which is all the paths under test
// need: `tgo serve` binds, prints and stops, and makes no request. NewSession
// and CheckSchema refuse, so a test that started generating, or compiled a
// schema, by accident fails rather than hanging.
type fakeServerEngine struct {
	name       string
	context    int
	vocab      int
	perSession int64

	// sessions is the pool size startServe asked for, which is what
	// [TestServePoolSizeIsTheAdmissionLimit] reads to check that the admission
	// arithmetic and the pool are one number rather than two that agree today.
	sessions int
}

func (f *fakeServerEngine) Name() string                { return f.name }
func (f *fakeServerEngine) Context() int                { return f.context }
func (f *fakeServerEngine) VocabSize() int              { return f.vocab }
func (f *fakeServerEngine) CacheBytesPerSession() int64 { return f.perSession }
func (f *fakeServerEngine) NewSession(context.Context, server.SessionSpec) (server.Session, error) {
	return nil, errors.New("this fake engine generates nothing")
}

func (f *fakeServerEngine) CheckSchema([]byte) error {
	return errors.New("this fake engine compiles nothing")
}

// fakeServable is the model `tgo serve` loads, without a device or a
// checkpoint.
//
// The four numbers are pairwise distinct and none of them divides another
// evenly by accident: a report that printed the per-session cache where the
// weights belong, or the context where the vocabulary belongs, is visible.
const (
	fakeWeightBytes = 300 << 20 // 300 MiB
	fakeCacheBytes  = 48 << 20  // 48 MiB per session
	fakeVocabSize   = 112
)

// useFakeServable installs a servable for the duration of a test and records
// what each command asked to load.
func useFakeServable(t *testing.T, info engineInfo) *[]string {
	t.Helper()
	var names []string
	prev := openServable
	openServable = func(dir, name string, o engineOptions) (servable, error) {
		names = append(names, name)
		return servable{
			Pool: func(sessions int) (server.Engine, error) {
				return &fakeServerEngine{
					name: name, context: o.Context, vocab: fakeVocabSize,
					perSession: info.CacheBytesPerSession, sessions: sessions,
				}, nil
			},
			Info:  info,
			Close: func() error { return nil },
		}, nil
	}
	t.Cleanup(func() { openServable = prev })
	return &names
}

// fakeInfo is what the fake loader resolved.
func fakeInfo(context int) engineInfo {
	return engineInfo{
		Precision: "f16", WeightBytes: fakeWeightBytes,
		CacheBytesPerSession: fakeCacheBytes, Context: context,
	}
}

func TestParseServeDefaults(t *testing.T) {
	o, err := parseServe([]string{filepath.Join("models", "qwen3")})
	if err != nil {
		t.Fatalf("parseServe: %v", err)
	}
	// 009-D8: loopback by default. A server with no authentication is not
	// exposed by omission, so the default is asserted rather than assumed.
	if o.Addr != server.DefaultAddr {
		t.Errorf("the default address is %q, want %q", o.Addr, server.DefaultAddr)
	}
	if host, _, _ := net.SplitHostPort(o.Addr); !net.ParseIP(host).IsLoopback() {
		t.Errorf("the default address %q is not loopback", o.Addr)
	}
	if o.Public {
		t.Error("--public defaults to true, so a public bind would need no flag")
	}
	if o.Name != "qwen3" {
		t.Errorf("the model id is %q, want the directory's name", o.Name)
	}
	if o.Engine.Context != defaultContext {
		t.Errorf("context = %d, want %d", o.Engine.Context, defaultContext)
	}
	if o.Engine.Device != tgo.AutoDevice {
		t.Errorf("device = %v, want auto", o.Engine.Device)
	}
}

func TestParseServeFlags(t *testing.T) {
	o, err := parseServe([]string{"--addr", "0.0.0.0:8080", "--public", "--precision", "int8",
		"--context", "1024", "--device", "cpu", "d"})
	if err != nil {
		t.Fatalf("parseServe: %v", err)
	}
	if o.Addr != "0.0.0.0:8080" || !o.Public || o.Engine.Context != 1024 || o.Engine.Device != tgo.CPU {
		t.Errorf("parseServe = %+v, want every flag carried through", o)
	}
	if got := o.Engine.Precision.String(); got != "int8" {
		t.Errorf("precision = %q, want int8", got)
	}
	// An empty --addr is the default rather than the wildcard, which is what
	// makes it safe: net.Listen("tcp", "") binds every interface.
	o, err = parseServe([]string{"--addr", "  ", "d"})
	if err != nil {
		t.Fatalf("parseServe with a blank --addr: %v", err)
	}
	if o.Addr != server.DefaultAddr {
		t.Errorf("a blank --addr became %q, want the loopback default", o.Addr)
	}
}

func TestParseServeRefusals(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"no directory", nil, "no model directory"},
		{"two directories", []string{"a", "b"}, "one model directory"},
		{"a flag after the directory", []string{"d", "--addr", "127.0.0.1:1"}, "flags go before it"},
		{"an unknown flag", []string{"--nope", "d"}, "nope"},
		{"an address with no port", []string{"--addr", "127.0.0.1", "d"}, "is not a host:port address"},
		{"an empty context", []string{"--context", "0", "d"}, "--context is 0"},
		{"a negative context", []string{"--context", "-1", "d"}, "--context is -1"},
		{"an unknown precision", []string{"--precision", "bf16", "d"}, "not f16, int8, int4 or auto"},
		{"an unknown device", []string{"--device", "cuda", "d"}, "not auto, cpu or metal"},
		{"a public address without the flag", []string{"--addr", "0.0.0.0:8080", "d"}, "--public"},
		{"the wildcard without the flag", []string{"--addr", ":8080", "d"}, "--public"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseServe(tc.args)
			if err == nil {
				t.Fatalf("parseServe(%v) was accepted", tc.args)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want one containing %q", err, tc.want)
			}
			if !errors.Is(err, errUsage) {
				t.Error("the refusal does not wrap errUsage, so main would not print the usage")
			}
		})
	}
}

// TestModelIDIsTheDirectorysName pins the id a request has to name. There is no
// flag for it, so the derivation is the whole contract.
func TestModelIDIsTheDirectorysName(t *testing.T) {
	for _, tc := range []struct{ dir, want string }{
		{filepath.Join("models", "Qwen3-0.6B"), "Qwen3-0.6B"},
		{filepath.Join("models", "Qwen3-0.6B") + string(filepath.Separator), "Qwen3-0.6B"},
		{"qwen", "qwen"},
		{".", "model"},
		{string(filepath.Separator), "model"},
	} {
		if got := modelID(tc.dir); got != tc.want {
			t.Errorf("modelID(%q) = %q, want %q", tc.dir, got, tc.want)
		}
	}
}

// TestKVAdmission is specs/009-server.md §6's arithmetic, and the three ways it
// has no answer.
func TestKVAdmission(t *testing.T) {
	// 1000 - 280 = 720 over 150 is 4 sessions with 120 left over, which the
	// floor drops. Four distinct numbers, none a multiple of another.
	a, err := kvAdmission(1000, 280, 150, 0)
	if err != nil {
		t.Fatalf("kvAdmission: %v", err)
	}
	if a.Budget != 720 || a.Fits != 4 || a.Sessions != 4 {
		t.Errorf("kvAdmission = %+v, want a budget of 720 and 4 sessions", a)
	}
	if a.Pool != 1000 || a.Weights != 280 || a.PerSession != 150 {
		t.Errorf("kvAdmission = %+v, want the three terms carried with the answer", a)
	}

	// The boundary the floor sits on: a budget that holds exactly one session
	// holds one, and is not the "lower --context" refusal below. That is the
	// machine which can only just run the model, which is the one an operator
	// is most likely to be on, and a `<` written as `<=` turns it away at
	// startup with a message telling it to shrink a context that already fits.
	a, err = kvAdmission(1000, 850, 150, 0)
	if err != nil {
		t.Fatalf("a budget of exactly one session was refused: %v", err)
	}
	if a.Budget != 150 || a.Fits != 1 || a.Sessions != 1 {
		t.Errorf("kvAdmission(1000, 850, 150) = %+v, want a budget of 150 and 1 session", a)
	}

	// 019-D2 reserves every pooled session's cache at startup, so the device's
	// capacity is a ceiling and not a target: an operator who asks for nothing
	// gets the default rather than every session the budget would hold.
	a, err = kvAdmission(100000, 280, 150, 0)
	if err != nil {
		t.Fatalf("kvAdmission on a large device: %v", err)
	}
	if a.Fits <= defaultSessions || a.Sessions != defaultSessions {
		t.Errorf("a device holding %d sessions reserved %d, want the default of %d",
			a.Fits, a.Sessions, defaultSessions)
	}

	// And an explicit ask is honoured up to what fits, and refused above it
	// rather than at the allocation that would fail part way through.
	if a, err := kvAdmission(1000, 280, 150, 3); err != nil || a.Sessions != 3 {
		t.Errorf("kvAdmission(..., 3) = %+v, %v; three of four sessions fit", a, err)
	}
	_, err = kvAdmission(1000, 280, 150, 5)
	if err == nil {
		t.Fatal("a pool of five was accepted where four fit")
	}
	for _, want := range []string{"--sessions 5", "4 session"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal %q does not say %q", err, want)
		}
	}

	for _, tc := range []struct {
		name                      string
		pool, weights, perSession int64
		want                      string
	}{
		{"a cache of no size", 1000, 280, 0, "no admission limit"},
		{"weights larger than the pool", 1000, 1400, 150, "leaving nothing"},
		{"a budget under one session", 1000, 900, 150, "lower --context"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := kvAdmission(tc.pool, tc.weights, tc.perSession, 0)
			if err == nil {
				t.Fatalf("kvAdmission(%d, %d, %d) was accepted", tc.pool, tc.weights, tc.perSession)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want one containing %q", err, tc.want)
			}
		})
	}
}

// TestServeReportsWhatAnOperatorGot holds the startup report against the server
// it describes: an operator reads this and nothing else before their first
// request.
func TestServeReportsWhatAnOperatorGot(t *testing.T) {
	useCPUDevice(t)
	names := useFakeServable(t, fakeInfo(1024))
	dir := syntheticDir(t)

	var stdout, stderr strings.Builder
	sv, err := startServe([]string{"--addr", "127.0.0.1:0", "--context", "1024", "--device", "cpu", dir},
		&stdout, &stderr)
	if err != nil {
		t.Fatalf("startServe: %v", err)
	}
	defer func() {
		sv.ln.Close()
		sv.release()
	}()

	out := stdout.String()
	if want := filepath.Base(dir); len(*names) != 1 || (*names)[0] != want {
		t.Errorf("the model was served as %v, want the directory's name %q", *names, want)
	}
	// The routes, so that an operator can point a client at one.
	for _, r := range serveRoutes {
		if !strings.Contains(out, r.Path) {
			t.Errorf("the report does not name %s:\n%s", r.Path, out)
		}
	}
	// The address that was bound, not the one that was asked for: --addr
	// 127.0.0.1:0 asks the kernel for a port.
	if !strings.Contains(out, sv.ln.Addr().String()) {
		t.Errorf("the report does not carry the bound address %s:\n%s", sv.ln.Addr(), out)
	}
	if strings.Contains(out, "127.0.0.1:0\n") {
		t.Error("the report printed the requested port rather than the bound one")
	}
	// The budget, with the two terms it came from (§6), and the limit the
	// server actually admits against.
	adm, err := kvAdmission(2147483647, fakeWeightBytes, fakeCacheBytes, 0)
	if err != nil {
		t.Fatalf("kvAdmission: %v", err)
	}
	if got := sv.srv.Concurrency(); got != adm.Sessions {
		t.Errorf("the server admits %d sessions and the report's arithmetic says %d", got, adm.Sessions)
	}
	for _, want := range []string{
		fmt.Sprintf("%d pooled", sv.srv.Concurrency()),
		"held until this process exits",
		humanBytes(fakeWeightBytes) + " weights",
		humanBytes(fakeCacheBytes) + " at 1024 positions",
		// The default is off, and the line says what turning it on buys.
		"--prefix-cache",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the report does not carry %q:\n%s", want, out)
		}
	}
	// Nothing was said about authentication, because loopback needs no
	// warning: a warning printed every time is a warning nobody reads.
	if strings.Contains(stderr.String(), "no authentication") {
		t.Errorf("a loopback bind warned about authentication:\n%s", stderr.String())
	}
}

// TestPublicAddressAgreesWithTheServer holds this package's restatement of
// 009-D8's rule against the server's own.
//
// The rule is checked twice on purpose -- here so that a command line which
// cannot work is refused before a 1.4 GiB checkpoint is loaded, and in
// server.Server.Listen, which is the enforcement. Two copies of one rule drift,
// and a copy that drifted towards permissive would bind an unauthenticated
// server to the network without the flag that exists to make that deliberate.
func TestPublicAddressAgreesWithTheServer(t *testing.T) {
	srv, err := server.New(&fakeServerEngine{
		name: "fixture", context: 1024, vocab: fakeVocabSize, perSession: fakeCacheBytes,
	}, server.WithNotice(nil))
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	for _, addr := range []string{
		"127.0.0.1:0", "localhost:0", "[::1]:0", "127.0.0.2:0", "LOCALHOST:0",
		"0.0.0.0:0", ":0", "[::]:0", "192.0.2.1:0", "example.invalid:0",
	} {
		t.Run(addr, func(t *testing.T) {
			host, _, err := net.SplitHostPort(addr)
			if err != nil {
				t.Fatalf("split %q: %v", addr, err)
			}
			mine := isPublicAddr(host)
			ln, err := srv.Listen(addr)
			if ln != nil {
				ln.Close()
			}
			theirs := err != nil && strings.Contains(err.Error(), "WithPublicBind")
			if mine != theirs {
				t.Errorf("this package calls %q public=%v and the server calls it public=%v (%v)",
					addr, mine, theirs, err)
			}
		})
	}
}

// TestServeRefusesANonLoopbackBindWithoutTheFlag is specs/009-server.md §7 and
// 009-D8 at the command line: an unauthenticated server is not exposed by
// omission.
func TestServeRefusesANonLoopbackBindWithoutTheFlag(t *testing.T) {
	useCPUDevice(t)
	loaded := useFakeServable(t, fakeInfo(defaultContext))
	dir := syntheticDir(t)

	for _, addr := range []string{"0.0.0.0:0", ":0", "[::]:0", "192.0.2.1:0"} {
		t.Run(addr, func(t *testing.T) {
			var stdout, stderr strings.Builder
			sv, err := startServe([]string{"--addr", addr, "--device", "cpu", dir}, &stdout, &stderr)
			if err == nil {
				sv.ln.Close()
				sv.release()
				t.Fatalf("--addr %s bound without --public", addr)
			}
			// And nothing was loaded: the refusal is a flag the user can fix,
			// so it comes before the minutes a checkpoint takes to open.
			if len(*loaded) != 0 {
				t.Errorf("a refused bind loaded the model first: %v", *loaded)
			}
			if !strings.Contains(err.Error(), "--public") {
				t.Errorf("error = %v, want one naming the flag that would allow it", err)
			}
			if !strings.Contains(err.Error(), "no authentication") {
				t.Errorf("error = %v, want one saying why the flag exists", err)
			}
			if !errors.Is(err, errUsage) {
				t.Error("the refusal does not wrap errUsage, so main would not print the usage")
			}
			if stdout.Len() != 0 {
				t.Errorf("a refused bind still printed a report: %q", stdout.String())
			}
		})
	}
}

// TestServePublicBindSaysItHasNoAuthentication is the other half of 009-D8: the
// flag is taken, and the line is printed.
func TestServePublicBindSaysItHasNoAuthentication(t *testing.T) {
	useCPUDevice(t)
	useFakeServable(t, fakeInfo(defaultContext))

	var stdout, stderr strings.Builder
	sv, err := startServe([]string{"--addr", "0.0.0.0:0", "--public", "--device", "cpu", syntheticDir(t)},
		&stdout, &stderr)
	if err != nil {
		t.Fatalf("startServe --public: %v", err)
	}
	defer func() {
		sv.ln.Close()
		sv.release()
	}()
	notice := stderr.String()
	for _, want := range []string{"no authentication", "reachable from the network"} {
		if !strings.Contains(notice, want) {
			t.Errorf("the public bind did not say %q:\n%s", want, notice)
		}
	}
}

// TestServeRefusesAModelThatWillNotLoad: the load failure reaches the user
// rather than a server bound to a model that is not there.
func TestServeRefusesAModelThatWillNotLoad(t *testing.T) {
	useCPUDevice(t)
	prev := openServable
	openServable = func(dir, name string, o engineOptions) (servable, error) {
		return servable{}, errFake
	}
	t.Cleanup(func() { openServable = prev })

	var stdout, stderr strings.Builder
	err := cmdServe([]string{"--addr", "127.0.0.1:0", "--device", "cpu", syntheticDir(t)}, &stdout, &stderr)
	if !errors.Is(err, errFake) {
		t.Fatalf("cmdServe = %v, want the loader's error", err)
	}
	if stdout.Len() != 0 {
		t.Errorf("a server that never loaded still printed a report: %q", stdout.String())
	}
}

// TestServeRefusesAModelTooLargeForTheDevice is §6's arithmetic reaching the
// user: the weights fit and a session's cache does not, which is a refusal at
// startup rather than an out-of-memory error under load.
func TestServeRefusesAModelTooLargeForTheDevice(t *testing.T) {
	useCPUDevice(t)
	info := fakeInfo(defaultContext)
	info.WeightBytes = 2147483647 // the whole CPU pool
	useFakeServable(t, info)

	var stdout, stderr strings.Builder
	_, err := startServe([]string{"--addr", "127.0.0.1:0", "--device", "cpu", syntheticDir(t)},
		&stdout, &stderr)
	if err == nil {
		t.Fatal("a model that leaves no room for a cache was served")
	}
	if !strings.Contains(err.Error(), "leaving nothing") {
		t.Errorf("error = %v, want one naming the memory", err)
	}
}

// useCountedServable installs a servable that counts how often it was released.
//
// Separate from [useFakeServable] because what it records is the other side of
// the same seam: that one says which model was opened, this one says whether it
// was given back.
func useCountedServable(t *testing.T, info engineInfo) *int {
	t.Helper()
	n := new(int)
	prev := openServable
	openServable = func(dir, name string, o engineOptions) (servable, error) {
		return servable{
			Pool: func(sessions int) (server.Engine, error) {
				return &fakeServerEngine{
					name: name, context: o.Context, vocab: fakeVocabSize,
					perSession: info.CacheBytesPerSession, sessions: sessions,
				}, nil
			},
			Info:  info,
			Close: func() error { *n++; return nil },
		}, nil
	}
	t.Cleanup(func() { openServable = prev })
	return n
}

// TestServeReleasesTheModelItLoaded: a loaded checkpoint holds device memory,
// so both ways out of `tgo serve` have to give it back -- a start that failed
// after the load, and a command that returned after its interrupt.
//
// Neither is visible from outside the process: the operating system reclaims
// everything at exit, so a leak here is a `tgo serve` embedded in something
// longer-lived holding a whole checkpoint and nothing saying so. The release is
// therefore counted at the seam rather than inferred from the listener, which
// stops answering whether or not the model was freed.
func TestServeReleasesTheModelItLoaded(t *testing.T) {
	useCPUDevice(t)
	closes := useCountedServable(t, fakeInfo(defaultContext))
	dir := syntheticDir(t)

	// A bind the kernel refuses rather than one 009-D8 refuses, so that the
	// failure lands after the weights are on the device: 99999 is not a port,
	// and parseServe does not check the range because net.Listen does.
	var stdout, stderr strings.Builder
	if _, err := startServe([]string{"--addr", "127.0.0.1:99999", "--device", "cpu", dir},
		&stdout, &stderr); err == nil {
		t.Fatal("--addr 127.0.0.1:99999 was bound")
	}
	if *closes != 1 {
		t.Fatalf("a start that failed after the load released the model %d times, want 1", *closes)
	}

	// And the ordinary exit, which is the path every run takes.
	interrupt, _ := useInterrupts(t)
	out, errs := &syncBuilder{}, &syncBuilder{}
	done := make(chan error, 1)
	go func() {
		done <- cmdServe([]string{"--addr", "127.0.0.1:0", "--device", "cpu", dir}, out, errs)
	}()
	awaitAddr(t, out)
	interrupt()
	if err := <-done; err != nil {
		t.Fatalf("cmdServe = %v, want nil after an interrupt", err)
	}
	if *closes != 2 {
		t.Errorf("the command returned having released the model %d times in all, want 2", *closes)
	}
}

// TestServeRoutesAreTheRoutesTheServerAnswers holds the printed list against a
// real server.
//
// The list is written in this package because server.Server exports no way to
// enumerate its mux, so without this test a route renamed upstream would be
// printed at an operator who then cannot reach it. A route the server does not
// serve answers 404; every route it does serve answers something else, even
// with a body it rejects.
func TestServeRoutesAreTheRoutesTheServerAnswers(t *testing.T) {
	srv, err := server.New(&fakeServerEngine{
		name: "fixture", context: 1024, vocab: fakeVocabSize, perSession: fakeCacheBytes,
	}, server.WithNotice(nil))
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	for _, r := range serveRoutes {
		t.Run(r.Method+" "+r.Path, func(t *testing.T) {
			req := httptest.NewRequest(r.Method, r.Path, strings.NewReader("{}"))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			srv.ServeHTTP(w, req)
			if w.Code == http.StatusNotFound {
				t.Errorf("`tgo serve` prints %s %s and the server answers 404", r.Method, r.Path)
			}
		})
	}
	// And the negative: a route nobody serves is a 404, so the assertion above
	// is not vacuous.
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, httptest.NewRequest("GET", "/v1/no-such-route", nil))
	if w.Code != http.StatusNotFound {
		t.Errorf("an unserved route answered %d, so the route check above proves nothing", w.Code)
	}
}

// TestServeUntilLetsAnInFlightRequestFinish is the graceful stop: a request
// that is still inside its handler when the signal arrives is served to
// completion, and only then does the command leave.
//
// The ordering is the whole test. A stop that closed every connection at once
// passes any version of this that merely observes the handler was entered at
// some earlier moment, because the handler is then free to finish before the
// shutdown starts. So the handler is held until the listener is provably shut
// -- a dial that is refused is the shutdown sequence having begun -- and only
// then released. What is asserted afterwards is the body the client read, not
// how long anything took.
func TestServeUntilLetsAnInFlightRequestFinish(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := ln.Addr().String()
	entered := make(chan struct{})
	release := make(chan struct{})
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		<-release
		io.WriteString(w, "finished")
	})

	ctx, cancel := context.WithCancel(context.Background())
	stopped := make(chan struct{})
	done := make(chan error, 1)
	var stderr syncBuilder
	go func() { done <- serveUntil(ctx, func() { close(stopped) }, ln, h, 5*time.Second, &stderr) }()

	body := make(chan string, 1)
	go func() {
		resp, err := http.Get("http://" + addr + "/slow")
		if err != nil {
			body <- "error: " + err.Error()
			return
		}
		defer resp.Body.Close()
		b, err := io.ReadAll(resp.Body)
		if err != nil {
			body <- "error: " + err.Error()
			return
		}
		body <- string(b)
	}()

	<-entered // the request is inside the handler
	cancel()  // and now the signal arrives
	<-stopped // serveUntil has seen it and released the signal handler
	awaitClosed(t, addr)
	close(release) // the handler finishes after the shutdown began, not before

	if got := <-body; got != "finished" {
		t.Errorf("the in-flight request read %q, want the handler's whole answer", got)
	}
	if err := <-done; err != nil {
		t.Fatalf("serveUntil = %v, want nil on a clean stop", err)
	}
	if !strings.Contains(stderr.String(), "in-flight") {
		t.Errorf("the stop said nothing about what it was waiting for:\n%s", stderr.String())
	}
}

// awaitClosed blocks until addr refuses a connection, which is the shutdown
// having closed the listener.
//
// A poll rather than a wait on a duration: rule 6 of this project's tests is
// that a duration is never asserted, and what is needed here is an ordering,
// not an interval. Every dial that succeeds is closed, so the poll does not
// leave the server a connection to wait for.
func awaitClosed(t *testing.T, addr string) {
	t.Helper()
	for range 5000 {
		c, err := net.DialTimeout("tcp", addr, time.Second)
		if err != nil {
			return
		}
		c.Close()
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("%s still accepts connections after the shutdown began", addr)
}

// TestServeUntilClosesWhatOutlastsTheGrace: a stream still open when the grace
// expires is cut and the reason is stated. The alternative is a process that
// does not exit.
func TestServeUntilClosesWhatOutlastsTheGrace(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		<-release
	})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	var stderr strings.Builder
	go func() { done <- serveUntil(ctx, func() {}, ln, h, 20*time.Millisecond, &stderr) }()
	go http.Get("http://" + ln.Addr().String() + "/forever") //nolint:errcheck // the client is cut on purpose

	<-entered
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("serveUntil = %v, want nil after it closed what would not finish", err)
	}
	if !strings.Contains(stderr.String(), "still in flight") {
		t.Errorf("the stop did not say it cut a request:\n%s", stderr.String())
	}
}

// TestServeUntilReportsAListenerThatFails: a listener that dies on its own is
// an error, and http.ErrServerClosed is not one.
func TestServeUntilReportsAListenerThatFails(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	ln.Close()
	var stderr strings.Builder
	err = serveUntil(context.Background(), func() {}, ln,
		http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}), time.Second, &stderr)
	if err == nil {
		t.Fatal("a listener that will not accept was reported as a clean stop")
	}
	if errors.Is(err, http.ErrServerClosed) {
		t.Error("a listener failure was reported as a shutdown")
	}
}

// useInterrupts replaces the signal source with a cancel a test can call, and
// returns it. Delivering a real SIGINT to the test binary is the alternative,
// and it is a race against the handler's registration.
func useInterrupts(t *testing.T) (cancel func(), stopped *int) {
	t.Helper()
	ctx, cancelCtx := context.WithCancel(context.Background())
	n := new(int)
	prev := interrupts
	interrupts = func() (context.Context, func()) { return ctx, func() { *n++ } }
	t.Cleanup(func() {
		interrupts = prev
		cancelCtx()
	})
	return cancelCtx, n
}

// TestCmdServeServesUntilInterrupted is the whole command: it loads, binds,
// reports, answers a request, and leaves when the interrupt arrives.
func TestCmdServeServesUntilInterrupted(t *testing.T) {
	useCPUDevice(t)
	useFakeServable(t, fakeInfo(1024))
	interrupt, stopped := useInterrupts(t)

	// A synchronised buffer, because the report is written by the goroutine
	// below and read by awaitAddr while it is still running.
	stdout, stderr := &syncBuilder{}, &syncBuilder{}
	done := make(chan error, 1)
	go func() {
		done <- cmdServe([]string{"--addr", "127.0.0.1:0", "--context", "1024",
			"--device", "cpu", syntheticDir(t)}, stdout, stderr)
	}()

	// The report is written before the command blocks, so the address it
	// printed is the one to ask. Waiting on the health route rather than on a
	// sleep: what is under test is that the server answers, and a duration
	// asserted here would be a duration measured on Windows.
	addr := awaitAddr(t, stdout)
	resp, err := http.Get("http://" + addr + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("/health answered %d, want 200", resp.StatusCode)
	}

	interrupt()
	if err := <-done; err != nil {
		t.Fatalf("cmdServe = %v, want nil after an interrupt", err)
	}
	if *stopped == 0 {
		t.Error("the signal handler was never released, so a second Ctrl-C would be swallowed")
	}
	// And the model was released: a served model holds device memory, and a
	// command that returned without freeing it leaks the whole checkpoint.
	if _, err := http.Get("http://" + addr + "/health"); err == nil {
		t.Error("the listener is still answering after the command returned")
	}
}

// awaitAddr reads the bound address out of the report the command printed.
//
// A poll rather than a sleep, and no duration is asserted: the report appears
// when the goroutine gets there, and how long that takes is not the contract.
func awaitAddr(t *testing.T, out fmt.Stringer) string {
	t.Helper()
	for range 2000 {
		if _, rest, ok := strings.Cut(out.String(), "listening  http://"); ok {
			if addr, _, ok := strings.Cut(rest, "\n"); ok {
				return strings.TrimSpace(addr)
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("the server never printed an address:\n%s", out.String())
	return ""
}

// syncBuilder is a strings.Builder a test can read while a command is still
// writing to it.
type syncBuilder struct {
	mu sync.Mutex
	b  strings.Builder
}

func (s *syncBuilder) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuilder) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// TestServeARealCheckpoint runs the whole command against the model TGO_MODEL
// names: the live loader, a real device, a real request over HTTP, and the
// graceful stop.
//
// Skipped by default (specs/000-decisions.md decision 8). It stays in the tree
// because openServable is the one path above that no fake exercises: every
// other test replaces it, so without this nothing checks that server.Wrap and
// tgo.Open agree with what this file asks of them.
func TestServeARealCheckpoint(t *testing.T) {
	dir := os.Getenv("TGO_MODEL")
	if dir == "" {
		t.Skip("TGO_MODEL is not set; this test loads a real checkpoint")
	}
	interrupt, _ := useInterrupts(t)
	stdout, stderr := &syncBuilder{}, &syncBuilder{}
	done := make(chan error, 1)
	go func() {
		done <- cmdServe([]string{"--addr", "127.0.0.1:0", "--context", "512", dir}, stdout, stderr)
	}()
	defer func() {
		interrupt()
		if err := <-done; err != nil {
			t.Errorf("cmdServe = %v", err)
		}
	}()

	addr := awaitAddr(t, stdout)
	t.Logf("\n%s", stdout.String())

	body := fmt.Sprintf(`{"model":%q,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`,
		modelID(dir))
	resp, err := http.Post("http://"+addr+"/v1/chat/completions", "application/json",
		strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read the answer: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("the request answered %d: %s", resp.StatusCode, got)
	}
	if !strings.Contains(string(got), "chat.completion") {
		t.Errorf("the answer is not an OpenAI completion: %s", got)
	}
	// The report is what an operator reads before their first request, so the
	// three things they need are held against a real model.
	for _, want := range []string{"Qwen3ForCausalLM", "/v1/chat/completions", "kv budget"} {
		if !strings.Contains(stdout.String(), want) {
			t.Errorf("the report does not carry %q", want)
		}
	}
}

// TestServeReportsThePoolAndWhatItReuses is specs/019-session-affinity.md §4
// where an operator reads it.
//
// Two costs and two lines. The pool is memory reserved at startup and held
// until the process exits, so the report carries what N sessions come to beside
// what the device would hold; the prefix cache changes what an answer says, so
// the report says which way it is set rather than leaving the operator to
// remember the flag they did not pass.
func TestServeReportsThePoolAndWhatItReuses(t *testing.T) {
	useCPUDevice(t)
	dir := syntheticDir(t)
	useFakeServable(t, fakeInfo(1024))

	for _, tc := range []struct {
		name string
		args []string
		want []string
	}{
		{"the default pool", nil, []string{
			fmt.Sprintf("%d pooled", defaultSessions),
			"held until this process exits",
			// The flag an operator would have to pass to get any reuse, in the
			// line that says they are getting none.
			"--prefix-cache reuses what a",
		}},
		{"reuse turned on", []string{"--prefix-cache"}, []string{
			"on, scoped to one pooled session",
			"in distribution rather than bit for bit",
		}},
		{"a pool of one", []string{"--sessions", "1"}, []string{
			"1 pooled",
			"1 x " + humanBytes(fakeCacheBytes),
		}},
		// The two lines that stop being true under a batch: what the number
		// counts, and what concurrency buys.
		{"batched", []string{"--batched"}, []string{
			"every in-flight request is in one forward pass",
			"the weights are\n                    read once for all of them",
			"on, shared across every session",
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr strings.Builder
			args := append([]string{"--addr", "127.0.0.1:0", "--context", "1024",
				"--device", "cpu"}, tc.args...)
			sv, err := startServe(append(args, dir), &stdout, &stderr)
			if err != nil {
				t.Fatalf("startServe: %v", err)
			}
			defer func() {
				sv.ln.Close()
				sv.release()
			}()
			out := stdout.String()
			for _, want := range tc.want {
				if !strings.Contains(out, want) {
					t.Errorf("the report does not carry %q:\n%s", want, out)
				}
			}
		})
	}
}

// TestServeRefusesAPoolLargerThanTheDeviceHolds is 019-D2 at the command line:
// every pooled session's cache is reserved at startup, so a pool that does not
// fit is refused before the bind rather than part way through the allocation.
func TestServeRefusesAPoolLargerThanTheDeviceHolds(t *testing.T) {
	useCPUDevice(t)
	dir := syntheticDir(t)
	useFakeServable(t, fakeInfo(1024))

	var stdout, stderr strings.Builder
	_, err := startServe([]string{"--addr", "127.0.0.1:0", "--context", "1024",
		"--device", "cpu", "--sessions", "100000", dir}, &stdout, &stderr)
	if err == nil {
		t.Fatal("a pool of a hundred thousand sessions was accepted")
	}
	for _, want := range []string{"--sessions 100000", "lower --sessions"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal %q does not say %q", err, want)
		}
	}
	if stdout.Len() != 0 {
		t.Errorf("a server that refused its pool still printed a report: %q", stdout.String())
	}
}

// TestServeRefusesANegativeSessionCount: zero is "as many as fit", and a
// negative is a pool that holds no conversation.
func TestServeRefusesANegativeSessionCount(t *testing.T) {
	_, err := parseServe([]string{"--sessions", "-1", "models/x"})
	if err == nil || !errors.Is(err, errUsage) {
		t.Fatalf("parseServe = %v, want a usage error", err)
	}
	if !strings.Contains(err.Error(), "at least one") {
		t.Errorf("the refusal %q does not say what a pool holds", err)
	}
}

// TestServePassesThePrefixCacheFlagToTheLoader: the flag reaches the option
// that turns reuse on, rather than being parsed and dropped.
func TestServePassesThePrefixCacheFlagToTheLoader(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want tgo.CacheScope
	}{
		{nil, tgo.CacheOff},
		// Bare, which has to keep meaning the session scope and has to not
		// swallow the model directory after it.
		{[]string{"--prefix-cache"}, tgo.CacheSession},
		{[]string{"--prefix-cache=off"}, tgo.CacheOff},
		{[]string{"--prefix-cache=session"}, tgo.CacheSession},
		{[]string{"--prefix-cache=process"}, tgo.CacheProcess},
	} {
		o, err := parseServe(append(append([]string(nil), tc.args...), "models/x"))
		if err != nil {
			t.Fatalf("parseServe(%v): %v", tc.args, err)
		}
		if o.Engine.PrefixCache != tc.want {
			t.Errorf("parseServe(%v).Engine.PrefixCache = %v, want %v",
				tc.args, o.Engine.PrefixCache, tc.want)
		}
		if tc.args != nil && o.Dir != "models/x" {
			t.Errorf("parseServe(%v).Dir = %q; the bare flag swallowed the model "+
				"directory", tc.args, o.Dir)
		}
	}
	if _, err := parseServe([]string{"--prefix-cache=everything", "models/x"}); err == nil {
		t.Error("an unknown scope was accepted")
	} else if !strings.Contains(err.Error(), "off, session or process") {
		t.Errorf("the refusal does not name the scopes: %v", err)
	}
}

// TestServePoolSizeIsTheAdmissionLimit is
// specs/019-session-affinity.md §4's "two numbers that must agree", asserted
// rather than only commented.
//
// The pool reserves N sessions' key/value cache at startup and the admitter
// lets N requests generate at once, and [startServe] arrives at both from one
// call to [kvAdmission]. A pool built one larger than the admission limit is
// memory an operator was never told about -- the report prints what the limit
// costs -- and one built smaller puts requests inside the engine waiting for a
// session, where the queue neither counts them nor times them out (§8.6).
// Neither shows in the report, so it is read from the size the pool was
// actually built with.
func TestServePoolSizeIsTheAdmissionLimit(t *testing.T) {
	useCPUDevice(t)
	dir := syntheticDir(t)
	info := fakeInfo(1024)

	var built *fakeServerEngine
	prev := openServable
	openServable = func(_, name string, o engineOptions) (servable, error) {
		return servable{
			Pool: func(sessions int) (server.Engine, error) {
				built = &fakeServerEngine{
					name: name, context: o.Context, vocab: fakeVocabSize,
					perSession: info.CacheBytesPerSession, sessions: sessions,
				}
				return built, nil
			},
			Info:  info,
			Close: func() error { return nil },
		}, nil
	}
	t.Cleanup(func() { openServable = prev })

	// Three sizes: the default, and two an operator asked for, neither of them
	// the default, so a pool wired to the constant rather than to the
	// arithmetic reads wrong on two of the three rows.
	for _, tc := range []struct {
		args []string
		want int
	}{
		{nil, defaultSessions},
		{[]string{"--sessions", "1"}, 1},
		{[]string{"--sessions", "3"}, 3},
	} {
		func() {
			built = nil
			var stdout, stderr strings.Builder
			args := append([]string{"--addr", "127.0.0.1:0", "--context", "1024",
				"--device", "cpu"}, tc.args...)
			sv, err := startServe(append(args, dir), &stdout, &stderr)
			if err != nil {
				t.Fatalf("startServe(%v): %v", tc.args, err)
			}
			defer func() {
				sv.ln.Close()
				sv.release()
			}()
			if built == nil {
				t.Fatalf("startServe(%v) built no pool", tc.args)
			}
			if built.sessions != tc.want {
				t.Errorf("startServe(%v) reserved a pool of %d, want %d",
					tc.args, built.sessions, tc.want)
			}
			if got := sv.srv.Concurrency(); got != built.sessions {
				t.Errorf("startServe(%v) admits %d requests at once over a pool of %d; "+
					"the two are one number", tc.args, got, built.sessions)
			}
		}()
	}
}

// TestServeBatchedImpliesTheProcessScope is 022-D1 at the command line: the
// batched path and the process scope are one configuration, because a batch
// refuses a model with no shared block pool.
//
// So --batched sets the scope rather than failing later with an error about a
// pool the operator never mentioned, and an operator who asked for a different
// scope is told the two cannot both hold rather than having theirs overwritten.
func TestServeBatchedImpliesTheProcessScope(t *testing.T) {
	for _, tc := range []struct {
		name  string
		args  []string
		scope tgo.CacheScope
	}{
		{"bare", []string{"--batched"}, tgo.CacheProcess},
		{"asked for the same scope", []string{"--batched", "--prefix-cache=process"},
			tgo.CacheProcess},
		{"not asked for at all", nil, tgo.CacheOff},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o, err := parseServe(append(tc.args, filepath.Join("models", "qwen3")))
			if err != nil {
				t.Fatalf("parseServe(%v): %v", tc.args, err)
			}
			if o.Engine.PrefixCache != tc.scope {
				t.Errorf("scope = %v, want %v", o.Engine.PrefixCache, tc.scope)
			}
			if want := len(tc.args) > 0; o.Engine.Batched != want {
				t.Errorf("Batched = %v, want %v", o.Engine.Batched, want)
			}
		})
	}
	for _, args := range [][]string{
		{"--batched", "--prefix-cache=session"},
		{"--batched", "--prefix-cache=off"},
	} {
		if _, err := parseServe(append(args, "d")); err == nil {
			t.Errorf("parseServe(%v) was accepted; a batch has no shared pool under "+
				"either of those scopes", args)
		}
	}
}

// TestServeReserveFitsTheContext is what keeps §3's promise payable on a short
// context: a reserve larger than the context admits nobody, because
// ceil((T+R)/B) is then more blocks than one sequence's share of the pool.
func TestServeReserveFitsTheContext(t *testing.T) {
	for _, c := range []struct{ context, want int }{
		{4096, tgo.DefaultReserve},
		{1024, tgo.DefaultReserve},
		{256, 128},
		{32, tgo.CacheBlock},
		{8, tgo.CacheBlock},
	} {
		if got := serveReserve(c.context); got != c.want {
			t.Errorf("serveReserve(%d) = %d, want %d", c.context, got, c.want)
		}
	}
}
