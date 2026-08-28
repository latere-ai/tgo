// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"latere.ai/x/pkg/llmdialect"
	"latere.ai/x/pkg/llmdialect/anthropic"
	"latere.ai/x/pkg/llmdialect/ir"
	"latere.ai/x/pkg/llmdialect/openaichat"
	"latere.ai/x/pkg/llmdialect/openairesp"
)

// pooledEngine is the part of [PoolEngine] that [New] needs and [Engine] does
// not declare: how many sessions the engine can hand out at once.
//
// An optional interface rather than a method on [Engine], because an engine
// that allocates per request ([Wrap]) has no such number -- what bounds it is
// device memory, which [WithKVBudget] divides.
type pooledEngine interface{ Sessions() int }

// queuedEngine is an engine that does its own admission, counted and bounded.
//
// It is the difference [019 §8.6](../specs/019-session-affinity.md) turns on. A
// pooled engine makes the surplus above its pool wait inside NewSession, where
// this package neither counts it nor times it out — so the Retry-After stops
// describing what that request waits, and [New] refuses the configuration. An
// engine that queues answers for its own wait: it bounds the depth, it bounds
// the budget, and it reports both, so the surplus is a wait the deployment can
// see and a 429 it can predict.
type queuedEngine interface {
	// AdmissionWait is the budget past which a queued request is refused,
	// which is the only interval a Retry-After can promise.
	AdmissionWait() time.Duration

	// AdmissionDepth is how many requests may wait at once.
	AdmissionDepth() int
}

// Server serves one [Engine] over the four request routes and the three
// informational ones. It is an [http.Handler]; nothing here starts a listener
// except [Server.Listen], which exists to refuse a public bind.
type Server struct {
	eng     Engine
	opt     options
	mux     *http.ServeMux
	admit   *admitter
	metrics *metrics
}

// New builds a server over one model.
//
// The routes are fixed: this package serves one model and does no routing
// (009-D5). Everything a deployment can change is an [Option], and each one
// that can be wrong is checked here rather than at the first request.
func New(eng Engine, opts ...Option) (*Server, error) {
	if eng == nil {
		return nil, fmt.Errorf("server: an engine is required")
	}
	o := defaults()
	for _, opt := range opts {
		if err := opt(&o); err != nil {
			return nil, err
		}
	}
	if o.kvBudget > 0 {
		per := eng.CacheBytesPerSession()
		if per <= 0 {
			return nil, fmt.Errorf("server: the engine reports no cache size, so a KV budget " +
				"cannot be divided into a concurrency; use WithConcurrency")
		}
		n := int(o.kvBudget / per)
		if n < 1 {
			return nil, fmt.Errorf("server: a KV budget of %d bytes does not hold one "+
				"session's cache, which is %d bytes", o.kvBudget, per)
		}
		o.concurrency = n
	}
	// A pooled engine has a semaphore of its own: Pool.Acquire waits when
	// every pooled session is leased. Admitting more requests than the pool
	// holds is bounded -- the surplus is concurrency minus the pool, and the
	// requests behind them still queue and still get their 429 -- but it is a
	// wait the admitter does not describe. Those requests hold a slot and
	// block inside NewSession, where they are not in the queue, so
	// [WithQueueWait] does not time them out and the queue depth does not
	// count them; what their caller gets is a wait past the Retry-After budget
	// instead of the refusal that budget promises. Both numbers are known
	// here, so the disagreement is refused here.
	_, queues := eng.(queuedEngine)
	if p, ok := eng.(pooledEngine); ok && !queues && o.concurrency > p.Sessions() {
		return nil, fmt.Errorf("server: admission allows %d requests to generate at once and "+
			"the engine pools %d session(s); lower the concurrency or pool more sessions",
			o.concurrency, p.Sessions())
	}

	s := &Server{eng: eng, opt: o, mux: http.NewServeMux(), metrics: newMetrics()}
	s.admit = newAdmitter(o.concurrency, o.queue, o.queueWait, s.metrics)

	s.mux.Handle("POST /v1/chat/completions", s.dialect(openaichat.NewFrontend()))
	s.mux.Handle("POST /v1/messages", s.dialect(anthropic.NewFrontend()))
	s.mux.Handle("POST /v1/responses", s.dialect(openairesp.NewFrontend()))
	s.mux.Handle("POST /v1/completions", s.dialect(newLegacyFrontend()))
	s.mux.HandleFunc("GET /v1/models", s.models)
	s.mux.HandleFunc("GET /health", s.health)
	s.mux.HandleFunc("GET /metrics", s.exposeMetrics)
	return s, nil
}

// ServeHTTP routes one request.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

// Concurrency is how many sessions may generate at once, after
// [WithKVBudget] has been divided out. Over a pooled engine it is the pool's
// size, which [New] refuses to let it exceed. A deployment prints it, and a
// test reads it.
func (s *Server) Concurrency() int { return s.opt.concurrency }

// dialect is the pipeline every request route shares.
//
// One order, and each step is where it is because of what the step before it
// can refuse: the body is bounded before it is parsed, the raw members are
// refused before a frontend rejects them in its own words, the model is checked
// before a session is built, and admission is last because a request that will
// not run must not first take a KV reservation from one that would.
func (s *Server) dialect(front llmdialect.Frontend) http.Handler {
	d := front.Name()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req, aerr := s.decode(front, w, r)
		if aerr != nil {
			s.fail(w, d, aerr)
			return
		}
		// Before anything is written, because the list is known now and a
		// header set after WriteHeader is a header nobody receives.
		if len(req.loss) > 0 {
			w.Header().Set("X-Tgo-Loss", header(req.loss))
			s.metrics.lost(req.loss)
		}

		release, aerr := s.admit.acquire(r.Context())
		if aerr != nil {
			s.fail(w, d, aerr)
			return
		}
		defer release()

		s.metrics.enter(string(d))
		defer s.metrics.leave(string(d))
		s.generate(w, r, front, req)
	})
}

// decode turns a request body into a checked, mapped request.
func (s *Server) decode(front llmdialect.Frontend, w http.ResponseWriter,
	r *http.Request) (*request, *apiError) {

	d := front.Name()
	// The writer is handed to MaxBytesReader so that an oversized body closes
	// the connection rather than leaving the client writing into a request
	// nobody is reading.
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, s.opt.maxBody))
	if err != nil {
		return nil, badRequest("tgo: reading the request body: %v", err)
	}
	top, err := topLevel(body)
	if err != nil {
		return nil, badRequest("tgo: %v", err)
	}
	if aerr := refuseRaw(d, top); aerr != nil {
		return nil, aerr
	}
	req, err := front.DecodeRequest(body)
	if err != nil {
		return nil, badRequest("tgo: %v", err)
	}
	if req.Model != s.eng.Name() {
		return nil, &apiError{kind: errNotFound, field: "model", reason: "not_found",
			msg: fmt.Sprintf("tgo: this server serves %q and the request asked for %q; "+
				"it serves one model and does no routing", s.eng.Name(), req.Model)}
	}
	ex, err := parseExtras(top)
	if err != nil {
		return nil, badRequest("tgo: %v", err)
	}
	out, aerr := adapt(d, req, ex, keys(top), s.eng)
	if aerr != nil {
		return nil, aerr
	}
	if lf, ok := front.(*legacyFrontend); ok {
		if aerr := lf.finish(top, out); aerr != nil {
			return nil, aerr
		}
	}
	return out, nil
}

// fail records a refusal and answers it in the caller's dialect.
func (s *Server) fail(w http.ResponseWriter, d ir.Dialect, e *apiError) {
	if e.reason != "" {
		s.metrics.reject(e.reason)
	}
	writeError(w, d, e)
}

// notice prints a line nobody asked for and everybody needs: a session that
// would not close, an encoder that failed with the status already sent.
func (s *Server) notice(format string, args ...any) {
	if s.opt.notice == nil {
		return
	}
	fmt.Fprintf(s.opt.notice, format+"\n", args...)
}

// models lists the one model this server serves.
func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id": s.eng.Name(), "object": "model", "created": 0, "owned_by": "tgo",
		}},
	})
}

// health says the process is up and which model it holds. It allocates
// nothing and touches no device, so it stays honest while the device is busy.
func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, map[string]any{
		"status": "ok", "model": s.eng.Name(), "context": s.eng.Context(),
		"concurrency": s.opt.concurrency,
	})
}

// exposeMetrics writes §6's series.
func (s *Server) exposeMetrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	s.metrics.write(w)
}

func writeJSON(w http.ResponseWriter, v any) {
	raw, err := json.Marshal(v)
	if err != nil {
		http.Error(w, `{"error":{"message":"tgo: the body could not be encoded"}}`,
			http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(raw)
}

// Listen opens the listener this server should be served on.
//
// It exists for one rule: an address that is not loopback needs
// [WithPublicBind], and taking it prints a line saying the server has no
// authentication (009-D8). A caller who builds their own listener has opted out
// of the check, which is why the check is here and not in [New].
func (s *Server) Listen(addr string) (net.Listener, error) {
	if addr == "" {
		addr = DefaultAddr
	}
	public, err := isPublic(addr)
	if err != nil {
		return nil, err
	}
	if public && !s.opt.public {
		return nil, fmt.Errorf("server: %s is not a loopback address and this server has no "+
			"authentication; pass WithPublicBind to bind it anyway", addr)
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	if public {
		s.notice("tgo: listening on %s, which is reachable from the network. This server "+
			"has no authentication and no rate limiting: anyone who can reach it can use "+
			"the model.", ln.Addr())
	}
	return ln, nil
}

// isPublic reports whether an address would be reachable from off the machine.
//
// An empty host is the wildcard, which is the case worth naming: ":8080" binds
// every interface and reads like it binds none.
func isPublic(addr string) (bool, error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false, fmt.Errorf("server: %q is not a host:port address: %w", addr, err)
	}
	if host == "" {
		return true, nil
	}
	if strings.EqualFold(host, "localhost") {
		return false, nil
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	if ip == nil {
		// A name this package cannot resolve to a loopback address is treated
		// as public: guessing the other way would be a hostname away from an
		// unauthenticated server on the network.
		return true, nil
	}
	if ip.IsUnspecified() {
		return true, nil
	}
	return !ip.IsLoopback(), nil
}
