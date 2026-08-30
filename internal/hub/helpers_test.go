// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package hub

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// newStatusServer answers every request with one status and body, which is how
// the API's refusals are staged without a whole fake repo behind them.
func newStatusServer(t *testing.T, code int, body string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(code)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}
