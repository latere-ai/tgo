// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"fmt"

	"github.com/latere-ai/tgo"
)

// scopeFlag is --prefix-cache: off, session or process.
//
// A flag with three values rather than a switch with two, because what the
// operator is deciding is *what may share with what* and that is one dimension
// with three settings, not a boolean with a special case. It was a boolean
// while the process scope was refused; making it a boolean again now would mean
// spelling the third setting as a second flag, and two flags for one dimension
// is two things to keep consistent.
//
// It reports itself as a boolean flag to the flag package, so `--prefix-cache`
// alone still means the session scope and does not swallow the model directory
// that follows it.
type scopeFlag struct {
	scope tgo.CacheScope
	set   bool
}

// IsBoolFlag lets `--prefix-cache` stand alone.
func (s *scopeFlag) IsBoolFlag() bool { return true }

func (s *scopeFlag) String() string {
	if s == nil {
		return "off"
	}
	return s.scope.String()
}

func (s *scopeFlag) Set(v string) error {
	s.set = true
	switch v {
	case "true", "session":
		s.scope = tgo.CacheSession
	case "false", "off":
		s.scope = tgo.CacheOff
	case "process":
		s.scope = tgo.CacheProcess
	default:
		return fmt.Errorf("%q is not a prefix-cache scope; it is off, session or process "+
			"(bare --prefix-cache is session)", v)
	}
	return nil
}
