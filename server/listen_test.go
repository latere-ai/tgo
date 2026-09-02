// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"strings"
	"testing"
)

// §7 and 009-D8. This server has no authentication, so it does not reach the
// network by omission: a non-loopback bind needs the flag, and taking it prints
// a line saying what was just exposed.

func TestANonLoopbackBindIsRefusedWithoutTheFlag(t *testing.T) {
	t.Parallel()
	for _, addr := range []string{"0.0.0.0:0", ":0", "[::]:0", "192.0.2.1:0",
		"an.example.test:0"} {
		t.Run(addr, func(t *testing.T) {
			t.Parallel()
			s := newTestServer(t, &fakeEngine{})
			ln, err := s.Listen(addr)
			if err == nil {
				_ = ln.Close()
				t.Fatalf("Listen(%q) bound an address reachable from the network", addr)
			}
			if !strings.Contains(err.Error(), "WithPublicBind") ||
				!strings.Contains(err.Error(), "authentication") {
				t.Errorf("the refusal does not say what to do or why: %v", err)
			}
		})
	}
}

func TestALoopbackBindNeedsNoFlag(t *testing.T) {
	t.Parallel()
	for _, addr := range []string{"127.0.0.1:0", "localhost:0", "[::1]:0"} {
		t.Run(addr, func(t *testing.T) {
			t.Parallel()
			s := newTestServer(t, &fakeEngine{})
			ln, err := s.Listen(addr)
			if err != nil {
				// A machine with no IPv6 cannot bind [::1], and that is the
				// machine's answer rather than this package's.
				if strings.Contains(err.Error(), "WithPublicBind") {
					t.Fatalf("Listen(%q) refused a loopback address", addr)
				}
				t.Skipf("this machine cannot bind %s: %v", addr, err)
			}
			_ = ln.Close()
		})
	}
}

func TestAPublicBindSaysWhatItJustDid(t *testing.T) {
	t.Parallel()
	var notice strings.Builder
	s, err := New(&fakeEngine{}, WithPublicBind(), WithNotice(&notice))
	if err != nil {
		t.Fatal(err)
	}
	ln, err := s.Listen("0.0.0.0:0")
	if err != nil {
		t.Fatalf("Listen with WithPublicBind: %v", err)
	}
	defer func() { _ = ln.Close() }()
	for _, want := range []string{"no authentication", "reachable from the network"} {
		if !strings.Contains(notice.String(), want) {
			t.Errorf("the notice does not say %q: %q", want, notice.String())
		}
	}
}

func TestAnAddressThatIsNotOneIsAnError(t *testing.T) {
	t.Parallel()
	s := newTestServer(t, &fakeEngine{})
	if _, err := s.Listen("not-a-host-port"); err == nil {
		t.Fatal("Listen accepted an address with no port")
	}
	if _, err := s.Listen("127.0.0.1:not-a-port"); err == nil {
		t.Fatal("Listen accepted a port that is not one")
	}
}

// The default is loopback, which is the whole of 009-D8.
func TestTheDefaultAddressIsLoopback(t *testing.T) {
	t.Parallel()
	public, err := isPublic(DefaultAddr)
	if err != nil {
		t.Fatal(err)
	}
	if public {
		t.Errorf("DefaultAddr = %q, which is reachable from the network", DefaultAddr)
	}
	// And an empty address resolves to it rather than to the wildcard. The
	// bind itself may fail on a machine already serving that port, which is
	// that machine's answer and not this rule's.
	s := newTestServer(t, &fakeEngine{})
	ln, err := s.Listen("")
	if err == nil {
		_ = ln.Close()
		return
	}
	if strings.Contains(err.Error(), "WithPublicBind") {
		t.Errorf("an empty address was treated as a public bind: %v", err)
	}
}

// A nil notice writer silences the lines rather than panicking on them.
func TestANilNoticeIsSilent(t *testing.T) {
	t.Parallel()
	s, err := New(&fakeEngine{}, WithNotice(nil))
	if err != nil {
		t.Fatal(err)
	}
	s.notice("this goes nowhere")
}
