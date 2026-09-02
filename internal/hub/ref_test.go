// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package hub

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseRefReadsIdsAndPaths(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want Ref
	}{
		{"Qwen/Qwen3-0.6B", Ref{Org: "Qwen", Repo: "Qwen3-0.6B", Revision: "main"}},
		{"Qwen/Qwen3-0.6B@refs/pr/1", Ref{Org: "Qwen", Repo: "Qwen3-0.6B", Revision: "refs/pr/1"}},
		{"acme/qwen-mini@0f1e2d", Ref{Org: "acme", Repo: "qwen-mini", Revision: "0f1e2d"}},
		{"gpt2", Ref{Org: noOrg, Repo: "gpt2", Revision: "main"}},
		{"./weights", Ref{Local: filepath.Clean("./weights")}},
		{"/opt/models/qwen", Ref{Local: filepath.FromSlash("/opt/models/qwen")}},
		{"../sibling/dir", Ref{Local: filepath.FromSlash("../sibling/dir")}},
		{"~/models/qwen", Ref{Local: filepath.FromSlash("~/models/qwen")}},
		{`C:\models\qwen`, Ref{Local: filepath.FromSlash(`C:\models\qwen`)}},
		{"a/b/c", Ref{Local: filepath.FromSlash("a/b/c")}},
	} {
		got, err := ParseRef(tc.in)
		if err != nil {
			t.Fatalf("ParseRef(%q): %v", tc.in, err)
		}
		if got != tc.want {
			t.Errorf("ParseRef(%q) = %+v, want %+v", tc.in, got, tc.want)
		}
	}
}

func TestParseRefRefusesWhatCannotBeADirectory(t *testing.T) {
	// Each component becomes a directory name under the cache, so a component
	// that is not a name is refused where it is read rather than where it is
	// joined.
	for _, in := range []string{
		"", "   ", "acme/", "@main", "acme/qwen@", "acme/qwen mini", "ac me/qwen",
		"acme/qwen:mini", "acme/qwen%20mini", "a/..",
	} {
		if _, err := ParseRef(in); err == nil {
			t.Errorf("ParseRef(%q) was accepted", in)
		}
	}
}

func TestRefRendersTheWayItIsRead(t *testing.T) {
	r, err := ParseRef("acme/qwen-mini@dev")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := r.ID(), "acme/qwen-mini"; got != want {
		t.Errorf("ID() = %q, want %q", got, want)
	}
	if got, want := r.String(), "acme/qwen-mini@dev"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
	if r.IsLocal() {
		t.Error("a repo id reported itself local")
	}

	bare, err := ParseRef("gpt2")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := bare.ID(), "gpt2"; got != want {
		t.Errorf("bare ID() = %q, want %q", got, want)
	}

	local, err := ParseRef("./weights")
	if err != nil {
		t.Fatal(err)
	}
	if !local.IsLocal() || local.String() != filepath.Clean("./weights") {
		t.Errorf("local ref = %+v", local)
	}
}

func TestSafePathRefusesEverythingThatEscapes(t *testing.T) {
	// A listing is a server's word for what a repo holds, so it is untrusted.
	for _, in := range []string{
		"", "/etc/passwd", "../../../.ssh/authorized_keys", "..",
		"a/../../b", `dir\file.json`, "./config.json", "a//b.json",
		"trailing/", "bad\x00name.json", "ctrl\nname.json",
	} {
		err := safePath(in)
		if err == nil {
			t.Errorf("safePath(%q) was accepted", in)
			continue
		}
		if !errors.Is(err, ErrUnsafePath) {
			t.Errorf("safePath(%q) = %v, want ErrUnsafePath", in, err)
		}
	}
	for _, in := range []string{
		"config.json", "model-00001-of-00003.safetensors", "onnx/model.onnx",
	} {
		if err := safePath(in); err != nil {
			t.Errorf("safePath(%q) = %v", in, err)
		}
	}
}

func TestCacheDirPrefersTgoCacheThenXdgThenHome(t *testing.T) {
	// t.Setenv forbids t.Parallel, which is why this is one test with three
	// phases rather than three parallel ones.
	t.Setenv("TGO_CACHE", filepath.FromSlash("/var/tgo"))
	t.Setenv("XDG_CACHE_HOME", filepath.FromSlash("/var/xdg"))
	got, err := CacheDir()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.FromSlash("/var/tgo"); got != want {
		t.Errorf("with TGO_CACHE: %q, want %q", got, want)
	}

	t.Setenv("TGO_CACHE", "")
	got, err = CacheDir()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(filepath.FromSlash("/var/xdg"), "tgo"); got != want {
		t.Errorf("with XDG_CACHE_HOME: %q, want %q", got, want)
	}

	t.Setenv("XDG_CACHE_HOME", "")
	got, err = CacheDir()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(got, filepath.Join(".cache", "tgo")) {
		t.Errorf("with neither: %q, want a path ending in .cache/tgo", got)
	}
}
