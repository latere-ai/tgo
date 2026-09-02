// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.design/x/accel"

	tgo "github.com/latere-ai/tgo"
	"github.com/latere-ai/tgo/model"
	"github.com/latere-ai/tgo/weights"
)

func TestRunDispatch(t *testing.T) {
	useCPUDevice(t)
	dir := syntheticDir(t)

	var stdout, stderr strings.Builder
	if err := run([]string{"info", dir}, &stdout, &stderr); err != nil {
		t.Fatalf("run info: %v", err)
	}
	if !strings.Contains(stdout.String(), "architecture") {
		t.Errorf("info wrote nothing recognisable:\n%s", stdout.String())
	}

	// The other two arms dispatch to the same parsers, against a fake engine:
	// what is under test is that `run` reaches them at all, since a switch that
	// sent bench to cmdRun would pass every test in this package but one.
	stdout.Reset()
	useFakeEngine(t, &fakeEngine{promptTokens: 2, ttft: time.Millisecond})
	if err := run([]string{"run", "--max-tokens", "2", dir}, &stdout, &stderr); err != nil {
		t.Fatalf("run run: %v", err)
	}
	if !strings.Contains(stdout.String(), "t0") {
		t.Errorf("run did not stream:\n%s", stdout.String())
	}
	stdout.Reset()
	if err := run([]string{"bench", "--tokens", "2", "--prompt-tokens", "2", "--warmup", "0", dir},
		&stdout, &stderr); err != nil {
		t.Fatalf("run bench: %v", err)
	}
	if !strings.Contains(stdout.String(), "# tgo bench") {
		t.Errorf("bench did not write the table:\n%s", stdout.String())
	}

	// serve and pull, through their own refusals: what is under test here is
	// the switch, so each arm only has to reach a message no other arm
	// produces. A `serve` case that fell through to the default would report an
	// unknown command, and one wired to cmdInfo would report a model directory
	// that could not be read.
	stdout.Reset()
	useFakeServable(t, fakeInfo(defaultContext))
	err := run([]string{"serve", "--addr", "0.0.0.0:8080", dir}, &stdout, &stderr)
	if err == nil || !strings.Contains(err.Error(), "--public") {
		t.Errorf("run serve = %v, want 009-D8's refusal from cmdServe", err)
	}
	stdout.Reset()
	err = run([]string{"pull", "./not-a-repo-id"}, &stdout, &stderr)
	if err == nil || !strings.Contains(err.Error(), "Hugging Face repo id") {
		t.Errorf("run pull = %v, want cmdPull's refusal", err)
	}

	stdout.Reset()
	if err := run([]string{"help"}, &stdout, &stderr); err != nil {
		t.Fatalf("run help: %v", err)
	}
	for _, want := range []string{"tgo run", "tgo bench", "--json", "--precision"} {
		if !strings.Contains(stdout.String(), strings.TrimPrefix(want, "tgo ")) {
			t.Errorf("the usage does not mention %q:\n%s", want, stdout.String())
		}
	}
}

func TestRunRefusals(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"no command", nil, "no command"},
		// A word that is not a command and never will be: "serve" used to
		// stand here, and it became one.
		{"an unknown command", []string{"quantize", "d"}, "unknown command \"quantize\""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr strings.Builder
			err := run(tc.args, &stdout, &stderr)
			if err == nil {
				t.Fatalf("run(%v) was accepted", tc.args)
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

// TestInfoRefusesAnUnreadableDirectory is the first thing a user gets wrong: a
// path that is not a model. The refusal names the file it could not read rather
// than reporting an empty model.
func TestInfoRefusesAnUnreadableDirectory(t *testing.T) {
	useCPUDevice(t)
	var stdout, stderr strings.Builder
	err := cmdInfo([]string{filepath.Join(t.TempDir(), "nope")}, &stdout, &stderr)
	if err == nil {
		t.Fatal("a directory with no config.json was accepted")
	}
	if !strings.Contains(err.Error(), "config") {
		t.Errorf("error = %v, want one naming the file it could not read", err)
	}
	if stdout.Len() != 0 {
		t.Errorf("a refused command wrote to stdout: %q", stdout.String())
	}
}

// TestInfoRefusesAModelThatIsNotAModel covers the other unreadable case: a
// config.json that parses as JSON and is not a model this build knows.
func TestInfoRefusesAnUnknownArchitecture(t *testing.T) {
	useCPUDevice(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"),
		[]byte(`{"architectures":["LlamaForCausalLM"]}`), 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	var stdout, stderr strings.Builder
	err := cmdInfo([]string{dir}, &stdout, &stderr)
	if err == nil || !strings.Contains(err.Error(), "Llama") {
		t.Fatalf("cmdInfo = %v, want a refusal naming the architecture", err)
	}
}

func TestInfoFlags(t *testing.T) {
	useCPUDevice(t)
	dir := syntheticDir(t)
	var stdout, stderr strings.Builder
	if err := cmdInfo([]string{"--context", "1024", "--precision", "int8", "--budget", "1073741824", dir},
		&stdout, &stderr); err != nil {
		t.Fatalf("cmdInfo: %v", err)
	}
	out := stdout.String()
	for _, want := range []string{"precision  int8", "at 1024 positions", "1.00 GiB"} {
		if !strings.Contains(out, want) {
			t.Errorf("the description does not carry %q:\n%s", want, out)
		}
	}
	// A bad flag is refused before the device is opened.
	if err := cmdInfo([]string{"--precision", "bf16", dir}, &stdout, &stderr); err == nil {
		t.Fatal("--precision bf16 was accepted")
	}
	if err := cmdInfo([]string{"--context", "0", dir}, &stdout, &stderr); err == nil {
		t.Fatal("--context 0 was accepted")
	}
}

// TestRealCheckpoint reads the checkpoint TGO_MODEL names.
//
// It is skipped by default (specs/000-decisions.md decision 8): the smallest
// Qwen3 is over a gigabyte, and a CI that downloads one is a CI nobody runs
// locally. It stays in the tree because every number above is computed from a
// config this repository wrote, and this is the one place the arithmetic meets
// a model it did not choose.
func TestRealCheckpoint(t *testing.T) {
	dir := os.Getenv("TGO_MODEL")
	if dir == "" {
		t.Skip("TGO_MODEL is not set; this test reads a real checkpoint")
	}
	useCPUDevice(t)
	var stdout, stderr strings.Builder
	if err := cmdInfo([]string{"--context", "4096", dir}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdInfo on %s: %v", dir, err)
	}
	out := stdout.String()
	t.Logf("\n%s", out)
	for _, want := range []string{"Qwen3ForCausalLM", "head_dim 128", "tied checkpoint", "kv cache"} {
		if !strings.Contains(out, want) {
			t.Errorf("the description does not carry %q:\n%s", want, out)
		}
	}
	// Qwen3-0.6B is 596 million parameters over 310 distinct tensors, and the
	// tied head makes a 311th plane on the device.
	rep, err := openAndDescribe(dir, describeOptions{Context: 4096})
	if err != nil {
		t.Fatalf("openAndDescribe: %v", err)
	}
	if rep.Model.Parameters < 500e6 || rep.Model.Parameters > 700e6 {
		t.Errorf("parameters = %d, want Qwen3-0.6B's ~596 million", rep.Model.Parameters)
	}
	if rep.Model.Planes <= rep.Model.Parameters {
		t.Error("the tied head is not counted as a second device plane")
	}
	// Two bytes for every element the loader narrows, four for every norm gain
	// (specs/004-model-graph.md §3 declares the gain ports f32 and the engine
	// uploads them there), so the footprint is strictly above two bytes per
	// device element and equal to the sum over the planes.
	b, err := model.Open(dir)
	if err != nil {
		t.Fatalf("model.Open: %v", err)
	}
	var want int64
	for _, sp := range b.Weights() {
		n := int64(1)
		for _, d := range sp.Shape {
			n *= int64(d)
		}
		want += planeBytes(sp.Kind, n, weights.F16)
	}
	if rep.Precision.F16Bytes != want {
		t.Errorf("f16 footprint = %d, want %d", rep.Precision.F16Bytes, want)
	}
	if rep.Precision.F16Bytes <= rep.Model.Planes*2 {
		t.Error("the f16 footprint is two bytes per element throughout; the gains are f32")
	}
	// specs/005-kv-cache.md §3: L=28, H_kv=8, d_h=128 is 2·28·8·128 = 57344
	// elements per position, 224 KiB in f32.
	if want := int64(224 * 1024); rep.Memory.KVBytesPerPosition != want {
		t.Errorf("kv per position = %d, want %d", rep.Memory.KVBytesPerPosition, want)
	}
}

// TestDeviceFlagReachesBothTheDescriptionAndTheEngine pins the wiring that
// makes --device mean one thing. The command describes the machine by opening a
// device itself and the engine opens another, and a flag that reached only one
// of them would print one backend's limits beside another backend's numbers.
func TestDeviceFlagReachesBothTheDescriptionAndTheEngine(t *testing.T) {
	asked := useCPUDevice(t)
	var opened []engineOptions
	prev := openEngine
	openEngine = func(dir string, o engineOptions) (engine, error) {
		opened = append(opened, o)
		return &fakeEngine{promptTokens: 1}, nil
	}
	t.Cleanup(func() { openEngine = prev })

	var stdout, stderr strings.Builder
	if err := cmdRun([]string{"--device", "cpu", "--max-tokens", "1", syntheticDir(t)},
		&stdout, &stderr); err != nil {
		t.Fatalf("cmdRun: %v", err)
	}
	if len(*asked) != 1 || (*asked)[0] != tgo.CPU {
		t.Errorf("the description opened %v, want the CPU the flag named", *asked)
	}
	if len(opened) != 1 || opened[0].Device != tgo.CPU {
		t.Errorf("the engine was opened with %v", opened)
	}
	// And an unknown device is refused before anything is opened.
	before := len(*asked)
	if err := cmdRun([]string{"--device", "cuda", syntheticDir(t)}, &stdout, &stderr); err == nil {
		t.Fatal("--device cuda was accepted")
	}
	if len(*asked) != before {
		t.Error("a refused device flag still opened a device")
	}
	if _, err := parseBench([]string{"--device", "cuda", "d"}); err == nil {
		t.Fatal("bench accepted --device cuda")
	}
	if err := cmdInfo([]string{"--device", "cuda", syntheticDir(t)}, &stdout, &stderr); err == nil {
		t.Fatal("info accepted --device cuda")
	}
	// The default asks for auto, which is what a machine with a GPU should get.
	o, err := parseBench([]string{"d"})
	if err != nil {
		t.Fatalf("parseBench: %v", err)
	}
	if o.Engine.Device != tgo.AutoDevice {
		t.Errorf("the default device is %v, want auto", o.Engine.Device)
	}
}

// TestUsageDocumentsEveryFlag holds the usage text against the flags the three
// commands declare, in both directions.
//
// A flag that exists and is undocumented is a flag nobody finds, and a
// documented flag that does not exist is a refusal a user cannot explain. The
// drift is silent otherwise: every other test in this package types the flag it
// means to exercise, so none of them ever reads the usage.
func TestUsageDocumentsEveryFlag(t *testing.T) {
	for _, tc := range []struct {
		command string
		set     *flag.FlagSet
	}{
		{"run", firstOf(runFlagSet())},
		{"bench", firstOf(benchFlagSet())},
		{"serve", firstOf(serveFlagSet())},
		{"info", firstOf(infoFlagSet())},
		{"pull", firstOf(pullFlagSet())},
	} {
		t.Run(tc.command, func(t *testing.T) {
			documented := usageFlags(t, tc.command)
			declared := map[string]bool{}
			tc.set.VisitAll(func(f *flag.Flag) {
				declared[f.Name] = true
				if !documented[f.Name] {
					t.Errorf("`tgo %s` accepts --%s and the usage does not mention it", tc.command, f.Name)
				}
			})
			for name := range documented {
				if !declared[name] {
					t.Errorf("the usage documents `tgo %s --%s`, which the command does not accept",
						tc.command, name)
				}
			}
		})
	}
}

// firstOf drops the value struct a flag-set constructor returns.
func firstOf(fs *flag.FlagSet, _ any) *flag.FlagSet { return fs }

// usageFlags reads one command's flag names out of the usage text.
func usageFlags(t *testing.T, command string) map[string]bool {
	t.Helper()
	_, rest, ok := strings.Cut(usage, command+" flags:\n")
	if !ok {
		t.Fatalf("the usage has no %q section", command+" flags:")
	}
	section, _, _ := strings.Cut(rest, "\n\n")
	found := map[string]bool{}
	for line := range strings.SplitSeq(section, "\n") {
		name, _, ok := strings.Cut(strings.TrimPrefix(strings.TrimSpace(line), "--"), " ")
		if !ok || !strings.HasPrefix(strings.TrimSpace(line), "--") {
			continue
		}
		found[name] = true
	}
	if len(found) == 0 {
		t.Fatalf("the %s section documents no flags", command)
	}
	return found
}

// TestDefaultPromptIsWhatTheUsageSays: the usage describes the default rather
// than printing it, so the description is the thing that can go stale.
func TestDefaultPromptIsWhatTheUsageSays(t *testing.T) {
	if !strings.Contains(defaultPrompt, "transformer") {
		t.Errorf("the usage calls the default prompt a question about transformers and it is %q",
			defaultPrompt)
	}
	if !strings.HasSuffix(defaultPrompt, "?") {
		t.Errorf("the default prompt %q is not a question", defaultPrompt)
	}
}

// TestCommandsReportADeviceThatWillNotOpen is the refusal a user gets on a
// machine whose accelerator is absent or busy, and the one --device makes
// reachable on purpose: a named device is refused rather than falling back,
// because a user who named one had a reason to.
func TestCommandsReportADeviceThatWillNotOpen(t *testing.T) {
	prev := openDevice
	openDevice = func(want tgo.Device) (*accel.Device, error) { return nil, errNoDevice }
	t.Cleanup(func() { openDevice = prev })
	useFakeEngine(t, &fakeEngine{promptTokens: 1})

	dir := syntheticDir(t)
	for name, fn := range map[string]func(args []string, stdout, stderr io.Writer) error{
		"info":  cmdInfo,
		"run":   cmdRun,
		"bench": cmdBench,
	} {
		t.Run(name, func(t *testing.T) {
			var stdout, stderr strings.Builder
			err := fn([]string{"--tokens", "1", "--prompt-tokens", "1", "--max-tokens", "1", dir},
				&stdout, &stderr)
			// run and info do not take --tokens, so each command is given the
			// arguments it accepts.
			if errors.Is(err, errUsage) {
				err = fn([]string{dir}, &stdout, &stderr)
			}
			if !errors.Is(err, errNoDevice) {
				t.Fatalf("`tgo %s` with no device = %v, want the device error", name, err)
			}
			if stdout.Len() != 0 {
				t.Errorf("a command that could not open a device wrote to stdout: %q", stdout.String())
			}
		})
	}
}

// errNoDevice stands in for a machine whose accelerator will not open.
var errNoDevice = errors.New("no device would open")
