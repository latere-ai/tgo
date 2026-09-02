// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

// Command tgo runs a model, measures it, and reports what it is.
//
//	tgo run   [--prompt P] [--max-tokens N] [--temp T] [--seed S] [--precision f16|int8|auto] <model-dir>
//	tgo bench [--tokens N] [--prompt-tokens N] [--batch N] [--json out.json] <model-dir>
//	tgo serve [--addr host:port] [--public] [--precision P] [--context C] [--slots N] [--kv P] [--prefix-cache S] [--batched] [--device D] <model-dir>
//	tgo info  <model-dir>
//	tgo pull  [--revision R] [--token T] <repo-id>
//
// The flags come before the model directory, which is [flag.FlagSet.Parse]'s
// rule and not a preference: parsing stops at the first argument that is not a
// flag, so a flag written after the directory is a second positional argument
// and is refused rather than applied.
//
// Every command that opens a model -- run, bench, serve and info -- takes
// --device auto|cpu|metal and --precision f16|int8|auto. `tgo pull` opens none
// and takes neither: it turns a repo id into a directory the other four can be
// pointed at. `tgo help` prints the whole surface, and
// TestUsageDocumentsEveryFlag holds it against the flags the five commands
// declare.
//
// The command is argument parsing and process wiring. Every number it prints is
// computed by a package under it -- model parses the config, bench aggregates
// the percentiles, weights decides the precision -- and what lives here is the
// decision of which of them to call and how to render the answer. That is why
// .lateregate.yaml exempts it from the coverage floor, so anything with a
// rule in it is a function in this package with a test, not a branch inside a
// flag handler.
//
// Six things are stated here because a reader of the
// output has to be able to check them:
//
//   - The precision choice is printed with the two footprints and the budget it
//     compared, never as a bare word (specs/001-weights.md §5). It is printed
//     after the model opens, so the word names what the loader resolved rather
//     than what this process predicted.
//   - `tgo info` computes that choice, specs/005-kv-cache.md §3's cache
//     arithmetic and the resident footprint from config.json and the declared
//     weight map alone. It loads no weights, which is why it answers in a third
//     of a second on a 1.4 GiB checkpoint, and it is pinned against a loaded
//     model by the TGO_MODEL tests in engine_test.go.
//   - Every benchmark number is printed with the hardware, the model, the
//     precision and the sampling policy, plus the Go version, GOOS/GOARCH and
//     the accel backend. A tokens-per-second figure without them is decoration
//     (017-D4).
//   - The batch axis is reported as an axis even though tgo does not batch yet.
//     It has one point, and the output says why rather than omitting the column
//     (017-D5, specs/008-scheduler.md).
//   - 017-D1's host/submit/device/readback breakdown is the deliverable of
//     `tgo bench`, and it is obtained: the engine takes the caller's recorder
//     through tgo.WithRecorder (017-D7, options.go), so every run here carries
//     the four terms. A record without them is a session opened without a
//     recorder, and it says so rather than marshalling a report of zeros that
//     reads as a measurement. See noBreakdownNote in record.go.
//   - `tgo serve` prints the admission limit with the three terms it was
//     divided out of, not as a bare count (specs/009-server.md §6). The
//     available memory in that arithmetic is the device's MaxPoolBytes, which
//     is a cap on one allocation rather than a report of free memory, so the
//     derivation is what lets an operator see that a 16 GiB machine admitting
//     one session is arithmetic and not a bug. See kvAdmission in serve.go.
package main

import (
	"errors"
	"fmt"
	"io"
	"os"
)

// errUsage marks a failure the user can fix by typing a different command, as
// opposed to one the machine produced. main prints the usage after it.
var errUsage = errors.New("usage")

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "tgo: %v\n", err)
		if errors.Is(err, errUsage) {
			fmt.Fprint(os.Stderr, usage)
		}
		os.Exit(1)
	}
}

// usage is the whole surface, in the order a new user meets it.
const usage = `
usage: tgo <command> [flags] <model-dir>
       tgo pull [flags] <repo-id>

commands:
  run     generate from a prompt, streaming tokens as they are produced
  bench   measure the host/submit/device/readback breakdown and write a report
  serve   serve the model over HTTP in three wire dialects
  info    print the architecture, the precision choice and what memory it costs
  pull    download a Hugging Face checkpoint and print where it landed

run flags:
  --prompt P          the prompt text (default: a question about transformers)
  --raw               send the prompt as typed, without the chat template
  --max-tokens N      stop after N generated tokens (default 128)
  --temp T            sampling temperature; 0 is greedy (default 0)
  --top-k K           keep the K largest candidates; 0 disables the stage
  --top-p P           nucleus mass in (0, 1]; 0 disables the stage
  --repeat-penalty R  divisive repetition penalty; 1 is none (default 1)
  --seed S            the sampler seed (default 0)
  --precision P       f16, int8, int4 or auto (default auto). Narrowing is a
                      last resort: auto takes the widest that fits, and int4 is
                      not uniformly more accurate than int8
  --context C         KV cache capacity in positions (default 4096)
  --device D          auto, cpu or metal (default auto)

bench flags:
  --tokens N          decode steps to measure (default 128)
  --prompt-tokens N   synthetic prompt length in tokens (default 128)
  --batch N           sequences in flight; only 1 is built today (default 1)
  --warmup N          steps to run and discard before measuring (default 8)
  --json out.json     write the machine-readable record here
  --temp T            sampling temperature; 0 is greedy (default 0)
  --seed S            the sampler seed (default 0)
  --precision P       f16, int8, int4 or auto (default auto). Narrowing is a
                      last resort: auto takes the widest that fits, and int4 is
                      not uniformly more accurate than int8
  --context C         KV cache capacity in positions (default 4096)
  --device D          auto, cpu or metal (default auto)

serve flags:
  --addr host:port    where to listen (default 127.0.0.1:11434, loopback)
  --public            allow a bind that is not loopback; this server has no
                      authentication, so it is a flag rather than a default
  --precision P       f16, int8, int4 or auto (default auto). Narrowing is a
                      last resort: auto takes the widest that fits, and int4 is
                      not uniformly more accurate than int8
  --context C         KV cache capacity per session, in positions (default 4096)
  --slots N           how many requests generate at once. Under --batched it is
                      the batch width and costs a page table each; otherwise it
                      is the pooled session count and costs a whole context of
                      key/value cache each (default 8 batched, 4 pooled, or
                      fewer if the device holds fewer)
  --kv P              shared block pool, in positions. Needs --prefix-cache
                      process, which --batched implies (default slots x context)
  --sessions N        deprecated alias for --slots: how many requests generate
                      at once, and
                      under --prefix-cache session how many conversations keep
                      their cache between turns. Under --prefix-cache process
                      the blocks are shared, so which session a request lands
                      on stops deciding what it reuses and this is concurrency
                      alone. Every one is reserved at startup and held until
                      the process exits (default 0, which takes 4, or fewer if
                      the device holds fewer)
  --prefix-cache S    off, session or process: reuse the key/value state a
                      conversation already paid for, so a turn prefills only
                      what is new. session keeps every block inside one
                      conversation; process shares them, so two conversations
                      with the same system prompt prefill it once between them
                      and a request's cache_salt is what keeps tenants apart.
                      A warm answer matches a cold one in distribution rather
                      than bit for bit (default off; bare --prefix-cache is
                      session)
  --batched           put every in-flight request in one forward pass instead
                      of giving each one a session of its own, so the weights
                      are read once for all of them rather than once each.
                      Implies --prefix-cache process, because sequences that
                      step together have different lengths and a per-session
                      cache would pad every one of them to the longest
                      (default off)
  --device D          auto, cpu or metal (default auto)

info flags:
  --context C         KV cache capacity to price (default 4096)
  --precision P       f16, int8, int4 or auto (default auto). Narrowing is a
                      last resort: auto takes the widest that fits, and int4 is
                      not uniformly more accurate than int8
  --budget B          device bytes the weights may occupy; 0 asks the device
  --device D          auto, cpu or metal (default auto)

pull flags:
  --revision R        branch, tag or commit sha (default the repo's main)
  --token T           Hugging Face access token (default $HF_TOKEN)
`

// run dispatches one command line. It returns an error rather than exiting so
// that every refusal below is reachable from a test.
func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("%w: no command; one of run, bench, serve, info or pull is required", errUsage)
	}
	switch cmd := args[0]; cmd {
	case "run":
		return cmdRun(args[1:], stdout, stderr)
	case "bench":
		return cmdBench(args[1:], stdout, stderr)
	case "serve":
		return cmdServe(args[1:], stdout, stderr)
	case "info":
		return cmdInfo(args[1:], stdout, stderr)
	case "pull":
		return cmdPull(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		_, _ = fmt.Fprint(stdout, usage)
		return nil
	default:
		return fmt.Errorf("%w: unknown command %q; the commands are run, bench, serve, info and pull",
			errUsage, cmd)
	}
}
