// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"fmt"
	"runtime"
	"runtime/debug"

	"golang.design/x/accel"

	tgo "github.com/latere-ai/tgo"
)

// hardware is the machine a number was produced on.
//
// specs/017-benchmarks.md 017-D4: a tokens-per-second figure without the
// hardware is decoration. Every field here is asked of the opened device rather
// than typed by whoever ran the benchmark, because the field of a report that a
// human fills in is the field that is wrong.
type hardware struct {
	Backend       string `json:"accel_backend"`
	Device        string `json:"device"`
	Vendor        string `json:"vendor"`
	Software      bool   `json:"software"`
	UnifiedMemory bool   `json:"unified_memory"`
	MaxPoolBytes  int64  `json:"max_pool_bytes"`
	CPUs          int    `json:"cpus"`
}

// environment is the build a number was produced by: the Go toolchain, the
// target, and the accel revision that decided every device time in the report.
//
// The accel version is read from the build info rather than written down. A
// regression check that compares two records has to be able to tell "tgo got
// slower" from "accel changed", which is the whole reason 017-D1 splits the
// step into four terms, and it cannot do that without knowing which accel each
// record was produced against.
type environment struct {
	Go     string `json:"go_version"`
	GOOS   string `json:"goos"`
	GOARCH string `json:"goarch"`
	Accel  string `json:"accel_version"`
}

// stampEnvironment reads the toolchain and the accel revision this binary was
// built from.
func stampEnvironment() environment {
	e := environment{
		Go:     runtime.Version(),
		GOOS:   runtime.GOOS,
		GOARCH: runtime.GOARCH,
		Accel:  "unknown",
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return e
	}
	for _, d := range info.Deps {
		if d.Path == accelModule {
			e.Accel = d.Version
			if d.Replace != nil {
				e.Accel = fmt.Sprintf("%s (replaced by %s %s)", d.Version, d.Replace.Path, d.Replace.Version)
			}
			break
		}
	}
	return e
}

// accelModule is the module path the device layer comes from.
const accelModule = "golang.design/x/accel"

// stampHardware describes an opened device.
func stampHardware(dev *accel.Device) hardware {
	i := dev.Info()
	return hardware{
		Backend:       i.Backend.String(),
		Device:        i.Name,
		Vendor:        i.Vendor,
		Software:      i.Software,
		UnifiedMemory: i.Capabilities.SharedMemoryKind,
		MaxPoolBytes:  int64(dev.Limits().MaxPoolBytes),
		CPUs:          runtime.NumCPU(),
	}
}

// openDevice opens the device a command describes itself against.
//
// It restates tgo's own device selection, which is unexported and reachable
// only by opening a model. `tgo info` reports the machine without loading a
// byte, and `tgo run` reads the device limit that decides the precision before
// the engine exists, so the three cases exist here too; [engineOptions.Device]
// carries the same choice into the engine so that the two open the same one.
//
// AutoDevice allows the CPU, because the CPU backend is a first-class accel
// backend and a machine with no GPU must still run the model and say so: the
// backend name is in every report, so a CPU number is never mistaken for a GPU
// one (017-D4). A named device is refused where there is none rather than
// falling back, because a user who named one had a reason to.
var openDevice = func(want tgo.Device) (*accel.Device, error) {
	var (
		dev *accel.Device
		err error
	)
	switch want {
	case tgo.CPU:
		dev, err = accel.OpenCPU(accel.CPUOptions{})
	case tgo.Metal:
		dev, err = accel.OpenBest(accel.Policy{Prefer: []accel.Backend{accel.BackendMetal}})
	default:
		dev, err = accel.OpenBest(accel.Policy{AllowCPU: true})
	}
	if err != nil {
		return nil, fmt.Errorf("opening the %v device: %w", want, err)
	}
	return dev, nil
}
