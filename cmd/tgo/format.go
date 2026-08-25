// Copyright 2026 Latere AI.
// Licensed under the Apache License, Version 2.0.

package main

import (
	"fmt"
	"time"
)

// humanBytes formats a byte count in binary units.
//
// It is spelled the way weights.humanBytes is, to the digit, because `tgo info`
// and the loader's own log line report the same footprints and a user reading
// both must not have to decide whether "1.40 GiB" and "1.4GB" are the same
// number.
func humanBytes(n int64) string {
	const unit = 1024
	if n < 0 {
		return "-" + humanBytes(-n)
	}
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit && exp < 3; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %ciB", float64(n)/float64(div), "KMGT"[exp])
}

// humanCount formats a parameter count in decimal units, which is the unit a
// model card uses: 0.6B is a name, not a measurement in mebibytes.
func humanCount(n int64) string {
	switch {
	case n < 0:
		return "-" + humanCount(-n)
	case n >= 1e9:
		return fmt.Sprintf("%.2fB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	case n >= 1e3:
		return fmt.Sprintf("%.2fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// humanDuration formats a duration at a fixed three significant digits, so that
// a column of them lines up and a p50 next to a p99 can be compared by eye.
func humanDuration(d time.Duration) string {
	switch {
	case d == 0:
		return "0"
	case d < time.Microsecond:
		return fmt.Sprintf("%dns", d.Nanoseconds())
	case d < time.Millisecond:
		return fmt.Sprintf("%.2fµs", float64(d)/float64(time.Microsecond))
	case d < time.Second:
		return fmt.Sprintf("%.2fms", float64(d)/float64(time.Millisecond))
	default:
		return fmt.Sprintf("%.2fs", d.Seconds())
	}
}

// percent formats a share of a step. The shares sum to one, so two decimals of
// a percent is the resolution at which the four terms still add up on the page.
func percent(f float64) string { return fmt.Sprintf("%.2f%%", f*100) }
