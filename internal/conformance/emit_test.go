// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package conformance

import (
	"os"
	"testing"
)

// TestEmitTheTable writes §2's generated table where TGO_EMIT_TABLE names, so
// regenerating the spec is a command rather than a paste out of a failure
// message. Off unless the variable is set.
func TestEmitTheTable(t *testing.T) {
	path := os.Getenv("TGO_EMIT_TABLE")
	if path == "" {
		t.Skip("TGO_EMIT_TABLE is unset")
	}
	if err := os.WriteFile(path, []byte(Document(Register())), 0o644); err != nil {
		t.Fatal(err)
	}
}
