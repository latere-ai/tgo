// SPDX-FileCopyrightText: 2026 Latere AI
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"encoding/json"

	"latere.ai/x/pkg/llmdialect/ir"
)

// §4's rule has two halves and this file is the first: refuse what changes the
// answer, naming the field. The second half -- record what does not -- is
// loss.go. The rule is testable rather than a list: a field is advisory when a
// request with it and a request without it produce the same tokens, so anything
// here is a field whose absence would produce different ones.
//
// The checks that need the decoded request (a schema, an image, a logit_bias id
// outside the vocabulary) are in adapt.go, beside the mapping that would
// otherwise drop them. The check here needs the raw body, because a frontend
// would refuse it first in its own words and with its own field name.

// definesN are the dialects with an n member.
//
// Anthropic Messages and OpenAI Responses have none, and a member a dialect
// does not define is an unrecognized key rather than a field that changes the
// answer: tgo ignores it either way, which is exactly §4's test for advisory.
// So it falls through to the loss report there, like every other stray key,
// and only the two surfaces that define it refuse it.
var definesN = map[ir.Dialect]bool{
	ir.DialectOpenAIChat: true,
	dialectLegacy:        true,
}

// refuseRaw refuses the members that must be answered before a frontend sees
// them.
func refuseRaw(d ir.Dialect, top map[string]json.RawMessage) *apiError {
	if !definesN[d] {
		return nil
	}
	raw, ok := top["n"]
	if !ok || isNull(raw) {
		return nil
	}
	var n int
	if err := json.Unmarshal(raw, &n); err != nil {
		return badRequest("tgo: n must be an integer: %v", err)
	}
	if n > 1 {
		return refusal("n", "tgo: n=%d is not supported: more than one completion per "+
			"request needs batching (specs/008-scheduler.md). Send %d requests", n, n)
	}
	return nil
}
