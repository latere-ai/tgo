module github.com/latere-ai/tgo

go 1.27.0

require (
	golang.design/x/accel v0.0.0-20260827130011-b99f0cfd4576
	golang.org/x/text v0.41.0
	latere.ai/x/pkg v0.41.0
)

require (
	github.com/ebitengine/purego v0.10.1 // indirect
	github.com/goccy/go-yaml v1.19.2 // indirect
	latere.ai/x/ci-gate v0.24.0 // indirect
)

tool latere.ai/x/ci-gate/cmd/lateregate
