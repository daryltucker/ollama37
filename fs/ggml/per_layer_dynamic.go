package ggml

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// getOffloadTypes returns the list of tensor quantization types that should be
// kept on CPU instead of GPU. This is controlled by the OLLAMA_CPU_OFFLOAD_TYPES
// environment variable.
//
// Example:
//
//	OLLAMA_CPU_OFFLOAD_TYPES=MXFP4,Q4_0,Q4_1,Q8_0,Q8_1
//
// Supported types include:
//   - MXFP4: 4-bit mixed precision
//   - Q4_0, Q4_1: INT4 quantization variants
//   - Q8_0, Q8_1: INT8 quantization variants
//   - Any other valid tensor type string (F16, BF16, Q2_K, etc.)
func getOffloadTypes() map[string]bool {
	// >> Tesla K80
	envValue := envconfig.CpuOffloadTypes()
	// << Tesla K80
	if envValue == "" {
		return nil
	}

	types := make(map[string]bool)
	for _, t := range strings.Split(envValue, ",") {
		t = strings.TrimSpace(t)
		if t != "" {
			// Normalize to uppercase to match TensorType.String() output
			types[strings.ToUpper(t)] = true
		}
	}

	if len(types) > 0 {
		slog.Info("per_layer_dynamic: CPU offload types configured", "types", getTypesList(types))
	}

	return types
}

// getTypesList returns the list of types from the map for logging.
func getTypesList(types map[string]bool) []string {
	list := make([]string, 0, len(types))
	for t := range types {
		list = append(list, t)
	}
	return list
}

// MarkTensorsForCPUOffload iterates through all tensors in the model and marks
// those with matching quantization types for CPU-only placement.
//
// This is a global, model-agnostic operation that runs after LoadModel but before
// GPU memory allocation. Tensors marked with CPUOnly=true will be excluded from
// GPU VRAM calculations (via Layer.Size()) and will remain in system memory.
//
// The types to offload are configured via the OLLAMA_CPU_OFFLOAD_TYPES environment
// variable. If not set, no tensors are marked for offload.
func (g *GGML) MarkTensorsForCPUOffload() {
	if g == nil {
		return
	}

	offloadTypes := getOffloadTypes()
	if len(offloadTypes) == 0 {
		return
	}

	offloadCount := 0
	var offloadedSize uint64

	for _, t := range g.Tensors().Items() {
		tensorType := t.Type()
		if offloadTypes[tensorType] {
			t.CPUOnly = true
			offloadCount++
			offloadedSize += t.Size()
			slog.Debug("per_layer_dynamic: marking tensor for CPU offload",
				"name", t.Name,
				"type", tensorType,
				"size", t.Size())
		}
	}

	if offloadCount > 0 {
		slog.Info("per_layer_dynamic: tensors marked for CPU offload",
			"count", offloadCount,
			"total_size", offloadedSize)
	}
}
