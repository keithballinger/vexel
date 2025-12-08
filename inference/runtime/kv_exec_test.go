package runtime_test

import (
	"testing"
	"vexel/inference/backend/cpu"
	"vexel/inference/runtime"
	"vexel/inference/tensor"
)

func TestBlockExecutionWithKV(t *testing.T) {
	spy := &SpyBackend{Backend: cpu.NewBackend()}
	block := runtime.NewBlockRuntime(spy)

	// Mock Inputs
	inputShape := tensor.NewShape(1, 4)
	input := tensor.NewTensor(inputShape, tensor.Float32, tensor.NewDevicePtr(tensor.CPU, 100))
	scratch := tensor.NewTensor(tensor.NewShape(1000), tensor.Float32, tensor.NewDevicePtr(tensor.CPU, 200))
	
	// Execute with sequence length > 1 (implies history)
	// We need to pass metadata to Execute saying "Current pos is X, Total Seq Len is Y".
	// Currently Execute signature is `(x, scratch)`.
	// We need `Execute(x, scratch, kvCache, layerIdx, pos)`.
	
	// This test asserts we change the signature (compiler fail) OR we fail to use it.
	
	// Let's assume we update Execute to take `RuntimeContext` struct?
	// Or just params.
	
	// I'll call with new signature and expect compilation failure, marking the "Red" phase.
	
	// block.Execute(input, scratch, nil, 0, 5) 
	
	// Execute with KV params (kvCache=nil for test, layer=0, pos=10)
	// Expectation: Execute signature should support this.
	block.Execute(input, scratch, nil, 0, 10)
}