package cpu_test

import (
	"testing"
	"unsafe"
	"vexel/inference/backend/cpu"
	"vexel/inference/tensor"
)

func TestRoPE(t *testing.T) {
	b := cpu.NewCPUBackend()

	// 1. Basic interleaved RoPE (LLaMA style)
	q := []float32{1, 0, 1, 0} // headDim=4, 1 head
	b.RoPE(tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&q[0]))), tensor.DevicePtr{}, 4, 1, 0, 1, 1, 0, 10000.0, false)

	// Expect rotation at pos 1
	if q[0] == 1.0 && q[1] == 0.0 {
		t.Errorf("RoPE LLaMA style should have rotated q[0], q[1]")
	}

	// 2. NeoX style RoPE (Phi-2 style)
	// headDim=4, ropeDim=4, 1 head, pos=1
	qNeox := []float32{1, 1, 0, 0}
	b.RoPE(tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&qNeox[0]))), tensor.DevicePtr{}, 4, 1, 0, 1, 1, 4, 10000.0, true)

	// In NeoX, pairs are (0, 2) and (1, 3) for dim=4
	// If q = [1, 1, 0, 0], rotated should change all 4 if angle != 0
	if qNeox[0] == 1.0 && qNeox[2] == 0.0 {
		t.Errorf("RoPE NeoX style should have rotated (0, 2) pair")
	}

	// 3. Partial RoPE (Phi-2 style)
	// headDim=4, ropeDim=2, 1 head, pos=1
	qPartial := []float32{1, 0, 1, 0}
	b.RoPE(tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&qPartial[0]))), tensor.DevicePtr{}, 4, 1, 0, 1, 1, 2, 10000.0, false)

	// Dimensions 2, 3 should be UNCHANGED
	if qPartial[2] != 1.0 || qPartial[3] != 0.0 {
		t.Errorf("Partial RoPE changed dimensions beyond ropeDim: got [%f, %f], want [1.0, 0.0]", qPartial[2], qPartial[3])
	}
	// Dimensions 0, 1 should be CHANGED
	if qPartial[0] == 1.0 && qPartial[1] == 0.0 {
		t.Errorf("Partial RoPE failed to rotate dimensions within ropeDim")
	}
}
