package lora

import (
	"unsafe"

	"vexel/inference/tensor"
)

// GPUAdapter holds LoRA adapter weights uploaded to GPU memory.
type GPUAdapter struct {
	Scale  float32
	Rank   int
	Layers []GPULayerAdapter
}

// GPULayerAdapter holds the GPU-resident LoRA A/B matrices for a single
// transformer layer's Q and V attention projections.
type GPULayerAdapter struct {
	QA, QB tensor.DevicePtr
	VA, VB tensor.DevicePtr
	HasQ   bool
	HasV   bool
}

// Allocator is the minimal interface required to upload weights to the GPU.
// Both Metal and any future backend satisfy this via backend.Backend.
type Allocator interface {
	AllocPermanent(bytes int) tensor.DevicePtr
	ToDevice(dst tensor.DevicePtr, src []byte)
}

// UploadToGPU transfers all LoRA weights in adapter to GPU memory using alloc
// and returns a GPUAdapter ready for use during inference.
func UploadToGPU(adapter *Adapter, alloc Allocator) (*GPUAdapter, error) {
	gpu := &GPUAdapter{
		Scale:  adapter.Scale,
		Rank:   adapter.Config.Rank,
		Layers: make([]GPULayerAdapter, len(adapter.Layers)),
	}
	for i, layer := range adapter.Layers {
		if layer.HasQ() {
			gpu.Layers[i].QA = uploadF32(alloc, layer.QA)
			gpu.Layers[i].QB = uploadF32(alloc, layer.QB)
			gpu.Layers[i].HasQ = true
		}
		if layer.HasV() {
			gpu.Layers[i].VA = uploadF32(alloc, layer.VA)
			gpu.Layers[i].VB = uploadF32(alloc, layer.VB)
			gpu.Layers[i].HasV = true
		}
	}
	return gpu, nil
}

// uploadF32 allocates GPU memory and copies the float32 slice into it.
func uploadF32(alloc Allocator, data []float32) tensor.DevicePtr {
	bytes := len(data) * 4
	ptr := alloc.AllocPermanent(bytes)
	if ptr.IsNil() {
		return ptr
	}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), bytes)
	alloc.ToDevice(ptr, src)
	return ptr
}

// GetLayer returns the GPULayerAdapter for layer idx, or nil when idx is out of
// range or the layer has neither Q nor V LoRA weights.
func (g *GPUAdapter) GetLayer(idx int) *GPULayerAdapter {
	if idx < 0 || idx >= len(g.Layers) {
		return nil
	}
	la := &g.Layers[idx]
	if !la.HasQ && !la.HasV {
		return nil
	}
	return la
}
