package scheduler

import (
	"math"
	"testing"
	"unsafe"

	"vexel/inference/backend/cpu"
	"vexel/inference/pkg/sampler"
	"vexel/inference/tensor"
)

func TestGetLogitsSlice(t *testing.T) {
	t.Run("cpu_tensor", func(t *testing.T) {
		// Create a small logits tensor on CPU
		vocabSize := 8
		logitsData := make([]float32, vocabSize)
		for i := range logitsData {
			logitsData[i] = float32(i) * 0.5
		}

		ptr := tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&logitsData[0])))
		logitsTensor := tensor.NewTensor(tensor.NewShape(1, vocabSize), tensor.Float32, ptr)

		b := cpu.NewCPUBackend()
		result := getLogitsSlice(logitsTensor, b, vocabSize)

		if result == nil {
			t.Fatal("getLogitsSlice returned nil for CPU tensor")
		}
		if len(result) != vocabSize {
			t.Fatalf("expected len %d, got %d", vocabSize, len(result))
		}
		for i, v := range result {
			expected := float32(i) * 0.5
			if math.Abs(float64(v-expected)) > 1e-6 {
				t.Errorf("result[%d] = %f, want %f", i, v, expected)
			}
		}
	})

	t.Run("gpu_tensor", func(t *testing.T) {
		// Use CPU backend to simulate GPU path (ToHost copy).
		// CPU backend treats everything as CPU memory, so ToHost is effectively a memcpy.
		vocabSize := 16
		logitsData := make([]float32, vocabSize)
		for i := range logitsData {
			logitsData[i] = float32(i) * -0.3
		}

		b := cpu.NewCPUBackend()
		// Alloc via backend simulates GPU allocation
		devicePtr := b.Alloc(vocabSize * 4)
		if devicePtr.IsNil() {
			t.Skip("CPU backend Alloc returned nil")
		}

		// Copy data to "device"
		srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&logitsData[0])), vocabSize*4)
		b.ToDevice(devicePtr, srcBytes)

		logitsTensor := tensor.NewTensor(tensor.NewShape(1, vocabSize), tensor.Float32, devicePtr)

		result := getLogitsSlice(logitsTensor, b, vocabSize)

		if result == nil {
			t.Fatal("getLogitsSlice returned nil for device tensor")
		}
		if len(result) != vocabSize {
			t.Fatalf("expected len %d, got %d", vocabSize, len(result))
		}
		for i, v := range result {
			expected := float32(i) * -0.3
			if math.Abs(float64(v-expected)) > 1e-6 {
				t.Errorf("result[%d] = %f, want %f", i, v, expected)
			}
		}
	})

	t.Run("nil_tensor", func(t *testing.T) {
		b := cpu.NewCPUBackend()
		nilTensor := tensor.Tensor{}
		result := getLogitsSlice(nilTensor, b, 100)
		if result != nil {
			t.Error("expected nil for nil tensor, got non-nil")
		}
	})

	t.Run("multi_token_logits", func(t *testing.T) {
		// When verification runs multiple tokens, logits has shape [seqLen, vocabSize].
		// getLogitsSlice should return all seqLen * vocabSize values.
		seqLen := 3
		vocabSize := 4
		totalElements := seqLen * vocabSize

		logitsData := make([]float32, totalElements)
		for i := range logitsData {
			logitsData[i] = float32(i)
		}

		ptr := tensor.NewDevicePtr(tensor.CPU, uintptr(unsafe.Pointer(&logitsData[0])))
		logitsTensor := tensor.NewTensor(tensor.NewShape(seqLen, vocabSize), tensor.Float32, ptr)

		b := cpu.NewCPUBackend()
		result := getLogitsSlice(logitsTensor, b, totalElements)

		if result == nil {
			t.Fatal("getLogitsSlice returned nil for multi-token tensor")
		}
		if len(result) != totalElements {
			t.Fatalf("expected len %d, got %d", totalElements, len(result))
		}
		for i, v := range result {
			if math.Abs(float64(v-float32(i))) > 1e-6 {
				t.Errorf("result[%d] = %f, want %f", i, v, float32(i))
			}
		}
	})
}

func TestGetTokenProbability(t *testing.T) {
	// Verify softmax-based probability computation
	logits := []float32{0, 0, 10, 0} // Token 2 has highest logit
	s := sampler.New(sampler.Config{Temperature: 0}, 42)

	prob := getTokenProbability(logits, 2, s)
	if prob < 0.99 {
		t.Errorf("Expected high probability for token 2, got %f", prob)
	}

	prob0 := getTokenProbability(logits, 0, s)
	if prob0 > 0.01 {
		t.Errorf("Expected low probability for token 0, got %f", prob0)
	}

	// Edge cases
	nilProb := getTokenProbability(nil, 0, s)
	if nilProb != 0 {
		t.Errorf("Expected 0 for nil logits, got %f", nilProb)
	}
}

func TestSpeculativeMetrics(t *testing.T) {
	m := SpeculativeMetrics{
		DraftTokensGenerated: 20,
		DraftTokensAccepted:  15,
		VerificationSteps:    5,
	}

	rate := m.AcceptanceRate()
	if math.Abs(rate-0.75) > 1e-6 {
		t.Errorf("AcceptanceRate = %f, want 0.75", rate)
	}

	speedup := m.Speedup()
	// (15 accepted + 5 verification steps) / 5 = 4.0
	if math.Abs(speedup-4.0) > 1e-6 {
		t.Errorf("Speedup = %f, want 4.0", speedup)
	}

	// Edge cases
	empty := SpeculativeMetrics{}
	if empty.AcceptanceRate() != 0 {
		t.Error("Expected 0 acceptance rate for empty metrics")
	}
	if empty.Speedup() != 1.0 {
		t.Errorf("Expected 1.0 speedup for empty metrics, got %f", empty.Speedup())
	}
}

func TestDefaultConfigs(t *testing.T) {
	spec := DefaultSpeculativeConfig()
	if spec.NumDraftTokens != 4 {
		t.Errorf("Expected 4 draft tokens, got %d", spec.NumDraftTokens)
	}

	self := DefaultSelfSpeculativeConfig()
	if self.DraftLayers != 8 {
		t.Errorf("Expected 8 draft layers, got %d", self.DraftLayers)
	}
	if self.NumDraftTokens != 4 {
		t.Errorf("Expected 4 draft tokens, got %d", self.NumDraftTokens)
	}
}
