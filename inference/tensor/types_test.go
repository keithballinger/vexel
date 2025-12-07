package tensor_test

import (
	"testing"
	"vexel/inference/tensor"
)

func TestDType(t *testing.T) {
	tests := []struct {
		name     string
		dtype    tensor.DType
		wantBytes int
		wantBits  int
	}{
		{"Float32", tensor.Float32, 4, 32},
		{"Float16", tensor.Float16, 2, 16},
		{"Int8", tensor.Int8, 1, 8},
		{"Uint8", tensor.Uint8, 1, 8},
		{"BFloat16", tensor.BFloat16, 2, 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.dtype.SizeBytes(); got != tt.wantBytes {
				t.Errorf("DType.SizeBytes() = %v, want %v", got, tt.wantBytes)
			}
			if got := tt.dtype.BitSize(); got != tt.wantBits {
				t.Errorf("DType.BitSize() = %v, want %v", got, tt.wantBits)
			}
		})
	}
}

func TestLocation(t *testing.T) {
	tests := []struct {
		name string
		loc  tensor.Location
	}{
		{"CPU", tensor.CPU},
		{"CUDA", tensor.CUDA},
		{"Metal", tensor.Metal},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Just verify the constant exists and can be stringified
			if tt.loc.String() == "" {
				t.Error("Location.String() returned empty string")
			}
		})
	}
}
