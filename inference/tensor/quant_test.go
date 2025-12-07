package tensor_test

import (
	"testing"
	"vexel/inference/tensor"
)

func TestQuantProfile(t *testing.T) {
	tests := []struct {
		name    string
		profile tensor.QuantProfile
	}{
		{"None", tensor.QuantNone},
		{"Q8_0", tensor.Q8_0},
		{"Q4_0", tensor.Q4_0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.profile.String() == "" {
				t.Error("QuantProfile.String() returned empty string")
			}
		})
	}
}
