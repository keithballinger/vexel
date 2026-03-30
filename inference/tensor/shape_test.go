package tensor_test

import (
	"reflect"
	"testing"
	"vexel/inference/tensor"
)

func TestShape(t *testing.T) {
	tests := []struct {
		name         string
		dims         []int
		wantNumElems int
		wantRank     int
		wantStrides  []int
	}{
		{
			name:         "Scalar",
			dims:         []int{},
			wantNumElems: 1,
			wantRank:     0,
			wantStrides:  []int{}, // Scalar has empty strides
		},
		{
			name:         "Vector",
			dims:         []int{5},
			wantNumElems: 5,
			wantRank:     1,
			wantStrides:  []int{1},
		},
		{
			name:         "Matrix",
			dims:         []int{2, 3},
			wantNumElems: 6,
			wantRank:     2,
			wantStrides:  []int{3, 1},
		},
		{
			name:         "3D Tensor",
			dims:         []int{2, 3, 4},
			wantNumElems: 24,
			wantRank:     3,
			wantStrides:  []int{12, 4, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := tensor.NewShape(tt.dims...)

			if got := s.NumElements(); got != tt.wantNumElems {
				t.Errorf("Shape.NumElements() = %v, want %v", got, tt.wantNumElems)
			}

			if got := s.Rank(); got != tt.wantRank {
				t.Errorf("Shape.Rank() = %v, want %v", got, tt.wantRank)
			}

			if got := s.StridesRowMajor(); !reflect.DeepEqual(got, tt.wantStrides) {
				t.Errorf("Shape.StridesRowMajor() = %v, want %v", got, tt.wantStrides)
			}

			// Test Equality
			s2 := tensor.NewShape(tt.dims...)
			if !s.Equal(s2) {
				t.Errorf("Shape.Equal() returned false for identical shapes")
			}
		})
	}
}

func TestShapeEqual(t *testing.T) {
	s1 := tensor.NewShape(2, 3)
	s2 := tensor.NewShape(2, 4)
	s3 := tensor.NewShape(3, 2)

	if s1.Equal(s2) {
		t.Error("Shape.Equal() should be false for (2,3) vs (2,4)")
	}
	if s1.Equal(s3) {
		t.Error("Shape.Equal() should be false for (2,3) vs (3,2)")
	}
}
