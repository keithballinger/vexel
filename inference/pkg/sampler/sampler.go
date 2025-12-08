package sampler

import "math"

// Argmax returns the index of the maximum value in the slice.
func Argmax(logits []float32) int {
	maxVal := float32(-math.MaxFloat32)
	maxIdx := 0
	
	for i, v := range logits {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
