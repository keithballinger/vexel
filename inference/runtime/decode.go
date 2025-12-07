package runtime

import (
	"fmt"
	"vexel/inference/tensor"
)

// DecodeStep performs a single decoding step for the batch.
func (m *ModelRuntime) DecodeStep(inputs BatchRuntimeInputs) (tensor.Tensor, error) {
	// 1. Prepare batch metadata (sequence lengths, block tables)
	// TODO: Flatten tokens and copy to device
	
	// 2. Embedding Lookup
	// TODO: Run embedding kernel
	
	// 3. Layer Loop
	// for _, layer := range m.layers {
	// 	err := layer.Execute(activations, kvCache)
	// 	if err != nil {
	// 		return tensor.Tensor{}, err
	// 	}
	// }
	
	// 4. Final Norm
	// TODO: Run RMSNorm
	
	// 5. Compute Logits (Output Head)
	// TODO: Matmul with lm_head
	
	// Placeholder: Return empty tensor for now to satisfy signature
	return tensor.Tensor{}, fmt.Errorf("DecodeStep not implemented")
}
