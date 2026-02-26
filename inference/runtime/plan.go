// Package runtime provides model-aware execution planning for Vexel.
//
// The execution planner selects kernel variants, precision paths, layouts,
// and fusion strategies based on model architecture and device capabilities.
package runtime

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Regime classifies the model into optimization categories.
type Regime string

const (
	// RegimeSmall is for small models (TinyLlama-ish) that are latency/launch-bound.
	// Focus: minimize kernel launches, favor fusion, avoid overhead from parallel reductions.
	RegimeSmall Regime = "small"

	// RegimeLarge is for large models (7B+) that are bandwidth/throughput-bound.
	// Focus: multi-output-per-simdgroup, tiling, vectorization, FFN optimization.
	RegimeLarge Regime = "large"

	// RegimeCustom allows manual override of all settings.
	RegimeCustom Regime = "custom"
)

// PrecisionPolicy controls floating-point precision choices.
type PrecisionPolicy struct {
	// AttentionF16 keeps Q/K/V in FP16 through RoPE, KV write, SDPA, and O-proj.
	AttentionF16 bool

	// AccumFP32 accumulates in FP32 inside kernels where needed for numerical stability.
	AccumFP32 bool

	// KVCacheF16 stores KV cache in FP16 format.
	KVCacheF16 bool
}

// KVPolicy controls KV cache layout and operations.
type KVPolicy struct {
	// Layout specifies the KV cache memory layout.
	// "head_major" = [numKVHeads, maxSeqLen, headDim] - optimal for decode SDPA.
	// "seq_major" = [maxSeqLen, numKVHeads, headDim] - legacy layout.
	Layout string

	// AppendKernel specifies which kernel to use for KV cache updates.
	// "scatter_kv_f16" = GPU scatter kernel (single dispatch).
	// "blit_copy" = multiple blit copies (fallback).
	AppendKernel string

	// HeadStride is the stride between KV heads in elements (maxSeqLen * headDim).
	HeadStride int
}

// KernelVariants specifies which kernel implementation to use for each operation.
type KernelVariants struct {
	// SDPA kernels
	SDPADecode  string // e.g., "sdpa_decode_f16", "sdpa_decode_f16_hd64", "sdpa_decode_f16_hd128"
	SDPAPrefill string // e.g., "flash_attention_2_f16", "sdpa_prefill_f32"

	// Projection kernels
	QKV string // e.g., "fused_rmsnorm_qkv_f16", "separate_qkv"
	Wo  string // e.g., "matvec_q4_0_nr2", "matvec_q4_0_nr4"

	// FFN kernels
	FFNGateUp string // e.g., "fused_rmsnorm_gateup", "fused_w1w3_silu", "separate"
	FFNDown   string // e.g., "matvec_q4_0_nr2", "matvec_q4_0_nr4"

	// Output head
	LMHead string // e.g., "q6k_nr2", "q6k_multi_output"
}

// FusionPolicy controls which operations are fused together.
type FusionPolicy struct {
	// FuseRMSNormQKV fuses RMSNorm with QKV projection.
	FuseRMSNormQKV bool

	// FuseRMSNormGateUp fuses RMSNorm with Gate/Up projection.
	FuseRMSNormGateUp bool

	// FuseW1W3 fuses W1 (gate) and W3 (up) into a single kernel.
	// For large models, this can be extended to include SiLU and multiply.
	FuseW1W3 bool

	// FuseSiLUMul fuses SiLU activation with element-wise multiply.
	FuseSiLUMul bool

	// FuseResidualAdd fuses residual additions into preceding kernels.
	FuseResidualAdd bool

	// FuseMLP uses FusedMLP kernel: SiLU(x@W1)*(x@W3) in single dispatch.
	// Replaces separate FusedRMSNorm+MatMul(W1) + FusedRMSNorm+MatMul(W3) + SiLUMul.
	// Only applies to decode (seqLen=1) with RMSNorm + Q4_0 weights.
	FuseMLP bool

	// FuseAddRMSNorm uses AddRMSNorm kernel: x+=residual, out=RMSNorm(x) in single dispatch.
	// Replaces separate Add1 + RMSNorm2. Applies to both decode and prefill.
	FuseAddRMSNorm bool
}

// TuningParams contains kernel-specific tuning parameters.
type TuningParams struct {
	// Nr0_FFN is the number of output rows per simdgroup for FFN kernels.
	// Higher values (2, 4) improve activation reuse but increase register pressure.
	Nr0_FFN int

	// Nr0_Proj is the number of output rows per simdgroup for projection kernels.
	Nr0_Proj int

	// TileSizeSDPA is the tile size for SDPA prefill (K positions per tile).
	TileSizeSDPA int

	// UnrollFactor for inner loops in matmul kernels.
	UnrollFactor int
}

// ExecutionPlan is a model- and device-specific configuration that selects
// which kernel variants and layout/precision/fusion policies Vexel uses.
type ExecutionPlan struct {
	// ModelName identifies the model this plan is for (for logging).
	ModelName string

	// Regime classifies the model's optimization category.
	Regime Regime

	// RegimeReason explains why this regime was chosen.
	RegimeReason string

	// Precision controls floating-point precision choices.
	Precision PrecisionPolicy

	// KV controls KV cache layout and operations.
	KV KVPolicy

	// Kernels specifies which kernel implementation to use for each operation.
	Kernels KernelVariants

	// Fusion controls which operations are fused together.
	Fusion FusionPolicy

	// Tuning contains kernel-specific tuning parameters.
	Tuning TuningParams

	// PrefillOverrides contains settings that differ for prefill vs decode.
	// If nil, use the main settings for both.
	PrefillOverrides *PrefillSettings
}

// PrefillSettings contains settings specific to the prefill phase.
type PrefillSettings struct {
	SDPAKernel string
	UseBatched bool // Use batched matmul instead of matvec
}

// ModelMeta contains model architecture information used for planning.
type ModelMeta struct {
	Name             string
	HiddenSize       int
	IntermediateSize int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	VocabSize        int
	MaxSeqLen        int

	// Quantization formats used (from GGUF metadata)
	QuantFormats map[string]int // e.g., {"Q4_0": 225, "Q6_K": 1, "F32": 65}

	// NormType specifies the normalization type (RMSNorm vs LayerNorm)
	NormType NormType
}

// DeviceMeta contains device capability information.
type DeviceMeta struct {
	// Name is the device name (e.g., "Apple M1 Pro")
	Name string

	// Generation is the Apple Silicon generation (1, 2, 3, etc.)
	Generation int

	// Variant is Pro/Max/Ultra (0=base, 1=Pro, 2=Max, 3=Ultra)
	Variant int

	// UnifiedMemoryGB is the total unified memory in GB.
	UnifiedMemoryGB int

	// BandwidthClass is a coarse bandwidth classification.
	// 0=low (M1 base), 1=medium (M1 Pro), 2=high (M1 Max/Ultra, M2+)
	BandwidthClass int
}

// PlanConfig contains optional configuration for plan building.
type PlanConfig struct {
	// PreferLatency optimizes for lower latency per token (vs throughput).
	PreferLatency bool

	// MaxMemoryMB limits memory usage for caches and scratch.
	MaxMemoryMB int

	// PreferSafeKernels avoids experimental kernel variants.
	PreferSafeKernels bool
}

// BuildExecutionPlan creates an execution plan based on model and device metadata.
func BuildExecutionPlan(model ModelMeta, device DeviceMeta, config *PlanConfig) *ExecutionPlan {
	if config == nil {
		config = &PlanConfig{}
	}

	plan := &ExecutionPlan{
		ModelName: model.Name,
	}

	// Step 1: Classify regime based on model characteristics
	plan.Regime, plan.RegimeReason = classifyRegime(model)

	// Step 2: Apply regime-specific defaults
	switch plan.Regime {
	case RegimeSmall:
		applySmallModelDefaults(plan, model, device)
	case RegimeLarge:
		applyLargeModelDefaults(plan, model, device)
	default:
		applyLargeModelDefaults(plan, model, device) // Default to large
	}

	// Step 3: Apply config overrides
	applyConfigOverrides(plan, config)

	// Step 3.5: Disable fused RMSNorm kernels for LayerNorm models
	// These fused kernels only work with RMSNorm, not LayerNorm
	if model.NormType == NormLayerNorm {
		plan.Fusion.FuseRMSNormQKV = false
		plan.Fusion.FuseRMSNormGateUp = false
	}

	// Step 4: Apply environment variable overrides
	applyEnvOverrides(plan)

	// Step 5: Compute derived values
	plan.KV.HeadStride = model.MaxSeqLen * model.HeadDim

	return plan
}

// classifyRegime determines the optimization regime based on model architecture.
func classifyRegime(model ModelMeta) (Regime, string) {
	// Check for environment override first
	if override := os.Getenv("VEXEL_FORCE_REGIME"); override != "" {
		return Regime(override), "forced via VEXEL_FORCE_REGIME"
	}

	// Heuristics for regime classification
	reasons := []string{}

	// Small model indicators
	smallScore := 0
	if model.HeadDim == 64 {
		smallScore++
		reasons = append(reasons, "head_dim=64")
	}
	if model.NumKVHeads <= 8 {
		smallScore++
		reasons = append(reasons, fmt.Sprintf("num_kv_heads=%d", model.NumKVHeads))
	}
	if model.IntermediateSize <= 8000 {
		smallScore++
		reasons = append(reasons, fmt.Sprintf("intermediate_size=%d", model.IntermediateSize))
	}
	if model.HiddenSize <= 2048 {
		smallScore++
		reasons = append(reasons, fmt.Sprintf("hidden_size=%d", model.HiddenSize))
	}

	// Large model indicators
	largeScore := 0
	if model.HeadDim >= 128 {
		largeScore++
		reasons = append(reasons, "head_dim>=128")
	}
	if model.IntermediateSize >= 12000 {
		largeScore++
		reasons = append(reasons, fmt.Sprintf("intermediate_size=%d", model.IntermediateSize))
	}
	if model.HiddenSize >= 4096 {
		largeScore++
		reasons = append(reasons, fmt.Sprintf("hidden_size=%d", model.HiddenSize))
	}
	if model.NumLayers >= 28 {
		largeScore++
		reasons = append(reasons, fmt.Sprintf("num_layers=%d", model.NumLayers))
	}

	reasonStr := strings.Join(reasons, ", ")

	if smallScore > largeScore {
		return RegimeSmall, fmt.Sprintf("small model regime (%s)", reasonStr)
	}
	return RegimeLarge, fmt.Sprintf("large model regime (%s)", reasonStr)
}

// applySmallModelDefaults sets defaults optimized for small models.
// Focus: minimize kernel launches, favor fusion, avoid overhead from parallel reductions.
func applySmallModelDefaults(plan *ExecutionPlan, model ModelMeta, device DeviceMeta) {
	// Precision: FP16 attention path saves bandwidth even on small models
	plan.Precision = PrecisionPolicy{
		AttentionF16: true,
		AccumFP32:    true,
		KVCacheF16:   true,
	}

	// KV: head-major layout with scatter kernel
	plan.KV = KVPolicy{
		Layout:       "head_major",
		AppendKernel: "scatter_kv_f16",
	}

	// Kernels: prefer fused variants, standard SDPA (don't over-optimize)
	plan.Kernels = KernelVariants{
		SDPADecode:  "sdpa_decode_f16", // Standard kernel, exp-latency bound anyway
		SDPAPrefill: "flash_attention_2_f16",
		QKV:         "fused_rmsnorm_qkv_f16",
		Wo:          "matvec_q4_0_nr2",
		FFNGateUp:   "fused_rmsnorm_gateup",
		FFNDown:     "matvec_q4_0_nr2",
		LMHead:      "q6k_nr2",
	}

	// Fusion: aggressive fusion to reduce launches
	plan.Fusion = FusionPolicy{
		FuseRMSNormQKV:    true,
		FuseRMSNormGateUp: true,
		FuseW1W3:          false, // Not implemented yet
		FuseSiLUMul:       true,
		FuseResidualAdd:   false, // Not implemented yet
		FuseMLP:           true,  // SiLU(x@W1)*(x@W3) in single kernel
		FuseAddRMSNorm:    true,  // Fused Add1+RMSNorm2
	}

	// Tuning: conservative settings
	plan.Tuning = TuningParams{
		Nr0_FFN:      2,
		Nr0_Proj:     2,
		TileSizeSDPA: 32,
		UnrollFactor: 4,
	}

	// Prefill uses batched matmul
	plan.PrefillOverrides = &PrefillSettings{
		SDPAKernel: "flash_attention_2_f16",
		UseBatched: true,
	}
}

// applyLargeModelDefaults sets defaults optimized for large models.
// Focus: throughput/bandwidth, multi-output-per-simdgroup, FFN optimization.
func applyLargeModelDefaults(plan *ExecutionPlan, model ModelMeta, device DeviceMeta) {
	// Precision: FP16 attention path is critical for bandwidth
	plan.Precision = PrecisionPolicy{
		AttentionF16: true,
		AccumFP32:    true,
		KVCacheF16:   true,
	}

	// KV: head-major layout with scatter kernel
	plan.KV = KVPolicy{
		Layout:       "head_major",
		AppendKernel: "scatter_kv_f16",
	}

	// Kernels: prefer high-throughput variants
	sdpaKernel := "sdpa_decode_f16"
	if model.HeadDim == 128 {
		sdpaKernel = "sdpa_decode_f16" // Could use specialized hd128 variant
	}

	plan.Kernels = KernelVariants{
		SDPADecode:  sdpaKernel,
		SDPAPrefill: "flash_attention_2_f16",
		QKV:         "fused_rmsnorm_qkv_f16",
		Wo:          "matvec_q4_0_nr4", // More outputs per simdgroup
		FFNGateUp:   "fused_rmsnorm_gateup",
		FFNDown:     "matvec_q4_0_nr4", // More outputs per simdgroup
		LMHead:      "q6k_nr2",
	}

	// Fusion: aggressive fusion, especially for FFN
	plan.Fusion = FusionPolicy{
		FuseRMSNormQKV:    true,
		FuseRMSNormGateUp: true,
		FuseW1W3:          false, // TODO: implement fused W1+W3 kernel
		FuseSiLUMul:       true,
		FuseResidualAdd:   false, // TODO: implement fused residual add
		FuseMLP:           true,  // SiLU(x@W1)*(x@W3) in single kernel
		FuseAddRMSNorm:    true,  // Fused Add1+RMSNorm2
	}

	// Tuning: aggressive settings for throughput
	plan.Tuning = TuningParams{
		Nr0_FFN:      4, // 4 outputs per simdgroup
		Nr0_Proj:     4,
		TileSizeSDPA: 64,
		UnrollFactor: 8,
	}

	// Prefill uses batched matmul
	plan.PrefillOverrides = &PrefillSettings{
		SDPAKernel: "flash_attention_2_f16",
		UseBatched: true,
	}
}

// applyConfigOverrides applies user configuration to the plan.
func applyConfigOverrides(plan *ExecutionPlan, config *PlanConfig) {
	if config.PreferLatency {
		// Reduce tile sizes and parallelism for lower latency
		plan.Tuning.Nr0_FFN = 2
		plan.Tuning.Nr0_Proj = 2
	}

	if config.PreferSafeKernels {
		// Use conservative kernel variants
		plan.Kernels.SDPADecode = "sdpa_decode_f16"
		plan.Kernels.Wo = "matvec_q4_0_nr2"
		plan.Kernels.FFNDown = "matvec_q4_0_nr2"
	}
}

// applyEnvOverrides applies environment variable overrides to the plan.
func applyEnvOverrides(plan *ExecutionPlan) {
	// Kernel overrides
	if v := os.Getenv("VEXEL_FORCE_SDPA"); v != "" {
		plan.Kernels.SDPADecode = v
	}
	if v := os.Getenv("VEXEL_FORCE_SDPA_PREFILL"); v != "" {
		plan.Kernels.SDPAPrefill = v
	}
	if v := os.Getenv("VEXEL_FORCE_FFN"); v != "" {
		plan.Kernels.FFNGateUp = v
		plan.Kernels.FFNDown = v
	}
	if v := os.Getenv("VEXEL_FORCE_FFN_GATEUP"); v != "" {
		plan.Kernels.FFNGateUp = v
	}
	if v := os.Getenv("VEXEL_FORCE_FFN_DOWN"); v != "" {
		plan.Kernels.FFNDown = v
	}
	if v := os.Getenv("VEXEL_FORCE_WO"); v != "" {
		plan.Kernels.Wo = v
	}
	if v := os.Getenv("VEXEL_FORCE_QKV"); v != "" {
		plan.Kernels.QKV = v
	}

	// Tuning overrides
	if v := os.Getenv("VEXEL_NR0_FFN"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			plan.Tuning.Nr0_FFN = n
		}
	}
	if v := os.Getenv("VEXEL_NR0_PROJ"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			plan.Tuning.Nr0_Proj = n
		}
	}
	if v := os.Getenv("VEXEL_TILE_SIZE_SDPA"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			plan.Tuning.TileSizeSDPA = n
		}
	}

	// Fusion overrides
	if v := os.Getenv("VEXEL_FUSE_W1W3"); v == "1" {
		plan.Fusion.FuseW1W3 = true
	} else if v == "0" {
		plan.Fusion.FuseW1W3 = false
	}
	if v := os.Getenv("VEXEL_FUSE_RMSNORM_QKV"); v == "0" {
		plan.Fusion.FuseRMSNormQKV = false
	}
	if v := os.Getenv("VEXEL_FUSE_RMSNORM_GATEUP"); v == "0" {
		plan.Fusion.FuseRMSNormGateUp = false
	}
	if v := os.Getenv("VEXEL_FUSE_MLP"); v == "0" {
		plan.Fusion.FuseMLP = false
	}
	if v := os.Getenv("VEXEL_FUSE_ADD_RMSNORM"); v == "0" {
		plan.Fusion.FuseAddRMSNorm = false
	}

	// Precision overrides
	if v := os.Getenv("VEXEL_KV_FP32"); v == "1" {
		plan.Precision.KVCacheF16 = false
	}
}

// String returns a human-readable summary of the execution plan.
func (p *ExecutionPlan) String() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Execution Plan for %s\n", p.ModelName))
	sb.WriteString(fmt.Sprintf("  Regime: %s (%s)\n", p.Regime, p.RegimeReason))
	sb.WriteString("\n")

	sb.WriteString("  Precision:\n")
	sb.WriteString(fmt.Sprintf("    AttentionF16: %v\n", p.Precision.AttentionF16))
	sb.WriteString(fmt.Sprintf("    KVCacheF16:   %v\n", p.Precision.KVCacheF16))
	sb.WriteString(fmt.Sprintf("    AccumFP32:    %v\n", p.Precision.AccumFP32))
	sb.WriteString("\n")

	sb.WriteString("  KV Cache:\n")
	sb.WriteString(fmt.Sprintf("    Layout:       %s\n", p.KV.Layout))
	sb.WriteString(fmt.Sprintf("    AppendKernel: %s\n", p.KV.AppendKernel))
	sb.WriteString("\n")

	sb.WriteString("  Kernels (Decode):\n")
	sb.WriteString(fmt.Sprintf("    SDPA:     %s\n", p.Kernels.SDPADecode))
	sb.WriteString(fmt.Sprintf("    QKV:      %s\n", p.Kernels.QKV))
	sb.WriteString(fmt.Sprintf("    Wo:       %s\n", p.Kernels.Wo))
	sb.WriteString(fmt.Sprintf("    FFNGateUp:%s\n", p.Kernels.FFNGateUp))
	sb.WriteString(fmt.Sprintf("    FFNDown:  %s\n", p.Kernels.FFNDown))
	sb.WriteString(fmt.Sprintf("    LMHead:   %s\n", p.Kernels.LMHead))
	sb.WriteString("\n")

	sb.WriteString("  Fusion:\n")
	sb.WriteString(fmt.Sprintf("    FuseRMSNormQKV:    %v\n", p.Fusion.FuseRMSNormQKV))
	sb.WriteString(fmt.Sprintf("    FuseRMSNormGateUp: %v\n", p.Fusion.FuseRMSNormGateUp))
	sb.WriteString(fmt.Sprintf("    FuseW1W3:          %v\n", p.Fusion.FuseW1W3))
	sb.WriteString(fmt.Sprintf("    FuseSiLUMul:       %v\n", p.Fusion.FuseSiLUMul))
	sb.WriteString(fmt.Sprintf("    FuseMLP:           %v\n", p.Fusion.FuseMLP))
	sb.WriteString(fmt.Sprintf("    FuseAddRMSNorm:    %v\n", p.Fusion.FuseAddRMSNorm))
	sb.WriteString("\n")

	sb.WriteString("  Tuning:\n")
	sb.WriteString(fmt.Sprintf("    Nr0_FFN:      %d\n", p.Tuning.Nr0_FFN))
	sb.WriteString(fmt.Sprintf("    Nr0_Proj:     %d\n", p.Tuning.Nr0_Proj))
	sb.WriteString(fmt.Sprintf("    TileSizeSDPA: %d\n", p.Tuning.TileSizeSDPA))

	return sb.String()
}

// LogPlan logs the execution plan to stdout.
func (p *ExecutionPlan) LogPlan() {
	fmt.Println("=====================================")
	fmt.Println("VEXEL EXECUTION PLAN")
	fmt.Println("=====================================")
	fmt.Print(p.String())
	fmt.Println("=====================================")
}
