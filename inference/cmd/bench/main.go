package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/tensor"
)

func main() {
	// 1. CLI Arguments

batchSize := flag.Int("batch", 128, "Max batch size")
seqLen := flag.Int("seq-len", 128, "Sequence length (input + output)")
numSeqs := flag.Int("num-seqs", 256, "Total number of sequences to process")
device := flag.String("device", "cpu", "Device (cpu, cuda, metal)")
flag.Parse()

	log.Printf("Starting benchmark on %s with Batch=%d, SeqLen=%d, TotalSeqs=%d", *device, *batchSize, *seqLen, *numSeqs)

	// 2. Setup (Memory Plan)
	cfg := runtime.Llama3_8B()
	plan := cfg.MemoryPlan(*batchSize, *seqLen, tensor.Q4_0)
	fmt.Printf("Estimated Memory Usage:\n Weights: %.2f GB\n KV Cache: %.2f GB\n Scratch: %.2f GB\n Total: %.2f GB\n",
		toGB(plan.Weights), toGB(plan.KV), toGB(plan.Scratch), toGB(plan.Total))

	// 3. Initialize Runtime (Stubbed for now)
	rt := &runtime.ModelRuntime{}
	schedCfg := scheduler.Config{
		MaxBatchSize: *batchSize,
		MaxSequences: *numSeqs,
	}
	sched, err := scheduler.NewScheduler(rt, schedCfg)
	if err != nil {
		log.Fatalf("Failed to create scheduler: %v", err)
	}

	// 4. Load Requests
	start := time.Now()
	for i := 0; i < *numSeqs; i++ {
		sched.AddSequence(scheduler.NewSequence(scheduler.SequenceID(i), "Benchmark Prompt"))
	}

	// 5. Run Benchmark Loop
	// Note: Since our runtime is stubbed/erroring, this will likely fail or exit fast.
	// In a real benchmark, we'd loop until all sequences are Finished.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	log.Println("Running...")
	if err := sched.Run(ctx); err != nil {
		log.Printf("Scheduler finished with: %v", err)
	}
	
duration := time.Since(start)
	
	// 6. Report
	metrics := sched.Metrics()
	throughput := float64(metrics.TotalTokens) / duration.Seconds()
	
	fmt.Printf("\nResults:\n")
	fmt.Printf(" Duration: %v\n", duration)
	fmt.Printf(" Total Tokens: %d\n", metrics.TotalTokens)
	fmt.Printf(" Throughput: %.2f tokens/sec\n", throughput)
}

func toGB(bytes int64) float64 {
	return float64(bytes) / 1024 / 1024 / 1024
}
