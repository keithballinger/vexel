#!/usr/bin/env python3
"""Post-P0 fix benchmark: Decode, Prefill, and Load Time comparison.

Measures Vexel vs llama.cpp on the same model/prompt/settings to see
the impact of the arena OOM fix and measure the current performance gap.

Usage:
    python3 benchmarks/bench_post_p0.py
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time

MODEL = os.path.expanduser("models/llama-2-7b.Q4_0.gguf")
VEXEL = os.path.expanduser("inference/cmd/vexel/vexel")
PROMPT_SHORT = "The quick brown fox jumped over the lazy dog"
PROMPT_128 = " ".join(
    ["The quick brown fox jumped over the lazy dog."] * 14
)  # ~128 tokens
PROMPT_512 = " ".join(
    ["The quick brown fox jumped over the lazy dog."] * 56
)  # ~512 tokens

WARMUP = 1
RUNS = 5


def run_llama(prompt, gen_tokens):
    """Run llama-completion and parse timing from output."""
    cmd = [
        "llama-completion",
        "-m", MODEL,
        "-p", prompt,
        "-n", str(gen_tokens),
        "--temp", "0",
        "-ngl", "99",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - start
    output = result.stderr + result.stdout

    metrics = {"wall_time_s": round(elapsed, 3)}

    # Parse llama.cpp timing stats
    for line in output.split("\n"):
        # Prompt eval (prefill)
        m = re.search(
            r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
            line,
        )
        if m:
            metrics["prefill_ms"] = float(m.group(1))
            metrics["prompt_tokens"] = int(m.group(2))
            metrics["prefill_tok_s"] = float(m.group(4))

        # Decode (eval)
        m = re.search(
            r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
            line,
        )
        if m and "prompt" not in line.lower():
            metrics["decode_ms"] = float(m.group(1))
            metrics["gen_tokens"] = int(m.group(2))
            metrics["decode_tok_s"] = float(m.group(4))

        # Load time
        m = re.search(r"load time\s*=\s*([\d.]+)\s*ms", line)
        if m:
            metrics["load_time_ms"] = float(m.group(1))

    return metrics


def run_vexel(prompt, gen_tokens):
    """Run Vexel and measure wall-clock time."""
    cmd = [
        VEXEL,
        "--model", MODEL,
        "generate",
        "--prompt", prompt,
        "--max-tokens", str(gen_tokens),
        "--temperature", "0",
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - start
    output = result.stdout + result.stderr

    # Check for errors
    if result.returncode != 0 or "OOM" in output or "panic" in output:
        return {"wall_time_s": round(elapsed, 3), "error": output[:200]}

    # Count generated text (rough approximation)
    # Vexel outputs just the generated text after loading messages
    lines = output.strip().split("\n")
    generated = ""
    collecting = False
    for line in lines:
        if collecting:
            generated += line + "\n"
        elif "kernel for LM head" in line or "Using Q6_K" in line:
            collecting = True

    # Rough token count: ~0.75 words per token
    word_count = len(generated.split())
    approx_tokens = max(int(word_count / 0.75), 1)

    return {
        "wall_time_s": round(elapsed, 3),
        "approx_tokens": approx_tokens,
        "approx_decode_tok_s": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0,
    }


def bench(name, engine_fn, prompt, gen_tokens, warmup=WARMUP, runs=RUNS):
    """Run benchmark with warmup and collect stats."""
    print(f"  Warming up ({warmup} runs)...", end="", flush=True)
    for _ in range(warmup):
        engine_fn(prompt, gen_tokens)
        print(".", end="", flush=True)
    print()

    print(f"  Measuring ({runs} runs)...", end="", flush=True)
    results = []
    for _ in range(runs):
        r = engine_fn(prompt, gen_tokens)
        results.append(r)
        print(".", end="", flush=True)
    print()

    return results


def summarize(results, key):
    """Compute mean and stdev for a key across results."""
    values = [r[key] for r in results if key in r and r[key] is not None]
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0
    return round(mean, 2), round(std, 2)


def print_divider():
    print("=" * 70)


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print_divider()
    print("  POST-P0 FIX BENCHMARK: Vexel vs llama.cpp")
    print("  Model: LLaMA 2 7B Q4_0")
    print(f"  Warmup: {WARMUP}, Runs: {RUNS}")
    print_divider()
    print()

    all_results = {}

    # --- Benchmark 1: Decode throughput (short prompt, many tokens) ---
    print("[1/4] DECODE THROUGHPUT (prompt=~10 tokens, gen=200 tokens)")
    print()

    print("  llama.cpp:")
    llama_decode = bench("llama-decode", run_llama, PROMPT_SHORT, 200)
    all_results["llama_decode"] = llama_decode

    print("  Vexel:")
    vexel_decode = bench("vexel-decode", run_vexel, PROMPT_SHORT, 200)
    all_results["vexel_decode"] = vexel_decode

    llama_mean, llama_std = summarize(llama_decode, "decode_tok_s")
    vexel_wall_mean, vexel_wall_std = summarize(vexel_decode, "wall_time_s")
    vexel_approx_mean, _ = summarize(vexel_decode, "approx_decode_tok_s")

    print(f"\n  llama.cpp decode: {llama_mean} ± {llama_std} tok/s")
    print(f"  Vexel wall time: {vexel_wall_mean} ± {vexel_wall_std} s")
    print(f"  Vexel approx decode: {vexel_approx_mean} tok/s (wall-clock)")
    print()

    # --- Benchmark 2: Prefill throughput (~128 tokens) ---
    print("[2/4] PREFILL 128 TOKENS (gen=10 tokens)")
    print()

    print("  llama.cpp:")
    llama_prefill128 = bench("llama-prefill128", run_llama, PROMPT_128, 10)
    all_results["llama_prefill128"] = llama_prefill128

    print("  Vexel:")
    vexel_prefill128 = bench("vexel-prefill128", run_vexel, PROMPT_128, 10)
    all_results["vexel_prefill128"] = vexel_prefill128

    llama_p128_mean, llama_p128_std = summarize(llama_prefill128, "prefill_tok_s")
    vexel_p128_wall, _ = summarize(vexel_prefill128, "wall_time_s")
    vexel_p128_err = [r for r in vexel_prefill128 if "error" in r]

    print(f"\n  llama.cpp prefill 128: {llama_p128_mean} ± {llama_p128_std} tok/s")
    if vexel_p128_err:
        print(f"  Vexel prefill 128: FAILED ({vexel_p128_err[0].get('error', '')[:100]})")
    else:
        print(f"  Vexel prefill 128: wall time {vexel_p128_wall} s (no internal timing)")
    print()

    # --- Benchmark 3: Prefill throughput (~512 tokens) ---
    print("[3/4] PREFILL 512 TOKENS (gen=10 tokens)")
    print()

    print("  llama.cpp:")
    llama_prefill512 = bench("llama-prefill512", run_llama, PROMPT_512, 10)
    all_results["llama_prefill512"] = llama_prefill512

    print("  Vexel:")
    vexel_prefill512 = bench("vexel-prefill512", run_vexel, PROMPT_512, 10)
    all_results["vexel_prefill512"] = vexel_prefill512

    llama_p512_mean, llama_p512_std = summarize(llama_prefill512, "prefill_tok_s")
    vexel_p512_wall, _ = summarize(vexel_prefill512, "wall_time_s")
    vexel_p512_err = [r for r in vexel_prefill512 if "error" in r]

    print(f"\n  llama.cpp prefill 512: {llama_p512_mean} ± {llama_p512_std} tok/s")
    if vexel_p512_err:
        print(f"  Vexel prefill 512: FAILED ({vexel_p512_err[0].get('error', '')[:100]})")
    else:
        print(f"  Vexel prefill 512: wall time {vexel_p512_wall} s (no internal timing)")
    print()

    # --- Benchmark 4: Load time ---
    print("[4/4] MODEL LOAD TIME (gen=1 token)")
    print()

    print("  llama.cpp:")
    llama_load = bench("llama-load", run_llama, "Hello", 1, warmup=0, runs=3)
    all_results["llama_load"] = llama_load

    print("  Vexel:")
    vexel_load = bench("vexel-load", run_vexel, "Hello", 1, warmup=0, runs=3)
    all_results["vexel_load"] = vexel_load

    llama_load_mean, llama_load_std = summarize(llama_load, "load_time_ms")
    vexel_load_wall, vexel_load_std = summarize(vexel_load, "wall_time_s")

    print(f"\n  llama.cpp load: {llama_load_mean} ± {llama_load_std} ms")
    print(f"  Vexel total wall (load+1tok): {vexel_load_wall} ± {vexel_load_std} s")
    print()

    # --- Summary ---
    print_divider()
    print("  SUMMARY")
    print_divider()
    print()

    # Previous results for comparison
    print("  DECODE (tok/s, higher=better):")
    print(f"    llama.cpp: {llama_mean} ± {llama_std} tok/s")
    print(f"    Vexel:     {vexel_approx_mean} tok/s (wall-clock, includes load)")
    print(f"    Pre-P0:    Vexel was 43.38 tok/s, llama.cpp was 78.45 tok/s")
    print()

    print("  PREFILL 128 tokens (tok/s, higher=better):")
    print(f"    llama.cpp: {llama_p128_mean} ± {llama_p128_std} tok/s")
    vexel_p128_status = "WORKING" if not vexel_p128_err else "FAILED"
    print(f"    Vexel:     {vexel_p128_status} (wall time {vexel_p128_wall} s)")
    print(f"    Pre-P0:    Vexel OOM'd at 128 tokens")
    print()

    print("  PREFILL 512 tokens (tok/s, higher=better):")
    print(f"    llama.cpp: {llama_p512_mean} ± {llama_p512_std} tok/s")
    vexel_p512_status = "WORKING" if not vexel_p512_err else "FAILED"
    print(f"    Vexel:     {vexel_p512_status} (wall time {vexel_p512_wall} s)")
    print(f"    Pre-P0:    Vexel OOM'd at 128+ tokens")
    print()

    # Save raw results
    output_path = os.path.join("benchmarks", "results", "post_p0_benchmark.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Raw data saved to: {output_path}")
    print_divider()


if __name__ == "__main__":
    main()
