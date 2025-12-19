# Development Philosophy
- **Idiomatic Go:** Prioritize clean, idiomatic Go code in all aspects of the project.
- **Performance First:** Maintain the high-performance characteristics of the underlying Metal kernels. Idiomatic Go should be the standard *unless* it introduces measurable performance degradation or correctness issues.
- **Pragmatic API Evolution:** During the early stages of development, the Go API will remain experimental. Breaking changes are acceptable to ensure the long-term quality and performance of the library.

# Quality Standards
- **Benchmarking as Documentation:** Performance improvements must be backed by data from the `perf_harness`.
- **Correctness Parity:** Every feature must be verified for output parity with llama.cpp using deterministic sampling.
- **Minimal Dependencies:** Keep the core Go module lean by minimizing external dependencies to ensure ease of integration for end-users.
