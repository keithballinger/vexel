# Track Spec: Go Client Library

## Overview
Develop a high-level, idiomatic Go client library (`vexel/client`) that simplifies interaction with the Vexel inference server. This library will abstract the complexities of gRPC/HTTP communication, making it easy for Go developers to integrate LLM capabilities into their applications.

## Goals
1.  **Idiomatic API:** Provide a clean, easy-to-use Go interface for inference.
2.  **Protocol Abstraction:** Hide the details of gRPC/HTTP/SSE.
3.  **Features:** Support text generation, streaming, and model management (if exposed).
4.  **Documentation:** Provide clear examples and documentation.

## Technical Details
-   **Package:** `vexel/client`
-   **Transport:** Utilize `net/http` for REST/SSE and `google.golang.org/grpc` for gRPC (optional/future).
-   **Structure:** `Client` struct with methods like `Generate(ctx, prompt) (string, error)` and `Stream(ctx, prompt) (<-chan string, error)`.
