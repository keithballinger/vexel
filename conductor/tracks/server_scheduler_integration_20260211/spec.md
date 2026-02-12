# Track Spec: Server & Scheduler Integration

## Overview
Connect the external API (gRPC/HTTP) to the internal inference scheduler. This enables the server to accept inference requests, queue them, and stream generated tokens back to clients.

## Goals
1.  **Request Handling:** Receive and validate generation requests.
2.  **Scheduling:** Submit requests to the `Scheduler`.
3.  **Streaming:** Push generated tokens to the response stream in real-time.

## Technical Details
-   **Scheduler Interface:** Use the existing or enhanced `inference/scheduler` package.
-   **Concurrency:** Handle multiple connections and asynchronous inference.
