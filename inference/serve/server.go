package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
	"vexel/inference/scheduler"
)

// seqCounter is a global atomic counter for generating unique sequence IDs.
// Using time.Now().UnixNano() caused ID collisions under concurrent requests.
var seqCounter atomic.Int64

// DefaultRequestTimeout is the default timeout for non-streaming generate requests.
const DefaultRequestTimeout = 120 * time.Second

// Config holds configuration for the inference server.
type Config struct {
	// RequestTimeout is the maximum duration for a non-streaming generate request.
	// Zero means no timeout (unlimited).
	RequestTimeout time.Duration
}

// nextSeqID returns a unique sequence ID using an atomic counter seeded with
// the current time. Safe for concurrent use.
func nextSeqID() scheduler.SequenceID {
	return scheduler.SequenceID(seqCounter.Add(1))
}

func init() {
	// Seed counter with current time so IDs are unique across restarts.
	seqCounter.Store(time.Now().UnixNano())
}

// Server handles HTTP/gRPC requests for inference.
type Server struct {
	scheduler *scheduler.Scheduler
	config    Config
	mux       *http.ServeMux
}

// NewServer creates a new inference server with default configuration.
func NewServer(sched *scheduler.Scheduler) *Server {
	return NewServerWithConfig(sched, Config{RequestTimeout: DefaultRequestTimeout})
}

// NewServerWithConfig creates a new inference server with the given configuration.
func NewServerWithConfig(sched *scheduler.Scheduler, cfg Config) *Server {
	s := &Server{
		scheduler: sched,
		config:    cfg,
		mux:       http.NewServeMux(),
	}
	s.routes()
	return s
}

func (s *Server) routes() {
	s.mux.HandleFunc("/generate", s.handleGenerate)
	s.mux.HandleFunc("/stream", s.handleStream)
	s.mux.HandleFunc("/health", s.handleHealth)
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

// handleHealth returns a simple health check response.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleStream handles streaming generation requests (SSE).
func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Prompt string `json:"prompt"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// 1. Create Sequence
	seqID := nextSeqID()
	seq := scheduler.NewSequence(seqID, req.Prompt)

	// 2. Add to Scheduler
	s.scheduler.AddSequence(seq)
	defer s.scheduler.RemoveSequence(seqID)

	// Stream from Scheduler
	for token := range seq.TokenChan() {
		data := map[string]string{"token": token}
		buf, _ := json.Marshal(data)

		fmt.Fprintf(w, "data: %s\n\n", buf)
		flusher.Flush()
	}
}

// handleGenerate handles non-streaming generation requests.
func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Prompt string `json:"prompt"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// 1. Create Sequence
	seqID := nextSeqID()
	seq := scheduler.NewSequence(seqID, req.Prompt)

	// 2. Add to Scheduler
	s.scheduler.AddSequence(seq)
	defer s.scheduler.RemoveSequence(seqID)

	// 3. Wait for completion and collect tokens
	var ctx context.Context
	var cancel context.CancelFunc
	if s.config.RequestTimeout > 0 {
		ctx, cancel = context.WithTimeout(r.Context(), s.config.RequestTimeout)
	} else {
		ctx, cancel = context.WithCancel(r.Context())
	}
	defer cancel()

	var tokens []string
	done := false
	for !done {
		select {
		case token, ok := <-seq.TokenChan():
			if !ok {
				done = true
			} else {
				tokens = append(tokens, token)
			}
		case <-ctx.Done():
			http.Error(w, "Request timed out", http.StatusRequestTimeout)
			return
		}
	}

	resp := map[string]string{
		"text": strings.Join(tokens, ""),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
