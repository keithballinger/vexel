package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
	"vexel/inference/scheduler"
)

// Server handles HTTP/gRPC requests for inference.
type Server struct {
	scheduler *scheduler.Scheduler
	mux       *http.ServeMux
}

// NewServer creates a new inference server.
func NewServer(sched *scheduler.Scheduler) *Server {
	s := &Server{
		scheduler: sched,
		mux:       http.NewServeMux(),
	}
	s.routes()
	return s
}

func (s *Server) routes() {
	s.mux.HandleFunc("/generate", s.handleGenerate)
	s.mux.HandleFunc("/stream", s.handleStream)
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
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
	seqID := scheduler.SequenceID(time.Now().UnixNano())
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
	// TODO: UUID generation. For now, pseudo-random or counter.
	// Since we don't have a shared counter yet, let's use a time-based ID for demo.
	seqID := scheduler.SequenceID(time.Now().UnixNano())
	seq := scheduler.NewSequence(seqID, req.Prompt)
	
		// 2. Add to Scheduler
		s.scheduler.AddSequence(seq)
		defer s.scheduler.RemoveSequence(seqID)
	
			// 3. Wait for completion and collect tokens
			ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
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
