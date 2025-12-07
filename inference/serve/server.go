package serve

import (
	"encoding/json"
	"fmt"
	"net/http"
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

	// Mock Streaming loop
	// TODO: Subscribe to Scheduler updates
	tokens := []string{"Mock", " ", "streaming", " ", "response", " ", "for: ", req.Prompt}

	for _, token := range tokens {
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

	// TODO: Integrate with Scheduler
	// For now, return a mock response to pass the test structure.
	// Real implementation will:
	// 1. Create Sequence
	// 2. Add to Scheduler
	// 3. Wait for completion (using a channel or future)
	
	resp := map[string]string{
		"text": "Mock response for: " + req.Prompt,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
