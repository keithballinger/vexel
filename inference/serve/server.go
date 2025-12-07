package serve

import (
	"encoding/json"
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
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
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
