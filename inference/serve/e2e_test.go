package serve

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
	"vexel/inference/runtime"
	"vexel/inference/scheduler"
)

func TestServer_E2E_Concurrent(t *testing.T) {
	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})
	server := NewServer(sched)
	ts := httptest.NewServer(server)
	defer ts.Close()

	// Simulate a simple scheduler "worker" that just returns the prompt backwards
	go func() {
		for {
			seqs := sched.GetSequences()
			for _, seq := range seqs {
				// Don't check state, just push if it has tokens and isn't closed
				// Actually, pushing and closing once is enough.
				// We can use the StateFinished to mark we are done with it here.
				if seq.State() != scheduler.StateFinished {
					seq.PushToken("Response for " + seq.ID().String())
					seq.SetState(scheduler.StateFinished)
					seq.Close()
				}
			}
			time.Sleep(100 * time.Microsecond)
		}
	}()

	var wg sync.WaitGroup
	numRequests := 5
	wg.Add(numRequests)

	for i := 0; i < numRequests; i++ {
		go func(id int) {
			defer wg.Done()
			reqBody, _ := json.Marshal(map[string]string{
				"prompt": "Request",
			})
			resp, err := http.Post(ts.URL+"/generate", "application/json", bytes.NewBuffer(reqBody))
			if err != nil {
				t.Errorf("Request %d failed: %v", id, err)
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				t.Errorf("Request %d got status %d", id, resp.StatusCode)
				return
			}

			var result map[string]string
			json.NewDecoder(resp.Body).Decode(&result)
			if result["text"] == "" {
				t.Errorf("Request %d got empty response", id)
			}
		}(i)
	}

	wg.Wait()
}

// Re-using contains helper or defining locally if needed.
