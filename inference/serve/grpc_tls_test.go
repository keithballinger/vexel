package serve_test

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"vexel/inference/runtime"
	"vexel/inference/scheduler"
	"vexel/inference/serve"
	"vexel/inference/serve/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// generateTestCert creates a self-signed CA and server certificate for testing.
// Returns the paths to the cert and key files in a temp directory.
func generateTestCert(t *testing.T) (certFile, keyFile, caFile string) {
	t.Helper()

	dir := t.TempDir()

	// Generate CA
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate CA key: %v", err)
	}

	caTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "Test CA"},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
	}

	caCertDER, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("create CA cert: %v", err)
	}

	caCert, err := x509.ParseCertificate(caCertDER)
	if err != nil {
		t.Fatalf("parse CA cert: %v", err)
	}

	// Write CA cert
	caFile = filepath.Join(dir, "ca.pem")
	writePEM(t, caFile, "CERTIFICATE", caCertDER)

	// Generate server certificate
	serverKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate server key: %v", err)
	}

	serverTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: "localhost"},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		IPAddresses:  []net.IP{net.ParseIP("127.0.0.1")},
		DNSNames:     []string{"localhost"},
	}

	serverCertDER, err := x509.CreateCertificate(rand.Reader, serverTemplate, caCert, &serverKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("create server cert: %v", err)
	}

	certFile = filepath.Join(dir, "server.pem")
	writePEM(t, certFile, "CERTIFICATE", serverCertDER)

	keyDER, err := x509.MarshalECPrivateKey(serverKey)
	if err != nil {
		t.Fatalf("marshal server key: %v", err)
	}
	keyFile = filepath.Join(dir, "server-key.pem")
	writePEM(t, keyFile, "EC PRIVATE KEY", keyDER)

	return certFile, keyFile, caFile
}

func writePEM(t *testing.T, path, blockType string, data []byte) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer f.Close()
	if err := pem.Encode(f, &pem.Block{Type: blockType, Bytes: data}); err != nil {
		t.Fatalf("encode PEM %s: %v", path, err)
	}
}

// TestGRPCWithTLS verifies that the gRPC server works over TLS.
func TestGRPCWithTLS(t *testing.T) {
	certFile, keyFile, caFile := generateTestCert(t)

	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	// Simulate token generation
	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				seqs[0].PushToken("secure")
				seqs[0].Close()
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()

	// Create TLS server
	tlsConfig, err := serve.LoadServerTLS(certFile, keyFile)
	if err != nil {
		t.Fatalf("LoadServerTLS: %v", err)
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()

	serverCreds := credentials.NewTLS(tlsConfig)
	s := grpc.NewServer(grpc.Creds(serverCreds))
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)

	go func() { _ = s.Serve(lis) }()
	defer s.Stop()

	// Create TLS client with CA certificate
	caCertData, err := os.ReadFile(caFile)
	if err != nil {
		t.Fatalf("read CA: %v", err)
	}
	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caCertData) {
		t.Fatal("failed to add CA cert")
	}
	clientCreds := credentials.NewTLS(&tls.Config{
		RootCAs: caPool,
	})

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(clientCreds))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)
	resp, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "tls test"})
	if err != nil {
		t.Fatalf("Generate over TLS: %v", err)
	}
	if resp.Text != "secure" {
		t.Errorf("expected 'secure', got %q", resp.Text)
	}
}

// TestGRPCWithMutualTLS verifies mutual TLS (mTLS) authentication.
func TestGRPCWithMutualTLS(t *testing.T) {
	certFile, keyFile, caFile := generateTestCert(t)

	sched, _ := scheduler.NewScheduler(&runtime.ModelRuntime{}, nil, scheduler.Config{})

	go func() {
		for {
			seqs := sched.GetSequences()
			if len(seqs) > 0 {
				seqs[0].PushToken("mutual")
				seqs[0].Close()
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()

	// Create mutual TLS server config
	tlsConfig, err := serve.LoadMutualTLS(certFile, keyFile, caFile)
	if err != nil {
		t.Fatalf("LoadMutualTLS: %v", err)
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()

	serverCreds := credentials.NewTLS(tlsConfig)
	s := grpc.NewServer(grpc.Creds(serverCreds))
	srv := serve.NewGRPCServer(sched)
	pb.RegisterInferenceServiceServer(s, srv)

	go func() { _ = s.Serve(lis) }()
	defer s.Stop()

	// Client with both CA and client cert (reuse server cert for simplicity)
	clientCert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		t.Fatalf("load client cert: %v", err)
	}
	caCertData, _ := os.ReadFile(caFile)
	caPool := x509.NewCertPool()
	caPool.AppendCertsFromPEM(caCertData)

	clientCreds := credentials.NewTLS(&tls.Config{
		Certificates: []tls.Certificate{clientCert},
		RootCAs:      caPool,
	})

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(clientCreds))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)
	resp, err := client.Generate(context.Background(), &pb.GenerateRequest{Prompt: "mtls test"})
	if err != nil {
		t.Fatalf("Generate over mTLS: %v", err)
	}
	if resp.Text != "mutual" {
		t.Errorf("expected 'mutual', got %q", resp.Text)
	}
}

// TestLoadServerTLSInvalidPaths verifies error handling for missing cert files.
func TestLoadServerTLSInvalidPaths(t *testing.T) {
	_, err := serve.LoadServerTLS("/nonexistent/cert.pem", "/nonexistent/key.pem")
	if err == nil {
		t.Error("expected error for nonexistent cert files")
	}
}

// TestLoadMutualTLSInvalidCA verifies error handling for invalid CA file.
func TestLoadMutualTLSInvalidCA(t *testing.T) {
	certFile, keyFile, _ := generateTestCert(t)
	_, err := serve.LoadMutualTLS(certFile, keyFile, "/nonexistent/ca.pem")
	if err == nil {
		t.Error("expected error for nonexistent CA file")
	}
}
