package vexel_test

import (
	"os"
	"testing"
)

func TestGoModExists(t *testing.T) {
	_, err := os.Stat("go.mod")
	if os.IsNotExist(err) {
		t.Fatal("go.mod file does not exist. Run 'go mod init' to initialize the module.")
	}
}
