package safetensors

import (
	"fmt"
	"os"
	"syscall"
)

// MappedFile represents a memory-mapped file.
type MappedFile struct {
	data []byte
	file *os.File
}

// Mmap maps the file at the given path into memory.
func Mmap(path string) (*MappedFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}

	size := info.Size()
	if size == 0 {
		return &MappedFile{data: nil, file: f}, nil
	}

	// Use syscall for mmap
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("syscall.mmap failed: %w", err)
	}

	return &MappedFile{
		data: data,
		file: f,
	}, nil
}

// Close unmaps the file and closes it.
func (m *MappedFile) Close() error {
	if m.data != nil {
		if err := syscall.Munmap(m.data); err != nil {
			return err
		}
	}
	return m.file.Close()
}

// Bytes returns the mapped data.
func (m *MappedFile) Bytes() []byte {
	return m.data
}
