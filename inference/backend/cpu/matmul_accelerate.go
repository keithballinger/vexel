//go:build darwin && cgo

package cpu

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

// cblas_sgemm wrapper for C = alpha * A @ B + beta * C
// For C = A @ B^T: use CblasTrans for B
void matmul_f32(float *a, float *b, float *c, int m, int n, int k, int transB) {
    // A: [M, K], B: [K, N] or [N, K] if transposed, C: [M, N]
    enum CBLAS_TRANSPOSE transA = CblasNoTrans;
    enum CBLAS_TRANSPOSE transBEnum = transB ? CblasTrans : CblasNoTrans;

    // lda = K (columns of A)
    // ldb = N if no trans, K if trans
    // ldc = N (columns of C)
    int lda = k;
    int ldb = transB ? k : n;
    int ldc = n;

    cblas_sgemm(CblasRowMajor, transA, transBEnum,
                m, n, k,
                1.0f,       // alpha
                a, lda,
                b, ldb,
                0.0f,       // beta
                c, ldc);
}
*/
import "C"

import (
	"unsafe"

	"vexel/inference/tensor"
)

// MatMulAccelerate performs C = A @ B using Apple's Accelerate framework.
// A: [M, K], B: [K, N], C: [M, N]
func (b *CPUBackend) MatMulAccelerate(a, bMat, out tensor.DevicePtr, m, n, k int) {
	aPtr := (*C.float)(unsafe.Pointer(a.Addr()))
	bPtr := (*C.float)(unsafe.Pointer(bMat.Addr()))
	outPtr := (*C.float)(unsafe.Pointer(out.Addr()))
	C.matmul_f32(aPtr, bPtr, outPtr, C.int(m), C.int(n), C.int(k), C.int(0))
}

// MatMulTransposedAccelerate performs C = A @ B^T using Apple's Accelerate framework.
// A: [M, K], B: [N, K] (stored row-major, so B^T is [K, N]), C: [M, N]
func (b *CPUBackend) MatMulTransposedAccelerate(a, bMat, out tensor.DevicePtr, m, n, k int) {
	aPtr := (*C.float)(unsafe.Pointer(a.Addr()))
	bPtr := (*C.float)(unsafe.Pointer(bMat.Addr()))
	outPtr := (*C.float)(unsafe.Pointer(out.Addr()))
	C.matmul_f32(aPtr, bPtr, outPtr, C.int(m), C.int(n), C.int(k), C.int(1))
}

// useAccelerate indicates that this build has Accelerate support
const useAccelerate = true
