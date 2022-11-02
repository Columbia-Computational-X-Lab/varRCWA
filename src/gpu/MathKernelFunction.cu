#include "MathKernelFunction.h"
#include <cstdio>
NAMESPACE_BEGIN(acacia::gpu::em)

__device__ Real MathKernelFunction_Buffer_;
__global__ void square_root_in_valid_branch_kernel(
  complex_t *lam, int N)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    Real nrm = sqrt(lam[i].x * lam[i].x + lam[i].y * lam[i].y);
    Real na = sqrt(0.5*(nrm + lam[i].x));
    Real nb = sqrt(0.5*(nrm - lam[i].x));
    if (lam[i].y >= 0.) {
      lam[i].x = na;
      lam[i].y = nb;
    } else {
      lam[i].x = -na;
      lam[i].y = nb;
    }
    
    if (abs(lam[i].y) < abs(lam[i].x) && lam[i].x < 0) {
      lam[i].x *= -1.;
      lam[i].y *= -1.;
    }
  }
}


void square_root_in_valid_branch(complex_t * lam, int nDim)
{
  square_root_in_valid_branch_kernel<<<(nDim+1024)/1024, 1024>>>(lam, nDim);
}

__global__ void right_inverse_diag_matrix_multiply_kernel(
  complex_t * A,
  const complex_t * D,
  int nDim)
{
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nDim && j < nDim) {
    complex_t d = D[j];
    Real nd2 = d.x*d.x+d.y*d.y;
    complex_t invd{d.x/nd2, -d.y/nd2};
    complex_t a = A[j*nDim+i];
    A[j*nDim+i].x = a.x*invd.x - a.y*invd.y;
    A[j*nDim+i].y = a.x*invd.y + a.y*invd.x;
  }
}

void right_inverse_diag_matrix_multiply(
  complex_t * A,
  const complex_t * D,
  int nDim)
{
  dim3 blockSize(32, 32, 1);
  int bx = (nDim + blockSize.x-1)/blockSize.x;
  int by = (nDim + blockSize.y-1)/blockSize.y;
  dim3 gridSize(bx, by, 1);
  right_inverse_diag_matrix_multiply_kernel<<<gridSize, blockSize>>>(A, D, nDim);
}

__global__ void phase_diag_matrix_multiply_kernel(
  bool isLeftMul,
  Real k0, Real dz,
  Real alpha, const complex_t * lam, const complex_t * A,
  Real beta, complex_t * B,
  int nDim)
{
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nDim && j < nDim) {
    int ind = j*nDim+i;
    int indG = -1;
    if (isLeftMul) {
      indG = i; // left mul -> row by row
    } else {
      indG = j; // right mul -> col by col
    }
    complex_t g = lam[indG];
    Real ikz_r = -g.y * dz / k0;
    Real ikz_i = g.x * dz / k0;
    Real ph_r = exp(ikz_r) * cos(ikz_i);
    Real ph_i = exp(ikz_r) * sin(ikz_i);
    complex_t a = A[ind];
    B[ind].x = beta * B[ind].x + alpha * (ph_r * a.x - ph_i * a.y);
    B[ind].y = beta * B[ind].y + alpha * (ph_r * a.y + ph_i * a.x);
  }
}

void phase_diag_matrix_multiply(
  bool isLeftMul,
  Real k0, Real dz,
  Real alpha, const complex_t * lam, const complex_t * A,
  Real beta, complex_t * B,
  int nDim)
{
  dim3 blockSize(32, 32, 1);
  int bx = (nDim + blockSize.x-1)/blockSize.x;
  int by = (nDim + blockSize.y-1)/blockSize.y;
  dim3 gridSize(bx, by, 1);
  phase_diag_matrix_multiply_kernel<<<gridSize, blockSize>>>(
    isLeftMul, k0, dz,
    alpha, lam, A,
    beta, B, nDim);
}

__global__ void add_phase_diagonal_kernel(Real k0, Real dz, 
  const complex_t * lam,
  complex_t * A, int N)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int ind = i*N+i;
    complex_t g = lam[i];
    Real ikz_r = -g.y * dz / k0;
    Real ikz_i = g.x * dz / k0;
    A[ind].x += exp(ikz_r) * cos(ikz_i);
    A[ind].y += exp(ikz_r) * sin(ikz_i);
  }
}

void add_phase_diagonal(Real k0, Real dz, 
  const complex_t * lam,
  complex_t * A, int nDim)
{
  add_phase_diagonal_kernel<<<(nDim+1024)/1024, 1024>>>(k0, dz, lam, A, nDim);
}

__global__ void cudaMemcpy2DDeviceToDevice_kernel(
  complex_t *dst, int lda,
  const complex_t *src, int ldb,
  int rows, int cols)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < rows && j < cols) {
    dst[i+j*lda].x = src[i+j*ldb].x;
    dst[i+j*lda].y = src[i+j*ldb].y;
  }
}


void cudaMemcpy2DDeviceToDevice(
  complex_t *dst, int lda,
  const complex_t *src, int ldb,
  int rows, int cols)
{
  dim3 blockSize(32, 32, 1);
  int bx = (rows + blockSize.x-1)/blockSize.x;
  int by = (cols + blockSize.y-1)/blockSize.y;
  dim3 gridSize(bx, by, 1);
  cudaMemcpy2DDeviceToDevice_kernel<<<gridSize, blockSize>>>(dst, lda, 
    src, ldb, rows, cols);
}

__global__ void setDecreaseVector_kernel(complex_t * A, int nDim)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nDim) {
    A[i].x = 1. / (1. + (double)i);
    A[i].y = 0.;
  }
}
void setDecreaseVector(complex_t * A, int nDim)
{
  setDecreaseVector_kernel<<<(nDim+1024)/1024, 1024>>>(A, nDim);
}


__global__ void add_uniform_real_diagonal_kernel(
  complex_t *A, Real x, int N)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int ind = i*N+i;
    A[ind].x += x;
  }
}

void add_uniform_real_diagonal(complex_t *A, Real x, int N, cudaStream_t stream)
{
  add_uniform_real_diagonal_kernel<<<(N+1024)/1024, 1024, 0, stream>>>(A, x, N);
}

__global__ void set_uniform_real_diagonal_kernel(complex_t * A, Real x, int N)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) {
    int ind = i*N+j;
    if (i == j) {
      A[ind].x = x;
    } else {
      A[ind].x = 0.;
    }
    A[ind].y = 0;
  }
}

void set_uniform_real_diagonal(complex_t * A, Real x, int N)
{
  dim3 blockSize(32, 32, 1);
  int bx = (N + blockSize.x-1)/blockSize.x;
  int by = (N + blockSize.y-1)/blockSize.y;
  dim3 gridSize(bx, by, 1);
  set_uniform_real_diagonal_kernel<<<gridSize, blockSize>>>(A, x, N);
}

__global__ void max_abs_complex_array_kernel(const complex_t * A, int N)
{
  MathKernelFunction_Buffer_ = 0.;
  for (int i = 0; i < N; ++i) {
    MathKernelFunction_Buffer_ = max(MathKernelFunction_Buffer_, 
      sqrt(A[i].x*A[i].x + A[i].y*A[i].y));
  }
}

Real max_abs_complex_array(const complex_t * A, int N)
{
  max_abs_complex_array_kernel<<<1, 1>>>(A, N);
  Real v = 0.;
  cudaMemcpyFromSymbol(&v, MathKernelFunction_Buffer_, 
    sizeof(MathKernelFunction_Buffer_), 0, cudaMemcpyDeviceToHost);
  return v;
}

NAMESPACE_END(acacia::gpu::em)