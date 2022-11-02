#pragma once
#include "types.cuh"
#include <cuda_runtime_api.h>
NAMESPACE_BEGIN(acacia::gpu::em)

void square_root_in_valid_branch(complex_t * lam, int nDim);

void right_inverse_diag_matrix_multiply(
  complex_t * A, // input and output
  const complex_t * D, // diagonal matrix represent by a vector
  int nDim);

// if isLeftMul B = beta * B + alpha * diag(ph) * A
// else B = beta * B +  alpha * A * diag(ph)
// A and B can be the same address
void phase_diag_matrix_multiply(
  bool isLeftMul,
  Real k0, Real dz,
  Real alpha,
  const complex_t * lam,
  const complex_t * A,
  Real beta,
  complex_t * B,
  int nDim);

void add_phase_diagonal(Real k0, Real dz, 
  const complex_t * lam,
  complex_t * A, int nDim);

void add_uniform_real_diagonal(complex_t *A, Real x, int N, cudaStream_t stream = 0);

void set_uniform_real_diagonal(complex_t * A, Real x, int N);


// don't use this function, it is extremely slow
// only for accuracy comparison with cpu code
Real max_abs_complex_array(const complex_t * A, int N);

// the following function is just for debugging
void cudaMemcpy2DDeviceToDevice(
  complex_t *dst, int lda,
  const complex_t *src, int ldb,
  int rows, int cols);

void setDecreaseVector(complex_t * A, int nDim);

NAMESPACE_END(acacia::gpu::em)