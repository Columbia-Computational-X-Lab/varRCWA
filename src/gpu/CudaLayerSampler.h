#pragma once

#include "types.cuh"
#include "math_constants.h"
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <memory>

NAMESPACE_BEGIN(acacia::gpu::em)

class CudaFourierSolver2D;

class CudaLayerSampler {
private:
  cublasHandle_t cublasH_;
  cusolverDnHandle_t cusolverH_;
  
  std::shared_ptr<CudaFourierSolver2D> fourier_;
public:
	CudaLayerSampler(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH, 
    std::shared_ptr<CudaFourierSolver2D> fourier):
    cublasH_(cublasH), cusolverH_(cusolverH), fourier_(fourier) {}
  
  // Kx, Ky both gpu matrices of size (nx*ny) x (nx*ny);
  // P, Q pre-allocated gpu matrix of size (2*nx*ny) x (2*nx*ny);
  // because of layout issue, we seperate each P, Q to 4 blocks
  void sample(
    const complex_t *eps,
    const complex_t *Kx, 
    const complex_t *Ky, 
    Real lambda, 
    int nx, int ny,
    complex_t *P00, complex_t *P01, complex_t *P10, complex_t *P11,
    complex_t *Q00, complex_t *Q01, complex_t *Q10, complex_t *Q11,
    complex_t *workspace, int *iworkspace);

  size_t workspace_buffer_size(int nx, int ny);
  size_t iworkspace_buffer_size(int nx, int ny);
};


NAMESPACE_END(acacia::gpu::em)