#include "builder.h"
#include "CudaLayerSampler.h"
#include "CudaFourierSolver2D.h"
#include <iostream>
#include <chrono>
#include <memory>
#include <utility>
#include <cusolverDn.h>

NAMESPACE_BEGIN(acacia::gpu::em)
void build_rcwa_sample(int nCx, int nCy,
  const Real *x, 
  const Real *y,
  const std::complex<Real> *eps)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cublasHandle_t cublasH = NULL;
  cublasCreate(&cublasH);

  int nHx = 12;
  int nHy = 10;
  int nx = 2 * nHx + 1;
  int ny = 2 * nHy + 1;
  Real lambda = 1.55;

  Real *cu_x=nullptr, *cu_y=nullptr;
  complex_t *cu_eps=nullptr;
  cudaMalloc(&cu_x, sizeof(Real)*nCx);
  cudaMalloc(&cu_y, sizeof(Real)*nCy);
  cudaMalloc(&cu_eps, sizeof(complex_t)*(nCx-1)*(nCy-1));

  cudaMemcpy(cu_x, x, sizeof(Real)*nCx, cudaMemcpyHostToDevice);
  cudaMemcpy(cu_y, y, sizeof(Real)*nCy, cudaMemcpyHostToDevice);
  cudaMemcpy(cu_eps, eps, sizeof(complex_t)*(nCx-1)*(nCy-1),
    cudaMemcpyHostToDevice);
  
  auto fourier = 
    std::make_shared<CudaFourierSolver2D>(cusolverH, nCx, nCy, cu_x, cu_y);

  CudaLayerSampler sampler(cublasH, cusolverH, fourier);

  
  size_t wSize = sampler.workspace_buffer_size(nx, ny);
  complex_t *workspace=nullptr;
  cudaMalloc(&workspace, sizeof(complex_t)*wSize);
  size_t iSize = sampler.iworkspace_buffer_size(nx, ny);
  int *iworkspace=nullptr;
  cudaMalloc(&iworkspace, sizeof(int)*iSize);

  int edgeSize = nx*ny;
  int blockSize = edgeSize * edgeSize;

  complex_t *Kx=nullptr, *Ky=nullptr;
  cudaMalloc(&Kx, sizeof(complex_t)*blockSize);
  cudaMalloc(&Ky, sizeof(complex_t)*blockSize);
  // TODO set and copy Kx, Ky from cpu

  complex_t *P00=nullptr, 
    *P01=nullptr, 
    *P10=nullptr, 
    *P11=nullptr;
  complex_t *Q00=nullptr,
    *Q01=nullptr,
    *Q10=nullptr,
    *Q11=nullptr;
  cudaMalloc(&P00, sizeof(complex_t)*blockSize);
  cudaMalloc(&P01, sizeof(complex_t)*blockSize);
  cudaMalloc(&P10, sizeof(complex_t)*blockSize);
  cudaMalloc(&P11, sizeof(complex_t)*blockSize);
  cudaMalloc(&Q00, sizeof(complex_t)*blockSize);
  cudaMalloc(&Q01, sizeof(complex_t)*blockSize);
  cudaMalloc(&Q10, sizeof(complex_t)*blockSize);
  cudaMalloc(&Q11, sizeof(complex_t)*blockSize);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  sampler.sample(cu_eps, Kx, Ky, lambda, nx, ny, 
    P00, P01, P10, P11,
    Q00, Q01, Q10, Q11,
    workspace, iworkspace);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop); 
  printf("loop time: %fms\n", milliseconds);
  
  cudaFree(P00);
  cudaFree(P01);
  cudaFree(P10);
  cudaFree(P11);
  cudaFree(Q00);
  cudaFree(Q01);
  cudaFree(Q10);
  cudaFree(Q11);

  cudaFree(Kx);
  cudaFree(Ky);

  cudaFree(iworkspace);
  cudaFree(workspace);
  cudaFree(cu_x);
  cudaFree(cu_y);
  cudaFree(cu_eps);
  if (cublasH) cublasDestroy(cublasH);
  if (cusolverH) cusolverDnDestroy(cusolverH);
}

NAMESPACE_END(acacia::gpu::em)