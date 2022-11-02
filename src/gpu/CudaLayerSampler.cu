#include "CudaLayerSampler.h"
#include "CudaFourierSolver2D.h"
#include "MathKernelFunction.h"
#include <cstdio>
#include <iostream>


NAMESPACE_BEGIN(acacia::gpu::em)

void CudaLayerSampler::sample(
  const complex_t *eps,
  const complex_t *Kx, 
  const complex_t *Ky, 
  Real lambda, 
  int nx, int ny,
  complex_t *P00, complex_t *P01, complex_t *P10, complex_t *P11,
  complex_t *Q00, complex_t *Q01, complex_t *Q10, complex_t *Q11,
  complex_t *workspace, int *iworkspace)
{
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // P00 = Kx * F2h;
	// P01 = k02 - Kx * F1h;
  // P10 = Ky * F2h - k02;
  // P11 = -Ky * F1h;
  // cudaEventRecord(start);
  int blockDim = nx*ny;
  int offset = blockDim*blockDim;
  complex_t *fe33 = workspace;
  complex_t *d_work = workspace + offset;
  fourier_->solve<ContinuousXY>(fe33, nx, ny, eps, d_work, iworkspace);
  cudaMemcpy(P00, Kx, sizeof(complex_t)*offset, cudaMemcpyDeviceToDevice);
  int *d_info = iworkspace + blockDim;
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds1 = 0;
  // cudaEventElapsedTime(&milliseconds1, start, stop);
  // TODO pass this matrix to do cpu solve
  // TODO make sure small matrices are well conditioned
  // cudaEventRecord(start);
  // gpu solver
  acacia_gpu_em_Complexgetrf_(cusolverH_, 
    blockDim, blockDim,
    fe33, blockDim,
    d_work,
    nullptr, d_info);
  int h_info;
  cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  // printf("info: %d\n", h_info);
  // fe33 is factorized
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, blockDim, blockDim,
    fe33, blockDim,
    nullptr,
    P00, blockDim,
    d_info);
  cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  // printf("info: %d\n", h_info);
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds2 = 0;
  // cudaEventElapsedTime(&milliseconds2, start, stop);
  // cpu solver
  // printf("at %d\n", __LINE__);
  // Eigen::MatrixXcs fe33_cpu(blockDim, blockDim);
  // Eigen::MatrixXcs Kx_cpu(blockDim, blockDim);
  // Eigen::MatrixXcs Ky_cpu(blockDim, blockDim);
  // cudaMemcpy(fe33_cpu.data(), fe33, sizeof(complex_t)*offset, cudaMemcpyDeviceToHost);
  // cudaMemcpy(Kx_cpu.data(), Kx, sizeof(complex_t)*offset, cudaMemcpyDeviceToHost);
  // cudaMemcpy(Ky_cpu.data(), Ky, sizeof(complex_t)*offset, cudaMemcpyDeviceToHost);
  // printf("at %d\n", __LINE__);
  // Eigen::BDCSVD<Eigen::MatrixXcs> SVDSolver(fe33_cpu, 
  //   Eigen::ComputeThinU | Eigen::ComputeThinV);
  // printf("at %d\n", __LINE__);
	// Eigen::MatrixXcs F1h(blockDim, blockDim);
  // // F1h = SVDSolver.solve(Kx_cpu);
	// Eigen::MatrixXcs F2h(blockDim, blockDim);
  // // F2h = SVDSolver.solve(Ky_cpu);
  // printf("at %d\n", __LINE__);
  // cudaMemcpy(P00, F1h.data(), sizeof(complex_t)*offset, cudaMemcpyHostToDevice);
  // printf("at %d\n", __LINE__);
  // P00 (cached) = F1h;
  // cudaEventRecord(start);
  Real k0 = 2 * Pi / lambda;
  complex_t alpha{-1., 0.};
  complex_t beta{0., 0.};
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    blockDim, blockDim, blockDim,
    &alpha, 
    Kx, blockDim,
    P00, blockDim,
    &beta,
    P01, blockDim);
  // P01 = -Kx * F1h(cached in P00);
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds3 = 0;
  // cudaEventElapsedTime(&milliseconds3, start, stop);
  
  // add_uniform_real_diagonal_kernel<<<(blockDim+1024)/1024, 1024>>>(
  //   P01, k0*k0, blockDim);
  add_uniform_real_diagonal(P01, k0*k0, blockDim);
  // P01 = k02 - Kx * F1h

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    blockDim, blockDim, blockDim,
    &alpha, 
    Ky, blockDim,
    P00, blockDim,
    &beta,
    P11, blockDim);
  // P11 = -Ky * F1h;
    
  cudaMemcpy(Q10, Ky, sizeof(complex_t)*offset, cudaMemcpyDeviceToDevice);
  // printf("at %d\n", __LINE__);
  // Q10 (cached) = Ky
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, blockDim, blockDim,
    fe33, blockDim,
    nullptr,
    Q10, blockDim,
    d_info);
  // cudaMemcpy(Q10, F2h.data(), sizeof(complex_t)*offset, cudaMemcpyHostToDevice);
  // printf("at %d\n", __LINE__);

  // Q10 = F2h
  complex_t alpha1 = {1., 0.};
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    blockDim, blockDim, blockDim,
    &alpha1, 
    Kx, blockDim,
    Q10, blockDim,
    &beta,
    P00, blockDim);
  // P00 = Kx * F2h;
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    blockDim, blockDim, blockDim,
    &alpha1, 
    Ky, blockDim,
    Q10, blockDim,
    &beta,
    P10, blockDim);
  // P10 = Ky * F2h;
  // add_uniform_real_diagonal_kernel<<<(blockDim+1024)/1024, 1024>>>(
  //   P10, -k0*k0, blockDim);
  acacia::gpu::em::add_uniform_real_diagonal(P10, -k0*k0, blockDim);
  // P10 = Ky * F2h - k02;
  
  complex_t *fe11 = fe33;
  // cudaEventRecord(start);
  fourier_->solve<DiscontinuousX>(fe11, nx, ny, eps, d_work, iworkspace);
  cudaMemcpy(Q10, fe11, sizeof(complex_t)*offset, cudaMemcpyDeviceToDevice);

  complex_t beta1{k0*k0, 0.};
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    blockDim, blockDim, blockDim,
    &alpha, 
    Ky, blockDim,
    Ky, blockDim,
    &beta1,
    Q10, blockDim);  
	// Q10 = -Ky * Ky + k02 * fe11;
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds4 = 0;
  // cudaEventElapsedTime(&milliseconds4, start, stop);
  complex_t *fe22 = fe11;
  fourier_->solve<DiscontinuousY>(fe22, nx, ny, eps, d_work, iworkspace);  
  cudaMemcpy(Q01, fe22, sizeof(complex_t)*offset, cudaMemcpyDeviceToDevice);
  complex_t beta2{-k0*k0, 0.};
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    blockDim, blockDim, blockDim,
    &alpha1, 
    Kx, blockDim,
    Kx, blockDim,
    &beta2,
    Q01, blockDim); 
	// Q01 = Kx * Kx - k02 * fe22;

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    blockDim, blockDim, blockDim,
    &alpha, 
    Kx, blockDim,
    Ky, blockDim,
    &beta,
    Q00, blockDim);   
	// Q00 = -Kx * Ky;
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    blockDim, blockDim, blockDim,
    &alpha1, 
    Ky, blockDim,
    Kx, blockDim,
    &beta,
    Q11, blockDim);
   
	// Q11 = Ky * Kx;
  // printf("complete\n");
  // std::cout << "sample inner: " << milliseconds1 << "\t"
  // << milliseconds2 << "\t"
  // << milliseconds3 << "\t"
  //   << milliseconds4 << std::endl;
}

size_t CudaLayerSampler::workspace_buffer_size(int nx, int ny)
{
  // buffer size needed by fourier solver
  size_t wSize = fourier_->workspace_buffer_size<ContinuousXY>(nx, ny);
  wSize = max(wSize, fourier_->workspace_buffer_size<DiscontinuousX>(nx, ny));
  wSize = max(wSize, fourier_->workspace_buffer_size<DiscontinuousY>(nx, ny));
  // buffer size needed by factorizing fe33
  int blockDim = nx*ny;
  int lwork;
  acacia_gpu_em_Complexgetrf_bufferSize(
    cusolverH_, blockDim, blockDim,
    nullptr, blockDim, &lwork);
  wSize = max(wSize, (size_t)lwork);
  // buffer size needed by fe33
  wSize += blockDim * blockDim;
  return wSize;
}

size_t CudaLayerSampler::iworkspace_buffer_size(int nx, int ny)
{
  size_t iSize = fourier_->iworkspace_buffer_size<ContinuousXY>(nx, ny);
  iSize = max(iSize, fourier_->iworkspace_buffer_size<DiscontinuousX>(nx, ny));
  iSize = max(iSize, fourier_->iworkspace_buffer_size<DiscontinuousY>(nx, ny));
  iSize = max(iSize, size_t(nx*ny+1));
  return iSize;
}


NAMESPACE_END(acacia::gpu::em)