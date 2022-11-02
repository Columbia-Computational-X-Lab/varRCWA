#include "RCWAIntegratorGPU.h"
#include "core/LayerSampler.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <magma_v2.h>
#include "rcwa/MKLEigenSolver.hpp"
#include "MathKernelFunction.h"
#include "utils/timer.hpp"


using namespace std;


RCWAIntegratorGPU::RCWAIntegratorGPU(scalar z0, scalar z1, 
  std::shared_ptr<LayerSampler> sampler):
  z0_(z0), z1_(z1), sampler_(sampler), N_(0), nEigVal_(0)
{
  magma_init();
  int nDim = sampler->nDim();
  int matSize = nDim*nDim;

  cublasCreate(&cublasH_);
  cusolverDnCreate(&cusolverH_);

  cudaMalloc(reinterpret_cast<void**>(&Tdd_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&Tuu_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&Rud_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&Rdu_), sizeof(acacia::gpu::complex_t)*matSize);

  cudaMalloc(reinterpret_cast<void**>(&tdd_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&tuu_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&rud_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&rdu_), sizeof(acacia::gpu::complex_t)*matSize);

  cudaMalloc(reinterpret_cast<void**>(&buffer_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&buffer2_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&buffer3_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&buffer4_), sizeof(acacia::gpu::complex_t)*matSize);

  int lwork;
  acacia_gpu_em_Complexgetrf_bufferSize(
    cusolverH_, nDim, nDim,
    nullptr, nDim, &lwork);
  cudaMalloc(reinterpret_cast<void**>(&dwork_), sizeof(acacia::gpu::complex_t)*lwork);

  cudaMalloc(reinterpret_cast<void**>(&iwork_), sizeof(int)*nDim);
  cudaMalloc(reinterpret_cast<void**>(&info_), sizeof(int));

  int nb = magma_get_zgehrd_nb(nDim);
  magma_lwork_ = nDim*(1 + 2*nb);
  int lwork2 = max(magma_lwork_, nDim*(5 + 2*nDim));
  magma_zmalloc_pinned(&magma_PQ_, matSize);
  magma_zmalloc_pinned(&magma_VR_, matSize);
  magma_zmalloc_pinned(&magma_work_, lwork2);
}

RCWAIntegratorGPU::~RCWAIntegratorGPU()
{
  magma_free_pinned(magma_PQ_);
  magma_free_pinned(magma_VR_);
  magma_free_pinned(magma_work_);

  cudaFree(info_);
  cudaFree(iwork_);

  cudaFree(dwork_);

  cudaFree(buffer_);
  cudaFree(buffer2_);
  cudaFree(buffer3_);
  cudaFree(buffer4_);

  cudaFree(tdd_);
  cudaFree(tuu_);
  cudaFree(rud_);
  cudaFree(rdu_);

  cudaFree(Tdd_);
  cudaFree(Tuu_);
  cudaFree(Rud_);
  cudaFree(Rdu_);

  if (cublasH_) cublasDestroy(cublasH_);
  if (cusolverH_) cusolverDnDestroy(cusolverH_);

  magma_finalize();
}

void RCWAIntegratorGPU::evaluateEigenvalueAndSetReference(
  std::shared_ptr<ReferenceSampling> ref)
{
  int nDim = sampler_->nDim();
  int matSize = nDim*nDim;
  acacia::gpu::complex_t alpha{1., 0.};
  acacia::gpu::complex_t beta{0., 0.};
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha, 
    ref->Pr, nDim,
    ref->Qr, nDim,
    &beta,
    buffer_, nDim);

  // ===============================
  // GPU version code
  // ===============================
  // example:
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);
  acacia::gpu::complex_t *w1=nullptr;
  double *rwork=nullptr;
  
  magma_zmalloc_cpu(&w1, nDim);
  magma_dmalloc_cpu(&rwork, 2*nDim);
  cudaMemcpy(magma_PQ_, buffer_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToHost);
  int info;
  // cudaEventRecord(start);
  magma_zgeev(MagmaNoVec, MagmaVec,
    nDim, magma_PQ_, nDim, w1,
    nullptr, 1, magma_VR_, nDim,
    magma_work_, magma_lwork_, rwork, &info);
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  cudaMemcpy(ref->W, magma_VR_, sizeof(acacia::gpu::complex_t)*matSize,
    cudaMemcpyHostToDevice);
  cudaMemcpy(ref->Lam, w1, sizeof(acacia::gpu::complex_t)*nDim,
    cudaMemcpyHostToDevice);
  magma_free_cpu(w1);
  magma_free_cpu(rwork);
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // cout << "eigval cost: " << milliseconds << " ms.\n";
  // ===============================
  // Eigen::MatrixXcs PQ(nDim, nDim);
  // cudaMemcpy(PQ.data(), buffer_, sizeof(acacia::gpu::complex_t)*matSize, 
  //   cudaMemcpyDeviceToHost);
    
  // auto t1 = sploosh::now();
  // MKLEigenSolver<Eigen::MatrixXcs> ces;
  // ces.compute(PQ);
  // auto t2 = sploosh::now();

  // Eigen::VectorXcs eigval = ces.eigenvalues();
  // Eigen::MatrixXcs W = ces.eigenvectors();
  
  // cudaMemcpy(ref->W, W.data(), sizeof(acacia::gpu::complex_t)*matSize,
  //   cudaMemcpyHostToDevice);
  // cudaMemcpy(ref->Lam, eigval.data(), sizeof(acacia::gpu::complex_t)*nDim,
  //   cudaMemcpyHostToDevice);
  
  // ================================
  acacia::gpu::em::square_root_in_valid_branch(ref->Lam, nDim);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha, 
    ref->Qr, nDim,
    ref->W, nDim,
    &beta,
    ref->V, nDim);
  acacia::gpu::em::right_inverse_diag_matrix_multiply(ref->V, ref->Lam, nDim);
  // ref.V = ref.Qr * ref.W * (ref.Lam)^{-1};

  // cout << "W: " << maxOnGPU(ref->W, matSize) << endl;
  // cout << "V: " << maxOnGPU(ref->V, matSize) << endl;
  // cout << "Lam: " << maxOnGPU(ref->Lam, nDim) << endl;

  cudaMemcpy(ref->facW, ref->W, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(ref->facV, ref->V, sizeof(acacia::gpu::complex_t)*matSize,
    cudaMemcpyDeviceToDevice);

  acacia_gpu_em_Complexgetrf_(cusolverH_, 
    nDim, nDim,
    ref->facW, nDim,
    dwork_,
    ref->pivW, info_);
  acacia_gpu_em_Complexgetrf_(cusolverH_, 
    nDim, nDim,
    ref->facV, nDim,
    dwork_,
    ref->pivV, info_);  
}

void RCWAIntegratorGPU::compute(int N)
{
  int nDim = sampler_->nDim();

  acacia::gpu::em::set_uniform_real_diagonal(Tuu_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Tdd_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rud_, 0., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rdu_, 0., nDim);

  samp0_ = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z0_, samp0_->Pr, samp0_->Qr);
  evaluateEigenvalueAndSetReference(samp0_);

  samp2_ = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z1_, samp2_->Pr, samp2_->Qr);
  evaluateEigenvalueAndSetReference(samp2_);

  auto samp = std::make_shared<ReferenceSampling>(nDim);

  leftRef_ = std::make_shared<ReferenceSampling>(nDim);

  copyRenferenceSampling(leftRef_, samp0_, nDim);
  scalar dz = (z1_ - z0_) / static_cast<scalar>(N);

  for (int i = 0; i < N; ++i) {
      cout << i << endl;
      scalar zi0 = z0_ + i * dz;
      scalar zi1 = zi0 + dz;
      // scalar z = 0.5 * (zi0 + zi1);
      scalar z = zi0;
      sampler_->sampleOnGPU(z, samp->Pr, samp->Qr);
      evaluateEigenvalueAndSetReference(samp);

      evaluateScatteringMatrix(samp, zi0, zi1, tdd_, tuu_, rud_, rdu_);
      reprojectLeftEnd(leftRef_, samp, tdd_, tuu_, rud_, rdu_);
      copyRenferenceSampling(leftRef_, samp, nDim);
      
      redhefferStarRightProduct(Tdd_, Tuu_, Rud_, Rdu_,
        tdd_, tuu_, rud_, rdu_);    
  }
  reprojectRightEnd(samp, samp2_, Tdd_, Tuu_, Rud_, Rdu_);
}

void RCWAIntegratorGPU::redhefferStarRightProduct(
  acacia::gpu::complex_t * Tdd,
  acacia::gpu::complex_t * Tuu,
  acacia::gpu::complex_t * Rud,
  acacia::gpu::complex_t * Rdu,
  const acacia::gpu::complex_t * tdd,
  const acacia::gpu::complex_t * tuu,
  const acacia::gpu::complex_t * rud,
  const acacia::gpu::complex_t * rdu)
{
  int nDim = sampler_->nDim();
  int matSize = nDim*nDim;
  acacia::gpu::complex_t alpha{-1., 0.};
  acacia::gpu::complex_t alpha1{1., 0.};
  acacia::gpu::complex_t beta{0., 0.};

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha, 
    rdu, nDim,
    Rud, nDim,
    &beta,
    buffer_, nDim);
  acacia::gpu::em::add_uniform_real_diagonal(buffer_, 1., nDim);
  // buffer_ = I - rdu * Rud;

  acacia_gpu_em_Complexgetrf_(cusolverH_, 
    nDim, nDim,
    buffer_, nDim,
    dwork_,
    iwork_, info_);
  
  cudaMemcpy(buffer2_, rdu, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    buffer_, nDim,
    iwork_,
    buffer2_, nDim,
    info_);
  // buffer2_ = C1 * rdu;
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    Rud, nDim,
    buffer2_, nDim,
    &beta,
    buffer3_, nDim);
  acacia::gpu::em::add_uniform_real_diagonal(buffer3_, 1., nDim);
  // buffer3_ = C2;
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer3_, nDim,
    Tuu, nDim,
    &beta,
    buffer2_, nDim);
  // buffer2_ = C2 * Tuu;

  cudaMemcpy(buffer3_, tdd, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);  
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    buffer_, nDim,
    iwork_,
    buffer3_, nDim,
    info_);
  // buffer3_ = C1 * tdd;

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    Tdd, nDim,
    rdu, nDim,
    &beta,
    buffer_, nDim);
  // buffer_ = Tdd * rdu;
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer_, nDim,
    buffer2_, nDim,
    &beta,
    buffer4_, nDim);
  // buffer4_ = Tdd * rdu * C2_Tuu;
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer4_, 1, Rdu, 1);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    Tdd, nDim,
    buffer3_, nDim,
    &beta,
    buffer_, nDim);
  cudaMemcpy(Tdd, buffer_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    tuu, nDim,
    Rud, nDim,
    &beta,
    buffer_, nDim);  
  // buffer_ = tuu * Rud;
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer_, nDim,
    buffer3_, nDim,
    &beta,
    Rud, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, rud, 1, Rud, 1);   

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    tuu, nDim,
    buffer2_, nDim,
    &beta,
    Tuu, nDim);
  
  // MatrixXcs invC1 = - rdu * Rud;
  // invC1.diagonal().array() += 1.0;
  // PartialPivLU<MatrixXcs> cycleSolver1(invC1);
  // MatrixXcs C1 = cycleSolver1.inverse(); 
  // MatrixXcs C2 = Rud_ * C1 * rdu; 
  // C2.diagonal().array() += 1.0;

  // Rdu = Rdu + Tdd * rdu * C2_Tuu;
  // Tdd = Tdd * C1_tdd; 
  // Rud = rud + tuu * Rud * C1_tdd;
  // Tuu = tuu * C2_Tuu;    
}


void RCWAIntegratorGPU::evaluateScatteringMatrix(
  std::shared_ptr<const ReferenceSampling> ref,
  scalar z0, scalar z1,
  acacia::gpu::complex_t * tdd,
  acacia::gpu::complex_t * tuu,
  acacia::gpu::complex_t * rud,
  acacia::gpu::complex_t * rdu)
{
  int nDim = sampler_->nDim();
  scalar lambda = sampler_->lambda();
  scalar k0 = 2 * Pi / lambda;
  scalar dz = z1 - z0;
  acacia::gpu::em::set_uniform_real_diagonal(tuu, 0., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(tdd, 0., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(rud, 0., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(rdu, 0., nDim);

  acacia::gpu::em::add_phase_diagonal(k0, dz, ref->Lam, tuu, nDim);
  acacia::gpu::em::add_phase_diagonal(k0, dz, ref->Lam, tdd, nDim);
}

void RCWAIntegratorGPU::reprojectLeftEnd(
  std::shared_ptr<const ReferenceSampling> lref,
  std::shared_ptr<const ReferenceSampling> rref,
  acacia::gpu::complex_t * tdd,
  acacia::gpu::complex_t * tuu,
  acacia::gpu::complex_t * rud,
  acacia::gpu::complex_t * rdu)
{
  acacia::gpu::complex_t alpha{-1., 0.};
  acacia::gpu::complex_t alpha1{1., 0.};
  acacia::gpu::complex_t beta{0., 0.};
  int nDim = sampler_->nDim();
  int matSize = nDim*nDim;
  cudaMemcpy(buffer_, lref->W, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    rref->facW, nDim,
    rref->pivW,
    buffer_, nDim,
    info_);
  // buffer_ = invW_W0;
  cudaMemcpy(buffer2_, lref->V, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    rref->facV, nDim,
    rref->pivV,
    buffer2_, nDim,
    info_);
  // buffer2_ = invV_V0;

  cudaMemcpy(buffer3_, buffer_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha, buffer2_, 1, buffer3_, 1);
  
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer2_, 1, buffer_, 1);
  
  acacia::gpu::complex_t mul{0.5, 0.};
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, buffer_, 1);
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, buffer3_, 1);
  // buffer_ = A;
  // buffer3_ = B;
  // const MatrixXcs& invW_W0 = solver->invW_Sol().solve(W0_);
  // const MatrixXcs& invV_V0 = solver->invV_Sol().solve(V0_);

  // const MatrixXcs& A2 = 0.5 * (invW_W0 + invV_V0);
  // const MatrixXcs& B2 = 0.5 * (invW_W0 - invV_V0);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha, 
    rdu, nDim,
    buffer3_, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer_, 1, buffer2_, 1);
  // buffer2_ = A2 - Rdu * B2;
  acacia_gpu_em_Complexgetrf_(cusolverH_, 
    nDim, nDim,
    buffer2_, nDim,
    dwork_,
    iwork_, info_);
  // PartialPivLU<MatrixXcs> inv_A2_minus_Rdu_B2;
  // inv_A2_minus_Rdu_B2.compute(A2 - Rdu * B2);

  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    buffer2_, nDim,
    iwork_,
    tdd, nDim,
    info_);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    rdu, nDim,
    buffer_, nDim,
    &beta,
    buffer4_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha, buffer3_, 1, buffer4_, 1);

  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    buffer2_, nDim,
    iwork_,
    buffer4_, nDim,
    info_);
  cudaMemcpy(rdu, buffer4_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  // Rdu = inv_A2_minus_Rdu_B2.solve(Rdu * A2 - B2);
  // Tdd = inv_A2_minus_Rdu_B2.solve(Tdd);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    tuu, nDim,
    buffer3_, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer2_, nDim,
    tdd, nDim,
    &beta,
    buffer4_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer4_, 1, rud, 1);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer3_, nDim,
    rdu, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer_, 1, buffer2_, 1);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    tuu, nDim,
    buffer2_, nDim,
    &beta,
    buffer4_, nDim);
  cudaMemcpy(tuu, buffer4_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);  
  // Rud = Tuu * B2 * Tdd + Rud;
  // Tuu = Tuu * (B2 * Rdu + A2);      
}


void RCWAIntegratorGPU::reprojectRightEnd(
  std::shared_ptr<const ReferenceSampling> lref,
  std::shared_ptr<const ReferenceSampling> rref,
  acacia::gpu::complex_t * tdd,
  acacia::gpu::complex_t * tuu,
  acacia::gpu::complex_t * rud,
  acacia::gpu::complex_t * rdu)
{
  acacia::gpu::complex_t alpha{-1., 0.};
  acacia::gpu::complex_t alpha1{1., 0.};
  acacia::gpu::complex_t beta{0., 0.};
  int nDim = sampler_->nDim();
  int matSize = nDim*nDim;
  cudaMemcpy(buffer_, rref->W, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    lref->facW, nDim,
    lref->pivW,
    buffer_, nDim,
    info_);
  // buffer_ = invW0_W;
  cudaMemcpy(buffer2_, rref->V, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    lref->facV, nDim,
    lref->pivV,
    buffer2_, nDim,
    info_);
  // buffer2_ = invV0_V;

  cudaMemcpy(buffer3_, buffer_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha, buffer2_, 1, buffer3_, 1);
  // buffer3_ = B;
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer2_, 1, buffer_, 1);
  // buffer_ = A;

  acacia::gpu::complex_t mul{0.5, 0.};
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, buffer_, 1);
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, buffer3_, 1);
  // const MatrixXcs& invW0_W = invW0_Sol.solve(samp1.W);
  // const MatrixXcs& invV0_V = invV0_Sol.solve(samp1.V);
  // const MatrixXcs& A = 0.5 * (invW0_W + invV0_V);
  // const MatrixXcs& B = 0.5 * (invW0_W - invV0_V);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha, 
    rud, nDim,
    buffer3_, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer_, 1, buffer2_, 1);
  acacia_gpu_em_Complexgetrf_(cusolverH_, 
    nDim, nDim,
    buffer2_, nDim,
    dwork_,
    iwork_, info_);
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    buffer2_, nDim,
    iwork_,
    tuu, nDim,
    info_);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha, 
    rud, nDim,
    buffer_, nDim,
    &beta,
    buffer4_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha, buffer3_, 1, buffer4_, 1);

  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    buffer2_, nDim,
    iwork_,
    buffer4_, nDim,
    info_);
  cudaMemcpy(rud, buffer4_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  
  // now buffer2, buffer4, is free
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    tdd, nDim,
    buffer3_, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer2_, nDim,
    tuu, nDim,
    &beta,
    buffer4_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer4_, 1, rdu, 1);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer3_, nDim,
    rud, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer_, 1, buffer2_, 1);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    tdd, nDim,
    buffer2_, nDim,
    &beta,
    buffer4_, nDim);
  cudaMemcpy(tdd, buffer4_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  // PartialPivLU<MatrixXcs> inv_A_minus_Rud_B_Sol;
  // inv_A_minus_Rud_B_Sol.compute(A - Rud_ * B);
  // Tuu_ = inv_A_minus_Rud_B_Sol.solve(Tuu_);
  // Rud_ = inv_A_minus_Rud_B_Sol.solve(Rud_ * A - B);
  // Rdu_ = Rdu_ + Tdd_ * B * Tuu_;
  // Tdd_ = Tdd_ * (B * Rud_ + A);  
}

Eigen::MatrixXcs RCWAIntegratorGPU::copy_from_gpu(const acacia::gpu::complex_t *A, 
  int row, int col) const
{
  Eigen::MatrixXcs temp(row, col);
  cudaMemcpy(temp.data(), A, sizeof(acacia::gpu::complex_t)*row*col, 
    cudaMemcpyDeviceToHost);
  return temp;
}
    

Eigen::MatrixXcs RCWAIntegratorGPU::Tdd() const
{
  return copy_from_gpu(Tdd_, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RCWAIntegratorGPU::Tuu() const
{
  return copy_from_gpu(Tuu_, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RCWAIntegratorGPU::Rud() const
{
  return copy_from_gpu(Rud_, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RCWAIntegratorGPU::Rdu() const
{
  return copy_from_gpu(Rdu_, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RCWAIntegratorGPU::W0() const
{
  return copy_from_gpu(samp0_->W, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RCWAIntegratorGPU::V0() const
{
  return copy_from_gpu(samp0_->V, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RCWAIntegratorGPU::W1() const
{
  return copy_from_gpu(samp2_->W, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RCWAIntegratorGPU::V1() const
{
  return copy_from_gpu(samp2_->V, sampler_->nDim(), sampler_->nDim());
}