#include "RedhefferIntegratorGPU.h"
#include "core/LayerSampler.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <magma_v2.h>
#include "rcwa/MKLEigenSolver.hpp"
#include "MathKernelFunction.h"
#include "utils/timer.hpp"

using namespace std;


ReferenceSampling::ReferenceSampling(int nDim, bool IsAllocForEig):
IsAllocForEig_(IsAllocForEig)
{
  cudaMalloc(reinterpret_cast<void**>(&(Pr)),
      sizeof(acacia::gpu::complex_t)*nDim*nDim);
  cudaMalloc(reinterpret_cast<void**>(&(Qr)),
    sizeof(acacia::gpu::complex_t)*nDim*nDim);

  if (IsAllocForEig) allocForEig(nDim);
}

void ReferenceSampling::allocForEig(int nDim)
{
  cudaMalloc(reinterpret_cast<void**>(&(W)),
      sizeof(acacia::gpu::complex_t)*nDim*nDim);
  cudaMalloc(reinterpret_cast<void**>(&(V)),
      sizeof(acacia::gpu::complex_t)*nDim*nDim);
  cudaMalloc(reinterpret_cast<void**>(&(Lam)),
      sizeof(acacia::gpu::complex_t)*nDim);
  cudaMalloc(reinterpret_cast<void**>(&(facW)),
      sizeof(acacia::gpu::complex_t)*nDim*nDim);
  cudaMalloc(reinterpret_cast<void**>(&(facV)),
      sizeof(acacia::gpu::complex_t)*nDim*nDim);
  cudaMalloc(reinterpret_cast<void**>(&(pivW)),
      sizeof(int)*(nDim+1));
  cudaMalloc(reinterpret_cast<void**>(&(pivV)),
      sizeof(int)*(nDim+1));
  IsAllocForEig_ = true;
}

ReferenceSampling::~ReferenceSampling()
{
  cudaFree(Pr);
  cudaFree(Qr);

  if (IsAllocForEig_) {
    cudaFree(W);
    cudaFree(V);
    cudaFree(Lam);

    cudaFree(facW);
    cudaFree(facV);  

    cudaFree(pivW);
    cudaFree(pivV);
  }
}

void copyPQ(std::shared_ptr<ReferenceSampling> lhs,
  std::shared_ptr<ReferenceSampling> rhs, int nDim)
{
  int matSize = nDim*nDim;
  cudaMemcpy(lhs->Pr, rhs->Pr, sizeof(acacia::gpu::complex_t)*matSize,
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(lhs->Qr, rhs->Qr, sizeof(acacia::gpu::complex_t)*matSize,
    cudaMemcpyDeviceToDevice);
}

void copyRenferenceSampling(std::shared_ptr<ReferenceSampling> lhs,
  std::shared_ptr<ReferenceSampling> rhs, int nDim)
{
  int matSize = nDim*nDim;
  cudaMemcpy(lhs->Pr, rhs->Pr, sizeof(acacia::gpu::complex_t)*matSize,
    cudaMemcpyDeviceToDevice);
  cudaMemcpy(lhs->Qr, rhs->Qr, sizeof(acacia::gpu::complex_t)*matSize,
    cudaMemcpyDeviceToDevice);
  if (lhs->hasEigMem() && rhs->hasEigMem()) {
    cudaMemcpy(lhs->W, rhs->W, sizeof(acacia::gpu::complex_t)*matSize,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(lhs->V, rhs->V, sizeof(acacia::gpu::complex_t)*matSize,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(lhs->facW, rhs->facW, sizeof(acacia::gpu::complex_t)*matSize,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(lhs->facV, rhs->facV, sizeof(acacia::gpu::complex_t)*matSize,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(lhs->Lam, rhs->Lam, sizeof(acacia::gpu::complex_t)*nDim,
      cudaMemcpyDeviceToDevice);
    cudaMemcpy(lhs->pivW, rhs->pivW, sizeof(int)*(nDim+1),
      cudaMemcpyDeviceToDevice);  
    cudaMemcpy(lhs->pivV, rhs->pivV, sizeof(int)*(nDim+1),
      cudaMemcpyDeviceToDevice);
  }
}


RedhefferIntegratorGPU::RedhefferIntegratorGPU(scalar z0, scalar z1, 
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
  cudaMalloc(reinterpret_cast<void**>(&buffer5_), sizeof(acacia::gpu::complex_t)*matSize);
  int lwork;
  acacia_gpu_em_Complexgetrf_bufferSize(
    cusolverH_, nDim, nDim,
    nullptr, nDim, &lwork);
  cudaMalloc(reinterpret_cast<void**>(&dwork_), sizeof(acacia::gpu::complex_t)*lwork);

  cudaMalloc(reinterpret_cast<void**>(&dA_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&dB_), sizeof(acacia::gpu::complex_t)*matSize);
  cudaMalloc(reinterpret_cast<void**>(&iwork_), sizeof(int)*nDim);
  cudaMalloc(reinterpret_cast<void**>(&info_), sizeof(int)*NUM_CUDA_STREAMS);

  int nb = magma_get_zgehrd_nb(nDim);
  magma_lwork_ = nDim*(1 + 2*nb);
  int lwork2 = max(magma_lwork_, nDim*(5 + 2*nDim));
  magma_zmalloc_pinned(&magma_PQ_, matSize);
  magma_zmalloc_pinned(&magma_VR_, matSize);
  magma_zmalloc_pinned(&magma_work_, lwork2);

  for (int i = 0; i < NUM_CUDA_STREAMS; i++) {
    cudaStreamCreate(&streams_[i]);
  }
}

RedhefferIntegratorGPU::~RedhefferIntegratorGPU()
{
  for (int i = 0; i < NUM_CUDA_STREAMS; i++) {
    cudaStreamDestroy(streams_[i]);
  }

  magma_free_pinned(magma_PQ_);
  magma_free_pinned(magma_VR_);
  magma_free_pinned(magma_work_);

  cudaFree(info_);
  cudaFree(iwork_);
  
  cudaFree(dA_); cudaFree(dB_);

  cudaFree(dwork_);
  
  cudaFree(buffer_);
  cudaFree(buffer2_);
  cudaFree(buffer3_);
  cudaFree(buffer4_);
  cudaFree(buffer5_);

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


void RedhefferIntegratorGPU::evaluateEigenvalueAndSetReference(
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


scalar RedhefferIntegratorGPU::maxOnGPU(const acacia::gpu::complex_t *A, int N)
{
  int index;
  acacia_gpu_em_Complexamax_(cublasH_, N, A, 1, &index);
  acacia::gpu::complex_t val;
  cudaMemcpy(&val, A+index-1, sizeof(acacia::gpu::complex_t), cudaMemcpyDeviceToHost);
  return sqrt(val.x*val.x + val.y*val.y);

  // the following implementation is very slow!
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);
  // scalar temp = acacia::gpu::em::max_abs_complex_array(A, N);
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // cout << "max on gpu cost: " << milliseconds << " ms.\n";
  // return temp;
}

void RedhefferIntegratorGPU::compute_direct(int N)
{
#if 0
  N_ = N;
  nEigVal_ = N+1;
  int nDim = sampler_->nDim();

  samp0_ = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z0_, samp0_->Pr, samp0_->Qr);
  evaluateEigenvalueAndSetReference(samp0_);

  acacia::gpu::em::set_uniform_real_diagonal(Tuu_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Tdd_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rud_, 0., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rdu_, 0., nDim);

  auto leftSamp = std::make_shared<ReferenceSampling>(nDim);
  auto midSamp = std::make_shared<ReferenceSampling>(nDim, false);
  auto rightSamp = std::make_shared<ReferenceSampling>(nDim);

  leftRef_ = std::make_shared<ReferenceSampling>(nDim);
  copyRenferenceSampling(leftSamp, samp0_, nDim);
  
  scalar dz = (z1_ - z0_) / N;

  for (int i = 0; i < N; ++i) {
    scalar zi0 = z0_ + i*dz;
    scalar zi1 = zi0 + dz;
    scalar zm = 0.5*(zi0+zi1);

    sampler_->sampleOnGPU(zm, midSamp->Pr, midSamp->Qr);
    sampler_->sampleOnGPU(zi1, rightSamp->Pr, rightSamp->Qr);
    evaluatePerturbationAndDifference(
      leftSamp, zi0, zi1,
      leftSamp->Pr, midSamp->Pr, rightSamp->Pr,
      leftSamp->Qr, midSamp->Qr, rightSamp->Qr,
      tdd_, tuu_, rud_, rdu_);
    if (i > 0) {
      reprojectLeftEnd(leftRef_, leftSamp, tdd_, tuu_, rud_, rdu_);
    }
    redhefferStarRightProduct(Tdd_, Tuu_, Rud_, Rdu_,
      tdd_, tuu_, rud_, rdu_);
    
    copyRenferenceSampling(leftRef_, leftSamp, nDim);
    evaluateEigenvalueAndSetReference(rightSamp);
    copyRenferenceSampling(leftSamp, rightSamp, nDim);
  }
  reprojectRightEnd(leftRef_, leftSamp, Tdd_, Tuu_, Rud_, Rdu_);
  samp2_ = leftSamp;
#else
  N_ = N;
  nEigVal_ = N+1;
  int nDim = sampler_->nDim();

  samp0_ = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z0_, samp0_->Pr, samp0_->Qr);
  evaluateEigenvalueAndSetReference(samp0_);

  acacia::gpu::em::set_uniform_real_diagonal(Tuu_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Tdd_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rud_, 0., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rdu_, 0., nDim);

  auto leftSamp = std::make_shared<ReferenceSampling>(nDim, false);
  auto midSamp = std::make_shared<ReferenceSampling>(nDim);
  auto rightSamp = std::make_shared<ReferenceSampling>(nDim, false);

  leftRef_ = std::make_shared<ReferenceSampling>(nDim);
  copyPQ(leftSamp, samp0_, nDim);
  copyRenferenceSampling(leftRef_, samp0_, nDim);

  scalar dz = (z1_ - z0_) / N;

  for (int i = 0; i < N; ++i) {
    scalar zi0 = z0_ + i*dz;
    scalar zi1 = zi0 + dz;
    scalar zm = 0.5*(zi0+zi1);

    sampler_->sampleOnGPU(zm, midSamp->Pr, midSamp->Qr);
    evaluateEigenvalueAndSetReference(midSamp);

    sampler_->sampleOnGPU(zi1, rightSamp->Pr, rightSamp->Qr);
    evaluatePerturbationAndDifference(
      midSamp, zi0, zi1,
      leftSamp->Pr, midSamp->Pr, rightSamp->Pr,
      leftSamp->Qr, midSamp->Qr, rightSamp->Qr,
      tdd_, tuu_, rud_, rdu_);
    
    reprojectLeftEnd(leftRef_, midSamp, tdd_, tuu_, rud_, rdu_);
    auto t1 = sploosh::now();
    redhefferStarRightProduct(Tdd_, Tuu_, Rud_, Rdu_,
      tdd_, tuu_, rud_, rdu_);
    auto t2 = sploosh::now();
    cout << "redheffer time: " << sploosh::duration_milli_d(t1, t2) << endl;

    copyPQ(leftSamp, rightSamp, nDim);
    copyRenferenceSampling(leftRef_, midSamp, nDim);
  }

  samp2_ = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z1_, samp2_->Pr, samp2_->Qr);
  evaluateEigenvalueAndSetReference(samp2_);
  reprojectRightEnd(leftRef_, samp2_, Tdd_, Tuu_, Rud_, Rdu_);
#endif
}

void RedhefferIntegratorGPU::compute_recursive(int maxN, scalar pe, scalar reBound)
{
#if 1
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  N_ = 0;
  nEigVal_ = 2;
  int nDim = sampler_->nDim();

  auto samp0 = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z0_, samp0->Pr, samp0->Qr);

  evaluateEigenvalueAndSetReference(samp0);

  auto samp2 = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z1_, samp2->Pr, samp2->Qr);
  evaluateEigenvalueAndSetReference(samp2);
  
  samp0_ = samp0;
  samp2_ = samp2;

  // cudaEventRecord(start);
  scalar zm = 0.5 * (z0_ + z1_);
  auto samp1 = std::make_shared<ReferenceSampling>(nDim, false);
  sampler_->sampleOnGPU(zm, samp1->Pr, samp1->Qr);

  leftRef_ = samp0;
  acacia::gpu::em::set_uniform_real_diagonal(Tuu_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Tdd_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rud_, 0., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rdu_, 0., nDim);
  // this function will use the buffer tuu_, tdd_, rud_, rdu_,
  // and update Tuu_, Tdd_, Rud_, Rdu_;
  
  recursiveSolve(samp0, z0_, z1_, pe, reBound,
    samp0, samp1, samp2, &N_, &nEigVal_);

  reprojectRightEnd(leftRef_, samp2, Tdd_,
    Tuu_, Rud_, Rdu_);
  // cudaEventRecord(stop);

  // cudaEventSynchronize(stop); 
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // cout << "total recursive cost: " << milliseconds << " ms.\n";
#else
  N_ = 0;
  nEigVal_ = 3;
  int nDim = sampler_->nDim();

  auto samp0 = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z0_, samp0->Pr, samp0->Qr);
  evaluateEigenvalueAndSetReference(samp0);
  
  auto samp2 = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(z1_, samp2->Pr, samp2->Qr);
  evaluateEigenvalueAndSetReference(samp2);
  
  samp0_ = samp0;
  samp2_ = samp2;

  // cudaEventRecord(start);
  scalar zm = 0.5 * (z0_ + z1_);
  auto samp1 = std::make_shared<ReferenceSampling>(nDim);
  sampler_->sampleOnGPU(zm, samp1->Pr, samp1->Qr);
  evaluateEigenvalueAndSetReference(samp1);

  leftRef_ = samp0;
  
  acacia::gpu::em::set_uniform_real_diagonal(Tuu_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Tdd_, 1., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rud_, 0., nDim);
  acacia::gpu::em::set_uniform_real_diagonal(Rdu_, 0., nDim);
  // this function will use the buffer tuu_, tdd_, rud_, rdu_,
  // and update Tuu_, Tdd_, Rud_, Rdu_;

  recursiveSolve_div3(samp1, z0_, z1_, pe, reBound,
    samp0, samp1, samp2, &N_, &nEigVal_);
  
  reprojectRightEnd(leftRef_, samp2, Tdd_,
    Tuu_, Rud_, Rdu_);
#endif
}

void RedhefferIntegratorGPU::reprojectLeftEnd(
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

// reproject the scattering matrix to make sure the right end 
// is represented by rref instead of lref
void RedhefferIntegratorGPU::reprojectRightEnd(
  std::shared_ptr<const ReferenceSampling> lref,
  std::shared_ptr<const ReferenceSampling> rref,
  acacia::gpu::complex_t * tdd,
  acacia::gpu::complex_t * tuu,
  acacia::gpu::complex_t * rud,
  acacia::gpu::complex_t * rdu)
{
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);

  acacia::gpu::complex_t alpha_m1{-1., 0.};
  acacia::gpu::complex_t alpha_p1{1., 0.};
  acacia::gpu::complex_t beta{0., 0.};
  acacia::gpu::complex_t mul{0.5, 0.};
  
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
  cudaMemcpy(buffer3_, buffer_, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  
  
  cudaMemcpy(buffer2_, rref->V, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice);
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    lref->facV, nDim,
    lref->pivV,
    buffer2_, nDim,
    info_);
  // buffer2_ = invV0_V;
  

  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_m1, buffer2_, 1, buffer3_, 1);
  // buffer3_ = B;
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_p1, buffer2_, 1, buffer_, 1);
  // buffer_ = A;

  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, buffer_, 1);
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, buffer3_, 1);
  // const MatrixXcs& invW0_W = invW0_Sol.solve(samp1.W);
  // const MatrixXcs& invV0_V = invV0_Sol.solve(samp1.V);
  // const MatrixXcs& A = 0.5 * (invW0_W + invV0_V);
  // const MatrixXcs& B = 0.5 * (invW0_W - invV0_V);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha_m1, 
    rud, nDim,
    buffer3_, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_p1, buffer_, 1, buffer2_, 1);
  acacia_gpu_em_Complexgetrf_(cusolverH_, 
    nDim, nDim,
    buffer2_, nDim,
    dwork_,
    iwork_, info_);
  // buffer2_ = fac(A-rud*B)
  
  
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    buffer2_, nDim,
    iwork_,
    tuu, nDim,
    info_);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha_m1, 
    rud, nDim,
    buffer_, nDim,
    &beta,
    buffer4_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_m1, buffer3_, 1, buffer4_, 1);

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
    &alpha_p1, 
    tdd, nDim,
    buffer3_, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha_p1, 
    buffer2_, nDim,
    tuu, nDim,
    &beta,
    buffer4_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_p1, buffer4_, 1, rdu, 1);

  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha_p1, 
    buffer3_, nDim,
    rud, nDim,
    &beta,
    buffer2_, nDim);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_p1, buffer_, 1, buffer2_, 1);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha_p1, 
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
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop); 
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // cout << "right project cost: " << milliseconds << " ms.\n";
}

void RedhefferIntegratorGPU::redhefferStarRightProduct(
  acacia::gpu::complex_t * Tdd,
  acacia::gpu::complex_t * Tuu,
  acacia::gpu::complex_t * Rud,
  acacia::gpu::complex_t * Rdu,
  const acacia::gpu::complex_t * tdd,
  const acacia::gpu::complex_t * tuu,
  const acacia::gpu::complex_t * rud,
  const acacia::gpu::complex_t * rdu)
{
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  
  int nDim = sampler_->nDim();
  int matSize = nDim*nDim;
  acacia::gpu::complex_t alpha{-1., 0.};
  acacia::gpu::complex_t alpha1{1., 0.};
  acacia::gpu::complex_t beta{0., 0.};

  // cudaEventRecord(start);

  cublasSetStream(cublasH_, streams_[0]);
  cusolverDnSetStream(cusolverH_, streams_[0]);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha, 
    rdu, nDim,
    Rud, nDim,
    &beta,
    buffer_, nDim);
  acacia::gpu::em::add_uniform_real_diagonal(buffer_, 1., nDim, streams_[0]);
  acacia_gpu_em_Complexgetrf_(cusolverH_, 
    nDim, nDim,
    buffer_, nDim,
    dwork_,
    iwork_, info_);
  // buffer_ = fac(I - rdu * Rud);
  cudaMemcpyAsync(buffer2_, rdu, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice, streams_[0]);
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
  acacia::gpu::em::add_uniform_real_diagonal(buffer3_, 1., nDim, streams_[0]);
  // buffer3_ = C2;
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer3_, nDim,
    Tuu, nDim,
    &beta,
    buffer2_, nDim);
  // buffer2_ = C2 * Tuu;
  cudaMemcpyAsync(buffer3_, tdd, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice, streams_[0]);  
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    buffer_, nDim,
    iwork_,
    buffer3_, nDim,
    info_);
  // buffer3_ = C1 * tdd;

  cublasSetStream(cublasH_, streams_[1]);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    Tdd, nDim,
    rdu, nDim,
    &beta,
    buffer4_, nDim);
  // buffer4_ = Tdd * rdu;

  cublasSetStream(cublasH_, streams_[2]);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    tuu, nDim,
    Rud, nDim,
    &beta,
    buffer5_, nDim);
  // buffer5_ = tuu * Rud

  // wait for buffer2, 3, 4, 5, finish
  cudaStreamSynchronize(streams_[0]);
  cudaStreamSynchronize(streams_[1]);
  cudaStreamSynchronize(streams_[2]);

  cublasSetStream(cublasH_, streams_[0]);
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha1, 
    buffer4_, nDim,
    buffer2_, nDim,
    &beta,
    buffer_, nDim);
  // buffer_ = Tdd * rdu * C2_Tuu;
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer_, 1, Rdu, 1);

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
    buffer5_, nDim,
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
  cudaStreamSynchronize(streams_[0]);
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop); 
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
  
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // cout << "redheffer product cost: " << milliseconds << " ms.\n";
}

// void RedhefferIntegratorGPU::redhefferStarRightProduct(
//   acacia::gpu::complex_t * Tdd,
//   acacia::gpu::complex_t * Tuu,
//   acacia::gpu::complex_t * Rud,
//   acacia::gpu::complex_t * Rdu,
//   const acacia::gpu::complex_t * tdd,
//   const acacia::gpu::complex_t * tuu,
//   const acacia::gpu::complex_t * rud,
//   const acacia::gpu::complex_t * rdu)
// {
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
  
//   int nDim = sampler_->nDim();
//   int matSize = nDim*nDim;
//   acacia::gpu::complex_t alpha{-1., 0.};
//   acacia::gpu::complex_t alpha1{1., 0.};
//   acacia::gpu::complex_t beta{0., 0.};
 
//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha, 
//     rdu, nDim,
//     Rud, nDim,
//     &beta,
//     buffer_, nDim);
//   acacia::gpu::em::add_uniform_real_diagonal(buffer_, 1., nDim);
//   acacia_gpu_em_Complexgetrf_(cusolverH_, 
//     nDim, nDim,
//     buffer_, nDim,
//     dwork_,
//     iwork_, info_);
//   // buffer_ = fac(I - rdu * Rud);
  
//   cudaEventRecord(start);

//   cudaMemcpy(buffer2_, rdu, sizeof(acacia::gpu::complex_t)*matSize, 
//     cudaMemcpyDeviceToDevice);
//   acacia_gpu_em_Complexgetrs_(cusolverH_, 
//     CUBLAS_OP_N, nDim, nDim,
//     buffer_, nDim,
//     iwork_,
//     buffer2_, nDim,
//     info_);
//   // buffer2_ = C1 * rdu;
//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha1, 
//     Rud, nDim,
//     buffer2_, nDim,
//     &beta,
//     buffer3_, nDim);
//   acacia::gpu::em::add_uniform_real_diagonal(buffer3_, 1., nDim);
//   // buffer3_ = C2;
//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha1, 
//     buffer3_, nDim,
//     Tuu, nDim,
//     &beta,
//     buffer2_, nDim);
//   // buffer2_ = C2 * Tuu;

//   cudaMemcpy(buffer3_, tdd, sizeof(acacia::gpu::complex_t)*matSize, 
//     cudaMemcpyDeviceToDevice);  
//   acacia_gpu_em_Complexgetrs_(cusolverH_, 
//     CUBLAS_OP_N, nDim, nDim,
//     buffer_, nDim,
//     iwork_,
//     buffer3_, nDim,
//     info_);
//   // buffer3_ = C1 * tdd;

  
//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha1, 
//     Tdd, nDim,
//     rdu, nDim,
//     &beta,
//     buffer_, nDim);
//   // buffer_ = Tdd * rdu;
//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha1, 
//     buffer_, nDim,
//     buffer2_, nDim,
//     &beta,
//     buffer4_, nDim);
//   // buffer4_ = Tdd * rdu * C2_Tuu;
//   acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, buffer4_, 1, Rdu, 1);

//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha1, 
//     Tdd, nDim,
//     buffer3_, nDim,
//     &beta,
//     buffer_, nDim);
//   cudaMemcpy(Tdd, buffer_, sizeof(acacia::gpu::complex_t)*matSize, 
//     cudaMemcpyDeviceToDevice);
  
//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha1, 
//     tuu, nDim,
//     Rud, nDim,
//     &beta,
//     buffer_, nDim);  
//   // buffer_ = tuu * Rud;
//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha1, 
//     buffer_, nDim,
//     buffer3_, nDim,
//     &beta,
//     Rud, nDim);
//   acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha1, rud, 1, Rud, 1);   

//   acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
//     nDim, nDim, nDim,
//     &alpha1, 
//     tuu, nDim,
//     buffer2_, nDim,
//     &beta,
//     Tuu, nDim);
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop); 
//   // MatrixXcs invC1 = - rdu * Rud;
//   // invC1.diagonal().array() += 1.0;
//   // PartialPivLU<MatrixXcs> cycleSolver1(invC1);
//   // MatrixXcs C1 = cycleSolver1.inverse();

//   // MatrixXcs C2 = Rud_ * C1 * rdu;
//   // C2.diagonal().array() += 1.0;

//   // Rdu = Rdu + Tdd * rdu * C2_Tuu;
//   // Tdd = Tdd * C1_tdd; 
//   // Rud = rud + tuu * Rud * C1_tdd;
//   // Tuu = tuu * C2_Tuu;  
  
//   float milliseconds = 0;
//   cudaEventElapsedTime(&milliseconds, start, stop);
//   cout << "redheffer product cost: " << milliseconds << " ms.\n";
// }

void RedhefferIntegratorGPU::recursiveSolve_div3(
  std::shared_ptr<ReferenceSampling> ref, 
  scalar z0, scalar z1, scalar pe, scalar reBound,
  std::shared_ptr<ReferenceSampling> samp0,
  std::shared_ptr<ReferenceSampling> samp1,
  std::shared_ptr<ReferenceSampling> samp2,
  int * N, int *nEigVal)
{
  int nDim = sampler_->nDim();
  scalar diffError, matVar;
  evaluatePerturbationAndDifference(  
    ref,
    z0, z1,
    samp0->Pr, samp1->Pr, samp2->Pr,
    samp0->Qr, samp1->Qr, samp2->Qr,
    tdd_,
    tuu_,
    rud_,
    rdu_,
    &diffError,
    &matVar);
  cout << "diffError: " << diffError << endl;
  // cout << "matVar: " << matVar << endl;
  if (diffError <= pe) {
    if (ref != leftRef_) {
      reprojectLeftEnd(leftRef_, ref, tdd_, tuu_, rud_, rdu_);
      leftRef_ = ref;
    }
    if (N != nullptr) {
      (*N)++;
    }
    redhefferStarRightProduct(Tdd_, Tuu_, Rud_, Rdu_,
      tdd_, tuu_, rud_, rdu_);
  } else {
    auto refMid = ref;
    scalar zm1 = 2. / 3. * z0 + 1. / 3. * z1;
    scalar zm2 = 1. / 3. * z0 + 2. / 3. * z1;
    cout << "divide at: " << zm1 << ", " << zm2 << endl;

    auto samp_m00 = std::make_shared<ReferenceSampling>(nDim, false);
    sampler_->sampleOnGPU(0.5*(z0 + zm1), samp_m00->Pr, samp_m00->Qr);
    auto samp_m01 = std::make_shared<ReferenceSampling>(nDim, false);
    sampler_->sampleOnGPU(zm1, samp_m01->Pr, samp_m01->Qr);
    auto samp_m02 = std::make_shared<ReferenceSampling>(nDim, false);
    sampler_->sampleOnGPU(zm2, samp_m02->Pr, samp_m02->Qr);
    auto samp_m03 = std::make_shared<ReferenceSampling>(nDim, false);
    sampler_->sampleOnGPU(0.5*(zm2 + z1), samp_m03->Pr, samp_m03->Qr);

    auto refLeft = ref;
    auto refRight = ref;
    if (matVar > reBound) {
      refLeft = samp_m00;
      refLeft->allocForEig(nDim);
      evaluateEigenvalueAndSetReference(refLeft);

      refRight = samp_m03;
      refRight->allocForEig(nDim);
      evaluateEigenvalueAndSetReference(refRight);

      if (nEigVal != nullptr) {
        (*nEigVal) += 2;
      } 
    }
    
    recursiveSolve_div3(refLeft, z0, zm1, pe, reBound,
      samp0, samp_m00, samp_m01, N, nEigVal);
    recursiveSolve_div3(refMid, zm1, zm2, pe, reBound,
      samp_m01, samp1, samp_m02, N, nEigVal);
    recursiveSolve_div3(refRight, zm2, z1, pe, reBound,
      samp_m02, samp_m03, samp2, N, nEigVal);
  }
}

void RedhefferIntegratorGPU::recursiveSolve(
  std::shared_ptr<ReferenceSampling> ref, 
  scalar z0, scalar z1, scalar pe, scalar reBound,
  std::shared_ptr<ReferenceSampling> samp0,
  std::shared_ptr<ReferenceSampling> samp1,
  std::shared_ptr<ReferenceSampling> samp2,
  int * N, int *nEigVal)
{
  int nDim = sampler_->nDim();
  
  scalar diffError, matVar;
  // auto t1 = sploosh::now();
  evaluatePerturbationAndDifference(  
    ref,
    z0, z1,
    samp0->Pr, samp1->Pr, samp2->Pr,
    samp0->Qr, samp1->Qr, samp2->Qr,
    tdd_,
    tuu_,
    rud_,
    rdu_,
    &diffError,
    &matVar);
  // auto t2 = sploosh::now();
  // cout << "var time: " << sploosh::duration_milli_d(t1, t2) << endl;
  
  
  // cout << "diffError: " << diffError << endl;
  if (diffError <= pe) {
    if (ref != leftRef_) {
      cout << "need to project!" << endl;
      reprojectLeftEnd(leftRef_, ref, tdd_, tuu_, rud_, rdu_);
      leftRef_ = ref;
    }
    if (N != nullptr) {
      (*N)++;
    }
    redhefferStarRightProduct(Tdd_, Tuu_, Rud_, Rdu_,
      tdd_, tuu_, rud_, rdu_);
  } else {
    auto refLeft = ref;
    auto refRight = ref;
    if (matVar > reBound) {
      refRight = samp1;
      refRight->allocForEig(nDim);
      if (nEigVal != nullptr) {
        (*nEigVal)++;
      }
      evaluateEigenvalueAndSetReference(refRight);
    }
    scalar zm = 0.5*(z0+z1);
    cout << "divide at: " << zm << endl;

    auto samp_m1 = std::make_shared<ReferenceSampling>(nDim, false);
    sampler_->sampleOnGPU(0.5*(z0+zm), samp_m1->Pr, samp_m1->Qr);
    recursiveSolve(refLeft, z0, zm, pe, reBound,
      samp0, samp_m1, samp1, N, nEigVal);
    
    auto samp_m2 = std::make_shared<ReferenceSampling>(nDim, false);
    sampler_->sampleOnGPU(0.5*(zm+z1), samp_m2->Pr, samp_m2->Qr);
    recursiveSolve(refRight, zm, z1, pe, reBound,
      samp1, samp_m2, samp2, N, nEigVal);
  }
}

void RedhefferIntegratorGPU::evaluateVarCoeff(
  std::shared_ptr<const ReferenceSampling> ref,
  const acacia::gpu::complex_t *P,
  const acacia::gpu::complex_t *Q,
  acacia::gpu::complex_t *dA,
  acacia::gpu::complex_t *dB,
  scalar *matVar)
{
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);

  int nDim = sampler_->nDim();
  int matSize = nDim * nDim;
  acacia::gpu::complex_t alpha_m1{-1., 0.};
  acacia::gpu::complex_t alpha_p1{1., 0.};
  acacia::gpu::complex_t beta{0., 0.};
  int *info_stream0 = info_;
  int *info_stream1 = info_+1;
  
  cublasSetStream(cublasH_, streams_[0]);
  cusolverDnSetStream(cusolverH_, streams_[0]);
  // buffer is a buffer with the same size as P
  cudaMemcpyAsync(buffer_, P, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice, streams_[0]);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_m1, ref->Pr, 1, buffer_, 1);
  // buffer_ = P - Pr
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha_p1, 
    buffer_, nDim,
    ref->V, nDim,
    &beta,
    dB, nDim);
  // dB = (P - Pr) * V;
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    ref->facW, nDim,
    ref->pivW,
    dB, nDim,
    info_stream0);
  // dB = invW_dP_V;

  cublasSetStream(cublasH_, streams_[1]);
  cusolverDnSetStream(cusolverH_, streams_[1]);
  cudaMemcpyAsync(buffer2_, Q, sizeof(acacia::gpu::complex_t)*matSize, 
    cudaMemcpyDeviceToDevice, streams_[1]);
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_m1, ref->Qr, 1, buffer2_, 1);
  // buffer_ = Q - Qr
  acacia_gpu_em_Complexgemm3m_(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
    nDim, nDim, nDim,
    &alpha_p1, 
    buffer2_, nDim,
    ref->W, nDim,
    &beta,
    dA, nDim);
  // dA = (Q - Qr) * W;
  acacia_gpu_em_Complexgetrs_(cusolverH_, 
    CUBLAS_OP_N, nDim, nDim,
    ref->facV, nDim,
    ref->pivV,
    dA, nDim,
    info_stream1); 
  cudaStreamSynchronize(streams_[0]);
  cudaStreamSynchronize(streams_[1]);
  // dA = invV_dQ_W;
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);

  cudaMemcpy(buffer_, dB, sizeof(acacia::gpu::complex_t)*matSize, cudaMemcpyDeviceToDevice);
  // buffer_ = invW_dP_V;
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_m1, dA, 1, dB, 1);
  // dB = invW_dP_V - invV_dQ_W;
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha_p1, buffer_, 1, dA, 1);
  // dA = invV_dQ_W + invW_dP_V;

  if (matVar != nullptr) {
    *matVar = max(*matVar, maxOnGPU(dA, matSize));
    *matVar = max(*matVar, maxOnGPU(dB, matSize));
  }

  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // cout << "varCoeff cost: " << milliseconds << " ms.\n";  
}

void RedhefferIntegratorGPU::evaluatePerturbationAndDifference(
  std::shared_ptr<const ReferenceSampling> ref,
  scalar z0, scalar z1,
  const acacia::gpu::complex_t *P0,
  const acacia::gpu::complex_t *P1,
  const acacia::gpu::complex_t *P2,
  const acacia::gpu::complex_t *Q0,
  const acacia::gpu::complex_t *Q1,
  const acacia::gpu::complex_t *Q2,
  acacia::gpu::complex_t * tdd,
  acacia::gpu::complex_t * tuu,
  acacia::gpu::complex_t * rud,
  acacia::gpu::complex_t * rdu,
  scalar * diffError,
  scalar * matVar)
{

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  // cudaEventRecord(start); 

  scalar k0 = sampler_->k0();
  scalar dz = z1 - z0;
  int nDim = sampler_->nDim();
  int matSize = nDim * nDim;

  if (matVar != nullptr) {
    *matVar = 0.;
  }

  evaluateVarCoeff(ref, P0, Q0, dA_, dB_, matVar);
  acacia::gpu::em::phase_diag_matrix_multiply(true, k0, dz, 
    1., ref->Lam, dA_, 0., tuu, nDim);
  // tuu = pf.asDiagonal() * dA0;
  acacia::gpu::em::phase_diag_matrix_multiply(false, k0, dz, 
    1., ref->Lam, dA_, 0., tdd, nDim);
  // tdd = dA0 * pf.asDiagonal();
  acacia::gpu::em::phase_diag_matrix_multiply(true, k0, dz, 
    1., ref->Lam, dB_, 0., rud, nDim);
  // rud = pf.asDiagonal() * dB0;
  acacia::gpu::em::phase_diag_matrix_multiply(false, k0, dz, 
    1., ref->Lam, rud, 0., rud, nDim); 
  // rud = pf.asDiagonal() * dB0 * pf.asDiagonal();
  cudaMemcpy(rdu, dB_, sizeof(acacia::gpu::complex_t)*matSize, cudaMemcpyDeviceToDevice);
  // rdu = dB0;

  evaluateVarCoeff(ref, P1, Q1, dA_, dB_, matVar);
  acacia::gpu::em::phase_diag_matrix_multiply(true, k0, 0.5*dz, 
    4., ref->Lam, dA_, 0., buffer_, nDim);
  // buffer_ = 4. * ph.asDiagonal() * dA1;
  acacia::gpu::em::phase_diag_matrix_multiply(false, k0, 0.5*dz, 
    1., ref->Lam, buffer_, 1., tuu, nDim);
  // tuu = pf.asDiagonal() * dA0 + 4. * ph.asDiagonal() * dA1 * ph.asDiagonal()
  acacia::gpu::em::phase_diag_matrix_multiply(false, k0, 0.5*dz, 
    1., ref->Lam, buffer_, 1., tdd, nDim);
  // tdd = dA0 * pf.asDiagonal() + 4. * ph.asDiagonal() * dA1 * ph.asDiagonal()
  acacia::gpu::em::phase_diag_matrix_multiply(true, k0, 0.5*dz, 
    4., ref->Lam, dB_, 0., buffer_, nDim);
  // buffer_ = 4. * ph.asDiagonal() * dB1
  acacia::gpu::em::phase_diag_matrix_multiply(false, k0, 0.5*dz, 
    1., ref->Lam, buffer_, 1., rud, nDim);
  // rud = pf.asDiagonal() * dB0 * pf.asDiagonal() 
  // + 4. * ph.asDiagonal() * dB1 * ph.asDiagonal();
  acacia::gpu::em::phase_diag_matrix_multiply(false, k0, 0.5*dz, 
    1., ref->Lam, buffer_, 1., rdu, nDim);
  // rdu = dB0 + 4. * ph.asDiagonal() * dB1 * ph.asDiagonal();

  evaluateVarCoeff(ref, P2, Q2, dA_, dB_, matVar);
  acacia::gpu::em::phase_diag_matrix_multiply(false, k0, dz, 
    1., ref->Lam, dA_, 1., tuu, nDim);
  acacia::gpu::em::phase_diag_matrix_multiply(true, k0, dz, 
    1., ref->Lam, dA_, 1., tdd, nDim);
  acacia::gpu::complex_t alpha{1., 0.};
  acacia_gpu_em_Complexaxpy_(cublasH_, matSize, &alpha, dB_, 1, rud, 1);
  acacia::gpu::em::phase_diag_matrix_multiply(true, k0, dz, 
    1., ref->Lam, dB_, 0., buffer_, nDim);
  // buffer_ = pf.asDiagonal() * dB2;
  acacia::gpu::em::phase_diag_matrix_multiply(false, k0, dz, 
    1., ref->Lam, buffer_, 1., rdu, nDim);
  // rdu +=  pf.asDiagonal() * dB2 * pf.asDiagonal();

  // scale the matrices
  acacia::gpu::complex_t mul{0., dz / (12. * k0)};
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, tuu, 1);
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, tdd, 1);
  mul.y *= -1.;
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, rud, 1);
  acacia_gpu_em_Complexscal_(cublasH_, matSize, &mul, rdu, 1);

  // tuu = j(z1_ - z0_) / (12*k0) * (pf * dA0 + 4. * ph * dA1 * ph + dA2 * pf);
  // tdd = j(z1_ - z0_) / (12*k0) * (dA0 * pf + 4. * ph * dA1 * ph + pf * dA2);
  // rud = -j(z1_ - z0_) / (12*k0) * (pf * dB0 * pf + 4. * ph * dB1 * ph + dB2);
  // rdu = -j(z1_ - z0_) / (12*k0) * (dB0 + 4. * ph * dB1 * ph + pf * dB2 * pf);
 
  // now evaluate the max norm
  if (diffError != nullptr) {
    *diffError = 0.;
    *diffError = max(*diffError, maxOnGPU(tuu, matSize));
    *diffError = max(*diffError, maxOnGPU(tdd, matSize));
    *diffError = max(*diffError, maxOnGPU(rud, matSize));
    *diffError = max(*diffError, maxOnGPU(rdu, matSize));
  }

  acacia::gpu::em::add_phase_diagonal(k0, dz, ref->Lam, tuu, nDim);
  // tuu += pf;
  acacia::gpu::em::add_phase_diagonal(k0, dz, ref->Lam, tdd, nDim);
  // tdd += pf;

  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop); 
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // cout << "difference evaluation cost: " << milliseconds << " ms.\n";
}

Eigen::MatrixXcs RedhefferIntegratorGPU::copy_from_gpu(
    const acacia::gpu::complex_t *A, int row, int col) const
{
  Eigen::MatrixXcs temp(row, col);
  cudaMemcpy(temp.data(), A, sizeof(acacia::gpu::complex_t)*row*col, 
    cudaMemcpyDeviceToHost);
  return temp;
}

Eigen::MatrixXcs RedhefferIntegratorGPU::Tdd() const
{
  return copy_from_gpu(Tdd_, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RedhefferIntegratorGPU::Tuu() const
{
  return copy_from_gpu(Tuu_, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RedhefferIntegratorGPU::Rud() const
{
  return copy_from_gpu(Rud_, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RedhefferIntegratorGPU::Rdu() const
{
  return copy_from_gpu(Rdu_, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RedhefferIntegratorGPU::W0() const
{
  return copy_from_gpu(samp0_->W, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RedhefferIntegratorGPU::V0() const
{
  return copy_from_gpu(samp0_->V, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RedhefferIntegratorGPU::W1() const
{
  return copy_from_gpu(samp2_->W, sampler_->nDim(), sampler_->nDim());
}

Eigen::MatrixXcs RedhefferIntegratorGPU::V1() const
{
  return copy_from_gpu(samp2_->V, sampler_->nDim(), sampler_->nDim());
}