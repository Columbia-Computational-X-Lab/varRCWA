#pragma once
#include "rcwa/defns.h"
#include "types.cuh"
#include <memory>
#include <utility>
#include <cusolverDn.h>
#include <cublas_v2.h>

class LayerSampler;

struct ReferenceSampling {
private:
  bool IsAllocForEig_;
public:
  acacia::gpu::complex_t *Pr;
  acacia::gpu::complex_t *Qr;

  acacia::gpu::complex_t *W;
  acacia::gpu::complex_t *V;
  acacia::gpu::complex_t *Lam;

  acacia::gpu::complex_t *facW;
  acacia::gpu::complex_t *facV;

  int *pivW;
  int *pivV;

  ReferenceSampling() = delete;
  ReferenceSampling(const ReferenceSampling& rhs) = delete;
  
  bool hasEigMem() const { return IsAllocForEig_; }
  void allocForEig(int nDim);
  ReferenceSampling(int nDim, bool IsAllocForEig=true);
  ~ReferenceSampling();
};

void copyRenferenceSampling(std::shared_ptr<ReferenceSampling> lhs,
  std::shared_ptr<ReferenceSampling> rhs, int nDim);



class RedhefferIntegratorGPU
{
private:
    scalar z0_;
    scalar z1_;

    std::shared_ptr<LayerSampler> sampler_;

    // primary memory
    acacia::gpu::complex_t *Tdd_;
    acacia::gpu::complex_t *Tuu_;
    acacia::gpu::complex_t *Rud_;
    acacia::gpu::complex_t *Rdu_;

    // buffer memory
    acacia::gpu::complex_t *tdd_;
    acacia::gpu::complex_t *tuu_;
    acacia::gpu::complex_t *rud_;
    acacia::gpu::complex_t *rdu_;


    int N_;
    int nEigVal_;

    // buffer
    cublasHandle_t cublasH_;
    cusolverDnHandle_t cusolverH_;
    
    acacia::gpu::complex_t *buffer_;
    acacia::gpu::complex_t *buffer2_;
    acacia::gpu::complex_t *buffer3_;
    acacia::gpu::complex_t *buffer4_;
    acacia::gpu::complex_t *buffer5_;

    acacia::gpu::complex_t *dwork_;

    // buffer for integral sampling
    acacia::gpu::complex_t *dA_, *dB_;

    std::shared_ptr<ReferenceSampling> leftRef_;

    // TODO just for debugging
    std::shared_ptr<ReferenceSampling> samp0_;
    std::shared_ptr<ReferenceSampling> samp2_;

    int *iwork_;
    int *info_;

    // pinned memory
    acacia::gpu::complex_t *magma_PQ_;
    acacia::gpu::complex_t *magma_VR_;
    acacia::gpu::complex_t *magma_work_;
    int magma_lwork_;

    static constexpr int NUM_CUDA_STREAMS = 4;
    cudaStream_t streams_[NUM_CUDA_STREAMS];
public:
    RedhefferIntegratorGPU(scalar z0, scalar z1, 
      std::shared_ptr<LayerSampler> sampler);
    ~RedhefferIntegratorGPU();
    
    void compute_recursive(int maxN = 10, scalar pe = 1e-2, scalar reBound = 1.);
    void compute_direct(int N);
    
    int N() const { return N_; }
    int nEigVal() const { return nEigVal_; }

    // return the scattering matrix to cpu memory
    Eigen::MatrixXcs Tdd() const;
    Eigen::MatrixXcs Tuu() const;
    Eigen::MatrixXcs Rud() const;
    Eigen::MatrixXcs Rdu() const;

    Eigen::MatrixXcs W0() const;
    Eigen::MatrixXcs V0() const;
    Eigen::MatrixXcs W1() const;
    Eigen::MatrixXcs V1() const;

private:
    Eigen::MatrixXcs copy_from_gpu(const acacia::gpu::complex_t *A, int row, int col) const;
    
    // reproject the scattering matrix to make sure the left end 
    // is represented by lref instead of rref
    void reprojectLeftEnd(
      std::shared_ptr<const ReferenceSampling> lref,
      std::shared_ptr<const ReferenceSampling> rref,
      acacia::gpu::complex_t * tdd,
      acacia::gpu::complex_t * tuu,
      acacia::gpu::complex_t * rud,
      acacia::gpu::complex_t * rdu);

    // reproject the scattering matrix to make sure the right end 
    // is represented by rref instead of lref
    void reprojectRightEnd(
      std::shared_ptr<const ReferenceSampling> lref,
      std::shared_ptr<const ReferenceSampling> rref,
      acacia::gpu::complex_t * tdd,
      acacia::gpu::complex_t * tuu,
      acacia::gpu::complex_t * rud,
      acacia::gpu::complex_t * rdu);
    
    void redhefferStarRightProduct(
      acacia::gpu::complex_t * Tdd,
      acacia::gpu::complex_t * Tuu,
      acacia::gpu::complex_t * Rud,
      acacia::gpu::complex_t * Rdu,
      const acacia::gpu::complex_t * tdd,
      const acacia::gpu::complex_t * tuu,
      const acacia::gpu::complex_t * rud,
      const acacia::gpu::complex_t * rdu);
    
    // assume a long vector
    scalar maxOnGPU(const acacia::gpu::complex_t *A, int N);

    void evaluateVarCoeff(
      std::shared_ptr<const ReferenceSampling> ref,
      const acacia::gpu::complex_t *P,
      const acacia::gpu::complex_t *Q,
      acacia::gpu::complex_t *dA,
      acacia::gpu::complex_t *dB,
      scalar *matVar=nullptr);

    void recursiveSolve(
      std::shared_ptr<ReferenceSampling> ref, 
      scalar z0, scalar z1, scalar pe, scalar reBound,
      std::shared_ptr<ReferenceSampling> samp0,
      std::shared_ptr<ReferenceSampling> samp1,
      std::shared_ptr<ReferenceSampling> samp2,
      int *N=nullptr, int *nEigVal=nullptr);
      
    void recursiveSolve_div3(
      std::shared_ptr<ReferenceSampling> ref, 
      scalar z0, scalar z1, scalar pe, scalar reBound,
      std::shared_ptr<ReferenceSampling> samp0,
      std::shared_ptr<ReferenceSampling> samp1,
      std::shared_ptr<ReferenceSampling> samp2,
      int *N=nullptr, int *nEigVal=nullptr);

    // assume Pr and Qr is already calculated
    // set the other variables
    void evaluateEigenvalueAndSetReference(
      std::shared_ptr<ReferenceSampling> ref);

    // referenceSampl is the precomputed reference point
    // assume tdd, tuu, rud, rdu are gpu memory preallocated
    void evaluatePerturbationAndDifference(
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
      scalar * diffError = nullptr,
      scalar * matVar = nullptr);
};