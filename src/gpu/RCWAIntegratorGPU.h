#pragma once
#include "rcwa/defns.h"
#include "types.cuh"
#include <memory>
#include <utility>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "RedhefferIntegratorGPU.h"

class RCWAIntegratorGPU
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

    acacia::gpu::complex_t *dwork_;
    
    int *iwork_;
    int *info_;

    std::shared_ptr<ReferenceSampling> samp0_;
    std::shared_ptr<ReferenceSampling> samp2_;

    std::shared_ptr<ReferenceSampling> leftRef_; 

    // pinned memory
    acacia::gpu::complex_t *magma_PQ_;
    acacia::gpu::complex_t *magma_VR_;
    acacia::gpu::complex_t *magma_work_;
    int magma_lwork_;   
public:
    RCWAIntegratorGPU(scalar z0, scalar z1, 
      std::shared_ptr<LayerSampler> sampler);
    ~RCWAIntegratorGPU();
    
    void compute(int N = 10);
    
    int N() const;
    int nEigVal() const;

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
      
    // assume Pr and Qr is already calculated
    // set the other variables
    void evaluateEigenvalueAndSetReference(
      std::shared_ptr<ReferenceSampling> ref);

    void evaluateScatteringMatrix(
      std::shared_ptr<const ReferenceSampling> ref,
      scalar z0, scalar z1,
      acacia::gpu::complex_t * tdd,
      acacia::gpu::complex_t * tuu,
      acacia::gpu::complex_t * rud,
      acacia::gpu::complex_t * rdu);
};