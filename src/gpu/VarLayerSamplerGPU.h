#pragma once

#include "rcwa/defns.h"
#include "types.cuh"
#include "core/LayerSampler.h"
#include <functional>
#include <unordered_map>
#include <memory>
#include <cusolverDn.h>
#include <cublas_v2.h>

namespace acacia::gpu::em {
  class CudaLayerSampler;
}

class VarLayerSamplerGPU:
    public LayerSampler
{
private:
  Eigen::VectorXs x_;
  Eigen::VectorXs y_;
  Eigen::MatrixXcs eps_;

  std::unordered_map<int, std::function<scalar(scalar)>> xrules_;
  std::unordered_map<int, std::function<scalar(scalar)>> yrules_;

  // pre-allocated gpu memory
  cublasHandle_t cublasH_;
  cusolverDnHandle_t cusolverH_;
  acacia::gpu::Real *x__;
  acacia::gpu::Real *y__;
  acacia::gpu::complex_t *eps__; 
  std::unique_ptr<acacia::gpu::em::CudaLayerSampler> sampler_;
  acacia::gpu::complex_t *workspace__;
  int *iworkspace__;
  acacia::gpu::complex_t *Kx__;
  acacia::gpu::complex_t *Ky__;
  acacia::gpu::complex_t *P00__, *P01__, *P10__, *P11__;
  acacia::gpu::complex_t *Q00__, *Q01__, *Q10__, *Q11__;
    
public: 
  VarLayerSamplerGPU(
      scalar lambda, 
      int nx, 
      int ny,
      const Eigen::VectorXs& x,
      const Eigen::VectorXs& y,
      const Eigen::MatrixXcs& eps,
      bool enablePML = true);
  ~VarLayerSamplerGPU();

  void addXRule(int layout, std::function<scalar(scalar)> rule);
  void addYRule(int layout, std::function<scalar(scalar)> rule);

  // P and Q are preallocated gpu memory
  void sampleOnGPU(scalar z,
    acacia::gpu::complex_t *P,
    acacia::gpu::complex_t *Q) override;
  
  void sample(scalar z, 
      Eigen::MatrixXcs& P,
      Eigen::MatrixXcs& Q) override;
private:
    void assignSimulationRegion() override;
};