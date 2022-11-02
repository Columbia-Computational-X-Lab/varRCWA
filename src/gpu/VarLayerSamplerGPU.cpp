#include "VarLayerSamplerGPU.h"
#include <iostream>
#include "utils/timer.hpp"
#include <cuda_runtime_api.h>
#include "gpu/CudaFourierSolver2D.h"
#include "gpu/CudaLayerSampler.h"
#include "gpu/MathKernelFunction.h"

using namespace std;
using namespace Eigen;

VarLayerSamplerGPU::VarLayerSamplerGPU(
    scalar lambda, 
    int nx, 
    int ny,
    const Eigen::VectorXs& x,
    const Eigen::VectorXs& y,
    const Eigen::MatrixXcs& eps,
    bool enablePML):
    LayerSampler(lambda, nx, ny, enablePML),
    x_(x), y_(y), eps_(eps),
    cublasH_(nullptr), cusolverH_(nullptr),
    x__(nullptr), y__(nullptr), eps__(nullptr)
{
    static_assert(sizeof(acacia::gpu::Real) == sizeof(scalar));
    static_assert(sizeof(acacia::gpu::complex_t) == sizeof(scalex));
    cublasCreate(&cublasH_);
    cusolverDnCreate(&cusolverH_);
    int nCx = x.size();
    int nCy = y.size();
    cudaMalloc(reinterpret_cast<void**>(&x__), 
      sizeof(acacia::gpu::Real)*nCx);
    cudaMalloc(reinterpret_cast<void**>(&y__), 
      sizeof(acacia::gpu::Real)*nCy);
    cudaMalloc(reinterpret_cast<void**>(&eps__), 
      sizeof(acacia::gpu::complex_t)*(nCx-1)*(nCy-1));
    cudaMemcpy(x__, x_.data(), sizeof(acacia::gpu::Real)*nCx, cudaMemcpyHostToDevice);
    cudaMemcpy(y__, y_.data(), sizeof(acacia::gpu::Real)*nCy, cudaMemcpyHostToDevice);
    cudaMemcpy(eps__, eps_.data(), sizeof(acacia::gpu::complex_t)*(nCx-1)*(nCy-1),
      cudaMemcpyHostToDevice);
    
    auto fourier = std::make_shared<acacia::gpu::em::CudaFourierSolver2D>(cusolverH_, 
      nCx, nCy, x__, y__);
    sampler_ = std::make_unique<acacia::gpu::em::CudaLayerSampler>(
      cublasH_, cusolverH_, fourier);
    size_t wSize = sampler_->workspace_buffer_size(nx, ny);
    cudaMalloc(reinterpret_cast<void**>(&workspace__), 
      sizeof(acacia::gpu::complex_t)*wSize);
    size_t iSize = sampler_->iworkspace_buffer_size(nx, ny);
    cudaMalloc(reinterpret_cast<void**>(&iworkspace__), 
      sizeof(int)*iSize);

    postInitialization();
    cudaMalloc(reinterpret_cast<void**>(&Kx__),
      sizeof(acacia::gpu::complex_t)*Kx_.size());
    cudaMalloc(reinterpret_cast<void**>(&Ky__),
      sizeof(acacia::gpu::complex_t)*Ky_.size());
    cudaMemcpy(Kx__, Kx_.data(), sizeof(acacia::gpu::complex_t)*Kx_.size(), 
      cudaMemcpyHostToDevice);
    cudaMemcpy(Ky__, Ky_.data(), sizeof(acacia::gpu::complex_t)*Ky_.size(), 
      cudaMemcpyHostToDevice);
    int blockDim = nx * ny;

    cudaMalloc(reinterpret_cast<void**>(&P00__),
      sizeof(acacia::gpu::complex_t)*blockDim*blockDim);
    cudaMalloc(reinterpret_cast<void**>(&P01__),
      sizeof(acacia::gpu::complex_t)*blockDim*blockDim);    
    cudaMalloc(reinterpret_cast<void**>(&P10__),
      sizeof(acacia::gpu::complex_t)*blockDim*blockDim);
    cudaMalloc(reinterpret_cast<void**>(&P11__),
      sizeof(acacia::gpu::complex_t)*blockDim*blockDim);
    cudaMalloc(reinterpret_cast<void**>(&Q00__),
      sizeof(acacia::gpu::complex_t)*blockDim*blockDim);
    cudaMalloc(reinterpret_cast<void**>(&Q01__),
      sizeof(acacia::gpu::complex_t)*blockDim*blockDim);    
    cudaMalloc(reinterpret_cast<void**>(&Q10__),
      sizeof(acacia::gpu::complex_t)*blockDim*blockDim);
    cudaMalloc(reinterpret_cast<void**>(&Q11__),
      sizeof(acacia::gpu::complex_t)*blockDim*blockDim);    
}

VarLayerSamplerGPU::~VarLayerSamplerGPU()
{
  cudaFree(P00__); cudaFree(P01__); cudaFree(P10__); cudaFree(P11__);
  cudaFree(Q00__); cudaFree(Q01__); cudaFree(Q10__); cudaFree(Q11__);
  cudaFree(Kx__);
  cudaFree(Ky__);
  cudaFree(iworkspace__);
  cudaFree(workspace__);
  cudaFree(x__);
  cudaFree(y__);
  cudaFree(eps__);
  if (cublasH_) {
    cublasDestroy(cublasH_);
  }
  if (cusolverH_) {
    cusolverDnDestroy(cusolverH_);
  }
}

void VarLayerSamplerGPU::assignSimulationRegion()
{
    int nCx = x_.size();
    int nCy = y_.size();
    Lx_ = x_(nCx-1) - x_(0);
    Ly_ = y_(nCy-1) - y_(0);

    b0_ = y_(0);
    b1_ = y_(1);
    u0_ = y_(nCy-2);
    u1_ = y_(nCy-1);

    l0_ = x_(0);
    l1_ = x_(1);
    r0_ = x_(nCx-2);
    r1_ = x_(nCx-1);
}


void VarLayerSamplerGPU::addXRule(int layout, std::function<scalar(scalar)> rule)
{
    xrules_[layout] = rule;
}

void VarLayerSamplerGPU::addYRule(int layout, std::function<scalar(scalar)> rule)
{
    yrules_[layout] = rule;
}

void VarLayerSamplerGPU::sampleOnGPU(scalar z,
  acacia::gpu::complex_t *P,
  acacia::gpu::complex_t *Q)
{
  int edgeSize = nx_*ny_;
  int matSize = edgeSize*edgeSize;

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  // auto t1 = sploosh::now();
  VectorXs x(x_);
  for (const auto & [i, f] : xrules_) { 
    x(i) = f(z);
  }
  VectorXs y(y_);
  for (const auto & [i, f] : yrules_) {
    y(i) = f(z);
  }
  // cout << "z=" << z << ", x: " << x.transpose() << endl;

  cudaMemcpy(x__, x.data(), sizeof(acacia::gpu::Real)*x.size(), 
    cudaMemcpyHostToDevice);
  cudaMemcpy(y__, y.data(), sizeof(acacia::gpu::Real)*y.size(),
    cudaMemcpyHostToDevice);
  // auto t2 = sploosh::now();
  // cudaEventRecord(start);
  sampler_->sample(eps__, Kx__, Ky__, lambda_, nx_, ny_, 
    P00__, P01__, P10__, P11__,
    Q00__, Q01__, Q10__, Q11__,
    workspace__, iworkspace__);
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // auto t3 = sploosh::now();

  cudaMemcpy2D(P, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    P00__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(P+2*matSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    P01__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(P+edgeSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    P10__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(P+2*matSize+edgeSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    P11__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(Q, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    Q00__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(Q+2*matSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    Q01__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(Q+edgeSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    Q10__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(Q+2*matSize+edgeSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    Q11__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToDevice);

  // acacia::gpu::em::cudaMemcpy2DDeviceToDevice(P, 2*edgeSize, P00__, 
  //   edgeSize, edgeSize, edgeSize);
  // acacia::gpu::em::cudaMemcpy2DDeviceToDevice(P+2*matSize, 2*edgeSize, P01__,
  //   edgeSize, edgeSize, edgeSize);
  // acacia::gpu::em::cudaMemcpy2DDeviceToDevice(P+edgeSize, 2*edgeSize, P10__,
  //   edgeSize, edgeSize, edgeSize);
  // acacia::gpu::em::cudaMemcpy2DDeviceToDevice(P+2*matSize+edgeSize, 2*edgeSize, P11__, 
  //   edgeSize, edgeSize, edgeSize);
  // acacia::gpu::em::cudaMemcpy2DDeviceToDevice(Q, 2*edgeSize, Q00__, 
  //   edgeSize, edgeSize, edgeSize);
  // acacia::gpu::em::cudaMemcpy2DDeviceToDevice(Q+2*matSize, 2*edgeSize, Q01__, 
  //   edgeSize, edgeSize, edgeSize);
  // acacia::gpu::em::cudaMemcpy2DDeviceToDevice(Q+edgeSize, 2*edgeSize, Q10__, 
  //   edgeSize, edgeSize, edgeSize);
  // acacia::gpu::em::cudaMemcpy2DDeviceToDevice(Q+2*matSize+edgeSize, 2*edgeSize, Q11__, 
  //   edgeSize, edgeSize, edgeSize);
    
  // auto t4 = sploosh::now();
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop); 
  // cout << "sample: " << sploosh::duration_milli_d(t1, t2) << "\t"
  //     << milliseconds << " with cpu: " << sploosh::duration_milli_d(t2, t3) << "\t"
  //     << sploosh::duration_milli_d(t3, t4) << endl;  
}

void VarLayerSamplerGPU::sample(scalar z, 
    Eigen::MatrixXcs& P,
    Eigen::MatrixXcs& Q)
{
  int edgeSize = nx_*ny_;
  int matSize = edgeSize*edgeSize;

  // acacia::gpu::complex_t *P__=nullptr;
  // acacia::gpu::complex_t *Q__=nullptr;

  // cout << "P size:" << 4*matSize << endl;
  // cudaMalloc((void**)&P__, sizeof(acacia::gpu::complex_t)*4*matSize);
  // cudaMalloc((void**)&Q__, sizeof(acacia::gpu::complex_t)*4*matSize);
  // sampleOnGPU(z, P__, Q__);
  // P.resize(2*edgeSize, 2*edgeSize);
  // Q.resize(2*edgeSize, 2*edgeSize);
  // cudaMemcpy(P.data(), P__, sizeof(acacia::gpu::complex_t)*4*matSize, 
  //   cudaMemcpyDeviceToHost);
  // cudaMemcpy(Q.data(), Q__, sizeof(acacia::gpu::complex_t)*4*matSize, 
  //   cudaMemcpyDeviceToHost);  
  // cudaFree(P__);
  // cudaFree(Q__);

  // cout << P.cwiseAbs().maxCoeff() << endl;
  // cout << Q.cwiseAbs().maxCoeff() << endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto t1 = sploosh::now();
  VectorXs x(x_);
  for (const auto & [i, f] : xrules_) { 
    x(i) = f(z);
  }
  VectorXs y(y_);
  for (const auto & [i, f] : yrules_) {
    y(i) = f(z);
  }

  cudaMemcpy(x__, x.data(), sizeof(acacia::gpu::Real)*x.size(), 
    cudaMemcpyHostToDevice);
  cudaMemcpy(y__, y.data(), sizeof(acacia::gpu::Real)*y.size(),
    cudaMemcpyHostToDevice);
  
  auto t2 = sploosh::now();
  cudaEventRecord(start);
  sampler_->sample(eps__, Kx__, Ky__, lambda_, nx_, ny_, 
    P00__, P01__, P10__, P11__,
    Q00__, Q01__, Q10__, Q11__,
    workspace__, iworkspace__);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  auto t3 = sploosh::now();
  P.resize(2*edgeSize, 2*edgeSize);
  Q.resize(2*edgeSize, 2*edgeSize);
  cudaMemcpy2D(P.data(), 2*edgeSize*sizeof(acacia::gpu::complex_t),
    P00__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(P.data()+2*matSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    P01__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(P.data()+edgeSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    P10__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(P.data()+2*matSize+edgeSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    P11__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(Q.data(), 2*edgeSize*sizeof(acacia::gpu::complex_t),
    Q00__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(Q.data()+2*matSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    Q01__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(Q.data()+edgeSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    Q10__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(Q.data()+2*matSize+edgeSize, 2*edgeSize*sizeof(acacia::gpu::complex_t),
    Q11__, edgeSize*sizeof(acacia::gpu::complex_t), 
    edgeSize*sizeof(acacia::gpu::complex_t), edgeSize, cudaMemcpyDeviceToHost);
  auto t4 = sploosh::now();
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop); 
  cout << "sample: " << sploosh::duration_milli_d(t1, t2) << "\t"
      << milliseconds << " with cpu: " << sploosh::duration_milli_d(t2, t3) << "\t"
      << sploosh::duration_milli_d(t3, t4) << endl;
}