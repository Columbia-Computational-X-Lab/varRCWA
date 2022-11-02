#pragma once
#include <cuComplex.h>

#if !defined(NAMESPACE_BEGIN)
#   define NAMESPACE_BEGIN(name) namespace name {
#endif
#if !defined(NAMESPACE_END)
#   define NAMESPACE_END(name) }
#endif

#define HAVE_CUBLAS
// #define ACA_USE_FLOAT32
NAMESPACE_BEGIN(acacia::gpu)
#ifdef ACA_USE_FLOAT32
  using complex_t = cuFloatComplex;
  using Real = float;
#else 
  using complex_t = cuDoubleComplex;
  using Real = double;
#endif

#ifdef ACA_USE_FLOAT32
  #define acacia_gpu_em_Complexgetrf_ cusolverDnCgetrf
  #define acacia_gpu_em_Complexgetrs_ cusolverDnCgetrs
  #define acacia_gpu_em_Complexgetrf_bufferSize cusolverDnCgetrf_bufferSize
  #define acacia_gpu_em_Complexgemm3m_ cublasCgemm3m
  #define acacia_gpu_em_Complexaxpy_ cublasCaxpy
  #define acacia_gpu_em_Complexscal_ cublasCscal
  #define acacia_gpu_em_Complexamax_ cublasIcamax
#else 
  #define acacia_gpu_em_Complexgetrf_ cusolverDnZgetrf
  #define acacia_gpu_em_Complexgetrs_ cusolverDnZgetrs
  #define acacia_gpu_em_Complexgemm3m_ cublasZgemm3m
  #define acacia_gpu_em_Complexgetrf_bufferSize cusolverDnZgetrf_bufferSize
  #define acacia_gpu_em_Complexaxpy_ cublasZaxpy
  #define acacia_gpu_em_Complexscal_ cublasZscal
  #define acacia_gpu_em_Complexamax_ cublasIzamax
#endif
NAMESPACE_END(acacia::gpu)
