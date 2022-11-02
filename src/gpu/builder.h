#pragma once
#include <complex>
#include "types.cuh"

NAMESPACE_BEGIN(acacia::gpu::em)
void build_rcwa_sample(
  int nCx, int nCy,
  const Real *x, 
  const Real *y,
  const std::complex<Real> *eps);
NAMESPACE_END(acacia::gpu::em)