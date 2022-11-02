#pragma once

#include "types.cuh"
#include "math_constants.h"
#include <cusolverDn.h>

NAMESPACE_BEGIN(acacia::gpu::em)

static constexpr Real Pi = CUDART_PI_F;

enum RCWAContinuityOption {
	ContinuousXY = 0,
	DiscontinuousX = 1,
	DiscontinuousY = 2
};

// all memory is on GPU
class CudaFourierSolver2D {
private:
  cusolverDnHandle_t cusolverH_;
  int nCx_, nCy_;
  const Real *x_;
  const Real *y_;
  
  static constexpr int LDA_ALIGNMENT = 32;
public:
	CudaFourierSolver2D(cusolverDnHandle_t cusolverH, 
    int nX, int nY, const Real *x, const Real *y):
    cusolverH_(cusolverH),
		nCx_(nX), nCy_(nY), x_(x), y_(y) { }

  template<RCWAContinuityOption Continuity_>
  size_t workspace_buffer_size(int nx, int ny) const;

  template<RCWAContinuityOption Continuity_>
  size_t iworkspace_buffer_size(int nx, int ny) const;

  size_t output_buffer_size(int nx, int ny) const { return nx*ny*nx*ny; }

  // output: Ff can be a gpu memory of size (nx*ny) x (nx*ny);
  // input: f of size (nCy-1)x(nCx-1);
  // assume all pointers are allocated out of this function
  // workspace and iworkspace should be allocated using the above functions
  template<RCWAContinuityOption Continuity_>
	void solve(complex_t *Ff, int nx, int ny, const complex_t *f, complex_t *workspace,
    int * iworkspace);
};

// require workspace size nx2 x ny2
// no require on ipiv
template<>
void CudaFourierSolver2D::solve<ContinuousXY>(
  complex_t *Ff, int nx, int ny, const complex_t *f, complex_t *workspace,
  int * iworkspace);


// require workspace size (nx2+ny2+2*nx*nx) x ngy
// require ipiv nx+1
template<>
void CudaFourierSolver2D::solve<DiscontinuousX>(
  complex_t *Ff, int nx, int ny, const complex_t *f, complex_t *workspace,
  int * iworkspace);

// require workspace size (ny2+nx2+2*ny*ny) x ngx
// require ipiv ny+1
template<>
void CudaFourierSolver2D::solve<DiscontinuousY>(
  complex_t *Ff, int nx, int ny, const complex_t *f, complex_t *workspace,
  int * iworkspace);
  
template<>
size_t CudaFourierSolver2D::iworkspace_buffer_size<ContinuousXY>(int nx, int ny) const;

template<>
size_t CudaFourierSolver2D::iworkspace_buffer_size<DiscontinuousX>(int nx, int ny) const;

template<>
size_t CudaFourierSolver2D::iworkspace_buffer_size<DiscontinuousY>(int nx, int ny) const;

template<>
size_t CudaFourierSolver2D::workspace_buffer_size<ContinuousXY>(int nx, int ny) const;

template<>
size_t CudaFourierSolver2D::workspace_buffer_size<DiscontinuousX>(int nx, int ny) const;

template<>
size_t CudaFourierSolver2D::workspace_buffer_size<DiscontinuousY>(int nx, int ny) const;
NAMESPACE_END(acacia::gpu::em)