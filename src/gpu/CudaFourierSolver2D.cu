#include "CudaFourierSolver2D.h"
#include <cstdio>

NAMESPACE_BEGIN(acacia::gpu::em)


// TODO stream
// set the initial data nHx x nHy
__global__ void constinuous_xy_fourier_kernel(complex_t *workspace, 
  int nx2, int ny2,
  int ncx, int ncy,
  const Real *x, const Real *y, const complex_t *f)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nx2 && j < ny2) {
    // assume column major here
    int ind = j*nx2+i;
    int m = i - (nx2-1)/2;
    int n = j - (ny2-1)/2;
    Real dx = x[ncx-1] - x[0];
    Real dy = y[ncy-1] - y[0];
    Real kx = m * Pi / dx;
    Real ky = n * Pi / dy;

    workspace[ind].x = 0.;
    workspace[ind].y = 0.;

    for (int ii = 0; ii < ncx-1; ++ii) {
      Real x0 = x[ii];
      Real x1 = x[ii+1];
      Real hr, hi;
      if (m == 0) {
        hr = (x1 - x0) / dx;
        hi = 0.;
      } else {
        Real coskxp = cos(kx*(x1+x0));
        Real sinkxp = sin(kx*(x1+x0));
        Real sinkxm = sin(kx*(x1-x0));
        hr = coskxp * sinkxm / (m * Pi);
        hi = -sinkxp * sinkxm / (m * Pi);
      }

      for (int jj = 0; jj < ncy-1; ++jj) {
        Real y0 = y[jj];
        Real y1 = y[jj+1];
        Real vr, vi;
        if (n == 0) {
          vr = (y1 - y0) / dy;
          vi = 0.;
        } else {
          Real coskyp = cos(ky*(y1+y0));
          Real sinkyp = sin(ky*(y1+y0));
          Real sinkym = sin(ky*(y1-y0));
          vr = coskyp * sinkym / (n * Pi);
          vi = -sinkyp * sinkym / (n * Pi);
        }

        complex_t fv = f[ii*(ncy-1)+jj];

        Real mulr = hr * vr - hi * vi;
        Real muli = hr * vi + hi * vr;
        workspace[ind].x += fv.x * mulr - fv.y * muli;
        workspace[ind].y += fv.x * muli + fv.y * mulr;
      }
    }
    // printf("m: %d, n: %d, Ff: %f+%fj\n", m, n, workspace[ind].x, workspace[ind].y);
  }
}


__global__ void block_toeplitz_copy_kernel(complex_t *Ff, 
  int nx, int ny, const complex_t *workspace)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nx*ny && col < nx*ny) {
    int m = row / ny;
    int n = row % ny;
    int j = col / ny;
    int l = col % ny;
  
    int diffi = m - j;
    int diffj = n - l;

    int srow = nx-1+diffi; // x index 
    int scol = ny-1+diffj; // y index

    int sind = scol*(2*nx-1)+srow; 
    int tind = col*(nx*ny) + row;

    Ff[tind] = workspace[sind];
    // printf("%d\t%d\t%e+%ej\n", row, col, Ff[tind].x, Ff[tind].y);
  }
}


template<>
void CudaFourierSolver2D::solve<ContinuousXY>(
  complex_t *Ff, int nx, int ny, const complex_t *f, complex_t *workspace,
  int * iworkspace)
{
  int nx2 = 2*nx-1;
  int ny2 = 2*ny-1;
  dim3 blockSize(LDA_ALIGNMENT, LDA_ALIGNMENT, 1);

  int bx = (nx2 + blockSize.x - 1)/blockSize.x;
  int by = (ny2 + blockSize.y - 1)/blockSize.y;
  dim3 gridSize(bx, by, 1);

  constinuous_xy_fourier_kernel<<<gridSize, blockSize>>>(workspace, nx2, ny2, 
    nCx_, nCy_, x_, y_, f);

  int edgeSize = nx*ny;
  int bx2 = (edgeSize + blockSize.x-1)/blockSize.x;
  int by2 = (edgeSize + blockSize.y-1)/blockSize.y;
  dim3 gridSize2(bx2, by2, 1);

  block_toeplitz_copy_kernel<<<gridSize2, blockSize>>>(Ff, nx, ny, workspace);
}

// for each row, calculate the toeplitz matrix of the inverse value
__global__ void row_by_row_toeplitz_of_fourier_inverse_kernel(
  complex_t *workspace, int nx2, int ny2, int ngy, int ncx, 
  const Real *x, const Real *y, const complex_t *f, bool isRow)
{
  // printf("gridDim: %d, %d, %d\n", gridDim.x, gridDim.y, gridDim.z);
  // printf("blockDim: %d, %d, %d\n", blockDim.x, blockDim.y, blockDim.z);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
 
  if (j < ngy) {
    // int ind = i*ngy+j;
    // we want to store the vector with the same j(ngy) together
    int ind = j*(nx2+ny2)+i;
    if (i < nx2) { // x component harmonics
      Real dx = x[ncx-1] - x[0];
      int m = i - (nx2-1)/2;
      workspace[ind].x = 0.;
      workspace[ind].y = 0.;
      for (int ii = 0; ii < ncx-1; ++ii) {
        int indv = -1;
        if (isRow) {
          indv = ii*ngy+j;
        } else {
          indv = j*(ncx-1)+ii;
        }
        complex_t fv = f[indv];
        Real dnm = fv.x*fv.x+fv.y*fv.y;
        Real ifr = fv.x / dnm;
        Real ifi = -fv.y / dnm;
        Real x0 = x[ii];
        Real x1 = x[ii+1];
        if (m == 0) {
          workspace[ind].x += ifr * (x1-x0) / dx;
          workspace[ind].y += ifi * (x1-x0) / dx;
        } else {
          Real kx = m*Pi / dx;
          Real coskxp = cos(kx*(x1+x0));
          Real sinkxp = sin(kx*(x1+x0));
          Real sinkxm = sin(kx*(x1-x0));
          Real hr = coskxp * sinkxm / (m*Pi);
          Real hi = -sinkxp * sinkxm / (m*Pi);
          workspace[ind].x += ifr * hr - ifi * hi;
          workspace[ind].y += ifi * hr + ifr * hi;
        }
      }
    } else if (i < nx2 + ny2) { // y component harmonics
      Real dy = y[ngy] - y[0];
      int n = i - nx2 - (ny2-1) / 2;
      Real y0 = y[j];
      Real y1 = y[j+1];
      if (n == 0) {
        workspace[ind].x = (y1 - y0) / dy;
        workspace[ind].y = 0.;
      } else {
        Real ky = n * Pi / dy;
        Real coskyp = cos(ky*(y1+y0));
        Real sinkyp = sin(ky*(y1+y0));
        Real sinkym = sin(ky*(y1-y0));
        workspace[ind].x = coskyp * sinkym / (n*Pi);
        workspace[ind].y = -sinkyp * sinkym / (n*Pi);
      }
    }
    if (i < nx2 + ny2) {
      // printf("%d, %d, %f+%fj\n", j, i, workspace[ind].x, workspace[ind].y);
    }
  }
}

__global__ void row_by_row_fill_in_toeplitz_linear_system_kernel(
  complex_t *lsys_ptr, int nx, int ny, int ngy, complex_t *workspace)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < ngy) {
    int ind = j * (2*nx*nx) + i;
    complex_t *vec = workspace+j*(2*nx-1+2*ny-1);
    if (i < nx*nx) { // fill in the toeplitz matrix
      int col = i / nx;
      int row = i % nx;
      lsys_ptr[ind].x = vec[(row-col)+nx-1].x;
      lsys_ptr[ind].y = vec[(row-col)+nx-1].y;
    } else if (i < 2*nx*nx) { // fill in the identity matrix
      int col = (i-nx*nx) / nx;
      int row = (i-nx*nx) % nx;
      if (row == col) {
        lsys_ptr[ind].x = 1.;
      } else {
        lsys_ptr[ind].x = 0.;
      }
      lsys_ptr[ind].y = 0.;
    }
  }
}

__global__ void sum_rows_mixing_toeplitz_matrix_kernel(
  complex_t *Ff, int nx, int ny, int ngy, 
  const complex_t *workspace, const complex_t *lsys_ptr)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nx*ny && col < nx*ny) {
    int m = row / ny;
    int n = row % ny;
    int j = col / ny;
    int l = col % ny;
    int ind = col*(nx*ny)+row;
    Ff[ind].x = 0.;
    Ff[ind].y = 0.;
    int indfy = n-l+ny-1;
    int indx = j*nx+m; // (m, j)
    for (int i = 0; i < ngy; ++i) {
      const complex_t *fy = workspace + i*(2*nx-1+2*ny-1)+2*nx-1;
      const complex_t *invfinvfx = lsys_ptr + i*(2*nx*nx) + nx*nx;
      Ff[ind].x += fy[indfy].x * invfinvfx[indx].x - fy[indfy].y * invfinvfx[indx].y;
      Ff[ind].y += fy[indfy].x * invfinvfx[indx].y + fy[indfy].y * invfinvfx[indx].x;
    }
    // printf("%d\t%d\t%d\t%d\t%d\n", row, col, indfy, m, j);
    // printf("nx: %d, ny: %d\n", nx, ny);
    // printf("%d\t%d\t%f+%fj\n", row, col, Ff[ind].x, Ff[ind].y);
  }
}

__global__ void sum_cols_mixing_toeplitz_matrix_kernel(
  complex_t *Ff, int nx, int ny, int ngx, 
  const complex_t *workspace, const complex_t *lsys_ptr)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nx*ny && col < nx*ny) {
    int m = row / ny;
    int n = row % ny;
    int j = col / ny;
    int l = col % ny;
    int ind = col*(nx*ny)+row;
    Ff[ind].x = 0.;
    Ff[ind].y = 0.;
    int indfx = m-j+nx-1;
    int indy = l*ny+n; // (n, l)
    for (int i = 0; i < ngx; ++i) {
      const complex_t *fx = workspace + i*(2*ny-1+2*nx-1)+2*ny-1;
      const complex_t *invfinvfy = lsys_ptr + i*(2*ny*ny) + ny*ny;
      Ff[ind].x += fx[indfx].x * invfinvfy[indy].x - fx[indfx].y * invfinvfy[indy].y;
      Ff[ind].y += fx[indfx].x * invfinvfy[indy].y + fx[indfx].y * invfinvfy[indy].x;
    }
    // printf("%d\t%d\t%d\t%d\t%d\n", row, col, indfy, m, j);
    // printf("nx: %d, ny: %d\n", nx, ny);
    // printf("%d\t%d\t%f+%fj\n", row, col, Ff[ind].x, Ff[ind].y);
  }  
}

template<>
void CudaFourierSolver2D::solve<DiscontinuousX>(
  complex_t *Ff, int nx, int ny, const complex_t *f, complex_t *workspace,
  int * iworkspace)
{
  int nx2 = 2*nx-1;
  int ny2 = 2*ny-1;
  int ngy = nCy_-1;

  dim3 blockSize(LDA_ALIGNMENT, LDA_ALIGNMENT, 1);

  int bx = (nx2+ny2 + blockSize.x - 1)/blockSize.x;
  int by = (ngy + blockSize.y - 1)/blockSize.y;

  dim3 gridSize(bx, by, 1);
  row_by_row_toeplitz_of_fourier_inverse_kernel<<<gridSize, blockSize>>>(
    workspace, nx2, ny2, ngy, nCx_, x_, y_, f, true);
  
  int offset = (nx2+ny2)*ngy;
  complex_t *lsys_ptr = workspace + offset;

  int bx2 = (2*nx*nx + blockSize.x - 1)/blockSize.x;
  dim3 gridSize2(bx2, by, 1);
  row_by_row_fill_in_toeplitz_linear_system_kernel<<<gridSize2, blockSize>>>(
    lsys_ptr, nx, ny, ngy, workspace);

  // printf("ngy: %d\n", ngy);
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);

  // const int CUDA_NUM_STREAMS = 6;
  // cudaStream_t streams[CUDA_NUM_STREAMS];
  // for (int i = 0; i < CUDA_NUM_STREAMS; ++i) {
  //   cudaStreamCreate(&streams[i]); 
  // }
  // TODO optimize using stream
  complex_t *d_work = lsys_ptr + (2*nx*nx)*ngy;
  int *d_info = iworkspace + nx;
  for (int i = 0; i < ngy; ++i) {
    // const int curStream = i % CUDA_NUM_STREAMS;
    // cusolverDnSetStream(cusolverH_, streams[curStream]);
    complex_t * invfinvfx = lsys_ptr + i*(2*nx*nx);
    complex_t * identity = invfinvfx + nx*nx;
    acacia_gpu_em_Complexgetrf_(cusolverH_, nx, nx,
      invfinvfx, nx,
      d_work,
      iworkspace, d_info);
    acacia_gpu_em_Complexgetrs_(cusolverH_, CUBLAS_OP_N, nx, nx,
      invfinvfx, nx,
      iworkspace,
      identity, nx,
      d_info);
  }

  // for (int i = 0; i < CUDA_NUM_STREAMS; ++i) {
  //   cudaStreamSynchronize(streams[i]);
  //   cudaStreamDestroy(streams[i]);
  // }

  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop); 
  // printf("loop time: %fms\n", milliseconds);

  int edgeSize = nx*ny;
  int bx3 = (edgeSize + blockSize.x-1)/blockSize.x;
  int by3 = (edgeSize + blockSize.y-1)/blockSize.y;
  dim3 gridSize3(bx3, by3, 1);
  sum_rows_mixing_toeplitz_matrix_kernel<<<gridSize3, blockSize>>>(
    Ff, nx, ny, ngy, workspace, lsys_ptr);
}

template<>
void CudaFourierSolver2D::solve<DiscontinuousY>(
  complex_t *Ff, int nx, int ny, const complex_t *f, complex_t *workspace,
  int * iworkspace)
{
  int nx2 = 2*nx-1;
  int ny2 = 2*ny-1;
  int ngx = nCx_-1;

  dim3 blockSize(LDA_ALIGNMENT, LDA_ALIGNMENT, 1);
  int bx = (ny2+nx2+blockSize.x - 1)/blockSize.x;
  int by = (ngx + blockSize.y - 1)/blockSize.y;

  dim3 gridSize(bx, by, 1);
  row_by_row_toeplitz_of_fourier_inverse_kernel<<<gridSize, blockSize>>>(
    workspace, ny2, nx2, ngx, nCy_, y_, x_, f, false);

  int offset = (ny2+nx2)*ngx;
  complex_t *lsys_ptr = workspace + offset;
  int bx2 = (2*ny*ny + blockSize.x - 1)/blockSize.x;
  dim3 gridSize2(bx2, by, 1);
  row_by_row_fill_in_toeplitz_linear_system_kernel<<<gridSize2, blockSize>>>(
    lsys_ptr, ny, nx, ngx, workspace);
  
  complex_t *d_work = lsys_ptr + (2*ny*ny)*ngx;
  int *d_info = iworkspace + ny;
  for (int i = 0; i < ngx; ++i) {
    complex_t * invfinvfy = lsys_ptr + i*(2*ny*ny);
    complex_t * identity = invfinvfy + ny*ny;
    acacia_gpu_em_Complexgetrf_(cusolverH_, ny, ny,
      invfinvfy, ny,
      d_work,
      iworkspace, d_info);
    acacia_gpu_em_Complexgetrs_(cusolverH_, CUBLAS_OP_N, ny, ny,
      invfinvfy, ny,
      iworkspace,
      identity, ny,
      d_info);
  }
  int edgeSize = nx*ny;
  int bx3 = (edgeSize + blockSize.x-1)/blockSize.x;
  int by3 = (edgeSize + blockSize.y-1)/blockSize.y;
  dim3 gridSize3(bx3, by3, 1);
  sum_cols_mixing_toeplitz_matrix_kernel<<<gridSize3, blockSize>>>(
    Ff, nx, ny, ngx, workspace, lsys_ptr);
}

template<>
size_t CudaFourierSolver2D::iworkspace_buffer_size<ContinuousXY>(int nx, int ny) const
{
  return 0;
}

template<>
size_t CudaFourierSolver2D::iworkspace_buffer_size<DiscontinuousX>(int nx, int ny) const
{
  return nx+1;
}

template<>
size_t CudaFourierSolver2D::iworkspace_buffer_size<DiscontinuousY>(int nx, int ny) const
{
  return ny+1;
}

template<>
size_t CudaFourierSolver2D::workspace_buffer_size<ContinuousXY>(int nx, int ny) const
{
  return (2*nx-1)*(2*ny-1);
}

template<>
size_t CudaFourierSolver2D::workspace_buffer_size<DiscontinuousX>(int nx, int ny) const
{
  int lwork;
  acacia_gpu_em_Complexgetrf_bufferSize(
    cusolverH_, nx, nx,
    nullptr, nx, &lwork);
  return (2*nx-1+2*ny-1+2*nx*nx) * (nCy_-1) + lwork;
  
}

template<>
size_t CudaFourierSolver2D::workspace_buffer_size<DiscontinuousY>(int nx, int ny) const
{
  int lwork;
  acacia_gpu_em_Complexgetrf_bufferSize(
    cusolverH_, ny, ny,
    nullptr, ny, &lwork);
  return (2*nx-1+2*ny-1+2*ny*ny) * (nCx_-1) + lwork;
}

NAMESPACE_END(acacia::gpu::em)