#include "GDSIISamplerGPU.h"
#include <string>
#include <limits>
#include <iostream>

#include "rcwa/Layer.h"
#include "utils/timer.hpp"
#include "gpu/CudaFourierSolver2D.h"
#include "gpu/CudaLayerSampler.h"
#include <cuda_runtime_api.h>

constexpr scalar MICRON = 1e-6;
constexpr scalar tolerance = 1e-12;

using namespace std;
using namespace Eigen;

GDSIISamplerGPU::GDSIISamplerGPU(scalar lambda, 
	int nx, 
	int ny,
	const std::string& filename,
	const Eigen::Vector3s& shift,
	scalar lPML, scalar rPML, scalar lspace, scalar rspace,
	scalar bPML, scalar uPML, scalar bspace, scalar uspace,
	const Material &permBackground, const Material &permDevice,
	scalar thickness) : 
	LayerSampler(lambda, nx, ny),
	shift_(shift),
	lPML_(lPML), rPML_(rPML), lspace_(lspace), rspace_(rspace),
	bPML_(bPML), uPML_(uPML), bspace_(bspace), uspace_(uspace),
	permBackground_(permBackground), permDevice_(permDevice),
	thickness_(thickness),
	x__(nullptr), xSize_(0),
	eps__(nullptr), epsSize_(0),
	iworkspace__(nullptr), iSize_(0),
	workspace__(nullptr), wSize_(0),
	y__(nullptr), Kx__(nullptr), Ky__(nullptr),
	P00__(nullptr), P01__(nullptr), P10__(nullptr), P11__(nullptr),
	Q00__(nullptr), Q01__(nullptr), Q10__(nullptr), Q11__(nullptr) {
	
	lib_ = gdstk::read_gds(filename.c_str(), MICRON);
	postInitialization();

	y_.resize(6);
	y_(0) = b0_;
	y_(1) = b1_;
	y_(2) = b1_ + bspace_;
	y_(3) = y_(2) + thickness_;
	y_(4) = u0_;
	y_(5) = u1_;

	cout << "y: " << y_.transpose() << endl;

	yIndicator_.resize(5);
	yIndicator_ << 0, 0, 1, 0, 0;
	// GPU allocation
	static_assert(sizeof(acacia::gpu::Real) == sizeof(scalar));
	static_assert(sizeof(acacia::gpu::complex_t) == sizeof(scalex));
	cublasCreate(&cublasH_);
	cusolverDnCreate(&cusolverH_);
	cudaMalloc(reinterpret_cast<void**>(&y__), sizeof(acacia::gpu::Real)*y_.size());
	cudaMemcpy(y__, y_.data(), sizeof(acacia::gpu::Real)*y_.size(), cudaMemcpyHostToDevice);

	cudaMalloc(reinterpret_cast<void**>(&Kx__), sizeof(acacia::gpu::complex_t)*Kx_.size());
	cudaMalloc(reinterpret_cast<void**>(&Ky__), sizeof(acacia::gpu::complex_t)*Ky_.size());
	cudaMemcpy(Kx__, Kx_.data(), sizeof(acacia::gpu::complex_t)*Kx_.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(Ky__, Ky_.data(), sizeof(acacia::gpu::complex_t)*Ky_.size(), cudaMemcpyHostToDevice);
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

GDSIISamplerGPU::~GDSIISamplerGPU() {
	// GPU destroy
  cudaFree(P00__); cudaFree(P01__); cudaFree(P10__); cudaFree(P11__);
  cudaFree(Q00__); cudaFree(Q01__); cudaFree(Q10__); cudaFree(Q11__);
  cudaFree(Kx__);
  cudaFree(Ky__);
	cudaFree(y__);

	if (x__) {
		cudaFree(x__);
	}
	if (eps__) {
		cudaFree(eps__);
	}
	if (workspace__) {
		cudaFree(workspace__);
	}
	if (iworkspace__) {
		cudaFree(iworkspace__);
	}
	if (cublasH_) {
		cublasDestroy(cublasH_);
	}
	if (cusolverH_) {
		cusolverDnDestroy(cusolverH_);
	}

	lib_.clear();
}

void GDSIISamplerGPU::sampleLayout(scalar z,
	Eigen::VectorXs& x,
	Eigen::MatrixXcs& eps) {
	
	gdstk::Cell *pcell = lib_.cell_array[0];
	vector<scalar> cross_section_xs;
	for (int n = 0; n < pcell->polygon_array.size; ++n) {
		gdstk::Polygon *ppoly = pcell->polygon_array[n];
		gdstk::Array<gdstk::Vec2> point_array = ppoly->point_array;
		for (int i = 0; i < point_array.size; ++i) {
			const gdstk::Vec2 &p0 = point_array[i];
			const gdstk::Vec2 &p1 = (i == point_array.size - 1) ? point_array[0] : point_array[i+1];
			if ((p0.y <= z && z < p1.y) || (p1.y <= z && z < p0.y)) {
				scalar h = p1.y - p0.y;
				// bias to p0
				if (fabs(h) < tolerance) {
						h += 2. * tolerance;
				}
				scalar alpha = (z - p0.y) / h;
				cross_section_xs.push_back((1 - alpha) * p0.x + alpha * p1.x);
			}
		}
	}
	std::sort(cross_section_xs.begin(), cross_section_xs.end());
	int Nx = cross_section_xs.size() + 4;
	x.resize(Nx);
	x(0) = l0_;
	x(1) = l1_;
	x(Nx-2) = r0_;
	x(Nx-1) = r1_;
	for (int i = 2; i <= Nx - 3; ++i) {
			x(i) = cross_section_xs[i-2] + shift_.x();
	}
	cout << "z=" << z << ", x=" << x.transpose() << endl;
	eps = MatrixXcs::Constant(y_.size()-1, Nx-1, permBackground_.permittivity(lambda_));

	for (int i = 0; i < y_.size()-1; ++i) {
		if (yIndicator_(i) == 1) {
			for (int j = 2; j <= Nx-4; j+=2) {
					eps(i, j) = permDevice_.permittivity(lambda_);
			}
		}
	}
}

void GDSIISamplerGPU::sample(scalar z, 
	Eigen::MatrixXcs& P,
	Eigen::MatrixXcs& Q) {
	
	int edgeSize = nx_*ny_;
	int matSize = edgeSize*edgeSize;
	int nCy = y_.size();

	VectorXs x;
	MatrixXcs eps;
	sampleLayout(z, x, eps);

	int xSize = x.size();
	if (x__ == nullptr || xSize_ < xSize) {
		if (x__ != nullptr) { cudaFree(x__); }
		cudaMalloc(reinterpret_cast<void**>(&x__), sizeof(acacia::gpu::Real)*xSize);
		xSize_ = xSize;
	}
	cudaMemcpy(x__, x.data(), sizeof(acacia::gpu::Real)*xSize, cudaMemcpyHostToDevice);

	int epsSize = eps.size();
	if (eps__ == nullptr || epsSize_ < epsSize) {
		if (eps__ != nullptr) { cudaFree(eps__); }
		cudaMalloc(reinterpret_cast<void**>(&eps__), sizeof(acacia::gpu::complex_t)*epsSize);
		epsSize_ = epsSize;
	}
	cudaMemcpy(eps__, eps.data(), sizeof(acacia::gpu::complex_t)*epsSize, cudaMemcpyHostToDevice);
	
	auto fourier = std::make_shared<acacia::gpu::em::CudaFourierSolver2D>(cusolverH_, 
		xSize, nCy, x__, y__);
	sampler_ = std::make_unique<acacia::gpu::em::CudaLayerSampler>(cublasH_, cusolverH_, fourier);

	size_t wSize = sampler_->workspace_buffer_size(nx_, ny_);
	if (workspace__ == nullptr || wSize_ < wSize) {
		if (workspace__ != nullptr) { cudaFree(workspace__); }
		cudaMalloc(reinterpret_cast<void**>(&workspace__), sizeof(acacia::gpu::complex_t)*wSize);
		wSize_ = wSize;
	}

	size_t iSize = sampler_->iworkspace_buffer_size(nx_, ny_);
	if (iworkspace__ == nullptr || iSize_ < iSize) {
		if (iworkspace__ != nullptr) { cudaFree(iworkspace__); }
		cudaMalloc(reinterpret_cast<void**>(&iworkspace__), sizeof(int)*iSize);
		iSize_ = iSize;
	}

	sampler_->sample(eps__, Kx__, Ky__, lambda_, nx_, ny_, 
		P00__, P01__, P10__, P11__,
		Q00__, Q01__, Q10__, Q11__,
		workspace__, iworkspace__);
	
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
}

void GDSIISamplerGPU::sampleOnGPU(scalar z,
	acacia::gpu::complex_t *P,
	acacia::gpu::complex_t *Q) {
	int edgeSize = nx_*ny_;
	int matSize = edgeSize*edgeSize;
	int nCy = y_.size();

	VectorXs x;
	MatrixXcs eps;
	sampleLayout(z, x, eps);

	int xSize = x.size();
	if (x__ == nullptr || xSize_ < xSize) {
		cout << "allocate x size: " << xSize << endl;
		if (x__ != nullptr) { cudaFree(x__); }
		cudaMalloc(reinterpret_cast<void**>(&x__), sizeof(acacia::gpu::Real)*xSize);
		xSize_ = xSize;
	}
	cudaMemcpy(x__, x.data(), sizeof(acacia::gpu::Real)*xSize, cudaMemcpyHostToDevice);

	int epsSize = eps.size();
	if (eps__ == nullptr || epsSize_ < epsSize) {
		if (eps__ != nullptr) { cudaFree(eps__); }
		cudaMalloc(reinterpret_cast<void**>(&eps__), sizeof(acacia::gpu::complex_t)*epsSize);
		epsSize_ = epsSize;
	}
	cudaMemcpy(eps__, eps.data(), sizeof(acacia::gpu::complex_t)*epsSize, cudaMemcpyHostToDevice);
	
	auto fourier = std::make_shared<acacia::gpu::em::CudaFourierSolver2D>(cusolverH_, 
		xSize, nCy, x__, y__);
	sampler_ = std::make_unique<acacia::gpu::em::CudaLayerSampler>(cublasH_, cusolverH_, fourier);

	size_t wSize = sampler_->workspace_buffer_size(nx_, ny_);
	if (workspace__ == nullptr || wSize_ < wSize) {
		if (workspace__ != nullptr) { cudaFree(workspace__); }
		cudaMalloc(reinterpret_cast<void**>(&workspace__), sizeof(acacia::gpu::complex_t)*wSize);
		wSize_ = wSize;
	}

	size_t iSize = sampler_->iworkspace_buffer_size(nx_, ny_);
	if (iworkspace__ == nullptr || iSize_ < iSize) {
		if (iworkspace__ != nullptr) { cudaFree(iworkspace__); }
		cudaMalloc(reinterpret_cast<void**>(&iworkspace__), sizeof(int)*iSize);
		iSize_ = iSize;
	}

	sampler_->sample(eps__, Kx__, Ky__, lambda_, nx_, ny_, 
		P00__, P01__, P10__, P11__,
		Q00__, Q01__, Q10__, Q11__,
		workspace__, iworkspace__);
	
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
}


void GDSIISamplerGPU::assignSimulationRegion() {
	// assume only one cell
	// TODO find max nCx
	gdstk::Cell *pcell = lib_.cell_array[0];

	minX_ = numeric_limits<scalar>::infinity();
	maxX_ = -numeric_limits<scalar>::infinity();
	minZ_ = numeric_limits<scalar>::infinity();
	maxZ_ = -numeric_limits<scalar>::infinity();

	for (int p = 0; p < pcell->polygon_array.size; ++p) {
			gdstk::Polygon *ppoly = pcell->polygon_array[p];
			gdstk::Array<gdstk::Vec2> point_array = ppoly->point_array;
			
			for (int i = 0; i < point_array.size; ++i) {
					const gdstk::Vec2 &p = point_array[i];
					scalar x = static_cast<scalar>(p.x);
					scalar y = static_cast<scalar>(p.y);
					minX_ = min(minX_, x);
					maxX_ = max(maxX_, x);
					minZ_ = min(minZ_, y);
					maxZ_ = max(maxZ_, y);
			}
	}

	minX_ += shift_.x();
	maxX_ += shift_.x();
	minZ_ += shift_.z();
	maxZ_ += shift_.z();

	b0_ = shift_.y();
	b1_ = b0_ + bPML_;
	u0_ = b1_ + bspace_ + thickness_ + uspace_;
	u1_ = u0_ + uPML_;

	l1_ = minX_ - lspace_;
	l0_ = l1_ - lPML_;
	r0_ = maxX_ + rspace_;
	r1_ = r0_ + rPML_;

	Lx_ = r1_ - l0_;
	Ly_ = u1_ - b0_;	
}