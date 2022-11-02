#include "GDSIISampler.h"

#include <string>
#include <limits>
#include <iostream>

#include "rcwa/Layer.h"
#include "utils/timer.hpp"
// #include <cuda_runtime_api.h>

constexpr scalar MICRON = 1e-6;
constexpr scalar tolerance = 1e-12;

using namespace std;
using namespace Eigen;

GDSIISampler::GDSIISampler(
    scalar lambda, 
    int nx, 
    int ny,
    const std::string& filename,
    const Eigen::Vector3s& shift, // x, y, z
    scalar lPML, scalar rPML, scalar lspace, scalar rspace,
    scalar bPML, scalar uPML, scalar bspace, scalar uspace,
    const Material &permBackground, const Material &permDevice,
    scalar thickness):
    LayerSampler(lambda, nx, ny), // assume always enable PML
    shift_(shift),
    lPML_(lPML), rPML_(rPML), lspace_(lspace), rspace_(rspace),
    bPML_(bPML), uPML_(uPML), bspace_(bspace), uspace_(uspace),
    permBackground_(permBackground), permDevice_(permDevice),
    thickness_(thickness)
{
    lib_ = gdstk::read_gds(filename.c_str(), MICRON);
    postInitialization();

    y_.resize(6);
    y_(0) = b0_;
    y_(1) = b1_;
    y_(2) = b1_ + bspace_;
    y_(3) = y_(2) + thickness_;
    y_(4) = u0_;
    y_(5) = u1_;

    yIndicator_.resize(5);
    yIndicator_ << 0, 0, 1, 0, 0;    
}


void GDSIISampler::assignSimulationRegion()
{
    // assume only one cell
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

void GDSIISampler::sampleLayout(scalar z,
    Eigen::VectorXs& x,
    Eigen::MatrixXcs& eps)
{
    gdstk::Cell *pcell = lib_.cell_array[0];
    vector<scalar> cross_section_xs;
    //cout << "poly num: " << pcell->polygon_array.size << endl;
    for (int n = 0; n < pcell->polygon_array.size; ++n) {
        gdstk::Polygon *ppoly = pcell->polygon_array[n];
        gdstk::Array<gdstk::Vec2> point_array = ppoly->point_array;

        for (int i = 0; i < point_array.size; ++i) {
            const gdstk::Vec2 &p0 = point_array[i];
            const gdstk::Vec2 &p1 = 
                (i == point_array.size - 1) ? point_array[0] : point_array[i+1];
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
    

    eps = MatrixXcs::Constant(y_.size()-1, Nx-1, 
        permBackground_.permittivity(lambda_));

    for (int i = 0; i < y_.size()-1; ++i) {
        if (yIndicator_(i) == 1) {
            for (int j = 2; j <= Nx-4; j+=2) {
                eps(i, j) = permDevice_.permittivity(lambda_);
            }
        }
    } 
}

void GDSIISampler::sample(scalar z, 
    Eigen::MatrixXcs& P,
    Eigen::MatrixXcs& Q)
{
    // auto t1 = sploosh::now();
    VectorXs x;
    MatrixXcs eps;
    sampleLayout(z, x, eps);
    // cout << x << endl;
    // auto t2 = sploosh::now();

    Layer layer(x, y_, eps);
    layer.waveEqnCoeff(Kx_, Ky_, lambda_, nx_, ny_, P, Q);

    // auto t3 = sploosh::now();
    // cout << "time: " << sploosh::duration_milli_d(t1, t2) << "\t"
    //     << sploosh::duration_milli_d(t2, t3) << endl;

}


// void GDSIISampler::sampleOnGPU(scalar z,
//     acacia::gpu::complex_t *P,
//     acacia::gpu::complex_t *Q) {
//     int edgeSize = nx_*ny_;
//     int matSize = edgeSize*edgeSize;
     
//     MatrixXcs hP(2*edgeSize, 2*edgeSize);
//     MatrixXcs hQ(2*edgeSize, 2*edgeSize);
//     sample(z, hP, hQ);
//     cudaMemcpy(P, hP.data(), 4*matSize*sizeof(acacia::gpu::complex_t), cudaMemcpyHostToDevice);
//     cudaMemcpy(Q, hQ.data(), 4*matSize*sizeof(acacia::gpu::complex_t), cudaMemcpyHostToDevice);
    
// }