#pragma once
#include "rcwa/BlockToeplitzMatrixXcs.h"
#include "gpu/types.cuh"

class LayerSampler
{
protected:
    Eigen::MatrixXcs Kx_;
    Eigen::MatrixXcs Ky_;

    scalar lambda_;
    int nx_;
    int ny_;

    scalar Lx_; // the spatial period along x direction
    scalar Ly_; // the spatial period along y direction

    // PML boundaries
    scalar b0_; // bottom of simulation region
    scalar b1_; // top of bottom PML region
    scalar u0_;
    scalar u1_;

    scalar l0_;
    scalar l1_;
    scalar r0_;
    scalar r1_;

    bool enablePML_; // if true, aperiodic boundary (PML); if false, periodic boundary
public:
    LayerSampler(scalar lambda, 
        int nx, 
        int ny,
        bool enablePML = true):
        lambda_(lambda),
        nx_(nx), ny_(ny),
        enablePML_(enablePML) {}
    virtual ~LayerSampler() {}
    
    scalar lambda() const { return lambda_; }
    scalar k0() const { return 2 * Pi / lambda_; }
    scalar Lx() const { return Lx_; }
    scalar Ly() const { return Ly_; }
    int nDim() const { return 2 * nx_ * ny_; }

    const Eigen::MatrixXcs& Kx() const { return Kx_; }
    const Eigen::MatrixXcs& Ky() const { return Ky_; }
    
    virtual void sample(scalar z, 
        Eigen::MatrixXcs& P,
        Eigen::MatrixXcs& Q) = 0;
        
    virtual void sampleOnGPU(scalar z,
        acacia::gpu::complex_t *P,
        acacia::gpu::complex_t *Q) = 0;
protected:
    // put this at the end of children's constructor
    void postInitialization();

    // assign Lx, Ly, b0, b1, u0, u1, l0, l1, r0, r1 here
    virtual void assignSimulationRegion() = 0;

    void evaluateAlphaBeta(Eigen::DiagonalMatrixXcs& alpha,
		Eigen::DiagonalMatrixXcs& beta);
        
    void evaluatePML(
		scalex gamma,
		BlockToeplitzMatrixXcs& Fx,
		BlockToeplitzMatrixXcs& Fy,
        scalar b0, scalar b1, scalar u0, scalar u1,
        scalar l0, scalar l1, scalar r0, scalar r1,
        scalar Lx, scalar Ly);

    void evaluateKMatrices(scalar b0, scalar b1, scalar u0, scalar u1,
        scalar l0, scalar l1, scalar r0, scalar r1,
        scalar Lx, scalar Ly);
};