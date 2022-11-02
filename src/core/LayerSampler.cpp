#include "LayerSampler.h"
#include "rcwa/PMLSolver1D.h"
#include "rcwa/MathFunction.h"
#include <iostream>

using namespace std;
using namespace Eigen;

void LayerSampler::evaluateAlphaBeta(Eigen::DiagonalMatrixXcs& alpha,
    Eigen::DiagonalMatrixXcs& beta)
{
	alpha.resize(nx_ * ny_);
	beta.resize(nx_ * ny_);

	int Nx = (nx_ - 1) / 2;
	int Ny = (ny_ - 1) / 2;

	scalar Kx = 2 * Pi / Lx_;
	scalar Ky = 2 * Pi / Ly_;

	for (int i = 0;  i < nx_; ++i)
	{
		for (int j = 0; j < ny_; ++j)
		{
			scalar m = i - Nx;
			scalar n = j - Ny;

			alpha.diagonal()(ny_ * i + j) = Kx * m;
			beta.diagonal()(ny_ * i + j) = Ky * n;
		}
	}    
}

void LayerSampler::evaluatePML(
    scalex gamma,
    BlockToeplitzMatrixXcs& Fx,
    BlockToeplitzMatrixXcs& Fy,
    scalar b0, scalar b1, scalar u0, scalar u1,
    scalar l0, scalar l1, scalar r0, scalar r1,
    scalar Lx, scalar Ly)
{
	int nPx = 2 * nx_ - 1;
	int nPy = 2 * ny_ - 1;
	int maxPx = (nPx - 1) / 2;
	int maxPy = (nPy - 1) / 2;

	PMLSolver1D bottom(b0, b1, Ly, gamma);
	PMLSolver1D top(u0, u1, Ly, gamma);
	VectorXcs fyy(nPy);

	for (int i = 0; i < nPy; ++i)
	{
		int n = i - maxPy;

		fyy(i) = (u0 - b1) / Ly * sinc(n*(u0 - b1) / Ly)
		 / exp(scalex(0, Pi) * scalex(n* (b1 + u0) / Ly));
		fyy(i) += bottom.solve(n, 1.);
		fyy(i) += top.solve(n, -1.);
	}

	PMLSolver1D left(l0, l1, Lx, gamma);
	PMLSolver1D right(r0, r1, Lx, gamma);
	VectorXcs fxx(nPx);

	for (int i = 0; i < nPx; ++i)
	{
		int m = i - maxPx;
		fxx(i) = (r0 - l1) * sinc(m*(r0 - l1) / Lx) / Lx
		 / exp(scalex(0, Pi * m * (l1 + r0) / Lx));
		fxx(i) += left.solve(m, 1.);
		fxx(i) += right.solve(m, -1.);
	}
	

	VectorXcs fyx(nPx);
	//scalar Kxm = 2 * Pi / Lx_;
	for (int i = 0; i < nPx; ++i)
	{
		int m = i - maxPx;
		if (m == 0)
		{
			fyx(i) = r1 - l0;
		}
		else
		{
			fyx(i) = (exp(scalex(0., -2 * Pi / Lx * m * r1))- exp(scalex(0., -2 * Pi / Lx * m * l0)))
			* scalex(0., Lx * Lx / (m * 2 * Pi));
		}
	}

	VectorXcs fxy(nPy);
	for (int i = 0; i < nPy; ++i)
	{
		int n = i - maxPy;
		if (n == 0)
		{
			fxy(i) = u1 - b0;
		}
		else
		{
			fxy(i) = (exp(scalex(0., -2 * Pi / Ly * n * u1)) - exp(scalex(0., - 2 * Pi / Ly * n * b0)))
			* scalex(0., Ly * Ly / (2 * n * Pi));
		}
	}

	MatrixXcs fy = 1. / Lx * fyx * fyy.transpose();
	MatrixXcs fx = 1. / Ly * fxx * fxy.transpose();

	Fx = BlockToeplitzMatrixXcs(fx);
	Fy = BlockToeplitzMatrixXcs(fy);    
}


void LayerSampler::evaluateKMatrices(
    scalar b0, scalar b1, scalar u0, scalar u1,
    scalar l0, scalar l1, scalar r0, scalar r1,
    scalar Lx, scalar Ly)
{
	DiagonalMatrixXcs alpha;
	DiagonalMatrixXcs beta;
	evaluateAlphaBeta(alpha, beta);

  if (enablePML_) {
    std::cerr << "use PML boundary condition!\n";
    BlockToeplitzMatrixXcs Fx(nx_, ny_), Fy(nx_, ny_);
    scalex gamma(1., -1.);
    gamma = scalex(1.) / gamma;
    evaluatePML(gamma, Fx, Fy, b0, b1, u0, u1,
          l0, l1, r0, r1, Lx, Ly);

    Kx_ = Fx * alpha;
    Ky_ = Fy * beta;
  } else {
    std::cerr << "use periodic boundary condition!\n";
    Kx_.resize(nx_*ny_, nx_*ny_);
    Ky_.resize(nx_*ny_, nx_*ny_);
    Kx_.setZero();
    Ky_.setZero();

    Kx_.diagonal() = alpha.diagonal();
    Ky_.diagonal() = beta.diagonal();
    std::cerr << "Kx max: " << Kx_.cwiseAbs().maxCoeff() << endl;
  }
}


void LayerSampler::postInitialization()
{
    assignSimulationRegion();
    evaluateKMatrices(b0_, b1_, u0_, u1_,
        l0_, l1_, r0_, r1_,
        Lx_, Ly_);
}