#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <utility>

#include <Eigen/Dense>

#include "RCWASolver.h"
#include "Layer.h"
#include "PMLSolver1D.h"
#include "MathFunction.h"
#include "BlockToeplitzMatrixXcs.h"

#include <fstream>

using namespace std;
using namespace Eigen;


RCWASolver::~RCWASolver()
{
}

RCWASolver::RCWASolver(const RCWASolver& solver)
{
	nx_ = solver.nx_;
	ny_ = solver.ny_;

	Lx_ = solver.Lx_;
	Ly_ = solver.Ly_;

	l0_ = solver.l0_;
	l1_ = solver.l1_;
	r0_ = solver.r0_;
	r1_ = solver.r1_;

	b0_ = solver.b0_;
	b1_ = solver.b1_;
	u0_ = solver.u0_;
	u1_ = solver.u1_;

	Kx_ = solver.Kx_;
	Ky_ = solver.Ky_;

	matKx_ = solver.matKx_;
	matKy_ = solver.matKy_;

	enablePML_ = solver.enablePML_;
	
	for (auto layer : solver.layers_)
	{
		// should make a deep copy of Layer
		layers_.push_back(std::make_shared<Layer>(*layer));
	}

	isLayerSolved_ = solver.isLayerSolved_;

	interfacesE_ = solver.interfacesE_;
	interfacesH_ = solver.interfacesH_;
	interfaceMap_ = solver.interfaceMap_;
	isInterfaceSolved_ = solver.isInterfaceSolved_;

	taus_ = solver.taus_;


	Tdd_ = solver.Tdd_;
	Rud_ = solver.Rud_;
	isScatterMatrixSolved_ = solver.isScatterMatrixSolved_;	
}


void RCWASolver::addLayer(const Layer& layer)
{
	if (layers_.empty())
	{
		Lx_ = layer.Lx();
		Ly_ = layer.Ly();

		b0_ = layer.b0();
		b1_ = layer.b1();
		u0_ = layer.u0();
		u1_ = layer.u1();

		l0_ = layer.l0();
		l1_ = layer.l1();
		r0_ = layer.r0();
		r1_ = layer.r1();
	} 
	else if (Lx_ != layer.Lx() || Ly_ != layer.Ly() ||
		b0_ != layer.b0() || b1_ != layer.b1() ||
		u0_ != layer.u0() || u1_ != layer.u1() ||
		l0_ != layer.l0() || l1_ != layer.l1() ||
		r0_ != layer.r0() || r1_ != layer.r1()) 
	{
		std::cerr << "The new layer does not match layers before!\n";
		return;
	}
	
	layers_.push_back(std::make_shared<Layer>(layer));

}

void RCWASolver::solve(
	scalar lambda,
	const std::vector<int>& layerStack,
	const std::vector<scalar>& thickness,
	const std::vector<scalar>* translation)
{
	solveLayers(lambda);
	solveInterfaces(layerStack, translation);
	solveScatterMatrix(layerStack, thickness);
}

void RCWASolver::evaluateKMatrices(
	int nx, int ny)
{
	DiagonalMatrixXcs alpha;
	DiagonalMatrixXcs beta;

	evaluateAlphaBeta(alpha, beta);

	int nDim = nx * ny;
	MatrixXcs Kx(nDim, nDim), Ky(nDim, nDim);
	Kx.setZero();
	Ky.setZero();


	if (enablePML_)
	{
		BlockToeplitzMatrixXcs Fx(nx, ny), Fy(nx, ny);
		scalex gamma(1., -1.);
		gamma = scalex(1.) / gamma;
		evaluatePML(gamma, Fx, Fy);

		Kx = Fx * alpha;
		Ky = Fy * beta;
	}
	else
	{
		Kx.diagonal() = alpha.diagonal();
		Ky.diagonal() = beta.diagonal();
		Kx_ = Kx.diagonal();
		Ky_ = Ky.diagonal();
	}


	matKx_ = Kx;
	matKy_ = Ky;	
}

void RCWASolver::evaluateKMatrices(MatrixXcs& Kx, MatrixXcs& Ky)
{
    evaluateKMatrices(nx_, ny_);
    Kx = matKx_;
    Ky = matKy_;
}

void RCWASolver::solveLayers(scalar lambda)
{
	evaluateKMatrices(nx_, ny_);

	if (layers_.empty())
	{
		std::cerr << "The layers are empty!\n";
		return;
	}

	for (auto layer : layers_)
	{
		layer->solve(matKx_, matKy_, lambda, nx_, ny_);	
	}

	isLayerSolved_ = true;
}

void RCWASolver::solveInterfaces(const std::vector<int>& layerStack,
	const std::vector<scalar>* translation)
{
	if (!isLayerSolved_)
	{
		std::cerr << "The layers have not been solved!\n";
		return;
	}

	// compute interface matrix
	std::clog << "compute interface matrix...";
	evaluateInterfaces(interfacesE_, interfacesH_, interfaceMap_, layerStack,
		translation);
	std::clog << "done!\n";
}


void RCWASolver::solveScatterMatrix(
	const std::vector<int>& layerStack,
	const std::vector<scalar>& thickness)
{
	if (!isInterfaceSolved_)
	{
		std::cerr << "The interfaces have not been solved!\n";
		return;
	}

	evaluateScatterMatrix(Tdd_, Rud_, layerStack, thickness);
}

void RCWASolver::evaluateAlphaBeta(
	Eigen::DiagonalMatrixXcs& alpha,
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

void RCWASolver::evaluatePML(
	scalex gamma,
	BlockToeplitzMatrixXcs& Fx,
	BlockToeplitzMatrixXcs& Fy)
{
	int nPx = 2 * nx_ - 1;
	int nPy = 2 * ny_ - 1;
	int maxPx = (nPx - 1) / 2;
	int maxPy = (nPy - 1) / 2;

	PMLSolver1D bottom(b0_, b1_, Ly_, gamma);
	PMLSolver1D top(u0_, u1_, Ly_, gamma);
	VectorXcs fyy(nPy);

	for (int i = 0; i < nPy; ++i)
	{
		int n = i - maxPy;

		fyy(i) = (u0_ - b1_) / Ly_ * sinc(n*(u0_ - b1_) / Ly_)
		 / exp(scalex(0, Pi) * scalex(n* (b1_ + u0_) / Ly_));
		fyy(i) += bottom.solve(n, 1.);
		fyy(i) += top.solve(n, -1.);
	}

	PMLSolver1D left(l0_, l1_, Lx_, gamma);
	PMLSolver1D right(r0_, r1_, Lx_, gamma);
	VectorXcs fxx(nPx);

	for (int i = 0; i < nPx; ++i)
	{
		int m = i - maxPx;
		fxx(i) = (r0_ - l1_) * sinc(m*(r0_ - l1_) / Lx_) / Lx_
		 / exp(scalex(0, Pi * m * (l1_ + r0_) / Lx_));
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
			fyx(i) = r1_ - l0_;
		}
		else
		{
			fyx(i) = (exp(scalex(0., -2 * Pi / Lx_ * m * r1_))- exp(scalex(0., -2 * Pi / Lx_ * m * l0_)))
			* scalex(0., Lx_ * Lx_ / (m * 2 * Pi));
		}
	}

	VectorXcs fxy(nPy);
	for (int i = 0; i < nPy; ++i)
	{
		int n = i - maxPy;
		if (n == 0)
		{
			fxy(i) = u1_ - b0_;
		}
		else
		{
			fxy(i) = (exp(scalex(0., -2 * Pi / Ly_ * n * u1_)) - exp(scalex(0., - 2 * Pi / Ly_ * n * b0_)))
			* scalex(0., Ly_ * Ly_ / (2 * n * Pi));
		}
	}

	MatrixXcs fy = 1. / Lx_ * fyx * fyy.transpose();
	MatrixXcs fx = 1. / Ly_ * fxx * fxy.transpose();

	Fx = BlockToeplitzMatrixXcs(fx);
	Fy = BlockToeplitzMatrixXcs(fy);
}

void RCWASolver::findGuidedModes(
	std::vector < std::pair<int, scalex> >& guidedModeRecords,
	int layerType,
	scalar k0,
	scalar imagCoeff,
	scalar realCoeffMin,
	scalar realCoeffMax)
{
	if (!isLayerSolved_)
	{
		std::cerr << "The layers have not been solved!\n";
		return;
	}

	const Eigen::VectorXcs& gamma = layers_[layerType]->gamma();
	
	for (int i = 0; i < gamma.size(); ++i)
	{
		scalex gamma_i = gamma(i) / k0;
		if (realCoeffMin <= gamma_i.real() &&
			gamma_i.real() <= realCoeffMax &&
			gamma_i.imag() < imagCoeff)
		{
			guidedModeRecords.push_back(std::pair<int, scalex>(i, gamma_i));
		}
	}

	std::sort(guidedModeRecords.begin(), guidedModeRecords.end(),
		[](const std::pair<int, scalex>& a, const std::pair<int, scalex>& b){
			return a.second.real() > b.second.real();
		});
}


void RCWASolver::evaluateScatterMatrix(
	Eigen::MatrixXcs& Tdd,
	Eigen::MatrixXcs& Rud,
	const std::vector<int>& layerStack,
	const std::vector<scalar>& thickness)
{
	int nDim = 2 * nx_ * ny_;
	int nL = layerStack.size();

	Tdd.resize(nDim, nDim);
	Rud.resize(nDim, nDim);

	Tdd.setIdentity();
	Rud.setZero();

	taus_.resize(0);
	// the following matrices are only used for debugging
	// MatrixXcs Rdu(nDim, nDim);
	// Rdu.setZero();
	// MatrixXcs Tuu(nDim, nDim);
	// Tuu.setIdentity();


	// RTCM algorithm
	MatrixXcs waveRud(nDim, nDim);
	MatrixXcs waveTdd(nDim, nDim);
	VectorXcs phiMinus(nDim);

	// note that the for loop starts from the last layer
	// and ends at the first layer

	for (int i = 0; i < nL-1; ++i)
	{
		int index = nL - i - 1;
		int layerIndex = layerStack[index];
		scalar thick = thickness[index];

		if (i == 0)
		{
			phiMinus.setOnes();
		}
		else
		{
			for (int j = 0; j < nDim; ++j)
			{
				phiMinus(j) = exp(scalex(0, 1)* layers_[layerIndex]->gamma()(j) * thick);
			}

			// clog << "layer index: " << layerIndex << endl;
			// clog << "gamma: " << layers_[layerIndex]->gamma()(1200) << endl;
			// clog << "thick: " << thick << endl;

			// clog << "phiMinus: " << phiMinus(1200) << endl;	
		}

		waveRud = phiMinus.asDiagonal() * Rud * phiMinus.asDiagonal();
		waveTdd = Tdd * phiMinus.asDiagonal();

		// debug
		// MatrixXcs waveTuu = phiMinus.asDiagonal() * Tuu;

		int interfaceIndex = interfaceMap_[index-1];
		MatrixXcs F = interfacesE_[interfaceIndex]
		+ interfacesE_[interfaceIndex] * waveRud;
		MatrixXcs G = interfacesH_[interfaceIndex]
		- interfacesH_[interfaceIndex] * waveRud;

		// clog << "interface index: " << interfaceIndex << endl;
		// clog << "interface E: " << interfacesE_[interfaceIndex](0, 0) << endl;
		// clog << "interface H: " << interfacesH_[interfaceIndex](0, 0) << endl;


		std::clog << "interface " << i << ", begin to solve tau...";
		PartialPivLU<MatrixXcs> solverTau(F + G);
		// FullPivLU<MatrixXcs> solverTau(F + G);
		MatrixXcs tau = solverTau.inverse();
		std::clog << "done!\n";
		
		taus_.push_back(tau);
		Rud = - 2. * G * tau;
		Rud.diagonal().array() += scalar(1.0);

		Tdd = 2. * waveTdd * tau;

		// clog << "waveTdd: " << waveTdd(240, 1200) << endl;
		// clog << "tau: " << tau(1200, 1235) << endl;

		// clog << "Tdd in RCWASolver: " << Tdd(240, 1235) << endl;

		// Tuu = (F * tau * interfacesH_[interfaceIndex] 
		// 	 + G * tau * interfacesE_[interfaceIndex]) * waveTuu;

		// Rdu = Rdu + waveTdd * tau * (interfacesH_[interfaceIndex]
		// 	- interfacesE_[interfaceIndex]) * waveTuu;
	}

	// const VectorXcs& gamma = layers_[0]->gamma();
	// VectorXcs phi(nDim);

	// for (int i = 0; i < nDim; ++i)
	// {
	// 	phi(i) = exp(scalex(0, 1) * gamma(i) * thickness[0]);
	// }

	// Tuu = phi.asDiagonal() * Tuu * phi.asDiagonal();
	// Rdu = phi.asDiagonal() * Rdu * phi.asDiagonal();

	// ofstream fout1("Tuu.txt");
	// fout1 << Tuu << endl;

	// ofstream fout2("Rdu.txt");
	// fout2 << Rdu << endl;
	
	isScatterMatrixSolved_ = true;
}


void RCWASolver::evaluateInterfaces(
	std::vector<Eigen::MatrixXcs>& interfacesE,
	std::vector<Eigen::MatrixXcs>& interfacesH,
	std::vector<int>& interfaceMap,
	const std::vector <int>& layerStack,
	const std::vector <scalar>* translation)
{
	interfacesE.resize(0);
	interfacesH.resize(0);
	
	int nDim = 2*nx_*ny_;
	int nL = layerStack.size();

	if (translation == nullptr)
	{
		std::set < std::pair<int, int> > layerPairs;

		for (int i = 0; i < nL - 1; ++i)
		{
			int layerIndex1 = layerStack[i];
			int layerIndex2 = layerStack[i+1];

			const std::pair<int, int>& pair = std::make_pair(layerIndex1, layerIndex2);
			layerPairs.insert(pair);
		}

		MatrixXcs interfaceE(nDim, nDim);
		MatrixXcs interfaceH(nDim, nDim);

		for (auto pair : layerPairs)
		{
			int i1 = pair.first;
			int i2 = pair.second;

			if (i1 == i2)
			{
				interfaceE.setIdentity();
				interfaceH.setIdentity();

			}
			else
			{
				CompleteOrthogonalDecomposition<MatrixXcs> solverE(
					layers_[i1]->eigvecE());
				interfaceE = solverE.solve(layers_[i2]->eigvecE());

				CompleteOrthogonalDecomposition<MatrixXcs> solverH(
					layers_[i1]->eigvecH());
				interfaceH = solverH.solve(layers_[i2]->eigvecH());
			}

			interfacesE.push_back(interfaceE);
			interfacesH.push_back(interfaceH);
		}


		interfaceMap.resize(nL-1);

		for (int i = 0; i < nL - 1; ++i)
		{
			int layerIndex1 = layerStack[i];
			int layerIndex2 = layerStack[i+1];

			int count = 0;
			for (auto pair : layerPairs)
			{
				if (pair.first == layerIndex1 &&
					pair.second == layerIndex2)
				{
					interfaceMap[i] = count;
				}

				count++;
			}
		}

	}
	else
	{
		if (translation->size() != nL)
		{
			cerr << "[ERROR]: The size of translation vector is not the same as the segment size!\n";
			return;
		}

		for (int i = 0; i < nL - 1; ++i)
		{
			int i1 = layerStack[i];
			int i2 = layerStack[i+1];
			scalar t1 = (*translation)[i];
			scalar t2 = (*translation)[i+1];

			MatrixXcs interfaceE(nDim, nDim);
			MatrixXcs interfaceH(nDim, nDim);

			MatrixXcs eigvecE0, eigvecH0, eigvecE1, eigvecH1;

			layers_[i1]->permuteEigVecX(eigvecE0, eigvecH0, t1, nx_, ny_);
			layers_[i2]->permuteEigVecX(eigvecE1, eigvecH1, t2, nx_, ny_);

			CompleteOrthogonalDecomposition<MatrixXcs> solverE(eigvecE0);
			interfaceE = solverE.solve(eigvecE1);

			CompleteOrthogonalDecomposition<MatrixXcs> solverH(eigvecH0);
			interfaceH = solverH.solve(eigvecH1);

			interfacesE.push_back(interfaceE);
			interfacesH.push_back(interfaceH);
		}

		interfaceMap.resize(nL-1);

		for (int i = 0; i < nL - 1; ++i)
		{
			interfaceMap[i] = i;
		}		
	}
	
	isInterfaceSolved_ = true;
}


scalar RCWASolver::transmittance(
	int inputMode,
	int outputMode) const
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The scatter matrix has not been solved!\n";
		return 1.0;
	}

	scalex tdd = Tdd_(outputMode, inputMode);
	return std::abs(tdd) * std::abs(tdd);
}

scalex RCWASolver::rud(int row, int col) const
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The scatter matrix has not been solved!\n";
		return scalex(0.);
	}

	return Rud_(row, col);
}

scalex RCWASolver::tdd(int row, int col) const
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The scatter matrix has not been solved!\n";
		return scalex(1.);
	}
	
	return Tdd_(row, col);
}

scalar RCWASolver::reflectance(
	int inputMode,
	int outputMode) const
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The scatter matrix has not been solved!\n";
		return 1.0;
	}

	scalex rud = Rud_(outputMode, inputMode);
	return std::abs(rud) * std::abs(rud);
}

scalar RCWASolver::transmittance(
	int inputMode) const
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The scatter matrix has not been solved!\n";
		return 1.0;
	}

	const VectorXcs& tdd = Tdd_.col(inputMode);
	scalar transmittance = 0;

	for (int i = 0; i < tdd.size(); ++i)
	{
		transmittance += std::abs(tdd(i)) * std::abs(tdd(i));
	}

	return transmittance;
}

scalar RCWASolver::reflectance(
	int inputMode) const
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The scatter matrix has not been solved!\n";
		return 1.0;
	}

	const VectorXcs& rud = Rud_.col(inputMode);
	scalar reflectance = 0;

	for (int i = 0; i < rud.size(); ++i)
	{
		reflectance += std::abs(rud(i)) * std::abs(rud(i));
	}

	return reflectance;
}

void RCWASolver::saveFieldImage(
	const std::string& filename,
	SliceType sliceType,
	scalar sliceCoord,
	FieldComponent fieldComponent,
	SaveOption opt,
	const std::vector<std::pair<int, scalex>>& inputCoeffs,
	const std::vector<int>& layerStack,
	const std::vector<scalar>& thickness,
	const std::vector<scalar>* translation)
{
	int nDim = 2 * nx_ * ny_;

	std::vector< VectorXcs > c_m;
	std::vector< VectorXcs > c_p;
	evaluateIntermediateField(c_m, c_p, inputCoeffs, layerStack, thickness);

	scalar startCoord1 = 0.;
	scalar L1 = 0.;
	scalar startCoord2 = 0.;
	scalar L2 = 0.;

	if (sliceType == sliceXZ)
	{
		startCoord1 = layers_[0]->l0();
		L1 = layers_[0]->Lx();
		startCoord2 = .0;
		L2 = .0;
		for (int i = 1; i < layerStack.size()-1; ++i)
		{
			L2 += thickness[i];
		}
	}
	else if (sliceType == sliceYZ)
	{
		startCoord1 = layers_[0]->b0();
		L1 = layers_[0]->Ly();
		startCoord2 = .0;
		L2 = .0;
		for (int i = 1; i < layerStack.size()-1; ++i)
		{
			L2 += thickness[i];
		}		
	}
	else if (sliceType == sliceXY)
	{
		startCoord1 = layers_[0]->l0();
		L1 = layers_[0]->Lx();
		startCoord2 = layers_[0]->b0();
		L2 = layers_[0]->Ly();
	}

	int nRes1 = 500;
	int nRes2 = static_cast<int>( nRes1 * L2 / L1);

	scalar step1 = L1 / nRes1;
	scalar step2 = L2 / nRes2;
	
	VectorXs sample1(nRes1);
	for (int i = 0; i < nRes1; ++i)
	{
		sample1(i) = startCoord1 + step1 * (i + 0.5); 
	}

	VectorXs sample2(nRes2);
	for (int i = 0; i < nRes2; ++i)
	{
		sample2(i) = startCoord2 + step2 * (i + 0.5);
	}

	auto SET_HARMONICS_2D = [&](scalar z, MatrixXcs& harmonics2d,
		int& layerType)
	{
		scalar z0 = .0;
		scalar z1 = .0;

		int layer0 = -1;
		int layer1 = -1;

		for (int j = 1; j < layerStack.size()-1; ++j)
		{
			z1 = z0 + thickness[j];
			if (z0 <= z && z <= z1)
			{
				layer0 = j-1;
				layer1 = j;
				break;
			}
			z0 = z1;
		}

		scalar z_p = z - z0;
		scalar z_m = z - z1;
		
		layerType = layerStack[layer1];

		scalar tx = 0.;
		if (translation != nullptr)
		{
			tx = (*translation)[layer1];
		}
		
		const VectorXcs& gamma = layers_[layerType]->gamma();

		VectorXcs phaseForward(nDim);
		VectorXcs phaseBackward(nDim);

		for (int j = 0; j < nDim; ++j)
		{
			phaseForward(j) = exp(scalex(0, 1) * z_p * gamma(j));
			phaseBackward(j) = exp(-scalex(0, 1) * z_m * gamma(j));
		}

		VectorXcs u_p = c_p[layer0].cwiseProduct(phaseForward);
		VectorXcs d_m = c_m[layer0].cwiseProduct(phaseBackward);

		VectorXcs harmonics1D;

		MatrixXcs eigvecE, eigvecH;
		layers_[layerType]->permuteEigVecX(eigvecE, eigvecH, tx, nx_, ny_);

		if (fieldComponent == Ex || fieldComponent == Ey)
		{

			//const MatrixXcs& eigvecE = layers_[layerType]->eigvecE();
			harmonics1D = eigvecE * (u_p + d_m);
		}
		else if (fieldComponent == Hx || fieldComponent == Hy)
		{
			// const MatrixXcs& eigvecH = layers_[layerType]->eigvecH();
			harmonics1D = eigvecH * (u_p - d_m);
		}
		else
		{
			std::cerr << "Evaluation for this component is not implemented!\n";
			return;
		}

		for (int i = 0; i < nx_; ++i)
		{
			for (int j = 0; j < ny_; ++j)
			{
				if (fieldComponent == Ex || 
					fieldComponent == Hx)
				{
					harmonics2d(i, j) = harmonics1D(j + i * ny_);
				}
				else if (fieldComponent == Ey ||
						 fieldComponent == Hy)
				{
					harmonics2d(i, j) = harmonics1D(j + i * ny_ + nx_ * ny_);
				}
				else 
				{
					std::cerr << "Evaluation for this component is not implemented!\n";
					return;					
				}
				
			}
		}
	};

	MatrixXcs field(nRes2, nRes1);
	if (sliceType == sliceXZ || sliceType == sliceYZ)
	{
		for (int i = 0; i < nRes2; ++i)
		{
			scalar z = sample2(i);
			MatrixXcs harmonics2D(nx_, ny_);
			int layerType;
			SET_HARMONICS_2D(z, harmonics2D, layerType);

			VectorXcs oneDirHarmonics;

			if (sliceType == sliceXZ)
			{
				VectorXcs waveletY(ny_);
				int maxY = (ny_ - 1) / 2;
				for (int n = -maxY; n <= maxY; ++n)
				{
					waveletY(n+maxY) 
					= exp(scalex(0, 2.* Pi * n / layers_[layerType]->Ly() * sliceCoord));
				}

				oneDirHarmonics = harmonics2D * waveletY;
			}
			else if (sliceType == sliceYZ)
			{
				VectorXcs waveletX(nx_);
				int maxX = (nx_ - 1) / 2;
				for (int m = -maxX; m <= maxX; ++m)
				{
					waveletX(m+maxX) = 
					exp(scalex(0, 2. * Pi * m / layers_[layerType]->Lx() * sliceCoord));
				}

				oneDirHarmonics = harmonics2D * waveletX;
			}		

			for (int j = 0; j < nRes1; ++j)
			{
				auto SET_FIELD = [&](int nh, scalar Lh)
				{
					scalar h = sample1(j);
					VectorXcs waveletH(nh);
					int maxH = (nh - 1) / 2;

					for (int m = -maxH; m <= maxH; ++m)
					{
						waveletH(m + maxH) 
						= exp(scalex(0, 2.* Pi * m / Lh * h));
					}

					field(i, j) = waveletH.dot(oneDirHarmonics);				
				};

				if (sliceType == sliceXZ)
				{
					SET_FIELD(nx_, layers_[layerType]->Lx());			
				}
				else if (sliceType == sliceYZ)
				{
					SET_FIELD(ny_, layers_[layerType]->Ly());
				}
			}
		}		
	}
	else if (sliceType == sliceXY)
	{
		scalar z = sliceCoord;
		MatrixXcs harmonics2D(nx_, ny_);
		int layerType;
		SET_HARMONICS_2D(z, harmonics2D, layerType);

		for (int i = 0; i < nRes2; ++i)
		{
			for (int j = 0; j < nRes1; ++j)
			{
				scalar x = sample1(j);
				scalar y = sample2(i);

				VectorXcs waveletX(nx_);
				int maxX = (nx_ - 1) / 2;
				for (int m = -maxX; m <= maxX; ++m)
				{
					waveletX(m+maxX) = exp(scalex(0, 2. * Pi * m / L1 * x));
				}

				VectorXcs waveletY(ny_);
				int maxY = (ny_ - 1) / 2;
				for (int n = -maxY; n <= maxY; ++n)
				{
					waveletY(n+maxY) = exp(scalex(0, 2. * Pi * n / L2 * y));
				}

				field(i, j) = waveletX.transpose() * harmonics2D * waveletY;
			}
		}
	}

	// MatrixVisualizer vis(field);
	// vis.setXTimes(1);
	// vis.save(filename, realpart);	
}

void RCWASolver::saveFieldImage(const std::string& filename,
	SliceType sliceType,
	scalar sliceCoord,
	FieldComponent fieldComponent,
	SaveOption opt,	
	int inputMode,
	const std::vector<int>& layerStack,
	const std::vector<scalar>& thickness,
	const std::vector<scalar>* translation)
{
	int nDim = 2 * nx_ * ny_;

	std::vector< VectorXcs > c_m;
	std::vector< VectorXcs > c_p;
	evaluateIntermediateField(c_m, c_p, inputMode, layerStack, thickness);

	scalar startCoord1 = 0.;
	scalar L1 = 0.;
	scalar startCoord2 = 0.;
	scalar L2 = 0.;

	if (sliceType == sliceXZ)
	{
		startCoord1 = layers_[0]->l0();
		L1 = layers_[0]->Lx();
		startCoord2 = .0;
		L2 = .0;
		for (int i = 1; i < layerStack.size()-1; ++i)
		{
			L2 += thickness[i];
		}
	}
	else if (sliceType == sliceYZ)
	{
		startCoord1 = layers_[0]->b0();
		L1 = layers_[0]->Ly();
		startCoord2 = .0;
		L2 = .0;
		for (int i = 1; i < layerStack.size()-1; ++i)
		{
			L2 += thickness[i];
		}		
	}
	else if (sliceType == sliceXY)
	{
		startCoord1 = layers_[0]->l0();
		L1 = layers_[0]->Lx();
		startCoord2 = layers_[0]->b0();
		L2 = layers_[0]->Ly();
	}

	int nRes1 = 500;
	int nRes2 = static_cast<int>( nRes1 * L2 / L1);

	scalar step1 = L1 / nRes1;
	scalar step2 = L2 / nRes2;
	
	VectorXs sample1(nRes1);
	for (int i = 0; i < nRes1; ++i)
	{
		sample1(i) = startCoord1 + step1 * (i + 0.5); 
	}

	VectorXs sample2(nRes2);
	for (int i = 0; i < nRes2; ++i)
	{
		sample2(i) = startCoord2 + step2 * (i + 0.5);
	}

	auto SET_HARMONICS_2D = [&](scalar z, MatrixXcs& harmonics2d,
		int& layerType)
	{
		scalar z0 = .0;
		scalar z1 = .0;

		int layer0 = -1;
		int layer1 = -1;

		for (int j = 1; j < layerStack.size()-1; ++j)
		{
			z1 = z0 + thickness[j];
			if (z0 <= z && z <= z1)
			{
				layer0 = j-1;
				layer1 = j;
				break;
			}
			z0 = z1;
		}

		scalar z_p = z - z0;
		scalar z_m = z - z1;
		
		layerType = layerStack[layer1];

		scalar tx = 0.;
		if (translation != nullptr)
		{
			tx = (*translation)[layer1];
		}
		
		const VectorXcs& gamma = layers_[layerType]->gamma();

		VectorXcs phaseForward(nDim);
		VectorXcs phaseBackward(nDim);

		for (int j = 0; j < nDim; ++j)
		{
			phaseForward(j) = exp(scalex(0, 1) * z_p * gamma(j));
			phaseBackward(j) = exp(-scalex(0, 1) * z_m * gamma(j));
		}

		VectorXcs u_p = c_p[layer0].cwiseProduct(phaseForward);
		VectorXcs d_m = c_m[layer0].cwiseProduct(phaseBackward);

		VectorXcs harmonics1D;

		MatrixXcs eigvecE, eigvecH;
		layers_[layerType]->permuteEigVecX(eigvecE, eigvecH, tx, nx_, ny_);

		if (fieldComponent == Ex || fieldComponent == Ey)
		{

			//const MatrixXcs& eigvecE = layers_[layerType]->eigvecE();
			harmonics1D = eigvecE * (u_p + d_m);
		}
		else if (fieldComponent == Hx || fieldComponent == Hy)
		{
			// const MatrixXcs& eigvecH = layers_[layerType]->eigvecH();
			harmonics1D = eigvecH * (u_p - d_m);
		}
		else
		{
			std::cerr << "Evaluation for this component is not implemented!\n";
			return;
		}

		for (int i = 0; i < nx_; ++i)
		{
			for (int j = 0; j < ny_; ++j)
			{
				if (fieldComponent == Ex || 
					fieldComponent == Hx)
				{
					harmonics2d(i, j) = harmonics1D(j + i * ny_);
				}
				else if (fieldComponent == Ey ||
						 fieldComponent == Hy)
				{
					harmonics2d(i, j) = harmonics1D(j + i * ny_ + nx_ * ny_);
				}
				else 
				{
					std::cerr << "Evaluation for this component is not implemented!\n";
					return;					
				}
				
			}
		}
	};

	MatrixXcs field(nRes2, nRes1);
	if (sliceType == sliceXZ || sliceType == sliceYZ)
	{
		for (int i = 0; i < nRes2; ++i)
		{
			scalar z = sample2(i);
			MatrixXcs harmonics2D(nx_, ny_);
			int layerType;
			SET_HARMONICS_2D(z, harmonics2D, layerType);

			VectorXcs oneDirHarmonics;

			if (sliceType == sliceXZ)
			{
				VectorXcs waveletY(ny_);
				int maxY = (ny_ - 1) / 2;
				for (int n = -maxY; n <= maxY; ++n)
				{
					waveletY(n+maxY) 
					= exp(scalex(0, 2.* Pi * n / layers_[layerType]->Ly() * sliceCoord));
				}

				oneDirHarmonics = harmonics2D * waveletY;
			}
			else if (sliceType == sliceYZ)
			{
				VectorXcs waveletX(nx_);
				int maxX = (nx_ - 1) / 2;
				for (int m = -maxX; m <= maxX; ++m)
				{
					waveletX(m+maxX) = 
					exp(scalex(0, 2. * Pi * m / layers_[layerType]->Lx() * sliceCoord));
				}

				oneDirHarmonics = harmonics2D * waveletX;
			}		

			for (int j = 0; j < nRes1; ++j)
			{
				auto SET_FIELD = [&](int nh, scalar Lh)
				{
					scalar h = sample1(j);
					VectorXcs waveletH(nh);
					int maxH = (nh - 1) / 2;

					for (int m = -maxH; m <= maxH; ++m)
					{
						waveletH(m + maxH) 
						= exp(scalex(0, 2.* Pi * m / Lh * h));
					}

					field(i, j) = waveletH.dot(oneDirHarmonics);				
				};

				if (sliceType == sliceXZ)
				{
					SET_FIELD(nx_, layers_[layerType]->Lx());			
				}
				else if (sliceType == sliceYZ)
				{
					SET_FIELD(ny_, layers_[layerType]->Ly());
				}
			}
		}		
	}
	else if (sliceType == sliceXY)
	{
		scalar z = sliceCoord;
		MatrixXcs harmonics2D(nx_, ny_);
		int layerType;
		SET_HARMONICS_2D(z, harmonics2D, layerType);

		for (int i = 0; i < nRes2; ++i)
		{
			for (int j = 0; j < nRes1; ++j)
			{
				scalar x = sample1(j);
				scalar y = sample2(i);

				VectorXcs waveletX(nx_);
				int maxX = (nx_ - 1) / 2;
				for (int m = -maxX; m <= maxX; ++m)
				{
					waveletX(m+maxX) = exp(scalex(0, 2. * Pi * m / L1 * x));
				}

				VectorXcs waveletY(ny_);
				int maxY = (ny_ - 1) / 2;
				for (int n = -maxY; n <= maxY; ++n)
				{
					waveletY(n+maxY) = exp(scalex(0, 2. * Pi * n / L2 * y));
				}

				field(i, j) = waveletX.transpose() * harmonics2D * waveletY;
			}
		}
	}


	// MatrixVisualizer vis(field);
	// vis.setXTimes(1);
	// vis.save(filename, realpart);
}


void RCWASolver::saveDeviceImage(const std::string& filename,
	SliceType sliceType,
	scalar sliceCoord,
	const std::vector<int>& layerStack,
	const std::vector<scalar>& thickness,
	const std::vector<scalar>* translation)
{
	scalar startCoord1 = 0.;
	scalar L1 = 0.;
	scalar startCoord2 = 0.;
	scalar L2 = 0.;

	if (sliceType == sliceXZ)
	{
		startCoord1 = layers_[0]->l0();
		L1 = layers_[0]->Lx();
		startCoord2 = .0;
		L2 = .0;
		for (int i = 1; i < layerStack.size()-1; ++i)
		{
			L2 += thickness[i];
		}
	}
	else if (sliceType == sliceYZ)
	{
		startCoord1 = layers_[0]->b0();
		L1 = layers_[0]->Ly();
		startCoord2 = .0;
		L2 = .0;
		for (int i = 1; i < layerStack.size()-1; ++i)
		{
			L2 += thickness[i];
		}
	}
	else if (sliceType == sliceXY)
	{
		startCoord1 = layers_[0]->l0();
		startCoord2 = layers_[0]->b0();
		L1 = layers_[0]->Lx();
		L2 = layers_[0]->Ly();
	}

	int nRes1 = 2000;
	int nRes2 = static_cast<int>( nRes1 * L2 / L1);

	scalar step1 = L1 / nRes1;
	scalar step2 = L2 / nRes2;
	
	VectorXs sample1(nRes1);
	for (int i = 0; i < nRes1; ++i)
	{
		sample1(i) = startCoord1 + step1 * (i + 0.5); 
	}

	VectorXs sample2(nRes2);
	for (int i = 0; i < nRes2; ++i)
	{
		sample2(i) = startCoord2 + step2 * (i + 0.5);
	}

	MatrixXcs device(nRes2, nRes1);

	if (sliceType == sliceXZ || sliceType == sliceYZ)
	{
		for (int i = 0; i < nRes2; ++i)
		{
			scalar z = sample2(i);

			// find the correspond layer
			scalar z0 = .0;
			scalar z1 = .0;

			//int layer0 = -1;
			int layer1 = -1;

			for (int j = 1; j < layerStack.size()-1; ++j)
			{
				z1 = z0 + thickness[j];
				if (z0 <= z && z < z1)
				{
					//layer0 = j-1;
					layer1 = j;
					break;
				}
				z0 = z1;
			}
			int layerType = layerStack[layer1];
			scalar tx = 0.;
			if (translation != nullptr)
			{
				tx = (*translation)[layer1];
			}

			for (int j = 0; j < nRes1; ++j)
			{
				if (sliceType == sliceXZ)
				{
					scalar x = sample1(j);
					device(i, j) = layers_[layerType]->eps(x-tx, sliceCoord);				
				}
				else if (sliceType == sliceYZ)
				{
					scalar y = sample1(j);
					device(i, j) = layers_[layerType]->eps(sliceCoord-tx, y);
				}

			}
		}		
	}
	else if (sliceType == sliceXY)
	{
		scalar z = sliceCoord;
		// find the correspond layer
		scalar z0 = .0;
		scalar z1 = .0;

		//int layer0 = -1;
		int layer1 = -1;

		for (int j = 1; j < layerStack.size()-1; ++j)
		{
			z1 = z0 + thickness[j];
			if (z0 <= z && z < z1)
			{
				layer1 = j;
				break;
			}
			z0 = z1;
		}
		int layerType = layerStack[layer1];
		scalar tx = 0.;
		if (translation != nullptr)
		{
			tx = (*translation)[layer1];
		}

		for (int i = 0; i < nRes2; ++i)
		{
			for (int j = 0; j < nRes1; ++j)
			{
				scalar x = sample1(j);
				scalar y = sample2(i);

				device(i, j) = layers_[layerType]->eps(x-tx, y);
			}
		}	
	}


	// MatrixVisualizer vis(device);
	// vis.setXTimes(1);
	// vis.save(filename, realpart);	
}


void RCWASolver::saveModeImage(const std::string& filename,
	int layerType, int modeIndex,
	FieldComponent opt, int nResX, int nResY)
{
	if (!isLayerSolved_)
	{
		std::cerr << "The layers have not been solved!\n";
		return;
	}

	// ModeVisualizer vis(layers_[layerType]);
	// vis.save(filename, opt, nx_, ny_, nResX, nResY, modeIndex);
}


void RCWASolver::evaluateIntermediateField(
	std::vector<Eigen::VectorXcs>& c_m,
	std::vector<Eigen::VectorXcs>& c_p,
	const std::vector<std::pair<int, scalex>>& inputCoeffs,
	const std::vector<int>& layerStack,
	const std::vector<scalar>& thickness)
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The total scatter matrix has not been solved!\n";
		return;
	}

	int nDim = 2 * nx_ * ny_;
	int nL = layerStack.size();

	std::vector< MatrixXcs > S11;
	std::vector< MatrixXcs > S12;
	std::vector< MatrixXcs > S21;
	std::vector< MatrixXcs > S22;

	std::vector< VectorXcs > phaseMatrices;
	std::vector< MatrixXcs > REFMatrices;


	MatrixXcs for_inv_s22;
	MatrixXcs Rud(nDim, nDim);
	Rud.setZero();

	REFMatrices.push_back(Rud);

	std::clog << "evaluate intermediate scatter matrices...";
	for (int i = 0; i < nL-1; ++i)
	{
		int index = nL - i - 1;
		int layerIndex = layerStack[index];
		scalar thick = thickness[index];

		VectorXcs phaseMatrix(nDim);

		if (i == 0)
		{
			phaseMatrix.setOnes();
		}
		else
		{
			for (int j = 0; j < nDim; ++j)
			{
				phaseMatrix(j) = exp(scalex(0, 1)* layers_[layerIndex]->gamma()(j) * thick);
			}			
		}

		phaseMatrices.push_back(phaseMatrix);

		int interfaceIndex = interfaceMap_[index-1];
		const MatrixXcs& interfaceE = interfacesE_[interfaceIndex];
		const MatrixXcs& interfaceH = interfacesH_[interfaceIndex];

		MatrixXcs t11 = 0.5 * (interfaceE + interfaceH);
		MatrixXcs t12 = 0.5 * (interfaceE - interfaceH);

		PartialPivLU<MatrixXcs> solver(t11);
		MatrixXcs inv_t22 = solver.inverse();

		MatrixXcs inv_t22_t12 = solver.solve(t12);

		S11.push_back(t11 - t12 * inv_t22_t12);
		S12.push_back(t12 * inv_t22);
		S21.push_back(-inv_t22_t12);
		S22.push_back(inv_t22);

		if (i == 0)
		{
			for_inv_s22 = t11;
		}

		MatrixXcs waveRud = phaseMatrix.asDiagonal() * Rud * phaseMatrix.asDiagonal();
		MatrixXcs G = interfaceH - interfaceH * waveRud;
		Rud = - 2. * G * taus_[i];
		Rud.diagonal().array() += scalar(1.0);

		REFMatrices.push_back(Rud);
	}
	std::clog << "done!\n";

	VectorXcs inputVec(nDim);
	inputVec.setZero();
	for (int i = 0; i < inputCoeffs.size(); ++i)
	{
		inputVec(inputCoeffs[i].first) = inputCoeffs[i].second;
	}

	// reflective wave (+) at last layer
	// VectorXcs R = Rud_.col(inputMode);
	VectorXcs R = Rud_ * inputVec;
	
	// transmittive wave (-) at first layer
	// VectorXcs T = Tdd_.col(inputMode);
	VectorXcs T = Tdd_ * inputVec;

	int nInnerLayers = nL - 2;

	std::clog << "solve each layer state...";

	c_m.resize(nInnerLayers);
	c_p.resize(nInnerLayers);
	std::vector<VectorXcs> d(nInnerLayers);

	c_m[nInnerLayers-1] = S12[0] * for_inv_s22 * T;

	CompleteOrthogonalDecomposition<MatrixXcs> tempSolver(S11[nL-2]);
	VectorXcs temp = tempSolver.solve(R - S12[nL-2] * inputVec
		/*S12[nL-2].col(inputMode)*/);
	c_p[0] = S21[nL-2] * temp + S22[nL-2] * inputVec/*S22[nL-2].col(inputMode)*/;
	d[nInnerLayers-1] = phaseMatrices[nL-2].cwiseProduct(c_p[0]);
	if (nInnerLayers >= 2)
	{
		for (int i = nInnerLayers-2; i >= 0; --i)
		{
			c_m[nInnerLayers-i-2] = REFMatrices[i+2] * d[i+1];
			CompleteOrthogonalDecomposition<MatrixXcs> solveri(S11[i+1]);
			VectorXcs tempi = solveri.solve(c_m[nInnerLayers-i-2] - S12[i+1] * d[i+1]);
			c_p[nInnerLayers-i-1] = S21[i+1] * tempi + S22[i+1]*d[i+1];
			d[i] = phaseMatrices[i+1].cwiseProduct(c_p[nInnerLayers-i-1]);
		}
	}
	std::clog << "done!\n";		
}


void RCWASolver::evaluateIntermediateField(
	std::vector<Eigen::VectorXcs>& c_m,
	std::vector<Eigen::VectorXcs>& c_p,
	int inputMode,
	const std::vector<int>& layerStack,
	const std::vector<scalar>& thickness)
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The total scatter matrix has not been solved!\n";
		return;
	}

	int nDim = 2 * nx_ * ny_;
	int nL = layerStack.size();

	std::vector< MatrixXcs > S11;
	std::vector< MatrixXcs > S12;
	std::vector< MatrixXcs > S21;
	std::vector< MatrixXcs > S22;

	std::vector< VectorXcs > phaseMatrices;
	std::vector< MatrixXcs > REFMatrices;


	MatrixXcs for_inv_s22;
	MatrixXcs Rud(nDim, nDim);
	Rud.setZero();

	REFMatrices.push_back(Rud);

	std::clog << "evaluate intermediate scatter matrices...";
	for (int i = 0; i < nL-1; ++i)
	{
		int index = nL - i - 1;
		int layerIndex = layerStack[index];
		scalar thick = thickness[index];

		VectorXcs phaseMatrix(nDim);

		if (i == 0)
		{
			phaseMatrix.setOnes();
		}
		else
		{
			for (int j = 0; j < nDim; ++j)
			{
				phaseMatrix(j) = exp(scalex(0, 1)* layers_[layerIndex]->gamma()(j) * thick);
			}			
		}

		phaseMatrices.push_back(phaseMatrix);

		int interfaceIndex = interfaceMap_[index-1];
		const MatrixXcs& interfaceE = interfacesE_[interfaceIndex];
		const MatrixXcs& interfaceH = interfacesH_[interfaceIndex];

		MatrixXcs t11 = 0.5 * (interfaceE + interfaceH);
		MatrixXcs t12 = 0.5 * (interfaceE - interfaceH);

		PartialPivLU<MatrixXcs> solver(t11);
		MatrixXcs inv_t22 = solver.inverse();

		MatrixXcs inv_t22_t12 = solver.solve(t12);

		S11.push_back(t11 - t12 * inv_t22_t12);
		S12.push_back(t12 * inv_t22);
		S21.push_back(-inv_t22_t12);
		S22.push_back(inv_t22);

		if (i == 0)
		{
			for_inv_s22 = t11;
		}

		MatrixXcs waveRud = phaseMatrix.asDiagonal() * Rud * phaseMatrix.asDiagonal();
		MatrixXcs G = interfaceH - interfaceH * waveRud;
		Rud = - 2. * G * taus_[i];
		Rud.diagonal().array() += scalar(1.0);

		REFMatrices.push_back(Rud);
	}
	std::clog << "done!\n";

	// reflective wave (+) at last layer
	VectorXcs R = Rud_.col(inputMode);
	// transmittive wave (-) at first layer
	VectorXcs T = Tdd_.col(inputMode);

	int nInnerLayers = nL - 2;

	std::clog << "solve each layer state...";

	c_m.resize(nInnerLayers);
	c_p.resize(nInnerLayers);
	std::vector<VectorXcs> d(nInnerLayers);

	c_m[nInnerLayers-1] = S12[0] * for_inv_s22 * T;

	CompleteOrthogonalDecomposition<MatrixXcs> tempSolver(S11[nL-2]);
	VectorXcs temp = tempSolver.solve(R - S12[nL-2].col(inputMode));
	c_p[0] = S21[nL-2] * temp + S22[nL-2].col(inputMode);
	d[nInnerLayers-1] = phaseMatrices[nL-2].cwiseProduct(c_p[0]);
	if (nInnerLayers >= 2)
	{
		for (int i = nInnerLayers-2; i >= 0; --i)
		{
			c_m[nInnerLayers-i-2] = REFMatrices[i+2] * d[i+1];
			CompleteOrthogonalDecomposition<MatrixXcs> solveri(S11[i+1]);
			VectorXcs tempi = solveri.solve(c_m[nInnerLayers-i-2] - S12[i+1] * d[i+1]);
			c_p[nInnerLayers-i-1] = S21[i+1] * tempi + S22[i+1]*d[i+1];
			d[i] = phaseMatrices[i+1].cwiseProduct(c_p[nInnerLayers-i-1]);
		}
	}
	std::clog << "done!\n";	
}


void RCWASolver::generateHorizontalPlaneWave(
	scalar px,
	scalar py,
	int layerType,
	Eigen::VectorXcs& c)
{
	VectorXcs delta(nx_ * ny_);
	delta.setZero();
	delta(ny_ * (nx_ - 1) / 2 + (ny_ - 1) / 2) = 1.0;
	
	layers_[layerType]->generatePlaneWave(delta, px, py, c);
}


void RCWASolver::scatterPlaneWave(
	const Eigen::VectorXcs& cInc,
	int refLayerType,
	int trnLayerType,
	scalar k0,
	VectorXs& REF,
	VectorXs& TRN)
{
	if (!isScatterMatrixSolved_)
	{
		std::cerr << "The scatter matrix has not been solved!\n";
		return;
	}

	VectorXcs cref = Rud_ * cInc;
	VectorXcs ctrn = Tdd_ * cInc;

	VectorXcs rxy, txy;
	layers_[refLayerType]->getHarmonics(cref, rxy);
	layers_[trnLayerType]->getHarmonics(ctrn, txy);

	int nDim = nx_ * ny_;
	VectorXcs rx = rxy.head(nDim);
	VectorXcs ry = rxy.tail(nDim);
	VectorXcs tx = txy.head(nDim);
	VectorXcs ty = txy.tail(nDim);

	VectorXcs Kzref(nDim);
	VectorXcs Kztrn(nDim);

	scalex eps_ref = layers_[refLayerType]->eps(.0, .0);
	scalex eps_trn = layers_[trnLayerType]->eps(.0, .0);

	scalex kref2 = k0 * k0 * eps_ref;
	scalex ktrn2 = k0 * k0 * eps_trn;


	for (int i = 0; i < nDim; ++i)
	{
		Kzref(i) = -sqrt(kref2 - Kx_(i) * Kx_(i) - Ky_(i) * Ky_(i));
		Kztrn(i) = sqrt(ktrn2 - Kx_(i) * Kx_(i) - Ky_(i) * Ky_(i));
	}

	VectorXcs rz(nDim), tz(nDim);

	for (int i = 0; i < nDim; ++i)
	{
		if (Kzref(i) == scalex(0.0, 0.0))
		{
			std::cerr << "One plane wave component moves traversely!\n";
			return;
		}
		else
		{
			rz(i) = scalar(1.0) / Kzref(i) * (Kx_(i) * rx(i) + Ky_(i) * ry(i));
		}

		if (Kztrn(i) == scalex(0.0, 0.0))
		{
			std::cerr << "One plane wave component moves traversely!\n";
			return;
		}
		else
		{
			tz(i) = scalar(1.0) / Kztrn(i) * (Kx_(i) * tx(i) + Ky_(i) * ty(i));
		}		
	}


	// rz = Kzref.cwiseInverse().asDiagonal() * (Kx_.asDiagonal() * rx + Ky_.asDiagonal() * ry);
	// tz = Kztrn.cwiseInverse().asDiagonal() * (Kx_.asDiagonal() * tx + Ky_.asDiagonal() * ty);

	VectorXs r, t;
	r.resize(nDim);
	t.resize(nDim);
	for (int i = 0; i < nDim; ++i)
	{
		r(i) = std::abs(rx(i)) * std::abs(rx(i))
			 + std::abs(ry(i)) * std::abs(ry(i))
			 + std::abs(rz(i)) * std::abs(rz(i));
		t(i) = std::abs(tx(i)) * std::abs(tx(i))
			 + std::abs(ty(i)) * std::abs(ty(i))
			 + std::abs(tz(i)) * std::abs(tz(i));
	}
	
	
	VectorXs Kzref_r(nDim);
	VectorXs Kztrn_r(nDim);
	for (int i = 0; i < nDim; ++i)
	{
		Kzref_r(i) = abs(Kzref(i).real());
		Kztrn_r(i) = abs(Kztrn(i).real());
	}

	// we only excite and receive plane wave in dielectric material
	// metal is not supported for excitation and receiver
	REF = 1. / (sqrt(eps_ref.real()) * k0) * Kzref_r.asDiagonal() * r;
	TRN = 1. / (sqrt(eps_trn.real()) * k0) * Kztrn_r.asDiagonal() * t;
}