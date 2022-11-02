#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <memory>
#include <vector>
#include <algorithm>

#include <Eigen/Core>
#include "rcwa/defns.h"
#include "core/VarLayerSampler.h"
#include "core/RedhefferIntegrator.h"
#include "core/RCWAIntegrator.h"
#include "core/DifferentialIntegrator.h"
#include "rcwa/WaveEquationSolver.h"
#include "utils/timer.hpp"

using namespace std;
using namespace Eigen;

scalar compareMatrix(
    const std::string& filename,
    const MatrixXcs& Tdd,
    string* cprFilename = nullptr)
{
    ifstream fin(filename);
    string temp;
    fin >> temp;

    MatrixXcs tempMatrix(Tdd.rows(), Tdd.cols());

    for (int i = 0; i < Tdd.rows(); ++i) {
        for (int j = 0; j < Tdd.cols(); ++j) {
            scalar val;
            fin >> val;
            tempMatrix(i, j).real(val);
        }
    }

    fin >> temp;

    for (int i = 0; i < Tdd.rows(); ++i) {
        for (int j = 0; j < Tdd.cols(); ++j) {
            scalar val;
            fin >> val;          
            tempMatrix(i, j).imag(val);
        }
    }

    MatrixXs diff = (tempMatrix - Tdd).cwiseAbs();
    cout << "residual: " << diff.maxCoeff() << endl;

    return diff.maxCoeff();
}

void saveMatrix(const std::string& filename, const MatrixXcs& Tdd) {
    ofstream fout(filename);
    fout << std::setprecision(12);
    fout << "real: " << endl;
    for (int i = 0; i < Tdd.rows(); ++i) {
        for (int j = 0; j < Tdd.cols(); ++j) {
            fout << Tdd(i, j).real() << "\t";
        }
        fout << endl;
    }
    fout << endl;
    
    fout << "imag: " << endl;
    for (int i = 0; i < Tdd.rows(); ++i) {
        for (int j = 0; j < Tdd.cols(); ++j) {
            fout << Tdd(i, j).imag() << "\t";
        }
        fout << endl;
    }
    fout << endl;
}

void sortEigvecCols(MatrixXcs& mat, vector<int>& indices)
{
  indices.resize(mat.cols());
  std::vector<VectorXcs> vec;
  for (int i = 0; i < mat.cols(); ++i) {
    vec.push_back(mat.col(i));
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), 
    [&](int i, int j)
      { 
        const VectorXcs& t1 = vec[i];
        const VectorXcs& t2 = vec[j];
        return t1(0).real() < t2(0).real() ||
        (t1(0).real() == t2(0).real() && t1(0).imag() < t2(0).imag()); 
      } );
  
  for (int i = 0; i < mat.cols(); ++i) {
    mat.col(i) = vec[indices[i]];
  }
}

MatrixXcs reorder_tuu(const MatrixXcs& mat,
  const vector<int>& ind1,
  const vector<int>& ind2)
{
  int row = mat.rows();
  int col = mat.cols();

  MatrixXcs temp(row, col);
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      int ii = ind1[i]; // output
      int jj = ind2[j]; // input
      temp(i, j) = mat(ii, jj);
    }
  }
  return temp;
}

scalar z0; // the starting z
scalar z1; // the ending z
int nHx, nHy; // half of the number of harmonic along x and y directions
scalar cl, co; // lower index, higher index
scalar lambda; // wavelength, micron
VectorXs x, y; // grid positions
MatrixXcs eps;
std::shared_ptr<VarLayerSampler> sampler;

void initialization()
{
  z0 = -0.1;
  z1 = 1.1;
  nHx = 12;
  nHy = 10;

  cl = 1.445 * 1.445;
  co = 3.48 * 3.48;

  lambda = 1.55;
  x.resize(8);
  x << .0, .5, 1.5, 2.6, 3.7, 3.9, 5., 5.5;

  y.resize(7);
  y << .0, .5, 1.5, 1.58, 1.8, 2.8, 3.3;

  eps.resize(6, 7);
  eps << cl, cl, cl, cl, cl, cl, cl,
  cl, cl, cl, cl, cl, cl, cl,
  cl, cl, cl, cl, cl, cl, cl,
  cl, cl, co, cl, cl, cl, cl,
  cl, cl, cl, cl, cl, cl, cl,
  cl, cl, cl, cl, cl, cl, cl;

  sampler = std::make_shared<VarLayerSampler>(lambda, 
      2*nHx+1, 2*nHy+1, 
      x, y, eps);
  

  // the following specifies some cases for testing
  // how x[i] changes with different z

  // var11
  // sampler->addXRule(3, [](scalar z){
  //     return (1. - z) * 3.7 + z * 2.6;
  // });

  // var12
  z1 = 10.1;
  sampler->addXRule(3, [](scalar z){
      if (z < 0.) {
        return 2.6;
      } else if (z > 10.) {
        return 3.7;
      } else {
        return (1. - 0.1 * z) * 2.6 + 0.1 * z * 3.7;
      }
  });  

  // var13
  // sampler->addXRule(3, [](scalar z){
  //     return 2.599 + 0.001* pow(1101., z);
  // });

  // var14
  // sampler->addXRule(3, [](scalar z){
  //     return 2.5999 + 0.0001* pow(11001., z);
  // });

  // var15
  // sampler->addXRule(3, [](scalar z){
  //     if (z < 0.1) {
  //         return (1. - 9.*z) * 2.6 + 9.*z * 3.7;
  //     } else {
  //        return 1.1/9. * (z-1.) + 3.7;
  //     }
  // });

  // var16
  // sampler->addXRule(3, [](scalar z){
  //     return 2.6 + 1.1 * sin(0.5*Pi*z);
  // });

  // var17
  // z1 = 20.;
  // // 1.8 -> 1.5
  // sampler->addXRule(2, [](scalar z){
  //     return 1.81 - 0.01 * pow(31., z/20);
  // });
  // // 2.3 -> 2.6
  // sampler->addXRule(3, [](scalar z){
  //     return 2.29 + 0.01 * pow(31., z/20);
  // });
}

void validateVarRCWA(scalar error, const std::string& filename)
{
  ofstream fout("difference_varRCWA.txt", ios::app);
  initialization();

  auto t1 = sploosh::now();
  RedhefferIntegrator integrator(z0, z1, sampler);
  integrator.compute(0, 1000000, error);
  auto t2 = sploosh::now();

  MatrixXcs Tuu = integrator.Tuu();

  vector<int> indices0;
  MatrixXcs W0 = integrator.W0();
  sortEigvecCols(W0, indices0);

  vector<int> indices1;
  MatrixXcs W1 = integrator.W1();
  sortEigvecCols(W1, indices1);
  Tuu = reorder_tuu(Tuu, indices1, indices0);
  scalar residual = compareMatrix(filename, Tuu);
  fout << error << "\t" << integrator.N() << "\t"
    << integrator.nEigVal() << "\t" 
    << sploosh::duration_milli_d(t1, t2) << "\t" 
    << residual << endl;
}

void simulateRCWA(int N, const std::string& prefix)
{
  initialization();
  RCWAIntegrator integrator2(z0, z1, sampler);
  integrator2.compute(N);
  MatrixXcs Tuu = integrator2.Tuu();
  
  vector<int> indices0;
  vector<int> indices1;
  MatrixXcs W0ref = integrator2.W0();
  sortEigvecCols(W0ref, indices0);
  MatrixXcs W1ref = integrator2.W1();
  sortEigvecCols(W1ref, indices1);
  Tuu = reorder_tuu(Tuu, indices1, indices0);

  stringstream ss;
  ss << prefix << "_" << N << ".txt";
  saveMatrix(ss.str(), Tuu);
}

void validateRCWA(int N, const std::string& filename)
{
  ofstream fout("difference_RCWA.txt", ios::app);
  initialization();
  auto t1 = sploosh::now();
  RCWAIntegrator integrator2(z0, z1, sampler);
  integrator2.compute(N);
  auto t2 = sploosh::now();
  MatrixXcs Tuu = integrator2.Tuu();
  
  vector<int> indices0;
  vector<int> indices1;
  MatrixXcs W0ref = integrator2.W0();
  sortEigvecCols(W0ref, indices0);
  MatrixXcs W1ref = integrator2.W1();
  sortEigvecCols(W1ref, indices1);
  Tuu = reorder_tuu(Tuu, indices1, indices0);

  scalar residual = compareMatrix(filename, Tuu);
  fout << N << "\t" 
    << sploosh::duration_milli_d(t1, t2) << "\t" 
    << residual << endl;
}

void validateDiffmethod(int N, const std::string& filename)
{
  ofstream fout("difference_diffmethod.txt", ios::app);
  initialization();

  auto t1 = sploosh::now();
  DifferentialIntegrator integrator(z0, z1, sampler);
  integrator.compute(N);
  auto t2 = sploosh::now();

  MatrixXcs Tuu = integrator.Tuu();

  vector<int> indices0;
  MatrixXcs W0 = integrator.W0();
  sortEigvecCols(W0, indices0);

  vector<int> indices1;
  MatrixXcs W1 = integrator.W1();
  sortEigvecCols(W1, indices1);
  Tuu = reorder_tuu(Tuu, indices1, indices0);
  scalar residual = compareMatrix(filename, Tuu);
  fout << N << "\t" 
    << sploosh::duration_milli_d(t1, t2) << "\t" 
    << residual << endl;
}

void findGuidedModes(
	std::vector < std::pair<int, scalex> >& guidedModeRecords,
  const Eigen::VectorXcs& gamma,
	scalar k0,
	scalar imagCoeff,
	scalar realCoeffMin,
	scalar realCoeffMax)
{
	for (int i = 0; i < gamma.size(); ++i)
	{
		scalex gamma_i = gamma(i) / k0/k0;
		if (realCoeffMin <= gamma_i.real() &&
			gamma_i.real() <= realCoeffMax &&
			gamma_i.imag() < imagCoeff)
		{
			guidedModeRecords.push_back(std::make_pair(i, gamma_i));
		}
	}

	std::sort(guidedModeRecords.begin(), guidedModeRecords.end(),
		[](const std::pair<int, scalex>& a, const std::pair<int, scalex>& b){
			return a.second.real() > b.second.real();
		});
}


scalar evaluateEnergy(
    const Eigen::VectorXcs& eigvecE,
    const Eigen::VectorXcs& eigvecH,
    int nx, int ny,
    scalar Lx, 
    scalar Ly,
    int nResX,
    int nResY,
    scalar x0, scalar x1,
    scalar y0, scalar y1)
{
    MatrixXcs kEx(nx, ny), 
        kEy(nx, ny), 
        kHx(nx, ny), 
        kHy(nx, ny);
    
    for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j)
		{
            int index = j + i * ny;
            kEx(i, j) = eigvecE(index);
            kHx(i, j) = eigvecH(index);
            kEy(i, j) = eigvecE(index + nx * ny);
            kHy(i, j) = eigvecH(index + nx * ny);
		}
	}

  int maxX = (nx - 1) / 2;
	int maxY = (ny - 1) / 2;

    scalar xStep = (x1 - x0) / nResX;
	scalar yStep = (y1 - y0) / nResY;

	VectorXs sampleX(nResX);
	for (int i = 0; i < nResX; ++i)
	{
		sampleX(i) = x0 + xStep * (i + 0.5); 
	}

	VectorXs sampleY(nResY);
	for (int i = 0; i < nResY; ++i)
	{
		sampleY(i) = y0 + yStep * (i + 0.5);
	}

    scalar energy = 0;

    for (int i = 0; i < nResY; ++i)
	{
		for (int j = 0; j < nResX; ++j)
		{
			scalar x = sampleX(j);
			scalar y = sampleY(i);

            VectorXcs waveletX(nx), waveletY(ny);

            for (int m = -maxX; m <= maxX; ++m)
            {
                waveletX(m+maxX) = exp(scalex(0, -2.* Pi * m / Lx * x));
            }
            for (int n = -maxY; n <= maxY; ++n)
            {
                waveletY(n+maxY) = exp(scalex(0, -2.* Pi * n / Ly * y));
            }

            scalex Ex = static_cast<scalex>(waveletX.transpose() * kEx * waveletY);
            scalex Hx = static_cast<scalex>(waveletX.transpose() * kHx * waveletY);
            scalex Ey = static_cast<scalex>(waveletX.transpose() * kEy * waveletY);
            scalex Hy = static_cast<scalex>(waveletX.transpose() * kHy * waveletY); 

            energy += 0.5 * (Ex * conj(Hy) - Ey * conj(Hx)).real() * xStep * yStep;    
        }
    }
    return energy;
}

// this is to normalize the scattering matrix in terms of energy
// and can be benchmarked against FDTD solver
void varRCWA_benchmark_guidedmodes(scalar error) {
  for (scalar lambda : { 1.55}) {
    z0 = -0.1;
    z1 = 10.1;
    nHx = 12;
    nHy = 10;

    cl = 1.445 * 1.445;
    co = 3.48 * 3.48;

    x.resize(7);
    x << .0, .5, 2.5, 3.6, 4.7, 6.7, 7.2;

    y.resize(7);
    y << .0, .5, 2., 2.08, 2.3, 3.8, 4.3;

    eps.resize(6, 6);
    eps << cl, cl, cl, cl, cl, cl,
    cl, cl, cl, cl, cl, cl,
    cl, cl, cl, cl, cl, cl,
    cl, cl, co, cl, cl, cl,
    cl, cl, cl, cl, cl, cl,
    cl, cl, cl, cl, cl, cl;

    sampler = std::make_shared<VarLayerSampler>(lambda, 
        2*nHx+1, 2*nHy+1, 
        x, y, eps);
    
    sampler->addXRule(3, [](scalar z){
        if (z < 0.) {
          return 3.6;
        } else if (z > 10.) {
          return 4.7;
        } else {
          return (1. - 0.1 * z) * 3.6 + 0.1 * z * 4.7;
        }
    });

    RedhefferIntegrator integrator(z0, z1, sampler);
    integrator.compute(0, 1000000, error);
    MatrixXcs Tuu = integrator.Tuu();

    const auto& gam0 = integrator.gam0();
    const auto& W0 = integrator.W0();
    const auto& V0 = integrator.V0();

    std::vector<std::pair<int, scalex>> guidedModeRecords0;
    findGuidedModes(guidedModeRecords0, gam0, 2 * Pi / lambda, 0.2, 1.445 , 3.48);

    for (int i = 0; i < guidedModeRecords0.size(); ++i) {
      const auto& rec = guidedModeRecords0[i];
      cout << rec.first << '\t' << rec.second.real() << endl;
    }

    const auto& gam1 = integrator.gam1();
    const auto& W1 = integrator.W1();
    const auto& V1 = integrator.V1();

    std::vector<std::pair<int, scalex>> guidedModeRecords1;
    findGuidedModes(guidedModeRecords1, gam1, 2*Pi/lambda, 0.2, 1.445, 3.48);
    for (int i = 0; i < guidedModeRecords1.size(); ++i) {
      const auto& rec = guidedModeRecords1[i];
      cout << rec.first << "\t" << rec.second.real() << endl;
    }

    ofstream fout("Tuu_varRCWA_benchmark.txt", ios::app);
    fout << "lambda: " << lambda << endl;
    for (int i = 0; i < guidedModeRecords1.size(); ++i) {
      int indi = guidedModeRecords1[i].first;
      const VectorXcs& eigvecE1 = W1.col(indi);
      const VectorXcs& eigvecH1 = V1.col(indi);
      scalar e1 = evaluateEnergy(eigvecE1, eigvecH1,
        2*nHx+1, 2*nHy+1, 7.2, 4.3, 500, 500, 
        0.5, 6.7,
        0.5, 3.8);
      cout << "e1: " << e1 << endl;
      for (int j = 0; j < guidedModeRecords0.size(); ++j) {
        int indj = guidedModeRecords0[j].first;
        const MatrixXcs& eigvecE0 = W0.col(indj);
        const MatrixXcs& eigvecH0 = V0.col(indj);
        scalar e0 = evaluateEnergy(eigvecE0, eigvecH0,
          2*nHx+1, 2*nHy+1, 7.2, 4.3, 500, 500, 
          0.5, 6.7,
          0.5, 3.8);
        cout << "e0: " << e0 << endl;
        scalex tuu = Tuu(indi, indj);
        fout <<  abs(tuu) * abs(tuu) * e1 / e0 << "\t";
      }
      fout << endl;
    }
  }
}

int main()
{
  // this will generate the file xxx_N.txt as a groundtruth
  // it takes some time (several hours) as this is RCWA with 8192 section
  simulateRCWA(8192, "var12_RCWA_Tuu_CPU");

  // validate the convergence of RCWA
  for (int i = 1; i <= 1024; i*=2) {
    validateRCWA(i, "var12_RCWA_Tuu_CPU_8192.txt");
  }

  // validate the convergence of differential method
  for (int i = 1; i <= 4096; i*=2) {
    validateDiffmethod(i, "var12_RCWA_Tuu_CPU_8192.txt");
  }

  // validate the convergence of VarRCWA
  for (scalar e = 1.; e >= 1e-5; e *= 0.2) {
    validateVarRCWA(e, "var12_RCWA_Tuu_CPU_8192.txt");
  }

  return 0;
}