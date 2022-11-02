#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <memory>
#include <vector>
#include <algorithm>

#include <Eigen/Core>
#include "rcwa/defns.h"
#include "gpu/builder.h"
#include "core/VarLayerSampler.h"
#include "gpu/VarLayerSamplerGPU.h"
#include "gpu/RedhefferIntegratorGPU.h"
#include "gpu/RCWAIntegratorGPU.h"
#include "gpu/DifferentialIntegratorGPU.h"
#include "core/RedhefferIntegrator.h"

#include "core/GDSIISampler.h"
#include "core/RCWAIntegrator.h"
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

scalar z0;
scalar z1;
int nHx, nHy;
scalar cl, co;
scalar lambda;
VectorXs x, y;
MatrixXcs eps;
std::shared_ptr<VarLayerSamplerGPU> sampler;

void initialization()
{
  z0 = 0.;
  z1 = 2.;

  nHx = 12;
  nHy = 12;

  cl = 1.;
  co = 3.48 * 3.48;

  lambda = 1.55;

  x.resize(4);
  x << .0, 0.1, 0.5, 0.65;

  y.resize(4);
  y << .0, 0.1, 0.5, 0.65;

  eps.resize(3, 3);
  eps << cl, cl, cl,
    cl, co, cl,
    cl, cl, cl;

  sampler = std::make_shared<VarLayerSamplerGPU>(lambda, 
      2*nHx+1, 2*nHy+1, 
      x, y, eps, false);

  sampler->addXRule(2, [](scalar z){
    if (z <= 1.) {
      return 0.5;
    } else if (z <= 2.) {
      return 0.2 * cos(0.5 * Pi * (z - 1.)) + 0.3;
    }
  });
}

void test2(scalar error)
{
  ofstream fout("difference_gpu_cpu.txt", ios::app);

  initialization();

  RedhefferIntegratorGPU integrator(z0, z1, sampler);
  integrator.compute_recursive(10000000, error, 0.);
  MatrixXcs Tuu = integrator.Tuu();

  RedhefferIntegrator integrator2(z0, z1, sampler);
  integrator2.compute(0, 100000000, error);
  MatrixXcs Tuu2 = integrator2.Tuu();

  cout << (Tuu - Tuu2).cwiseAbs().maxCoeff() << endl;

  vector<int> indices0;
  MatrixXcs W0 = integrator.W0();
  sortEigvecCols(W0, indices0);

  vector<int> indices1;
  MatrixXcs W1 = integrator.W1();
  sortEigvecCols(W1, indices1);

  Tuu = reorder_tuu(Tuu, indices1, indices0);
  MatrixXcs W0ref = integrator2.W0();
  sortEigvecCols(W0ref, indices0);
  MatrixXcs W1ref = integrator2.W1();
  sortEigvecCols(W1ref, indices1);
  Tuu2 = reorder_tuu(Tuu2, indices1, indices0);
  cout << "residual: " << (Tuu - Tuu2).cwiseAbs().maxCoeff() << endl;
  fout << error << "\t" << (Tuu - Tuu2).cwiseAbs().maxCoeff() << endl;
}

void validateVarRCWA(scalar error, const std::string& filename)
{
  ofstream fout("difference_varRCWA.txt", ios::app);
  initialization();

  auto t1 = sploosh::now();
  RedhefferIntegratorGPU integrator(z0, z1, sampler);
  integrator.compute_recursive(10000000, error, 0.);
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

void validateVarRCWA_direct(int N, const std::string& filename)
{
  ofstream fout("difference_varRCWA.txt", ios::app);
  initialization();

  auto t1 = sploosh::now();
  RedhefferIntegratorGPU integrator(z0, z1, sampler);
  integrator.compute_direct(N);
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
  fout << integrator.N() << "\t"
    << integrator.nEigVal() << "\t" 
    << sploosh::duration_milli_d(t1, t2) << "\t" 
    << residual << endl;
}

void simulateRCWA(int N, const std::string& prefix)
{
  initialization();
  RCWAIntegratorGPU integrator2(z0, z1, sampler);
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
  RCWAIntegratorGPU integrator2(z0, z1, sampler);
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
  DifferentialIntegratorGPU integrator2(z0, z1, sampler);
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

int main()
{
  // simulateRCWA(1024, "Metasurface_03_RCWA_GPU");
  // for (int i = 1; i <= 512; i *= 2) {
  //   validateDiffmethod(i, "Metasurface_03_RCWA_GPU_1024.txt");
  // }
  // for (scalar e = 2.; e >= 1e-4; e *= 0.2) {
  //   validateVarRCWA(e, "Metasurface_03_RCWA_GPU_1024.txt");
  // }
  // for (int i = 1; i <= 512; i *= 2) {
  //   validateRCWA(i, "Metasurface_03_RCWA_GPU_1024.txt");
  // }
  validateRCWA(1024, "Metasurface_03_RCWA_GPU_1024.txt");
  return 0;
}