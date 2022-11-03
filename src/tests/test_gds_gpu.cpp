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
#include "gpu/GDSIISamplerGPU.h"
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

void test1()
{
  int nHx = 12;
  int nHy = 10;
  int nx = 2*nHx+1;
  int ny = 2*nHy+1;
  scalar lambda = 1.55;
  
  scalex cl = 1.445 * 1.445;
  scalex co = 3.48 * 3.48;
  
  VectorXs x(8);
  x << .0, .5, 1.5, 2.6, 3.7, 3.9, 5., 5.5;
  VectorXs y(7);
  y << .0, .5, 1.5, 1.58, 1.8, 2.8, 3.3;
  MatrixXcs eps(6, 7);
  eps << cl, cl, cl, cl, cl, cl, cl,
    cl, cl, cl, cl, cl, cl, cl,
    cl, cl, cl, cl, cl, cl, cl,
    cl, cl, co, cl, cl, cl, cl,
    cl, cl, cl, cl, cl, cl, cl,
    cl, cl, cl, cl, cl, cl, cl;
  Vector3s shift {1.62, 0., 0.};

  auto sampler = std::make_shared<GDSIISamplerGPU>(lambda, 2*nHx+1, 2*nHy+1, 
      "constraint_linear_smooth.gds",
      shift, 0.5, 0.5, 1., 1.3);
  MatrixXcs P1, Q1;
  sampler->sample(6., P1, Q1);

  auto sampler2 = std::make_shared<GDSIISampler>(lambda, 2*nHx+1, 2*nHy+1, 
      "constraint_linear_smooth.gds",
      shift, 0.5, 0.5, 1., 1.3);
  MatrixXcs P2, Q2;
  sampler2->sample(6., P2, Q2);

  cout << (P1-P2).cwiseAbs().maxCoeff() << "\t"
    << (Q1-Q2).cwiseAbs().maxCoeff() << endl;
}

scalar z0;
scalar z1;
int nHx, nHy;
scalar cl, co;
scalar lambda;
VectorXs x, y;
MatrixXcs eps;
std::shared_ptr<GDSIISamplerGPU> sampler;

void initialization()
{
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

  // contour
  Vector3s shift {2.3647, 0., 0.};
  z0 = -1.63;
  z1 = 1.36;
	sampler = std::make_shared<GDSIISamplerGPU>(lambda, 2*nHx+1, 2*nHy+1, 
        "contour_extract.gds",
        shift, 0.5, 0.5, 1., 1.3);  
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

void simulateVarRCWA(scalar error, const std::string& prefix)
{
  initialization();
  RedhefferIntegratorGPU integrator(z0, z1, sampler);
  integrator.compute_recursive(10000000, error, 0.);
  MatrixXcs Tuu = integrator.Tuu();
  
  vector<int> indices0;
  vector<int> indices1;
  MatrixXcs W0ref = integrator.W0();
  sortEigvecCols(W0ref, indices0);
  MatrixXcs W1ref = integrator.W1();
  sortEigvecCols(W1ref, indices1);
  Tuu = reorder_tuu(Tuu, indices1, indices0);

  stringstream ss;
  ss << prefix << "_" << error << ".txt";
  saveMatrix(ss.str(), Tuu);
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
  simulateRCWA(8192*8, "contour_RCWA_Tuu_GPU");
  
  return 0;
}