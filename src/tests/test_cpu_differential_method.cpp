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

scalar z0;
scalar z1;
int nHx, nHy;
scalar cl, co;
scalar lambda;
VectorXs x, y;
MatrixXcs eps;
std::shared_ptr<VarLayerSampler> sampler;

void initialization()
{
  z0 = 0.;
  
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
  
  // length_1
  // z1 = 1.;
  // sampler->addXRule(3, [](scalar z){
  //     return (1. - z) * 3.7 + z * 2.6;
  // });

  // p vary
  // scalar p = 0.05;
  // sampler->addXRule(3, [&](scalar z){
  //     return (1. - z) * (p*3.7 + (1.-p)*2.6) + z * 2.6;
  // });

  // length_05
  // z1 = 0.5;
  // sampler->addXRule(3, [](scalar z){
  //     return (1. - 2.*z) * 3.7 + 2.*z * 2.6;
  // });

  // length_03
  // z1 = 0.3;
  // sampler->addXRule(3, [](scalar z){
  //     return (1. - 1./0.3 *z) * 3.7 + 1./0.3*z * 2.6;
  // });

  // length_02
  // z1 = 0.2;
  // sampler->addXRule(3, [](scalar z){
  //     return (1. - 5. *z) * 3.7 + 5.*z * 2.6;
  // });

  // length_01
  // z1 = 0.1;
  // sampler->addXRule(3, [](scalar z){
  //     return (1. - 10. *z) * 3.7 + 10.*z * 2.6;
  // });

  // length_005
  z1 = 0.05;
  sampler->addXRule(3, [](scalar z){
      return (1. - 20. *z) * 3.7 + 20.*z * 2.6;
  });
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

int main()
{
  // simulateRCWA(256, "RCWA_CPU_p_005");
  // validateDiffmethod(1, "RCWA_CPU_p_005_256.txt");
  // validateVarRCWA(10., "var11_RCWA_Tuu_CPU_1024.txt");
  validateRCWA(1, "RCWA_CPU_length_005_256.txt");

  // for (int i = 1; i < 256; i*=2) {
  //   validateRCWA(i, "var11_RCWA_Tuu_CPU_1024.txt");
  // }

  // for (int i = 1; i <= 64; i*=2) {
  //   validateDiffmethod(i, "var11_RCWA_Tuu_CPU_1024.txt");
  // }

  // for (scalar e = 1.; e >= 0.0005; e*=0.5) {
  //   validateVarRCWA(e, "var11_RCWA_Tuu_CPU_1024.txt");
  // }
  // validateVarRCWA(1., "var11_RCWA_Tuu_CPU_1024.txt");
  return 0;
}