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

#include "core/RCWAIntegrator.h"
#include "core/DifferentialIntegrator.h"
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
std::shared_ptr<GDSIISampler> sampler;

void initialization()
{
  nHx = 12;
  nHy = 10;

  cl = 1.445 * 1.445;
  co = 3.48 * 3.48;

  lambda = 1.55;

  // contour
  Vector3s shift {2.3647, 0., 0.};
  z0 = -1.63;
  z1 = 1.36;
	sampler = std::make_shared<GDSIISampler>(lambda, 2*nHx+1, 2*nHy+1, 
        "contour_extract.gds",
        shift, 0.5, 0.5, 1., 1.3);  
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

void simulateVarRCWA(scalar error, const std::string& prefix)
{
  initialization();
  RedhefferIntegrator integrator(z0, z1, sampler);
  integrator.compute(0, 1000000, error);
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
  DifferentialIntegrator integrator2(z0, z1, sampler);
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
  // test1();
  // simulateRCWA(8192*4, "contour_RCWA_Tuu_CPU");
  // for (int i = 1; i <= 2048; i*=2) {
  //   validateRCWA(i, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  // }
  // for (int N = 1; N <= 4096; N*=2) {
  //   validateDiffmethod(N, "contour_RCWA_Tuu_CPU_32768.txt");
  // }
  validateRCWA(4096, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  validateRCWA(8192, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  validateRCWA(8192*2, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  validateVarRCWA(0.00032*0.5, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  validateVarRCWA(0.00032*0.2, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  // for (scalar e = 1.; e >= 0.0001; e*=0.2) {
  //   validateVarRCWA(e, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  // }
  // validateVarRCWA(1., "contour_RCWA_Tuu_CPU_32768.txt");
  // validateVarRCWA(0.0016, "contour_RCWA_Tuu_CPU_32768.txt");
  // validateDiffmethod(2048, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  // validateDiffmethod(8192*2, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  // simulateVarRCWA(0.00005, "contour_varRCWA_Tuu_CPU");
  // validateVarRCWA(0.0001, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  // validateDiffmethod(4096, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  // validateDiffmethod(8192, "contour_varRCWA_Tuu_CPU_5e-05.txt");
  
  return 0;
}