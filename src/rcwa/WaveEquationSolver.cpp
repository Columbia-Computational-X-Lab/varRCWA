#include <iostream>
#include "WaveEquationSolver.h"
#include "MKLEigenSolver.hpp"

using namespace std;
using namespace Eigen;

struct GammaSqrtOp
{
    GammaSqrtOp() {} 

    inline const scalex operator() (const scalex& g) const
    {
        scalex gv = sqrt(g);
        if ( gv.imag() < 0 ) gv *= -1.;
        // now gv is at the upper half of the complex plane

        if (abs(gv.imag()) < abs(gv.real()) && gv.real() < 0)
            gv *= -1.;
        return gv;
    }
};

void WaveEquationSolver::solve(
    scalar lambda, const Eigen::MatrixXcs& P, const Eigen::MatrixXcs& Q,
    Eigen::MatrixXcs& PQ,
    Eigen::MatrixXcs& eigvecE, Eigen::MatrixXcs& eigvecH, 
    Eigen::VectorXcs& keff) const
{
    PQ = P*Q;
    MKLEigenSolver<MatrixXcs> ces;
    ces.compute(PQ);

    keff = ces.eigenvalues().unaryExpr(GammaSqrtOp());    // eigenvalue square root
    eigvecE = ces.eigenvectors();
    eigvecH = Q * eigvecE * keff.cwiseInverse().asDiagonal();
}
