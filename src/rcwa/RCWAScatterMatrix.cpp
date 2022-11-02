#include <iostream>
#include "RCWAScatterMatrix.h"
#include "WaveEquationSolver.h"
#include <chrono>

namespace dtmm
{

// template <typename MatrixType>
// void cycleMatrix(const MatrixType& rhs,
//     MatrixType& res)
// {
//     int N = 100;
//     res = rhs;
//     res.diagonal().array() += 1.;

//     MatrixType multiplier = rhs;
//     for (int i = 0; i < N; ++i)
//     {
//         multiplier *= rhs;
//         res += multiplier;
//     }
// }


void RCWAScatterMatrix::compute(scalar lambda, scalar wz,
        const RCWAScatterMatrix::MatrixType& P,
        const RCWAScatterMatrix::MatrixType& Q,
        const RCWAScatterMatrix::MatrixType& W0,
        const RCWAScatterMatrix::MatrixType& V0,
        const RCWAScatterMatrix::MatrixType& gamma0)
{
    using namespace Eigen;
   
    W0_ = W0;
    V0_ = V0;

    int nDim = P.rows();
    
    WaveEquationSolver wes;
    wes.solve(lambda, P, Q, PQ_, W_, V_, gamma_);

    // -----------------------------------------------
    // auto t1 = std::chrono::steady_clock::now();
    const scalar invk0 = lambda / (2.*Pi);
    normalizedLz_ = invk0*wz;   // wz / k0
    const scalar ss = normalizedLz_;
    X_ = gamma_.unaryExpr([&ss](const scalex& g) { return exp(scalex(0,1)*g*ss); });

    invWSol_.compute(W_);
    F0_ = invWSol_.solve(W0_);  // W^{-1}W_0

    invVSol_.compute(V_);

    G0_ = invVSol_.solve(V0_);

    A_ = F0_ + G0_;
    B_ = F0_ - G0_;

    invASol_.compute(A_);
    invAB_  = invASol_.solve(B_);  // invA*B

    MatrixXcs XA = X_.asDiagonal() * A_;       // XA
    MatrixXcs XB = X_.asDiagonal() * B_;       // XB
    invAXA_ = invASol_.solve(XA);
    invAXB_ = invASol_.solve(XB);

    const MatrixXcs I = MatrixXcs::Identity(nDim, nDim);
    I_minus_invAXB2_ = I - invAXB_*invAXB_;
    inv_I_minus_invAXB2_Sol_.compute(I_minus_invAXB2_);
    // inv_I_minus_invAXB2_ = inv_I_minus_invAXB2_Sol_.inverse();

    invAXB_invAXA_invAB_ = invAXB_*invAXA_ - invAB_;
    // Rud_ = inv_I_minus_invAXB2_ * invAXB_invAXA_invAB_;
    Rud_ = inv_I_minus_invAXB2_Sol_.solve(invAXB_invAXA_invAB_);
    invAXA_invAXB_invAB_ = invAXA_ - invAXB_*invAB_;
    // Tuu_ = inv_I_minus_invAXB2_ * invAXA_invAXB_invAB_;
    Tuu_ = inv_I_minus_invAXB2_Sol_.solve(invAXA_invAXB_invAB_);


    // auto t2 = std::chrono::steady_clock::now();
    // printf("time for (1)~(4): %f\n",(std::chrono::duration_cast
    // < std::chrono::duration<double> >
    // (t2 - t1)).count()); 
}


void RCWAScatterMatrix::compute_v2(scalar lambda, scalar wz,
    const MatrixType& W,
    const MatrixType& V,
    const MatrixType& gamma,
    const MatrixType& W0,
    const MatrixType& V0,
    const MatrixType& gamma0)
{
   using namespace Eigen;

    // int nDim = 2 * nx_ * ny_;

    W0_ = W0;
    V0_ = V0;

    int nDim = W0.rows();

    // WaveEquationSolver wes;
    // wes.solve(lambda, P, Q, PQ_, W_, V_, gamma_);
    W_ = W;
    V_ = V;
    gamma_ = gamma;

    // -----------------------------------------------
    //auto t1 = sploosh::now();

    const scalar invk0 = lambda / (2.*Pi);
    normalizedLz_ = invk0*wz;   // wz / k0
    const scalar ss = normalizedLz_;
    X_ = gamma_.unaryExpr([&ss](const scalex& g) { return exp(scalex(0,1)*g*ss); });

    invWSol_.compute(W_);
    MatrixXcs F0 = invWSol_.solve(W0_);  // W^{-1}W_0
    // N x n

    PartialPivLU<MatrixXcs> invSolver(nDim);
    invSolver.compute(V_);
    MatrixXcs G0 = invSolver.solve(V0_); // V^{-1}V_0
    // N x n

    const MatrixXcs& A = F0 + G0;
    const MatrixXcs& B = F0 - G0;

    invSolver.compute(A); // invA

    invAB_  = invSolver.solve( B );  // invA*B
    MatrixXcs XA = X_.asDiagonal() * A;       // XA
    MatrixXcs XB = X_.asDiagonal() * B;       // XB
    invAXA_ = invSolver.solve(XA);
    invAXB_ = invSolver.solve(XB);

    const MatrixXcs I = MatrixXcs::Identity(nDim, nDim);
    I_minus_invAXB2_ = I - invAXB_*invAXB_;
    inv_I_minus_invAXB2_Sol_.compute(I_minus_invAXB2_);

    invAXB_invAXA_invAB_ = invAXB_*invAXA_ - invAB_;
    Rud_ = inv_I_minus_invAXB2_Sol_.solve(invAXB_invAXA_invAB_);
    invAXA_invAXB_invAB_ = invAXA_ - invAXB_*invAB_;
    Tuu_ = inv_I_minus_invAXB2_Sol_.solve(invAXA_invAXB_invAB_);    
}
// void RCWAScatterMatrix::evaluateHomogeneousLayer(
//     scalar lambda, scalex eps,
//     RCWAScatterMatrix::MatrixType& eigvecE,
//     RCWAScatterMatrix::MatrixType& eigvecH,
//     RCWAScatterMatrix::VectorType& keff)
// {
//     WaveEquationCoeff coeff(nx_, ny_, Lx_, Ly_);
//     MatrixType Kx, Ky;
//     coeff.evaluateKMatrices(Kx, Ky);
    
//     scalar k0 = 2 * Pi / lambda;
//     scalex k02 = scalex(k0 * k0);
//     int blockDim = nx_ * ny_;

//     MatrixType Q(2*blockDim, 2*blockDim);
//     Q.topLeftCorner(blockDim, blockDim) = -Kx * Ky;
//     Q.topRightCorner(blockDim, blockDim) = Kx * Kx;
//     Q.topRightCorner(blockDim, blockDim).diagonal().array() -= k02 * eps;

//     Q.bottomLeftCorner(blockDim, blockDim) = -Ky * Ky;
//     Q.bottomLeftCorner(blockDim, blockDim).diagonal().array() += k02 * eps;

//     Q.bottomRightCorner(blockDim, blockDim) = Ky * Kx;

//     eigvecE.resize(2*blockDim, 2*blockDim);
//     eigvecE.setIdentity();

//     keff.resize(2*blockDim);
//     for (int i = 0; i < blockDim; ++i)
//     {
//         scalex gamma_i = sqrt(eps * k02 - Kx(i, i) * Kx(i, i) - Ky(i, i) * Ky(i, i));

//         if (gamma_i.imag() < 0)
//         {
//             gamma_i = -gamma_i;
//         }

        
//         if (abs(gamma_i.imag()) < abs(gamma_i.real()) && gamma_i.real() < 0)
//         {
//             gamma_i = -gamma_i;
//         }

//         keff(i) = keff(i + blockDim) = gamma_i;
//     }

//     //eigvecH.resize(2*blockDim, 2*blockDim);
//     eigvecH = Q * keff.cwiseInverse().asDiagonal() * (1. / k0);
// }

} // end of namespace dtmm
