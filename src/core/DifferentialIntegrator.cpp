#include "DifferentialIntegrator.h"
#include "LayerSampler.h"
#include "utils/timer.hpp"
#include "rcwa/RCWAScatterMatrix.h"
#include "rcwa/WaveEquationSolver.h"
#include <iostream>
#include <Eigen/Dense>
#include <numbers>

using namespace std;
using namespace Eigen;

DifferentialIntegrator::DifferentialIntegrator(scalar z0, scalar z1, 
    std::shared_ptr<LayerSampler> sampler):
    z0_(z0), z1_(z1), sampler_(sampler)
{
    
}


void DifferentialIntegrator::compute(int N)
{
    int nDim = sampler_->nDim();
    scalar lambda = sampler_->lambda();
    scalar k0 = 2. * Pi / lambda;

    sampler_->sample(z1_, P1_, Q1_);
    MatrixXcs PQ;
    WaveEquationSolver waveEqn_Sol;
    waveEqn_Sol.solve(lambda, P1_, Q1_, PQ, W1_, V1_, gam1_);

    PartialPivLU<MatrixXcs> invW1_Sol;
    invW1_Sol.compute(W1_);
    PartialPivLU<MatrixXcs> invV1_Sol;
    invV1_Sol.compute(V1_);

    Tdd_ = MatrixXcs::Identity(nDim, nDim);
    Tuu_ = MatrixXcs::Identity(nDim, nDim);
    Rud_ = MatrixXcs::Zero(nDim, nDim);
    Rdu_ = MatrixXcs::Zero(nDim, nDim);

    scalar dz = (z1_ - z0_) / static_cast<scalar>(N);

    const MatrixXcs identity = MatrixXcs::Identity(nDim, nDim);
    MatrixXcs L11(nDim, nDim), L12(nDim, nDim),
        L21(nDim, nDim), L22(nDim, nDim);

    MatrixXcs P0, Q0, P1, Q1, P2, Q2;
    sampler_->sample(z0_, P0, Q0);
    for (int i = 0; i < N; ++i) {
        cout << i << endl;
        scalar zl = z0_ + i * dz;
        scalar zr = z0_ + (i+1) * dz;
        scalar zm = 0.5 * (zl + zr);
        sampler_->sample(zm, P1, Q1);
        sampler_->sample(zr, P2, Q2);

        L11 = identity
            -dz*dz/(6.*k0*k0)*(P1*Q0 + P1*Q1 + P2*Q1 - dz*dz/(4.*k0*k0)*P2*Q1*P1*Q0);
        L12 = scalex(0., dz/(6.*k0))
            *(P0 + 4.*P1 + P2-dz*dz/(2.*k0*k0)*(P1*Q1*P0+P2*Q1*P1));
        L21 = scalex(0., dz/(6.*k0))
            *(Q0 + 4.*Q1 + Q2-dz*dz/(2.*k0*k0)*(Q1*P1*Q0+Q2*P1*Q1));
        L22 = identity
            -dz*dz/(6.*k0*k0)*(Q1*P0 + Q1*P1 + Q2*P1 - dz*dz/(4.*k0*k0)*Q2*P1*Q1*P0);
        
        cout << L11.cwiseAbs().maxCoeff() << endl;
        cout << L12.cwiseAbs().maxCoeff() << endl;
        cout << L21.cwiseAbs().maxCoeff() << endl;
        cout << L22.cwiseAbs().maxCoeff() << endl;

        MatrixXcs invW1_L11 = invW1_Sol.solve(L11);
        MatrixXcs invW1_L12 = invW1_Sol.solve(L12);
        MatrixXcs invV1_L21 = invV1_Sol.solve(L21);
        MatrixXcs invV1_L22 = invV1_Sol.solve(L22);

        const MatrixXcs& A11 = 0.5 * (invW1_L11 + invV1_L21);
        const MatrixXcs& A12 = 0.5 * (invW1_L12 + invV1_L22);
        const MatrixXcs& A21 = 0.5 * (invW1_L11 - invV1_L21);
        const MatrixXcs& A22 = 0.5 * (invW1_L12 - invV1_L22);

        const MatrixXcs& T11 = A11 * W1_ + A12 * V1_;
        const MatrixXcs& T12 = A11 * W1_ - A12 * V1_;
        const MatrixXcs& T21 = A21 * W1_ + A22 * V1_;
        const MatrixXcs& T22 = A21 * W1_ - A22 * V1_;


        cout << T11.cwiseAbs().maxCoeff() << endl;
        cout << T12.cwiseAbs().maxCoeff() << endl;
        cout << T21.cwiseAbs().maxCoeff() << endl;
        cout << T22.cwiseAbs().maxCoeff() << endl;

        PartialPivLU<MatrixXcs> tdd_sol(T22);
        MatrixXcs tdd = tdd_sol.inverse();
        MatrixXcs rud = T12 * tdd;
        MatrixXcs rdu = -tdd_sol.solve(T21);
        MatrixXcs tuu = T11 + T12 * rdu;

        cout << tuu.cwiseAbs().maxCoeff() << endl;
        cout << rud.cwiseAbs().maxCoeff() << endl;
        cout << rdu.cwiseAbs().maxCoeff() << endl;
        cout << tdd.cwiseAbs().maxCoeff() << endl;
        MatrixXcs invC1 = - rdu * Rud_;
        invC1.diagonal().array() += 1.0;
        PartialPivLU<MatrixXcs> cycleSolver1(invC1);
        MatrixXcs C1 = cycleSolver1.inverse();
        MatrixXcs C2 = Rud_ * C1 * rdu;
        C2.diagonal().array() += 1.0;
        Rdu_ = Rdu_ + Tdd_ * rdu * C2 * Tuu_;
        Tdd_ = Tdd_ * C1 * tdd;
        Rud_ = rud + tuu * Rud_ * C1 * tdd;
        Tuu_ = tuu * C2 * Tuu_;

        P0 = P2; Q0 = Q2;
    }

    sampler_->sample(z0_, P0_, Q0_);
    waveEqn_Sol.solve(lambda, P0_, Q0_, PQ, W0_, V0_, gam0_);

    const MatrixXcs& invW1_W0 = invW1_Sol.solve(W0_);
    const MatrixXcs& invV1_V0 = invV1_Sol.solve(V0_);

    const MatrixXcs& A = 0.5 * (invW1_W0 + invV1_V0);
    const MatrixXcs& B = 0.5 * (invW1_W0 - invV1_V0);

    PartialPivLU<MatrixXcs> inv_A_minus_Rdu_B;
    inv_A_minus_Rdu_B.compute(A - Rdu_ * B);
    // cout << inv_A_minus_Rdu_B.rcond() << endl;

    Rdu_ = inv_A_minus_Rdu_B.solve(Rdu_ * A - B);
    Tdd_ = inv_A_minus_Rdu_B.solve(Tdd_);
    Rud_ = Tuu_ * B * Tdd_ + Rud_;
    Tuu_ = Tuu_ * (B * Rdu_ + A);
}