#include "RCWAIntegrator.h"
#include "LayerSampler.h"
#include "utils/timer.hpp"
#include "rcwa/RCWAScatterMatrix.h"
#include "rcwa/WaveEquationSolver.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

RCWAIntegrator::RCWAIntegrator(scalar z0, scalar z1, 
    std::shared_ptr<LayerSampler> sampler):
    z0_(z0), z1_(z1), sampler_(sampler)
{
    
}


void RCWAIntegrator::compute(int N)
{
    int nDim = sampler_->nDim();
    scalar lambda = sampler_->lambda();

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

    for (int i = 0; i < N; ++i) {
        cout << i << endl;
        scalar z = z0_ + (i+0.5) * dz; //midpoint division
        // scalar z = z0_ + i * dz; // endpoint division
        MatrixXcs P, Q;
        sampler_->sample(z, P, Q);
        dtmm::RCWAScatterMatrix rcwa;
        rcwa.compute(lambda, dz, P, Q, W1_, V1_, gam1_);

        const MatrixXcs& tdd = rcwa.Tdd();
        const MatrixXcs& rdu = rcwa.Rdu();
        const MatrixXcs& tuu = rcwa.Tuu();
        const MatrixXcs& rud = rcwa.Rud();

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
    }

    sampler_->sample(z0_, P0_, Q0_);
    waveEqn_Sol.solve(sampler_->lambda(), P0_, Q0_, PQ, W0_, V0_, gam0_);    

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