#include "RedhefferIntegrator.h"
#include "LayerSampler.h"
#include "utils/timer.hpp"
#include "rcwa/WaveEquationSolver.h"

#include <limits>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;

RedhefferIntegrator::RedhefferIntegrator(scalar z0, scalar z1, 
    std::shared_ptr<LayerSampler> sampler):
    z0_(z0), z1_(z1), sampler_(sampler), 
    P0_(sampler->nDim(), sampler->nDim()),
    Q0_(sampler->nDim(), sampler->nDim()),
    W0_(sampler->nDim(), sampler->nDim()),
    V0_(sampler->nDim(), sampler->nDim()),
    P1_(sampler->nDim(), sampler->nDim()),
    Q1_(sampler->nDim(), sampler->nDim()),
    W1_(sampler->nDim(), sampler->nDim()),
    V1_(sampler->nDim(), sampler->nDim()),
    gam0_(sampler->nDim()),
    gam1_(sampler->nDim()),
    PQ_(sampler->nDim(), sampler->nDim()),
    leftW_(sampler->nDim(), sampler->nDim()),
    leftV_(sampler->nDim(), sampler->nDim()),
    invsol_(sampler->nDim()),
    Tdd_(sampler->nDim(), sampler->nDim()),
    Tuu_(sampler->nDim(), sampler->nDim()),
    Rud_(sampler->nDim(), sampler->nDim()),
    Rdu_(sampler->nDim(), sampler->nDim()),
    tdd_(sampler->nDim(), sampler->nDim()),
    tuu_(sampler->nDim(), sampler->nDim()),
    rud_(sampler->nDim(), sampler->nDim()),
    rdu_(sampler->nDim(), sampler->nDim()),
    order_(0), nEigVal_(0), N_(0)
{
}

// if 0, triple division
// if 1, double division
#if 0
void RedhefferIntegrator::compute(int p, int maxN, scalar pe)
{
    order_ = p;
    int nDim = sampler_->nDim();
    scalar lambda = sampler_->lambda();

    MatrixXcs PQ(nDim, nDim);
    WaveEquationSolver waveEqn_Sol;
    sampler_->sample(z0_, P0_, Q0_);
    waveEqn_Sol.solve(lambda, P0_, Q0_, PQ, W0_, V0_, gam0_);
    sampler_->sample(z1_, P1_, Q1_);
    waveEqn_Sol.solve(lambda, P1_, Q1_, PQ, W1_, V1_, gam1_);
    nEigVal_ = 2;
    N_ = 0;

    PartialPivLU<MatrixXcs> invWSol(W0_);
    PartialPivLU<MatrixXcs> invVSol(V0_);

    scalar zm = 0.5 * (z0_ + z1_);
    MatrixXcs Pm, Qm;
    sampler_->sample(zm, Pm, Qm);

    Tdd_.setIdentity();
    Tuu_.setIdentity();
    Rud_.setZero();
    Rdu_.setZero();

    leftW_ = W0_;
    leftV_ = V0_;

    recursive_solve(P0_, Q0_, W0_, V0_, invWSol, invVSol, gam0_,
      z0_, z1_, pe,
      P0_, Pm, P1_,
      Q0_, Qm, Q1_,
      &N_, &nEigVal_);

    // right projection
    invWSol.compute(leftW_);
    invVSol.compute(leftV_);
    const auto& invW0_W = invWSol.solve(W1_);
    const auto& invV0_V = invVSol.solve(V1_);
    const auto& A = 0.5 * (invW0_W + invV0_V);
    const auto& B = 0.5 * (invW0_W - invV0_V);

    invsol_.compute(A - Rud_ * B);
    Tuu_ = invsol_.solve(Tuu_);
    Rud_ = invsol_.solve(Rud_ * A - B);
    Rdu_ = Rdu_ + Tdd_ * B * Tuu_;
    Tdd_ = Tdd_ * (B * Rud_ + A);
}

void RedhefferIntegrator::recursive_solve(
  const Eigen::MatrixXcs& Pr,
  const Eigen::MatrixXcs& Qr,
  const Eigen::MatrixXcs& W,
  const Eigen::MatrixXcs& V,
  Eigen::PartialPivLU<Eigen::MatrixXcs>& invWSol,
  Eigen::PartialPivLU<Eigen::MatrixXcs>& invVSol,
  const Eigen::VectorXcs& gam,
  scalar z0, scalar z1, scalar pe,
  const Eigen::MatrixXcs& P0,
  const Eigen::MatrixXcs& Pm,
  const Eigen::MatrixXcs& P1,
  const Eigen::MatrixXcs& Q0,
  const Eigen::MatrixXcs& Qm,
  const Eigen::MatrixXcs& Q1,
  int *N, int *nEigVal) {
  scalar lambda = sampler_->lambda();
  scalar diffError {0.};
  evaluatePerturbationAndDifference(Pr, Qr, W, V, invWSol, invVSol,
    gam, z0, z1, P0, Pm, P1, Q0, Qm, Q1,
    tdd_, tuu_, rud_, rdu_, &diffError);
  if (diffError <= pe) {
    if (N != nullptr) {
      ++(*N);
    }
    // projection
    const auto& invW_W0 = invWSol.solve(leftW_);
    const auto& invV_V0 = invVSol.solve(leftV_);
    const auto& A = 0.5 * (invW_W0 + invV_V0);
    const auto& B = 0.5 * (invW_W0 - invV_V0);

    invsol_.compute(A - rdu_ * B);
    rdu_ = invsol_.solve(rdu_ * A - B);
    tdd_ = invsol_.solve(tdd_);
    rud_ = tuu_ * B * tdd_ + rud_;
    tuu_ = tuu_ * A + tuu_ * B * rdu_;

    leftW_ = W;
    leftV_ = V;
    // redheffer
    MatrixXcs invC1 = - rdu_ * Rud_;
    invC1.diagonal().array() += 1.0;
    invsol_.compute(invC1);
    const MatrixXcs& C1 = invsol_.inverse();
    MatrixXcs C2 = Rud_ * C1 * rdu_;
    C2.diagonal().array() += 1.0;

    Rdu_ = Rdu_ + Tdd_ * rdu_ * C2 * Tuu_;
    Tdd_ = Tdd_ * C1 * tdd_;
    Rud_ = rud_ + tuu_ * Rud_ * C1 * tdd_;
    Tuu_ = tuu_ * C2 * Tuu_;
  } else {
    scalar zm = 0.5 * (z0 + z1);
    cout << "divide at " << zm << endl;
    MatrixXcs Pt, Qt;
    sampler_->sample(0.5*(z0 + zm), Pt, Qt);
    recursive_solve(Pr, Qr, W, V, invWSol, invVSol, 
      gam, z0, zm, pe, P0, Pt, Pm,
      Q0, Qt, Qm, N, nEigVal);
    sampler_->sample(0.5*(zm + z1), Pt, Qt);

    MatrixXcs Wm, Vm;
    VectorXcs gamm;
    WaveEquationSolver waveEqn_Sol;
    waveEqn_Sol.solve(lambda, Pm, Qm, PQ_, Wm, Vm, gamm);

    if (nEigVal != nullptr) {
      ++(*nEigVal);
    }

    PartialPivLU<MatrixXcs> invWmSol(Wm);
    PartialPivLU<MatrixXcs> invVmSol(Vm);
    recursive_solve(Pm, Qm, Wm, Vm, invWmSol, invVmSol, gamm,
      zm, z1, pe, Pm, Pt, P1,
      Qm, Qt, Q1, N, nEigVal);
  } 
}
#else
void RedhefferIntegrator::compute(int p, int maxN, scalar pe)
{
    order_ = p;
    int nDim = sampler_->nDim();
    scalar lambda = sampler_->lambda();

    MatrixXcs PQ(nDim, nDim);
    WaveEquationSolver waveEqn_Sol;
    sampler_->sample(z0_, P0_, Q0_);
    waveEqn_Sol.solve(lambda, P0_, Q0_, PQ, W0_, V0_, gam0_);
    sampler_->sample(z1_, P1_, Q1_);
    waveEqn_Sol.solve(lambda, P1_, Q1_, PQ, W1_, V1_, gam1_);
    nEigVal_ = 2;
    N_ = 0;

    scalar zm = 0.5 * (z0_ + z1_);
    MatrixXcs Pm, Qm, Wm, Vm;
    VectorXcs gamm;
    sampler_->sample(zm, Pm, Qm);
    waveEqn_Sol.solve(lambda, Pm, Qm, PQ, Wm, Vm, gamm);
    PartialPivLU<MatrixXcs> invWSol(Wm);
    PartialPivLU<MatrixXcs> invVSol(Vm);

    Tdd_.setIdentity();
    Tuu_.setIdentity();
    Rud_.setZero();
    Rdu_.setZero();

    leftW_ = W0_;
    leftV_ = V0_;

    recursive_solve(Pm, Qm, Wm, Vm, invWSol, invVSol, gamm,
      z0_, z1_, pe,
      P0_, Pm, P1_,
      Q0_, Qm, Q1_,
      &N_, &nEigVal_);

    // right projection
    invWSol.compute(leftW_);
    invVSol.compute(leftV_);
    const auto& invW0_W = invWSol.solve(W1_);
    const auto& invV0_V = invVSol.solve(V1_);
    const auto& A = 0.5 * (invW0_W + invV0_V);
    const auto& B = 0.5 * (invW0_W - invV0_V);

    invsol_.compute(A - Rud_ * B);
    Tuu_ = invsol_.solve(Tuu_);
    Rud_ = invsol_.solve(Rud_ * A - B);
    Rdu_ = Rdu_ + Tdd_ * B * Tuu_;
    Tdd_ = Tdd_ * (B * Rud_ + A);
}

void RedhefferIntegrator::recursive_solve(
  const Eigen::MatrixXcs& Pr,
  const Eigen::MatrixXcs& Qr,
  const Eigen::MatrixXcs& W,
  const Eigen::MatrixXcs& V,
  Eigen::PartialPivLU<Eigen::MatrixXcs>& invWSol,
  Eigen::PartialPivLU<Eigen::MatrixXcs>& invVSol,
  const Eigen::VectorXcs& gam,
  scalar z0, scalar z1, scalar pe,
  const Eigen::MatrixXcs& P0,
  const Eigen::MatrixXcs& Pm,
  const Eigen::MatrixXcs& P1,
  const Eigen::MatrixXcs& Q0,
  const Eigen::MatrixXcs& Qm,
  const Eigen::MatrixXcs& Q1,
  int *N, int *nEigVal) {
  scalar lambda = sampler_->lambda();
  scalar diffError {0.};
  evaluatePerturbationAndDifference(Pr, Qr, W, V, invWSol, invVSol,
    gam, z0, z1, P0, Pm, P1, Q0, Qm, Q1,
    tdd_, tuu_, rud_, rdu_, &diffError);

  if (diffError <= pe) {
    if (N != nullptr) {
      ++(*N);
    }
    // projection
    const auto& invW_W0 = invWSol.solve(leftW_);
    const auto& invV_V0 = invVSol.solve(leftV_);
    const auto& A = 0.5 * (invW_W0 + invV_V0);
    const auto& B = 0.5 * (invW_W0 - invV_V0);

    invsol_.compute(A - rdu_ * B);
    rdu_ = invsol_.solve(rdu_ * A - B);
    tdd_ = invsol_.solve(tdd_);
    rud_ = tuu_ * B * tdd_ + rud_;
    tuu_ = tuu_ * A + tuu_ * B * rdu_;

    leftW_ = W;
    leftV_ = V;
    // redheffer
    MatrixXcs invC1 = - rdu_ * Rud_;
    invC1.diagonal().array() += 1.0;
    invsol_.compute(invC1);
    const MatrixXcs& C1 = invsol_.inverse();
    MatrixXcs C2 = Rud_ * C1 * rdu_;
    C2.diagonal().array() += 1.0;

    Rdu_ = Rdu_ + Tdd_ * rdu_ * C2 * Tuu_;
    Tdd_ = Tdd_ * C1 * tdd_;
    Rud_ = rud_ + tuu_ * Rud_ * C1 * tdd_;
    Tuu_ = tuu_ * C2 * Tuu_;
  } else {
    scalar zm0 = 2. / 3. * z0 + 1. / 3. * z1;
    scalar zm1 = 1. / 3. * z0 + 2. / 3. * z1;
    cout << "divide at " << zm0 << ", " << zm1 << endl;

    MatrixXcs Pt, Qt, Pk, Qk;
    sampler_->sample(0.5*(z0 + zm0), Pt, Qt);
    sampler_->sample(zm0, Pk, Qk);

    MatrixXcs Wt, Vt;
    VectorXcs gamt;
    PartialPivLU<MatrixXcs> invWtSol, invVtSol;

    WaveEquationSolver waveEqnSol;
    waveEqnSol.solve(lambda, Pt, Qt, PQ_, Wt, Vt, gamt);
    invWtSol.compute(Wt);
    invVtSol.compute(Vt);

    recursive_solve(Pt, Qt, Wt, Vt, invWtSol, invVtSol, gamt,
      z0, zm0, pe, P0, Pt, Pk,
      Q0, Qt, Qk, N, nEigVal);

    sampler_->sample(zm1, Pt, Qt);
    recursive_solve(Pr, Qr, W, V, invWSol, invVSol, gam,
      zm0, zm1, pe, Pk, Pm, Pt,
      Qk, Qm, Qt, N, nEigVal);

    sampler_->sample(0.5*(zm1 + z1), Pk, Qk);
    waveEqnSol.solve(lambda, Pk, Qk, PQ_, Wt, Vt, gamt);
    invWtSol.compute(Wt);
    invVtSol.compute(Vt);

    recursive_solve(Pk, Qk, Wt, Vt, invWtSol, invVtSol, gamt,
      zm1, z1, pe, Pt, Pk, P1,
      Qt, Qk, Q1, N, nEigVal);
  }
}
#endif

void RedhefferIntegrator::evaluatePerturbationAndDifference(
  const Eigen::MatrixXcs& Pr,
  const Eigen::MatrixXcs& Qr,
  const Eigen::MatrixXcs& W,
  const Eigen::MatrixXcs& V,
  const Eigen::PartialPivLU<Eigen::MatrixXcs>& invWSol,
  const Eigen::PartialPivLU<Eigen::MatrixXcs>& invVSol,
  const Eigen::VectorXcs& gam,
  scalar z0, scalar z1,
  const Eigen::MatrixXcs& P0,
  const Eigen::MatrixXcs& Pm,
  const Eigen::MatrixXcs& P1,
  const Eigen::MatrixXcs& Q0,
  const Eigen::MatrixXcs& Qm,
  const Eigen::MatrixXcs& Q1,
  Eigen::MatrixXcs& tdd,
  Eigen::MatrixXcs& tuu,
  Eigen::MatrixXcs& rud,
  Eigen::MatrixXcs& rdu,
  scalar * diffError) {
  if (diffError != nullptr) {
    *diffError = 0.;
  }
  // auto t1 = sploosh::now();
    
  scalar k0 = 2. * Pi / sampler_->lambda();
  MatrixXcs invW_dP_V = invWSol.solve((P0 - Pr)*V);
  MatrixXcs invV_dQ_W = invVSol.solve((Q0 - Qr)*W);

  const MatrixXcs& dA0 = invW_dP_V + invV_dQ_W;
  const MatrixXcs& dB0 = invW_dP_V - invV_dQ_W;

  invW_dP_V = invWSol.solve((Pm - Pr)*V);
  invV_dQ_W = invVSol.solve((Qm - Qr)*W); 

  const MatrixXcs& dA1 = invW_dP_V + invV_dQ_W;
  const MatrixXcs& dB1 = invW_dP_V - invV_dQ_W;

  invW_dP_V = invWSol.solve((P1 - Pr)*V);
  invV_dQ_W = invVSol.solve((Q1 - Qr)*W);

  const MatrixXcs& dA2 = invW_dP_V + invV_dQ_W;
  const MatrixXcs& dB2 = invW_dP_V - invV_dQ_W;

  VectorXcs ph = gam.unaryExpr([&k0, &z0, &z1](const scalex& g) 
        { return exp(scalex(0., 0.5 * (z1 - z0) / k0) * g); });
  VectorXcs pf = gam.unaryExpr([&k0, &z0, &z1](const scalex& g) 
        { return exp(scalex(0., (z1 - z0) / k0) * g); });

  // to evaluate the integral, we use Simpson's rule
  // int f dx = 1/6 * (f(0) + 4*f(1/2) + f(1)) * Dx
  
  tuu.noalias() = scalex(0., (z1 - z0) / (6*2*k0)) 
    * (pf.asDiagonal() * dA0 
      + 4. * ph.asDiagonal() * dA1 * ph.asDiagonal() 
      + dA2 * pf.asDiagonal());
  if (diffError != nullptr) {
    *diffError = max(*diffError, tuu.cwiseAbs().maxCoeff());
  }
  tuu.diagonal() += pf;

  rud.noalias() = scalex(0., -(z1 - z0) / (6*2*k0)) 
    * (pf.asDiagonal() * dB0 * pf.asDiagonal()
       + 4. * ph.asDiagonal() * dB1 * ph.asDiagonal() + dB2);
  if (diffError != nullptr) {
    *diffError = max(*diffError, rud.cwiseAbs().maxCoeff());
  }
  rdu.noalias() = scalex(0., -(z1 - z0) / (6*2*k0))
    * (dB0 + 4. * ph.asDiagonal() * dB1 * ph.asDiagonal() 
        + pf.asDiagonal() * dB2 * pf.asDiagonal());
  if (diffError != nullptr) {
    *diffError = max(*diffError, rdu.cwiseAbs().maxCoeff());
  }
  tdd.noalias() = scalex(0., (z1 - z0) / (6*2*k0)) 
    * (dA0 * pf.asDiagonal() + 4. * ph.asDiagonal() * dA1 * ph.asDiagonal() 
     + pf.asDiagonal() * dA2);
  if (diffError != nullptr) {
    *diffError = max(*diffError, tdd.cwiseAbs().maxCoeff());
  }
  tdd.diagonal() += pf;
  // auto t2 = sploosh::now();
  // cout << "time: " << sploosh::duration_milli_d(t1, t2) << endl;
}
