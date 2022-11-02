#pragma once
#include "rcwa/defns.h"
#include <memory>
#include <utility>
#include <Eigen/Dense>

class LayerSampler;
class SolverInterface;

class RedhefferIntegrator
{
private:
    scalar z0_;
    scalar z1_;
    std::shared_ptr<LayerSampler> sampler_;

    Eigen::MatrixXcs P0_, Q0_, W0_, V0_;
    Eigen::MatrixXcs P1_, Q1_, W1_, V1_;
    Eigen::VectorXcs gam0_, gam1_;

    Eigen::MatrixXcs PQ_;

    Eigen::MatrixXcs leftW_, leftV_;
    Eigen::PartialPivLU<Eigen::MatrixXcs> invsol_;

    Eigen::MatrixXcs Tdd_, Tuu_, Rud_, Rdu_;
    Eigen::MatrixXcs tdd_, tuu_, rud_, rdu_;

    // logging
    int order_;
    int nEigVal_;
    int N_;
public:
    RedhefferIntegrator(scalar z0, scalar z1,
      std::shared_ptr<LayerSampler> sampler);
    void compute(int p, int maxN = 10, scalar pe = 1e-2);
    
    int N() const { return N_; }
    int nEigVal() const { return nEigVal_; }
    const Eigen::MatrixXcs& Tdd() const { return Tdd_; }
    const Eigen::MatrixXcs& Tuu() const { return Tuu_; }
    const Eigen::MatrixXcs& Rud() const { return Rud_; }
    const Eigen::MatrixXcs& Rdu() const { return Rdu_; }

    const Eigen::MatrixXcs& W0() const { return W0_; }
    const Eigen::MatrixXcs& V0() const { return V0_; }
    const Eigen::VectorXcs& gam0() const { return gam0_; }
    const Eigen::MatrixXcs& W1() const { return W1_; }
    const Eigen::MatrixXcs& V1() const { return V1_; }
    const Eigen::VectorXcs& gam1() const { return gam1_; }

private:
    void recursive_solve(
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
      int *N = nullptr, int *nEigVal = nullptr);

  void evaluatePerturbationAndDifference(
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
    scalar * diffError = nullptr);
};
