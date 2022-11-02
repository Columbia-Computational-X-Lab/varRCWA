#pragma once
#include "rcwa/defns.h"
#include <memory>
#include <utility>

class LayerSampler;

class RCWAIntegrator
{
private:
    scalar z0_;
    scalar z1_;
    std::shared_ptr<LayerSampler> sampler_;

    Eigen::MatrixXcs P0_, Q0_, W0_, V0_;
    Eigen::MatrixXcs P1_, Q1_, W1_, V1_;
    Eigen::VectorXcs gam0_, gam1_;

    Eigen::MatrixXcs Tdd_;
    Eigen::MatrixXcs Tuu_;
    Eigen::MatrixXcs Rud_;
    Eigen::MatrixXcs Rdu_;

public:
    RCWAIntegrator(scalar z0, scalar z1, std::shared_ptr<LayerSampler> sampler);
    void compute(int N);

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
      
};
