#include "VarLayerSampler.h"
#include "rcwa/Layer.h"
#include <iostream>
#include "utils/timer.hpp"

using namespace std;
using namespace Eigen;

VarLayerSampler::VarLayerSampler(
    scalar lambda, 
    int nx, 
    int ny,
    const Eigen::VectorXs& x,
    const Eigen::VectorXs& y,
    const Eigen::MatrixXcs& eps,
    bool enablePML):
    LayerSampler(lambda, nx, ny, enablePML),
    x_(x), y_(y), eps_(eps)
{
    postInitialization();
}

void VarLayerSampler::assignSimulationRegion()
{
    int nCx = x_.size();
    int nCy = y_.size();
    Lx_ = x_(nCx-1) - x_(0);
    Ly_ = y_(nCy-1) - y_(0);

    b0_ = y_(0);
    b1_ = y_(1);
    u0_ = y_(nCy-2);
    u1_ = y_(nCy-1);

    l0_ = x_(0);
    l1_ = x_(1);
    r0_ = x_(nCx-2);
    r1_ = x_(nCx-1);
}


void VarLayerSampler::addXRule(int layout, std::function<scalar(scalar)> rule)
{
    xrules_[layout] = rule;
}

void VarLayerSampler::addYRule(int layout, std::function<scalar(scalar)> rule)
{
    yrules_[layout] = rule;
}

void VarLayerSampler::sample(scalar z, 
    Eigen::MatrixXcs& P,
    Eigen::MatrixXcs& Q)
{
    VectorXs x(x_);
    VectorXs y(y_);
    for (const auto & [i, f] : xrules_) { x(i) = f(z); }
    for (const auto & [i, f] : yrules_) { y(i) = f(z); }
    // cout << "z: " << z << ", " << x.transpose() << endl;
    // cout << "z: " << z << ", " << y.transpose() << endl;
    auto t1 = sploosh::now();
    Layer layer(x, y, eps_);
    auto t2 = sploosh::now();
    layer.waveEqnCoeff(Kx_, Ky_, lambda_, nx_, ny_, P, Q);
    auto t3 = sploosh::now();
    // cout << "sample: " << sploosh::duration_milli_d(t1, t2) << "\t"
    //     << sploosh::duration_milli_d(t2, t3) << endl;
}