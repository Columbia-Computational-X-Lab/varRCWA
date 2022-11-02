#pragma once
#include "defns.h"
#include <Eigen/Core>
#include <vector>
#include "Parameterization.h"

enum FourierSolverOption
{
	ContinuousXY = 0,
	DiscontinuousX = 1,
	DiscontinuousY = 2
};


class FourierSolver2D
{
private:
	Eigen::VectorXs coordX_;
	Eigen::VectorXs coordY_;

	scalar dx_;
	scalar dy_;

public:
	FourierSolver2D(const Eigen::VectorXs& coordX,
		const Eigen::VectorXs& coordY):
		coordX_(coordX), coordY_(coordY) {
		dx_ = coordX_(coordX_.size() - 1) - coordX_(0);
		dy_ = coordY_(coordY_.size() - 1) - coordY_(0);
	}

        
	void solve(const Eigen::MatrixXcs& f,
		int nx, int ny,
		FourierSolverOption opt,
		Eigen::MatrixXcs& Ff,
                std::vector<Eigen::MatrixXcs> * dFfs = nullptr,
                const Parameterization& para_x = {}, // the parameterization for x coordinates
                const Parameterization& para_y = {}); // the parameterization for y coordinates
        // notice that here x and y coordinates share the same set of parameters
        // therefore x = a_1 * p_1 + ... + a_k * p_k + x_0
        //           y = b_1 * p_1 + ... + b_k * p_k + y_0

};