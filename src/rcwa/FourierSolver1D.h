#pragma once
#include "defns.h"
#include <Eigen/Core>

#include "Parameterization.h"

class FourierSolver1D
{
private:
    Eigen::VectorXs x_;
	scalar d_;
    
public:
	FourierSolver1D(const Eigen::VectorXs& x, scalar d): 
	x_(x), d_(d) { }
        
	void solve(const Eigen::VectorXcs& f,
		int M,
		Eigen::VectorXcs& Ff);

        void solve_v2(const Eigen::VectorXcs& f,
                int M,
                Eigen::VectorXcs& Ff);

        void dSolve(const Eigen::VectorXcs& f,
                int M,
                int layout,
                Eigen::VectorXcs& dFf);
                
        void dSolve_v2(const Eigen::VectorXcs& f,
                int M,
                int layout,
                Eigen::VectorXcs& dFf);
        
        /**
         * @param dFf the gradient of f
         *            each col w. r. t. each parameter
         */
        void dSolve(const Eigen::VectorXcs& f,
                int M,
                const Parameterization& parameterization,
                Eigen::MatrixXcs& dFfs,
                int nPara = -1);

        void dSolve_v2(const Eigen::VectorXcs& f,
                int M,
                const Parameterization& parameterization,
                Eigen::MatrixXcs& dFfs,
                int nPara = -1);
        // when nPara is larger than number of parameters
        // set the remain parameters to be zero

        void reorder(Eigen::VectorXi& index);

        void refill(const Eigen::VectorXcs& f,
                const Eigen::VectorXi& rindex,
                Eigen::VectorXcs& newf);
                
};
