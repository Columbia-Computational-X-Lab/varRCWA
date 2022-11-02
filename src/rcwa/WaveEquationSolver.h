#include "defns.h"

class WaveEquationSolver
{
public:
        WaveEquationSolver() {}
    
        void solve(
            scalar lambda,
            const Eigen::MatrixXcs& P, 
            const Eigen::MatrixXcs& Q,
            Eigen::MatrixXcs& PQ,
            Eigen::MatrixXcs& eigvecE,
            Eigen::MatrixXcs& eigvecH,
            Eigen::VectorXcs& keff) const;
};