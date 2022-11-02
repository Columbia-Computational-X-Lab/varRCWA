#ifndef RCWA_SCATTERING_MATRIX_INC
#   define RCWA_SCATTERING_MATRIX_INC

#include <Eigen/Dense>
#include "defns.h"

namespace dtmm 
{

class RCWAScatterMatrix
{
    public:
        typedef Eigen::MatrixXcs                MatrixType;
        typedef Eigen::VectorXcs                VectorType;
        typedef Eigen::PartialPivLU<MatrixType> SolverType;

        // RCWAScatterMatrix(int nx, int ny, scalar Lx, scalar Ly):
        //         nx_(nx), ny_(ny), Lx_(Lx), Ly_(Ly) { }
        RCWAScatterMatrix() {}

        /*
         * Compute scattering matrix and store all the intermediate 
         * matrices needed for derivative computation
         */
        void compute(scalar lambda, scalar wz, 
                     const MatrixType& P,
                     const MatrixType& Q,
                     const MatrixType& W0,
                     const MatrixType& V0,
                     const MatrixType& gamma0);

        void compute_v2(scalar lambda, scalar wz,
                    const MatrixType& W,
                    const MatrixType& V,
                    const MatrixType& gamma,
                    const MatrixType& W0,
                    const MatrixType& V0,
                    const MatrixType& gamma0);
        
        inline const MatrixType& Tuu() const
        {   return Tuu_; }
        inline const MatrixType& Rud() const
        {   return Rud_; }
        inline const MatrixType& Rdu() const
        {   return Rud_; }
        inline const MatrixType& Tdd() const
        {   return Tuu_; }
        inline const MatrixType& W() const
        {   return W_; }
        inline const MatrixType& V() const
        {   return V_; }
        inline const MatrixType& PQ() const
        {   return PQ_; }
        inline const MatrixType& F0() const
        {   return F0_; }
        inline const MatrixType& G0() const
        {   return G0_; }
        inline const MatrixType& W0() const
        {   return W0_; }
        inline const MatrixType& V0() const
        {   return V0_; }
        inline scalar Lz_normalized() const
        {   return normalizedLz_; }

        inline const VectorType& Gamma() const
        {   return gamma_; }
        inline const VectorType& X() const
        {   return X_; }

        inline const MatrixType& A() const
        {   return A_; }
        inline const MatrixType& B() const
        {   return B_; }
        
        inline const MatrixType& invAB() const
        {   return invAB_; }
        inline const MatrixType& invAXA() const
        {   return invAXA_; }
        inline const MatrixType& invAXB() const
        {   return invAXB_; }
        inline const MatrixType& I_minus_invAXB2() const
        {   return I_minus_invAXB2_; }
        inline const MatrixType& invAXB_invAXA_invAB() const
        {   return invAXB_invAXA_invAB_; }
        inline const MatrixType& invAXA_invAXB_invAB() const
        {   return invAXA_invAXB_invAB_; }
        // inline const MatrixType& inv_I_minus_invAXB2() const
        // {   return inv_I_minus_invAXB2_; }

        inline const SolverType& invW_sol() const  // W^{-1}
        {   return invWSol_; }
        inline const SolverType& invV_sol() const 
        {   return invVSol_; }
        inline const SolverType& invA_sol() const
        {   return invASol_; }

        inline const SolverType& inv_I_minus_invAXB2_sol() const
        {   return inv_I_minus_invAXB2_Sol_; }

    private:
	// // assume no PML layer
	// void evaluateHomogeneousLayer(
	// 	scalar lambda, scalex eps,
	// 	MatrixType& eigvecE,
	// 	MatrixType& eigvecH,
	// 	VectorType& keff);

    private:
        scalar  normalizedLz_;

        MatrixType Tuu_, Rud_;
        MatrixType W0_, V0_; 
        MatrixType PQ_, W_, V_;
        MatrixType F0_, G0_;
        MatrixType A_, B_;
        MatrixType invAB_, invAXB_, invAXA_;
        MatrixType I_minus_invAXB2_;
        // MatrixType inv_I_minus_invAXB2_;

        MatrixType invAXB_invAXA_invAB_;
        MatrixType invAXA_invAXB_invAB_;

        VectorType gamma_, X_;

        SolverType invWSol_;
        SolverType invVSol_;
        SolverType inv_I_minus_invAXB2_Sol_;   // I - (invA*X*B)^2
        SolverType invASol_;

    //public:
    //    MatrixType ret_;    // for sanity check
};

}

#endif
