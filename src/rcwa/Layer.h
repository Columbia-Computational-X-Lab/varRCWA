#pragma once
#include "defns.h"
#include <Eigen/Core>
#include <vector>
#include "Parameterization.h"


class Layer
{
private:
	Eigen::VectorXs coordX_;
	Eigen::VectorXs coordY_;
	Eigen::MatrixXcs eps_;

	int nCx_;
	int nCy_;


	bool isSolved_;
	Eigen::MatrixXcs eigvecE_;
	Eigen::MatrixXcs eigvecH_;
	Eigen::VectorXcs gamma_;

public:
	Layer(const Eigen::VectorXs& coordX,
		const Eigen::VectorXs& coordY,
		const Eigen::MatrixXcs& eps);
	
	const Eigen::MatrixXcs& eps() const { return eps_; }
	scalex eps(scalar x, scalar y) const;
	
	scalar Lx() const { return coordX_[nCx_-1] - coordX_[0]; }
	scalar Ly() const { return coordY_[nCy_-1] - coordY_[0]; }

	scalar b0() const { return coordY_[0]; }
	scalar b1() const { return coordY_[1]; }
	scalar u0() const { return coordY_[nCy_-2]; }
	scalar u1() const { return coordY_[nCy_-1]; }

	scalar l0() const { return coordX_[0]; }
	scalar l1() const { return coordX_[1]; }
	scalar r0() const { return coordX_[nCx_-2]; }
	scalar r1() const { return coordX_[nCx_-1]; }

	scalar coordX(int layout) const { return coordX_[layout]; }
	scalar coordY(int layout) const { return coordY_[layout]; }
	const Eigen::VectorXs& coordX() const { return coordX_; }
	const Eigen::VectorXs& coordY() const { return coordY_; }
	void setCoordX(const Eigen::VectorXs& coordX) { coordX_ = coordX; isSolved_ = false; }
	void setCoordY(const Eigen::VectorXs& coordY) { coordY_ = coordY; isSolved_ = false; }

	void setCoordX(int layout, scalar coordX) { coordX_[layout] = coordX; isSolved_ = false; }
	void setCoordY(int layout, scalar coordY) { coordY_[layout] = coordY; isSolved_ = false; }
	
	void setEps(int row, const Eigen::VectorXcs& reps) { eps_.row(row) = reps; isSolved_ = false; }
	int nCx() const { return nCx_; }
	int nCy() const { return nCy_; }

	bool isSolved() const { return isSolved_; }

    void waveEqnCoeff(const Eigen::MatrixXcs& Kx, 
        const Eigen::MatrixXcs& Ky, 
        scalar lambda, 
        int nx, 
        int ny,
        Eigen::MatrixXcs& P,
        Eigen::MatrixXcs& Q,
        std::vector<Eigen::MatrixXcs>* dP = nullptr,
        std::vector<Eigen::MatrixXcs>* dQ = nullptr,
        const Parameterization& para_x = {},
        const Parameterization& para_y = {},
        Eigen::MatrixXcs * dPl = nullptr,
        Eigen::MatrixXcs * dQl = nullptr);
        
	void solve(const Eigen::MatrixXcs& Kx, 
        const Eigen::MatrixXcs& Ky, 
        scalar lambda, 
        int nx, 
        int ny);

	void permuteEigVecX(
		Eigen::MatrixXcs& eigvecE,
		Eigen::MatrixXcs& eigvecH,
		scalar dx, 
		int nx, 
		int ny);

	void dpermuteEigVecX(
		Eigen::MatrixXcs& deigvecE,
		Eigen::MatrixXcs& deigvecH,
		scalar dx,
		int nx, 
		int ny);
    
	const Eigen::VectorXcs& gamma() const { return gamma_; }
	const Eigen::MatrixXcs& eigvecE() const { return eigvecE_; }
	const Eigen::MatrixXcs& eigvecH() const { return eigvecH_; }

	void setSolutions(
		const Eigen::MatrixXcs& eigvecE,
		const Eigen::MatrixXcs& eigvecH,
		const Eigen::VectorXcs& gamma);
	
	void generatePlaneWave(
		const Eigen::VectorXcs& delta,
		scalar px,
		scalar py,
		Eigen::VectorXcs& c);
	
	void getHarmonics(
		const Eigen::VectorXcs& c,
		Eigen::VectorXcs& harmonics);
};
