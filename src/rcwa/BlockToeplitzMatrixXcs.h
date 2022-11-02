#pragma once

#include "defns.h"
#include <Eigen/Core>


class BlockToeplitzMatrixXcs
{
private:
	// size of matrix should be (nx_ * ny_) * (nx_ * ny_)
	int nx_;
	int ny_;

	// real storage only need (2 * nx_ - 1) * (2 * ny_ - 1)
	// ====================================================
	// Storage
	// ====================================================
	// * * * * * * * * * * * * * * * * 
	// *       *       *       *
	// *       *       *       *
	// *       *       *       *
	// * * * *
	// * 
	// *
	// *
	// * * * *
	// * 
	// *
	// *
	// * * * *
	// * 
	// *
	// *	
	// ====================================================

	Eigen::MatrixXcs data_;

public:
	BlockToeplitzMatrixXcs(int nx, int ny): nx_(nx), ny_(ny)
	{
		data_.resize(2 * nx_ - 1, 2 * ny_ - 1);
	}
	
	BlockToeplitzMatrixXcs(const Eigen::MatrixXcs& mat);
	scalex operator()(int row, int col) const;
	Eigen::MatrixXcs toDense() const;

	Eigen::MatrixXcs operator*(const Eigen::DiagonalMatrixXcs& rhs) const;
};
