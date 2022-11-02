#pragma once
#include "defns.h"
#include <Eigen/Core>


class ToeplitzMatrixXcs
{
private:
	// size of matrix should be (n_, n_)
	int n_;

	// real storage only need (2 * n_ - 1)
	// ====================================================
	// Storage
	// ====================================================
	// * * * * 
	// *      
	// *       
	// *      
	// ====================================================
	Eigen::VectorXcs data_;

public:
	ToeplitzMatrixXcs(int n): n_(n)
	{
		data_.resize(2 * n_ - 1);
	}
	
	ToeplitzMatrixXcs(const Eigen::VectorXcs& vec);
	scalex operator()(int row, int col) const;
	Eigen::MatrixXcs toDense() const;
	Eigen::MatrixXcs inverse() const;
};