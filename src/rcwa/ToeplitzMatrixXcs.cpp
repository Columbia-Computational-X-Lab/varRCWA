#include "ToeplitzMatrixXcs.h"

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

ToeplitzMatrixXcs::ToeplitzMatrixXcs(const Eigen::VectorXcs& vec)
{
	n_ = (vec.size() + 1) / 2;
	data_.resize(2 * n_ - 1);
	for (int i = 0; i < 2 * n_ - 1; ++i)
	{
		int diff = n_ - i - 1;
		data_(i) = vec(n_ + diff - 1); // 2n - i - 2
	}
}

scalex ToeplitzMatrixXcs::operator()(int row, int col) const
{
	return data_(n_ - 1 - (row - col));
}


Eigen::MatrixXcs ToeplitzMatrixXcs::toDense() const
{
	MatrixXcs result(n_, n_);

	for (int i = 0; i < n_; ++i)
	{
		for (int j = 0; j < n_; ++j)
		{
			result(i, j) = (*this)(i, j);
		}
	}

	return result;
}

// need optimization
Eigen::MatrixXcs ToeplitzMatrixXcs::inverse() const
{
	return this->toDense().inverse();
}