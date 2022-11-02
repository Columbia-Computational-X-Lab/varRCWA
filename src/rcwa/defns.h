#pragma once

#include <Eigen/Core>
#include <complex>
#include <vector>

typedef double scalar;
typedef std::complex<scalar> scalex;

namespace Eigen
{
	typedef Eigen::Matrix<scalar, 2, 1> Vector2s;
	typedef Eigen::Matrix<scalar, 3, 1> Vector3s;
	typedef Eigen::Matrix<scalar, Eigen::Dynamic, 1> VectorXs;
	typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

	typedef Eigen::DiagonalMatrix<scalex, Eigen::Dynamic> DiagonalMatrixXcs;
	typedef Eigen::Matrix<scalex, Eigen::Dynamic, 1> VectorXcs;
	typedef Eigen::Matrix<scalex, Eigen::Dynamic, Eigen::Dynamic> MatrixXcs;

	typedef std::vector<Eigen::MatrixXcs> MatrixDerivativeXcs;

}

constexpr scalar Pi = 3.1415926535897932;

enum FieldComponent
{
	Ex = 0,
	Ey = 1,
	Ez = 2,
	Hx = 3,
	Hy = 4,
	Hz = 5
};

enum SliceType
{
	sliceXY = 0,
	sliceXZ = 1,
	sliceYZ = 2,
};

enum SaveOption
{
	modulation = 0,
	realpart = 1,
	imagpart = 2
};

enum ColormapOption
{
	parula = 0,
	hue = 1
};
