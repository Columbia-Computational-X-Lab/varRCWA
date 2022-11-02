#include <complex>

#include "PMLSolver1D.h"
#include "defns.h"
#include "MathFunction.h"

using namespace std;


inline scalex expipi(scalar x) 
{
	return exp(scalex(0, Pi * x));
}

scalex PMLSolver1D::solve(int n, scalar coeff) const
{
	scalar bm = (b1_ - b0_) / L_;
	scalar bp = (b1_ + b0_) / L_;

	scalex result = scalar(4.) * sinc(n*bm - 0.5) * expipi(-(n*bp + coeff * 0.5));
	result += scalar(8.) * expipi(-(n*bp)) * sinc(n*bm);
	result += -scalar(2.) * gamma_ * expipi(-(n*bp)) * sinc(n*bm);
	result += scalar(4.) * sinc(n*bm + 0.5) * expipi(-(n*bp- coeff * 0.5));
	result += gamma_ * sinc(n*bm - 1.) * expipi(-(n*bp+ coeff * 1.));
	result += gamma_ * sinc(n*bm + 1.) * expipi(-(n*bp-coeff * 1.));

	result *= bm / scalar(16.);
	
	return result;
}