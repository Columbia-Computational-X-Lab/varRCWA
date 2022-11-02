#pragma once

#include "defns.h"

class PMLSolver1D
{
private:
	scalar b0_;
	scalar b1_;
	scalar L_;

	scalex gamma_;
public:
	PMLSolver1D(scalar b0, scalar b1, scalar L, scalex gamma):
	b0_(b0), b1_(b1), L_(L), gamma_(gamma) {}

	scalex solve(int n, scalar coeff) const;
};