#include "FourierSolver1D.h"

#include <iostream>
#include <algorithm>

#include "MathFunction.h"

using namespace std;
using namespace Eigen;

void FourierSolver1D::solve(const Eigen::VectorXcs& f,
	int M,
	Eigen::VectorXcs& Ff)
{
	if (f.size() != x_.size() - 1)
	{
		std::cerr << "The function configuration does not match the coordinates!\n";
	}

	Ff.resize(M);
	Ff.setZero();

	for (int i = 0; i < M; ++i)
	{
		int m = i - (M-1) / 2;
		for (int j = 1; j < x_.size(); ++j)
		{
			scalar x1 = x_(j);
			scalar x0 = x_(j-1);
			if (m == 0)
			{
				Ff(i) += f(j-1) / d_ * (x1 - x0);
			}
			else
			{
				Ff(i) += f(j-1) / d_ * exp(-scalex(0, m*Pi*(x1 + x0)/d_))
				* sinc(m * (x1 - x0) / d_) * (x1 - x0);			
			}
		}
	}
}

// assume the first and the last coordinate is not moving
void FourierSolver1D::solve_v2(const Eigen::VectorXcs& f,
        int M,
        Eigen::VectorXcs& Ff)
{
	if (f.size() != x_.size() - 1)
	{
		std::cerr << "The function configuration does not match the coordinates!\n";
	}

        VectorXi index;
        reorder(index);
        VectorXi rindex(index.size());
        for (int i = 0; i < index.size(); ++i)
        {
                rindex(index(i)) = i;
        }

        VectorXcs newf;
        refill(f, rindex, newf);

        // cout << newf.transpose() << endl;

	Ff.resize(M);
	Ff.setZero();

	for (int i = 0; i < M; ++i)
	{
		int m = i - (M-1) / 2;
		for (int j = 1; j < x_.size(); ++j)
		{
			scalar x1 = x_(index(j));
			scalar x0 = x_(index(j-1));
                        scalex eps = newf(j-1);

			if (m == 0)
			{
				Ff(i) += eps / d_ * (x1 - x0);
			}
			else
			{
				Ff(i) += eps / d_ 
                                * exp(-scalex(0, m*Pi*(x1 + x0)/d_))
				* sinc(m * (x1 - x0) / d_) * (x1 - x0);			
			}
		}
	}        
}


void FourierSolver1D::dSolve(
        const Eigen::VectorXcs& f,
	int M,
	int layout,
	Eigen::VectorXcs& dFf)
{
	if (f.size() != x_.size() - 1)
	{
		cerr << "[ERROR]: The function configuration does not match the coordinates!\n";
	}

	if (layout < 0 || layout > x_.size() - 1)
	{
		cerr << "[ERROR]: invalid layout!\n";
	}


	dFf.resize(M);
	dFf.setZero();

	for (int i = 0; i < M; ++i)
	{
		int m = i - (M-1) / 2;

		if (m == 0)
		{
			if (layout > 0)
			{
				dFf(i) += f(layout-1) / d_;
				
			}
			if (layout < x_.size() - 1)
			{
				dFf(i) += -f(layout) / d_;
			}
		}
		else
		{
			if (layout > 0)
			{
				dFf(i) += f(layout-1) / d_ * exp(-scalex(0, 2*m*Pi*x_(layout)/d_));
			}

			if (layout < x_.size() - 1)
			{
				dFf(i) += -f(layout) / d_ * exp(-scalex(0, 2*m*Pi*x_(layout)/d_));				
			}

		}
	}	
}

void FourierSolver1D::reorder(Eigen::VectorXi& index)
{
        index.resize(x_.size());
        for (int i = 0; i < x_.size(); ++i)
        {
                index(i) = i;
        }
        sort(index.data(), index.data() + index.size(), 
                [this](int i1, int i2) {return x_(i1) < x_(i2); });
}

void FourierSolver1D::refill(
        const Eigen::VectorXcs& f,
        const Eigen::VectorXi& rindex,
        Eigen::VectorXcs& newf)
{
        // check if inverse order exists
        bool needRefill = false;
        for (int i = 0; i < rindex.size()-1; ++i)
        {
                if (rindex(i) != i) 
                {
                        needRefill = true;
                        break;
                }
        }

        newf.resize(f.size());
        newf = f;

        // refill material
        // following painter's order
        if (needRefill)
        {
                for (int i = 0; i < f.size(); ++i)
                {
                        // only fill with correct order
                        if (x_(i) >= x_(i+1)) continue;
                        for (int j = 0; j < newf.size(); ++j)
                        {
                                if (rindex(i) <= j && 
                                    rindex(i+1) >= j+1)
                                {
                                        newf(j) = f(i);
                                }
                        }
                }
        }    
}

void FourierSolver1D::dSolve_v2(const Eigen::VectorXcs& f,
        int M,
        int layout,
        Eigen::VectorXcs& dFf)
{
	if (f.size() != x_.size() - 1)
	{
		cerr << "[ERROR]: The function configuration does not match the coordinates!\n";
	}

	if (layout <= 0 || layout >= x_.size() - 1)
	{
		cerr << "[ERROR]: invalid layout!\n";
	}

        VectorXi index;
        reorder(index);
        VectorXi rindex(x_.size());
        for (int i = 0; i < index.size(); ++i)
        {
                rindex(index(i)) = i;
        }

        VectorXcs newf;
        refill(f, rindex, newf);

        //cout << index.transpose() << endl;
        //cout << newf.transpose() << endl;

	dFf.resize(M);
	dFf.setZero();

        // TODO handle this derivative
	for (int i = 0; i < M; ++i)
	{
		int m = i - (M-1) / 2;

                // index in sorted configure
                int rlayout = rindex(layout);
                // coordinate x_(layout);
                // f(rlayout-1)
                // f(rlayout)

		if (m == 0)
		{
                        dFf(i) += newf(rlayout-1) / d_;
			dFf(i) += -newf(rlayout) / d_;
		}
		else
		{
			dFf(i) += newf(rlayout-1) / d_ 
                                * exp(-scalex(0, 2*m*Pi*x_(layout)/d_));
                        dFf(i) += -newf(rlayout) / d_ 
                                * exp(-scalex(0, 2*m*Pi*x_(layout)/d_));
		}
	}       
}

void FourierSolver1D::dSolve(const Eigen::VectorXcs& f,
        int M,
        const Parameterization& parameterization,
        Eigen::MatrixXcs& dFfs,
        int nPara)
{
        using std::max;
        nPara = max(nPara, count(parameterization));

        if (nPara < 0)
        {
                cerr << "[ERROR]: The number of parameters should be positive!\n";
        }

        
        MatrixXcs gradient(M, parameterization.size());
        MatrixXcs jacobian(parameterization.size(), nPara);

        int currIndex = 0;
        for (const auto& map : parameterization)
        {
                int layout = map.first;
                int nValidPara = map.second.size();

                VectorXcs dFf;
                dSolve(f, M, layout, dFf);

                gradient.col(currIndex) = dFf;
                jacobian.row(currIndex).leftCols(nValidPara) = map.second;
                ++currIndex;
        }
        
        dFfs = gradient * jacobian;
}


void FourierSolver1D::dSolve_v2(const Eigen::VectorXcs& f,
        int M,
        const Parameterization& parameterization,
        Eigen::MatrixXcs& dFfs,
        int nPara)
{
        using std::max;
        nPara = max(nPara, count(parameterization));

        if (nPara < 0)
        {
                cerr << "[ERROR]: The number of parameters should be positive!\n";
        }
        
        MatrixXcs gradient(M, parameterization.size());
        MatrixXcs jacobian(parameterization.size(), nPara);

        int currIndex = 0;
        for (const auto& map : parameterization)
        {
                int layout = map.first;
                int nValidPara = map.second.size();

                VectorXcs dFf;
                dSolve_v2(f, M, layout, dFf);

                gradient.col(currIndex) = dFf;
                jacobian.row(currIndex).leftCols(nValidPara) = map.second;
                ++currIndex;
        }
        
        dFfs = gradient * jacobian;        
}