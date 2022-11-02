#include "FourierSolver2D.h"

#include "FourierSolver1D.h"
#include "ToeplitzMatrixXcs.h"
#include "BlockToeplitzMatrixXcs.h"
#include <iostream>
// #include <fstream>

using namespace std;
using namespace Eigen;

void FourierSolver2D::solve(const Eigen::MatrixXcs& f,
	int nx, int ny,
	FourierSolverOption opt,
	Eigen::MatrixXcs& Ff,
    std::vector<Eigen::MatrixXcs> * dFfs,
    const Parameterization& para_x,
    const Parameterization& para_y)
{
  if (is_varied(para_y))
  {
          cerr << "[ERROR]: vary y is not currently supported!\n";
          return;
  }

  using std::max;
  int nPara = max(count(para_x), count(para_y));
  // cout << "nPara: " << nPara << endl;

	Ff.resize(nx*ny, nx*ny);
	Ff.setZero();

  if (dFfs != nullptr && nPara > 0)
  {
      dFfs->resize(nPara);
      for (int i = 0; i < nPara; ++i)
      {
          (*dFfs)[i].resize(nx*ny, nx*ny);
          (*dFfs)[i].setZero();
      }
  }

	if (opt == DiscontinuousX)
	{
		FourierSolver1D xSolver(coordX_, dx_);
    VectorXi index;
    xSolver.reorder(index);
    VectorXi rindex(index.size());
    for (int i = 0; i < index.size(); ++i) {
        rindex(index(i)) = i;
    }
    
    // ofstream fout("FfxFfy_i.txt");

		for (int i = 0; i < coordY_.size()-1; ++i) {
      VectorXcs newf;
      xSolver.refill(f.row(i), rindex, newf);
			VectorXcs invFi = newf.cwiseInverse();

			VectorXcs fInvFi;
			xSolver.solve(invFi, 2*nx-1, fInvFi);
			ToeplitzMatrixXcs tFInvFi(fInvFi);
			MatrixXcs tFFi = tFInvFi.inverse();
                        
      // the derivative w. r. t. each para
      vector<MatrixXcs> dtFFis(nPara);
      if (is_varied(para_x))
      {
          // VectorXcs dfInvFi;
          MatrixXcs dfInvFi; // 2*nx-1, nPara
          xSolver.dSolve_v2(f.row(i).cwiseInverse(), 
              2*nx-1, 
              para_x, 
              dfInvFi,
              nPara);
          // cout << dfInvFi.rows() << ", " << dfInvFi.cols()
          //         << endl;
          
          for (int p = 0; p < nPara; ++p)
          {
                  ToeplitzMatrixXcs dtFInvFi(dfInvFi.col(p));
                  dtFFis[p] = -tFFi * dtFInvFi.toDense() * tFFi;
          }	
      }

			VectorXs tempCoordY(2);
			tempCoordY(0) = coordY_(i);
			tempCoordY(1) = coordY_(i+1);
			FourierSolver1D ySolver(tempCoordY, dy_);
			VectorXcs fY1;
			VectorXcs identity(1);
			identity(0) = scalex(1.0, 0.0);
			ySolver.solve(identity, 2*ny-1, fY1);


      // fout << fInvFi.transpose() << "\t" << fY1.transpose() << endl;
      // vector<VectorXcs> dfY1s(nPara);
      // if (is_varied(para_y))
      // {
      //         for (int p = 0; p < nPara; ++p)
      //         {
      //                 dfY1s[p].resize(2*ny-1);
      //                 dfY1s[p].setZero();
      //         }

      //         // here we have n parameters
      //         // we need to see if the parameter relies
      //         // on the 
      // }

			for (int ii = 0; ii < nx; ++ii)
			{
				for (int jj = 0; jj < ny; ++jj)
				{
					int m = ii - (nx - 1) / 2;
					int n = jj - (ny - 1) / 2;

					for (int kk = 0; kk < nx; ++kk)
					{
						for (int ll = 0; ll < ny; ++ll)
						{
							int j = kk - (nx - 1) / 2;
							int l = ll - (ny - 1) / 2;
							Ff(ii * ny + jj, kk * ny + ll) += fY1(n-l + ny-1) * tFFi(ii, kk);

              // fout << ii*ny+jj << "\t" << kk*ny+ll << "\t"
              //   << n-l+ny-1 << "\t" << ii << kk << endl;

              if (dFfs != nullptr && nPara > 0)
              {
                  for (int p = 0; p < nPara; ++p)
                  {
                      (*dFfs)[p](ii * ny + jj, kk * ny + ll) += 
                          fY1(n-l + ny - 1) 
                          * dtFFis[p](m +(nx-1)/2, j+(nx-1)/2);
                  }
                      
              }
						}
					}
				}
			}
		}		
	}
	else if (opt == DiscontinuousY)
	{
        FourierSolver1D rSolver(coordX_, dx_);
        VectorXi index;
        rSolver.reorder(index);
        VectorXi rindex(index.size());
        for (int i = 0; i < index.size(); ++i)
        {
                rindex(index(i)) = i;
        }
        MatrixXcs newf(f.rows(), f.cols());
        // cout << "old f: " << endl;
        // cout << f << endl;

        for (int i = 0; i < f.rows(); ++i)
        {
                VectorXcs f0;
                rSolver.refill(f.row(i), rindex, f0);
                newf.row(i) = f0;
        }
        // cout << "new f: " << endl;
        // cout << newf << endl;

    // ofstream fout("colFxyFxy.txt");

		FourierSolver1D ySolver(coordY_, dy_);
		for (int i = 0; i < coordX_.size()-1; ++i)
		{
			VectorXcs invFi = newf.col(i).cwiseInverse();
			VectorXcs fInvFi;
			ySolver.solve(invFi, 2*ny-1, fInvFi);
			ToeplitzMatrixXcs tFInvFi(fInvFi);
			MatrixXcs tFFi = tFInvFi.inverse();

			VectorXs tempCoordX(2);
			tempCoordX(0) = coordX_(index(i));
			tempCoordX(1) = coordX_(index(i+1));
			FourierSolver1D xSolver(tempCoordX, dx_);
			VectorXcs fX1;
			VectorXcs identity(1);
			identity(0) = scalex(1.0, 0.0);
			xSolver.solve(identity, 2*nx-1, fX1);

      // fout << fInvFi.transpose() << "\t" << fX1.transpose() << endl;

            vector<VectorXcs> dfX1s(nPara);
            if (is_varied(para_x))
            {
                dfX1s.resize(nPara);
                for (int p = 0; p < nPara; ++p)
                {
                    dfX1s[p].resize(2*nx-1);
                    dfX1s[p].setZero();
                }

				if (para_x.count(index(i)) == 1)
				{
					VectorXcs dfX1_0;
					xSolver.dSolve(identity, 2*nx-1, 0, dfX1_0);
					for (int p = 0; p < nPara; ++p)
                    {
                        dfX1s[p] += para_x.at(index(i))(p) * dfX1_0;
                    }
				}
				
                if (para_x.count(index(i+1)) == 1)
				{
					VectorXcs dfX1_1;
					xSolver.dSolve(identity, 2*nx-1, 1, dfX1_1);
					for (int p = 0; p < nPara; ++p)
                    {
                        dfX1s[p] += para_x.at(index(i+1))(p) * dfX1_1;
                    }
				}
            }

			for (int ii = 0; ii < nx; ++ii)
			{
				for (int jj = 0; jj < ny; ++jj)
				{
					int m = ii - (nx - 1) / 2;
					int n = jj - (ny - 1) / 2;

					for (int kk = 0; kk < nx; ++kk)
					{
						for (int ll = 0; ll < ny; ++ll)
						{
							int j = kk - (nx - 1) / 2;
							int l = ll - (ny - 1) / 2;
							Ff(ii * ny + jj, kk * ny + ll) += fX1(m-j + nx - 1) * tFFi(jj, ll);

              if (dFfs != nullptr && is_varied(para_x))
							{
                  for (int p = 0; p < nPara; ++p)
                  {
                      (*dFfs)[p](ii * ny + jj, kk * ny + ll) += 
                      dfX1s[p](m-j + nx - 1) * tFFi(n +(ny-1)/2, l+(ny-1)/2);
                  }
							}
						}
					}
				}
			}			
		}		
	}
	else if (opt == ContinuousXY)
	{
		MatrixXcs fourier2D(2*nx-1, 2*ny-1);
		fourier2D.setZero();

        vector<MatrixXcs> dfourier2D(nPara);
        
        if (is_varied(para_x))
        {
            for (int p = 0; p < nPara; ++p)
            {
                dfourier2D[p].resize(2*nx-1, 2*ny-1);
                dfourier2D[p].setZero();
            }
        }
        // here notice that x and y is swappable

		// FourierSolver1D ySolver(coordY_, dy_);
		// for (int i = 0; i < coordX_.size()-1; ++i)
		// {
		// 	VectorXcs Fi = f.col(i);
		// 	VectorXcs fFi;
		// 	ySolver.solve(Fi, 2*ny-1, fFi);

		// 	VectorXs tempCoordX(2);
		// 	tempCoordX(0) = coordX_(i);
		// 	tempCoordX(1) = coordX_(i+1);
		// 	FourierSolver1D xSolver(tempCoordX, dx_);
		// 	VectorXcs fX1;
		// 	VectorXcs identity(1);
		// 	identity(0) = scalex(1.0, 0.0);
		// 	xSolver.solve(identity, 2*nx-1, fX1);

		// 	fourier2D += fX1 * fFi.transpose();
		// }

        FourierSolver1D xSolver(coordX_, dx_);
        VectorXi index;
        xSolver.reorder(index);
        VectorXi rindex(index.size());
        for (int i = 0; i < index.size(); ++i)
        {
                rindex(index(i)) = i;
        }


        for (int i = 0; i < coordY_.size()-1; ++i)
        {
            VectorXcs newf;
            xSolver.refill(f.row(i), rindex, newf);
            VectorXcs Fi = newf;
            VectorXcs fFi;
            xSolver.solve(Fi, 2*nx-1, fFi);

            MatrixXcs dfFis;
            if (is_varied(para_x))
            {
                xSolver.dSolve_v2(f.row(i), 
                    2*nx-1, 
                    para_x, 
                    dfFis,
                    nPara);
            }

            VectorXs tempCoordY(2);
            tempCoordY(0) = coordY_(i);
            tempCoordY(1) = coordY_(i+1);
            FourierSolver1D ySolver(tempCoordY, dy_);
            VectorXcs fY1;
            VectorXcs identity(1);
            identity << scalex(1.0, 0.);
            ySolver.solve(identity, 2*ny-1, fY1);

            fourier2D += fFi * fY1.transpose();

            if (nPara > 0)
            {
                    for (int p = 0; p < nPara; ++p)
                    {
                            dfourier2D[p] += dfFis.col(p) * fY1.transpose();
                    }
            }

        }
    // ofstream fout("fourier2D.txt");
    // fout << fourier2D << endl;

    // ofstream fout("fourier2D.txt");
    // fout << fourier2D << endl;

		Ff = BlockToeplitzMatrixXcs(fourier2D).toDense();

        if (dFfs != nullptr && nPara > 0)
        {
            for (int p = 0; p < nPara; ++p)
            {
                (*dFfs)[p] = BlockToeplitzMatrixXcs(dfourier2D[p]).toDense();
            }
        }
	}

}