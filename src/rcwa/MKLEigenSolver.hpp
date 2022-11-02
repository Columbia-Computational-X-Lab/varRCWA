#pragma once
#include "defns.h"
#include <Eigen/Core>
#include <mkl.h>

template <class MatrixType>
class MKLEigenSolver
{
private:
    typedef typename MatrixType::Scalar ScalarType;
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorType;

    MatrixType V_;
	VectorType d_;

public:
	MKLEigenSolver() {}
	void compute(const MatrixType& A);
	const MatrixType& eigenvectors() const { return V_; }
	const VectorType& eigenvalues() const { return d_; }
};


template <class MatrixType>
void MKLEigenSolver<MatrixType>::compute(const MatrixType& A)
{
    int n = A.rows();
    if constexpr (std::is_same_v<ScalarType, std::complex<double>>) {
        lapack_complex_double * a = nullptr;
        lapack_complex_double * w = nullptr;
        lapack_complex_double * vr = nullptr;

        a = new lapack_complex_double[n*n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                a[i * n + j].real = A(i, j).real();
                a[i * n + j].imag = A(i, j).imag();
            }
        }
        w = new lapack_complex_double[n];
        vr = new lapack_complex_double[n*n];

        LAPACKE_zgeev(LAPACK_ROW_MAJOR,
            'N',
            'V',
            n,
            a,
            n,
            w,
            nullptr,
            n,
            vr,
            n);

        V_.resize(n , n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                V_(i, j) = scalex(vr[i * n + j].real, vr[i * n + j].imag);
            }
        }

        d_.resize(n);
        for (int i = 0; i < n; ++i)
        {
            d_(i) = scalex(w[i].real, w[i].imag);
        }

        delete[] a;
        delete[] w;
        delete[] vr;
    } else if constexpr (std::is_same_v<ScalarType, std::complex<float>>) {
        
        lapack_complex_float * a = nullptr;
        lapack_complex_float * w = nullptr;
        lapack_complex_float * vr = nullptr;

        a = new lapack_complex_float[n*n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                a[i * n + j].real = A(i, j).real();
                a[i * n + j].imag = A(i, j).imag();
            }
        }
        w = new lapack_complex_float[n];
        vr = new lapack_complex_float[n*n];

        LAPACKE_cgeev(LAPACK_ROW_MAJOR,
            'N',
            'V',
            n,
            a,
            n,
            w,
            nullptr,
            n,
            vr,
            n);

        V_.resize(n , n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                V_(i, j) = scalex(vr[i * n + j].real, vr[i * n + j].imag);
            }
        }

        d_.resize(n);
        for (int i = 0; i < n; ++i)
        {
            d_(i) = scalex(w[i].real, w[i].imag);
        }

        delete[] a;
        delete[] w;
        delete[] vr;
    } else {
        ScalarType::unimplemented;
    }
}