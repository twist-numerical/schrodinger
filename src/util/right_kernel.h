#ifndef STRANDS_RIGHT_KERNEL_H
#define STRANDS_RIGHT_KERNEL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

namespace strands::internal {

    template<typename Derived>
    Eigen::Matrix<typename Eigen::MatrixBase<Derived>::Scalar,
            Eigen::MatrixBase<Derived>::ColsAtCompileTime, Eigen::Dynamic,
            (Eigen::AutoAlign | (Eigen::MatrixBase<Derived>::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)),
            Eigen::MatrixBase<Derived>::MaxColsAtCompileTime, Eigen::MatrixBase<Derived>::MaxColsAtCompileTime>
    rightKernel(const Eigen::MatrixBase<Derived> &A,
                typename Eigen::NumTraits<typename Eigen::MatrixBase<Derived>::Scalar>::Real threshold) {
        // Householder method

        typedef typename Eigen::MatrixBase<Derived> MatrixType;
        typedef typename MatrixType::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
        typedef typename Eigen::Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime,
                (Eigen::AutoAlign | (MatrixType::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)),
                MatrixType::MaxColsAtCompileTime, MatrixType::MaxRowsAtCompileTime> TransposeType;
        typedef typename Eigen::Matrix<Scalar, MatrixType::ColsAtCompileTime, Eigen::Dynamic,
                (Eigen::AutoAlign | (MatrixType::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)),
                MatrixType::MaxColsAtCompileTime, MatrixType::MaxColsAtCompileTime> KernelType;

        Eigen::Index rows = A.rows();
        Eigen::Index cols = A.cols();

        TransposeType T = A.transpose();
        Eigen::ColPivHouseholderQR<Eigen::Ref<TransposeType>> QR(T);

        const Eigen::Index diag = std::min(rows, cols);

        if (threshold < 0)
            threshold = RealScalar(diag) * Eigen::NumTraits<Scalar>::epsilon();

        const typename decltype(QR)::MatrixType &qr = QR.matrixQR();
        typename decltype(QR)::HouseholderSequenceType householder = QR.householderQ().setLength(QR.nonzeroPivots());

        Eigen::Index count = cols - diag;
        for (Eigen::Index i = 0; i < diag; ++i)
            if (std::abs(qr(i, i)) < threshold)
                ++count;

        // std::cout << "dense: " << count << ", " << diag << std::endl;

        KernelType K(cols, count);
        Eigen::Index j = 0;
        for (Eigen::Index i = 0; i < cols; ++i)
            if (diag <= i || std::abs(qr(i, i)) < threshold)
                K.col(j++) = householder * Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Unit(cols, i);

        assert(j == count);

        return K;


/*
        // SVD method
        // typedef typename Eigen::MatrixBase<Derived> MatrixType;
        int rows = A.rows();
        int cols = A.cols();

        Eigen::BDCSVD<Derived> svd;
        svd.setThreshold(threshold);
        svd.compute(A, Eigen::ComputeFullV);

        int rank = svd.rank();
        // rank = 1700;
        int kernelSize = cols - rank;
        printf("Matrix A: %dx%d, rank: %d, kernel: %d\n", rows, cols, rank, kernelSize);
        return svd.matrixV().rightCols(kernelSize);
*/
    }

#ifdef STRANDS_SLEPC
    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
    sparseRightKernel(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &eigenA, Scalar threshold = -1);
#endif
}

#endif //STRANDS_RIGHT_KERNEL_H
