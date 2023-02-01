#ifndef STRANDS_RECTANGULAR_PENCIL_H
#define STRANDS_RECTANGULAR_PENCIL_H

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <type_traits>
#include <string>
#include <cstdio>
#include <chrono>

namespace strands::internal {

    template<bool withEigenvectors, typename MatrixType>
    class RectangularPencil {
        // https://doi.org/10.1109/ISCAS.1991.176121
        typedef typename MatrixType::Scalar Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
        typedef Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> MatrixXcs;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
        typedef Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> VectorXcs;

        VectorXcs m_eigenvalues;
        MatrixXcs m_eigenvectors;
    public:
        RectangularPencil(const MatrixType &A, const MatrixType &B, Scalar threshold = 1e-8) {
            // New truncation method

            if (B.rows() == 0 || B.cols() == 0) {
                m_eigenvalues(0, 0);
                if constexpr (withEigenvectors) {
                    m_eigenvectors(0, 0);
                }
                return;
            }

            Eigen::BDCSVD<MatrixType> svdB;
            svdB.setThreshold(threshold);
            svdB.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

            int r = (int) svdB.rank();
            MatrixXs B_Ut = svdB.matrixU().leftCols(r);
            VectorXs B_St = svdB.singularValues().head(r);
            MatrixXs B_Vt = svdB.matrixV().leftCols(r);

            // (r x m) * (m x n) * (n x r) * (r x r)
            MatrixXs G = B_Ut.adjoint() * A * B_Vt;
            G *= B_St.array().inverse().matrix().asDiagonal();

            Eigen::EigenSolver<MatrixXs> eigenSolver;
            eigenSolver.compute(G, withEigenvectors);
            m_eigenvalues = eigenSolver.eigenvalues();
            if constexpr (withEigenvectors) {
                m_eigenvectors = B_Vt * eigenSolver.eigenvectors();
            }
        }

        const VectorXcs &eigenvalues() {
            return m_eigenvalues;
        }

        template<bool enable = true, typename = typename std::enable_if<withEigenvectors && enable>::type>
        const MatrixXcs &eigenvectors() {
            return m_eigenvectors;
        }
    };

}

#endif //STRANDS_RECTANGULAR_PENCIL_H
