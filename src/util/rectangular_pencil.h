#ifndef SCHRODINGER2D_RECTANGULAR_PENCIL_H
#define SCHRODINGER2D_RECTANGULAR_PENCIL_H

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <type_traits>
#include <string>

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
    RectangularPencil(const MatrixType &A, const MatrixType &B) {

        // Original method
/*
        Eigen::BDCSVD<MatrixType> svd;
        svd.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

        MatrixXs G = svd.matrixU().adjoint() * A * svd.matrixV();
        G *= svd.singularValues().array().topRows(svd.rank()).inverse().matrix().asDiagonal();

        Eigen::EigenSolver<MatrixXs> eigenSolver;
        eigenSolver.compute(G, withEigenvectors);
        m_eigenvalues = eigenSolver.eigenvalues();
        if constexpr (withEigenvectors) {
            // m_eigenvectors = svd.matrixV().adjoint() * eigenSolver.eigenvectors();
            m_eigenvectors = svd.matrixV() * eigenSolver.eigenvectors();
        }
*/

        // Trucated SVDs method (rank M truncation)
        int M = 200;

        Eigen::BDCSVD<MatrixType> svdA;
        svdA.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        MatrixXs A_Ut = svdA.matrixU().leftCols(M);
        MatrixXs A_St = svdA.singularValues().head(M).asDiagonal();
        MatrixXs A_Vt = svdA.matrixV().leftCols(M);

        Eigen::BDCSVD<MatrixType> svdB;
        svdB.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
        printf("U: %ld, %ld; V: %ld, %ld\n", svdB.matrixU().rows(), svdB.matrixU().cols(), svdB.matrixV().rows(), svdB.matrixV().cols());
        MatrixXs B_Ut = svdB.matrixU().leftCols(M);
        MatrixXs B_St = svdB.singularValues().head(M);
        MatrixXs B_Vt = svdB.matrixV().leftCols(M);

        MatrixXs G = B_Ut.adjoint() * A_Ut * A_St * A_Vt.adjoint() * svdB.matrixV();
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

#endif //SCHRODINGER2D_RECTANGULAR_PENCIL_H
