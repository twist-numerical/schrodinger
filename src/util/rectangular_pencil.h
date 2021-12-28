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

        Eigen::BDCSVD<MatrixType> svd;
        svd.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

        MatrixXs G = svd.matrixU().adjoint() * A * svd.matrixV();
        G *= svd.singularValues().array().topRows(svd.rank()).inverse().matrix().asDiagonal();

        // printf("row: %ld, col: %ld", G.rows(), G.cols());

        Eigen::EigenSolver<MatrixXs> eigenSolver;
        eigenSolver.compute(G, withEigenvectors);
        m_eigenvalues = eigenSolver.eigenvalues();
        if constexpr (withEigenvectors) {
            // m_eigenvectors = svd.matrixV().adjoint() * eigenSolver.eigenvectors();
            m_eigenvectors = svd.matrixV() * eigenSolver.eigenvectors();
        }


        // Trucated SVDs method (rank M truncation)
        /*
        int M = 205;

        printf("A: %ld, %ld\n", A.rows(), A.cols());
        printf("B: %ld, %ld\n", B.rows(), B.cols());
        Eigen::BDCSVD<MatrixType> svdA;
        svdA.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        for (int i = 0; i < svdA.singularValues().size(); i++) {
            printf("%f ", svdA.singularValues()(i));
        }
        printf("\n");

        printf("A U: %ld, %ld; V: %ld, %ld\n", svdA.matrixU().rows(), svdA.matrixU().cols(), svdA.matrixV().rows(), svdA.matrixV().cols());
        MatrixXs A_Ut = svdA.matrixU().leftCols(M);
        MatrixXs A_St = svdA.singularValues().head(M).asDiagonal();
        MatrixXs A_Vt = svdA.matrixV().leftCols(M);

        Eigen::BDCSVD<MatrixType> svdB;
        svdB.compute(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
        printf("B U: %ld, %ld; V: %ld, %ld\n", svdB.matrixU().rows(), svdB.matrixU().cols(), svdB.matrixV().rows(), svdB.matrixV().cols());
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
         */

        /*
        // Method: https://www.keisu.t.u-tokyo.ac.jp/data/2014/METR14-27.pdf
        int m = A.rows();
        int n = A.cols();
        MatrixType BA = MatrixType::Zero(m, n*2);
        BA.block(0, 0, m, n) = B;
        BA.block(0, n, m, n) = A;

        // printf("m = %ld, n = %ld\n", A.rows(), A.cols());

        Eigen::BDCSVD<MatrixType> svd;
        svd.compute(BA, Eigen::ComputeFullU | Eigen::ComputeFullV);

        MatrixXs V = svd.matrixV();
        MatrixXs V11 = V.block(0, 0, n, n);
        MatrixXs V21 = V.block(n, 0, n, n);

        Eigen::GeneralizedEigenSolver<MatrixXs> ges;
        ges.compute(V21.adjoint(), V11.adjoint());

        m_eigenvalues = ges.eigenvalues();
        if constexpr (withEigenvectors) {
            m_eigenvectors = ges.eigenvectors();
        }
         */



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
