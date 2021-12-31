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

        // printf("row: %ld, col: %ld", G.rows(), G.cols());

        Eigen::EigenSolver<MatrixXs> eigenSolver;
        eigenSolver.compute(G, withEigenvectors);
        m_eigenvalues = eigenSolver.eigenvalues();
        if constexpr (withEigenvectors) {
            // m_eigenvectors = svd.matrixV().adjoint() * eigenSolver.eigenvectors();
            m_eigenvectors = svd.matrixV() * eigenSolver.eigenvectors();
        }
         */


        // Trucated SVDs method (rank M truncation)

        /*
        // Determine rank of A - lambda*B
        Eigen::ColPivHouseholderQR<MatrixType> QR(A+20*B);
        int r = QR.rank();
        printf("Rank %d\n", r);

        printf("A: %ld, %ld\n", A.rows(), A.cols());
        printf("B: %ld, %ld\n", B.rows(), B.cols());
        Eigen::BDCSVD<MatrixType> svdA;
        svdA.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        int M = svdA.matrixU().cols() - 0;

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

            // Double-check eigenvalues to filter out the wrong ones
            for (int i = 0; i < m_eigenvalues.size(); i++) {
                VectorXcs v = m_eigenvectors.col(i);
                std::complex<Scalar> E = m_eigenvalues(i);
                VectorXcs res = A * v - E*B*v;

                printf("Eigenvalue: %f+%fi, vector error %f\n", E.real(), E.imag(), res.norm());
            }
        }
         */

        // Algorithm 3
        int m = A.rows();
        int n = A.cols();
        MatrixType BA = MatrixType::Zero(m, n*2);
        BA.block(0, 0, m, n) = A;
        BA.block(0, n, m, n) = B;

        printf("m = %d, n = %d\n", m, n);

        Eigen::BDCSVD<MatrixType> svd;
        svd.compute(BA, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Rank-M truncation
        int M = std::min(m, n) - 0;
        printf("Matrix V: %dx%d\n", (int)svd.matrixV().rows(), (int)svd.matrixV().cols());
        MatrixXs Vt = svd.matrixV().leftCols(M);

        // Another SVD
        MatrixXs V_12 = MatrixType::Zero(n, 2*M);
        V_12.block(0, 0, n, M) = Vt.topRows(n);
        V_12.block(0, M, n, M) = Vt.bottomRows(n);
        Eigen::BDCSVD<MatrixXs> svdV;
        svdV.compute(V_12, Eigen::ComputeFullU | Eigen::ComputeFullV);

        MatrixXs V_V = svdV.matrixV();
        printf("Matrix V: %dx%d\n", (int)V_V.rows(), (int)V_V.cols());
        MatrixXs V1_t = V_V.block(0, 0, M, M);
        MatrixXs V2_t = V_V.block(M, 0, M, M);

        Eigen::GeneralizedEigenSolver<MatrixXs> ges;
        ges.compute(V1_t, V2_t);

        m_eigenvalues = ges.eigenvalues();
        if constexpr (withEigenvectors) {
            m_eigenvectors = ges.eigenvectors();
        }

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
