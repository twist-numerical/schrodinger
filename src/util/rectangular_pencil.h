#ifndef SCHRODINGER2D_RECTANGULAR_PENCIL_H
#define SCHRODINGER2D_RECTANGULAR_PENCIL_H

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <type_traits>
#include <string>
#include <cstdio>
#include <chrono>

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
    RectangularPencil(const MatrixType &A, const MatrixType &B, int method=-1) {
        // Default method
        if (method == -1) method = 4;

        // Pseudo-inverse method
        if (method == 0) {
            Scalar threshold = 1e-8;
            Eigen::BDCSVD<MatrixType> svdA;
            svdA.setThreshold(threshold);
            svdA.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Eigen::BDCSVD<MatrixType> svdB;
            svdB.setThreshold(threshold);
            svdB.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

            int M = svdB.rank();

            MatrixXs A_Ut = svdA.matrixU().leftCols(M);
            MatrixXs A_St = svdA.singularValues().head(M).asDiagonal();
            MatrixXs A_Vt = svdA.matrixV().leftCols(M);

            MatrixXs B_Ut = svdB.matrixU().leftCols(M);
            VectorXs B_St = svdB.singularValues().head(M);
            MatrixXs B_Vt = svdB.matrixV().leftCols(M);

            MatrixXs G = B_Vt * B_St.array().inverse().matrix().asDiagonal() * B_Ut.adjoint()
                    * A_Ut * A_St * A_Vt.adjoint();

            Eigen::EigenSolver<MatrixXs> eigenSolver;
            eigenSolver.compute(G, withEigenvectors);
            m_eigenvalues = eigenSolver.eigenvalues();
            if constexpr (withEigenvectors) {
                m_eigenvectors = eigenSolver.eigenvectors();
            }
        }

        // B^T method
        if (method == 1) {
            Eigen::GeneralizedEigenSolver<MatrixXs> ges;
            ges.compute(B.adjoint() * A, B.adjoint() * B);

            // leave out infinite eigenvalues
            Eigen::Index size = ges.alphas().size();
            m_eigenvalues = VectorXcs::Zero(size);
            if constexpr (withEigenvectors) m_eigenvectors = MatrixXcs::Zero(A.cols(), size);

            int count = 0;
            for (int i = 0; i < size; i++) {
                if (std::abs(ges.betas()(i)) > 1e-16) {
                    m_eigenvalues(count) = ges.alphas()(i) / ges.betas()(i);
                    if constexpr (withEigenvectors) {
                        m_eigenvectors.col(count) = ges.eigenvectors().col(i);
                    }
                    count++;
                }
            }

            m_eigenvalues.resize(count);
            if constexpr (withEigenvectors) m_eigenvectors.resize(Eigen::NoChange, count);
        }

        // Truncated SVDs method (rank M truncation)
        if (method == 2 || method == 3) {
            // Determine rank of A - lambda*B
            Scalar threshold = method == 2 ? 1e-8 : 1e-12;

            Eigen::BDCSVD<MatrixType> rankSvd;
            rankSvd.setThreshold(threshold);
            rankSvd.compute(A + 20 * B);
            int r = rankSvd.rank();

            printf("Solving pencil, A, B: %dx%d\n", (int) A.rows(), (int) A.cols());
            printf("Pencil normal rank %d\n", r);

            Eigen::BDCSVD<MatrixType> svdA;
            svdA.setThreshold(threshold);
            svdA.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Eigen::BDCSVD<MatrixType> svdB;
            svdB.setThreshold(threshold);
            svdB.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

            printf("Rank of A: %d\n", (int) svdA.rank());
            printf("Rank of B: %d\n", (int) svdB.rank());

            int M = (int) svdB.rank();

            MatrixXs A_Ut = svdA.matrixU().leftCols(M);
            MatrixXs A_St = svdA.singularValues().head(M).asDiagonal();
            MatrixXs A_Vt = svdA.matrixV().leftCols(M);

            MatrixXs B_Ut = svdB.matrixU().leftCols(M);
            VectorXs B_St = svdB.singularValues().head(M);
            MatrixXs B_Vt = svdB.matrixV().leftCols(M);

            // (r x m) * (m x r) * (r x r) * (r x n) * (n x r) * (r x r)
            MatrixXs G = B_Ut.adjoint() * A_Ut * A_St * A_Vt.adjoint() * B_Vt;

            // (r x m) * (m x n) * (n x r) * (r x r)
            // MatrixXs G = B_Ut.adjoint() * A * B_Vt;
            G *= B_St.array().inverse().matrix().asDiagonal();

            Eigen::EigenSolver<MatrixXs> eigenSolver;
            eigenSolver.compute(G, withEigenvectors);
            m_eigenvalues = eigenSolver.eigenvalues();
            if constexpr (withEigenvectors) {
                m_eigenvectors = B_Vt * eigenSolver.eigenvectors();

/*
                // argsort values
                VectorXcs Ev = m_eigenvalues;
                std::vector<size_t> idx(m_eigenvalues.size());
                for (int i = 0; i < m_eigenvalues.size(); i++) idx[i] = i;
                std::stable_sort(idx.begin(), idx.end(),
                                 [&Ev](size_t i1, size_t i2) { return Ev(i1).real() < Ev(i2).real(); });

                // Double-check eigenvalues to filter out the wrong ones
                for (int i = 0; i < m_eigenvalues.size(); i++) {
                    int ii = (int) idx[i];
                    VectorXcs v = m_eigenvectors.col(ii).normalized();
                    std::complex<Scalar> E = m_eigenvalues(ii);
                    VectorXcs res = A * v - E * B * v;

                    printf("Eigenvalue: %f+%fi, vector error %f\n", E.real(), E.imag(), res.template lpNorm<2>());
                }
*/
            }
        }

        // New truncation method
        if (method == 4 || method == 5) {
            Scalar threshold = method == 4 ? 1e-8 : 1e-12;

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

            printf("Rank of B: %d\n", (int)svdB.rank());

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

        /*
        // Write A and B to file
        FILE* fh = fopen("matA.csv", "w+"); fprintf(fh, "Matrix %dx%d\n", (int)A.rows(), (int)A.cols());
        for (int i = 0; i < A.rows(); i++) for (int j = 0; j < A.cols(); j++) fprintf(fh, "%.17g%s", A(i,j), j == A.cols()-1?"\n":", ");
        fclose(fh);

        fh = fopen("matB.csv", "w+"); fprintf(fh, "Matrix %dx%d\n", (int)B.rows(), (int)B.cols());
        for (int i = 0; i < B.rows(); i++) for (int j = 0; j < B.cols(); j++) fprintf(fh, "%.17g%s", B(i,j), j == B.cols()-1?"\n":", ");
        fclose(fh);
         */

        // Algorithm 3
        if (method == 6) {
            int m = A.rows();
            int n = A.cols();
            MatrixType BA = MatrixType::Zero(m, n * 2);
            BA.block(0, 0, m, n) = A;
            BA.block(0, n, m, n) = B;

            printf("m = %d, n = %d\n", m, n);

            Eigen::BDCSVD<MatrixType> svd;
            svd.compute(BA, Eigen::ComputeFullU | Eigen::ComputeFullV);

            // Rank-M truncation
            int M = std::min(m, n) - 0;
            printf("Matrix V: %dx%d\n", (int) svd.matrixV().rows(), (int) svd.matrixV().cols());
            MatrixXs Vt = svd.matrixV().leftCols(M);

            // Another SVD
            MatrixXs V_12 = MatrixType::Zero(n, 2 * M);
            V_12.block(0, 0, n, M) = Vt.topRows(n);
            V_12.block(0, M, n, M) = Vt.bottomRows(n);
            Eigen::BDCSVD<MatrixXs> svdV;
            svdV.compute(V_12, Eigen::ComputeFullU | Eigen::ComputeFullV);

            MatrixXs V_V = svdV.matrixV();
            printf("Matrix V: %dx%d\n", (int) V_V.rows(), (int) V_V.cols());
            MatrixXs V1_t = V_V.block(0, 0, M, M);
            MatrixXs V2_t = V_V.block(M, 0, M, M);

            Eigen::GeneralizedEigenSolver<MatrixXs> ges;
            ges.compute(V1_t, V2_t);

            m_eigenvalues = ges.eigenvalues();
            if constexpr (withEigenvectors) {
                m_eigenvectors = ges.eigenvectors();
            }
        }

        // Total least squares method
        // https://www.keisu.t.u-tokyo.ac.jp/data/2014/METR14-27.pdf
        if (method == 7) {
            int m = A.rows();
            int n = A.cols();

            if (m < n) {
                m_eigenvalues = VectorXcs::Zero(0);
                if constexpr (withEigenvectors) {
                    m_eigenvectors = MatrixXcs::Zero(n, 0);
                }
                return;
            }

            MatrixType BA = MatrixType::Zero(m, n * 2);
            BA.block(0, 0, m, n) = B;
            BA.block(0, n, m, n) = A;

            // printf("m = %ld, n = %ld\n", A.rows(), A.cols());

            Eigen::BDCSVD<MatrixType> svd;
            svd.compute(BA, Eigen::ComputeThinU | Eigen::ComputeThinV);

            printf("SVD rank: %d\n", (int) svd.rank());

            MatrixXs V = svd.matrixV();
            MatrixXs V11 = V.block(0, 0, n, n);
            MatrixXs V21 = V.block(n, 0, n, n);

            Eigen::GeneralizedEigenSolver<MatrixXs> ges;
            ges.compute(V21.adjoint(), V11.adjoint());

            m_eigenvalues = ges.eigenvalues();
            if constexpr (withEigenvectors) {
                m_eigenvectors = ges.eigenvectors();
            }

            // Correct original matrices and print them out (noise-free version)
            /*
            MatrixXs R = -svd.matrixU().rightCols(n)
                         * svd.singularValues().bottomRows(n).asDiagonal()
                         * svd.matrixV().rightCols(n).adjoint();

            //MatrixXs A_hat = A + R.rightCols(n);
            //MatrixXs B_hat = B + R.leftCols(n);
            MatrixXs A_hat = A;
            MatrixXs B_hat = B;

            printf("R max %f, error %f\n", R.maxCoeff(), R.norm());

            FILE* fh = fopen("matA.csv", "w+"); fprintf(fh, "Matrix %dx%d\n", (int)A_hat.rows(), (int)A_hat.cols());
            for (int i = 0; i < A_hat.rows(); i++) for (int j = 0; j < A_hat.cols(); j++) fprintf(fh, "%.17g%s", A_hat(i,j), j == A_hat.cols()-1?"\n":", ");
            fclose(fh);

            fh = fopen("matB.csv", "w+"); fprintf(fh, "Matrix %dx%d\n", (int)B_hat.rows(), (int)B_hat.cols());
            for (int i = 0; i < B_hat.rows(); i++) for (int j = 0; j < B_hat.cols(); j++) fprintf(fh, "%.17g%s", B_hat(i,j), j == B_hat.cols()-1?"\n":", ");
            fclose(fh);
             */
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
