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

            // check valid
            for (int i = 0; i < m_eigenvectors.cols(); i++) {
                Scalar E = m_eigenvalues.real()(i);
                VectorXs v = m_eigenvectors.real().col(i);
                VectorXs residue = A*v - E * (B * v);
                printf("Eigenvalue: %f, residue avg: %f, residue norm: %f\n", E, residue.sum() / residue.size(), residue.norm());
            }
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
