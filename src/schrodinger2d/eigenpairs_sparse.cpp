#include "eigenpairs.h"
#include <Eigen/Sparse>

using namespace schrodinger;

struct DirectionGetter {
    bool getX;

    template<typename T>
    constexpr const T &operator()(const PerDirection<T> &p) {
        return getX ? p.x : p.y;
    }
};

DirectionGetter xDirection{true};
DirectionGetter yDirection{false};

template<typename Scalar>
class PermutedBeta {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

    Eigen::PermutationMatrix<Eigen::Dynamic> permutation;
    std::vector<MatrixXs> leastSquares;
    std::vector<MatrixXs> blocks;
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;

    PermutedBeta(const Schrodinger2D<Scalar> *schrodinger, DirectionGetter direction) : permutation{
            Eigen::Index(schrodinger->intersections.size())} {
        auto &threads = direction(schrodinger->threads);
        leastSquares.reserve(threads.size());
        blocks.reserve(threads.size());
        Eigen::Index pIndex = 0;
        for (auto &thread: threads) {
            for (auto intersection: thread.intersections)
                permutation.indices()(pIndex++) = intersection->index;

            Eigen::Index m = thread.intersections.size();
            Eigen::Index n = thread.eigenpairs.size();
            rows += m;
            cols += n;
            MatrixXs B(m, n);
            {
                Eigen::Index i = 0;
                for (auto intersection: thread.intersections)
                    B.row(i++) = direction(intersection->evaluation);
            }
            VectorXs lambda(n);
            {
                Eigen::Index i = 0;
                for (auto &ef: thread.eigenpairs)
                    lambda(i++) = std::get<0>(ef);
            }

            leastSquares.push_back(B.colPivHouseholderQr().solve(MatrixXs::Identity(m, m)));
            blocks.push_back(B * lambda.asDiagonal() * leastSquares.back());
        }
    }

    SparseMatrix lstsqMatrix() {
        SparseMatrix r{cols, rows};
        Eigen::VectorXi toReserve{rows};
        {
            Eigen::Index i = 0;
            for (auto &lstsq: leastSquares) {
                toReserve.middleRows(i, lstsq.rows()) = Eigen::VectorXi::Constant(lstsq.rows(), lstsq.cols());
                i += lstsq.rows();
            }
        }
        r.reserve(toReserve);
        {
            Eigen::Index i = 0;
            Eigen::Index j = 0;
            for (auto &lstsq: leastSquares) {
                for (int i1 = 0; i1 < lstsq.rows(); ++i1)
                    for (int j1 = 0; j1 < lstsq.cols(); ++j1)
                        r.insert(i + i1, j + j1) = lstsq(i1, j1);
                i += lstsq.rows();
                j += lstsq.cols();
            }
        }
        return permutation.transpose() * r * permutation;
    }

    SparseMatrix fullMatrix() {
        SparseMatrix r{rows, rows};
        Eigen::VectorXi toReserve{rows};
        {
            Eigen::Index i = 0;
            for (auto &block: blocks) {
                toReserve.middleRows(i, block.rows()) = Eigen::VectorXi::Constant(block.rows(), block.cols());
                i += block.rows();
            }
        }
        r.reserve(toReserve);
        {
            Eigen::Index i = 0;
            Eigen::Index j = 0;
            for (auto &block: blocks) {
                for (int i1 = 0; i1 < block.rows(); ++i1)
                    for (int j1 = 0; j1 < block.cols(); ++j1)
                        r.insert(i + i1, j + j1) = block(i1, j1);
                i += block.rows();
                j += block.cols();
            }
        }
        return permutation.transpose() * r * permutation;
    }
};

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>>
sparseEigenpairs(const Schrodinger2D<Scalar> *schrodinger, int) {
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

    PermutedBeta<Scalar> Bx{schrodinger, xDirection};
    PermutedBeta<Scalar> By{schrodinger, yDirection};
    Eigen::EigenSolver<MatrixXs> eigenSolver;
    eigenSolver.compute(MatrixXs(Bx.fullMatrix() + By.fullMatrix()), true);

    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> values = eigenSolver.eigenvalues();
    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> vectors = eigenSolver.eigenvectors();
    SparseMatrix lstsq_x = Bx.lstsqMatrix();
    SparseMatrix lstsq_y = By.lstsqMatrix();
    VectorXs norms = (lstsq_x * vectors - lstsq_y * vectors).colwise().norm();

    std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>> result;
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (std::abs(values(i).imag()) < 1e-8 && norms(i) < 1e-8) {
            if constexpr(withEigenfunctions) {
                VectorXs v(Bx.cols + By.cols);
                v.topRows(Bx.cols) = lstsq_x * vectors.col(i).real();
                v.bottomRows(By.cols) = lstsq_y * vectors.col(i).real();
                result.emplace_back(
                        values(i).real(),
                        std::make_unique<typename Schrodinger2D<Scalar>::Eigenfunction>(
                                schrodinger, values(i).real(), v)
                );
            } else {
                std::cout << values(i).real() << std::endl;
                result.emplace_back(values(i).real());
            }
        }
    }
    return result;
}

#define SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, withEigenfunctions) \
template \
std::vector<typename std::conditional_t<(withEigenfunctions), std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>> \
sparseEigenpairs<Scalar, withEigenfunctions>(const Schrodinger2D<Scalar> *, int);

#define SCHRODINGER_INSTANTIATE(Scalar) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, false) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, true)

#include "instantiate.h"
