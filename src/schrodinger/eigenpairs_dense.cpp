#include "eigenpairs.h"

#include <numeric>
#include "../util/right_kernel.h"
#include "../util/rectangular_pencil.h"

using namespace schrodinger;

template<typename Scalar>
PerDirection<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, 2> computeBeta(const Schrodinger<Scalar, 2> *self) {
    using MatrixXs = typename Schrodinger<Scalar, 2>::MatrixXs;
    size_t rows = self->intersections.size();

    PerDirection<MatrixXs, 2> beta;
    beta.x = MatrixXs::Zero(rows, self->columns.x);
    beta.y = MatrixXs::Zero(rows, self->columns.y);

    size_t row = 0;
    for (const auto &intersection: self->intersections) {
        beta.x.row(row).segment(
                intersection->thread.x->offset, intersection->thread.x->eigenpairs.size()
        ) = intersection->evaluation.x;
        beta.y.row(row).segment(
                intersection->thread.y->offset, intersection->thread.y->eigenpairs.size()
        ) = intersection->evaluation.y;

        ++row;
    }
    assert(row == rows);

    return beta;
}

template<typename Scalar>
PerDirection<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 2> computeLambda(const Schrodinger<Scalar, 2> *self) {
    using VectorXs = typename Schrodinger<Scalar, 2>::VectorXs;

    PerDirection<VectorXs, 2> lambda;
    lambda.x = VectorXs(self->columns.x);
    lambda.y = VectorXs(self->columns.y);

    for (const auto &x: self->threads.x) {
        Eigen::Index offset = x.offset;
        for (const auto &ef: x.eigenpairs)
            lambda.x(offset++) = ef.first;
    }
    for (const auto &y: self->threads.y) {
        Eigen::Index offset = y.offset;
        for (const auto &ef: y.eigenpairs)
            lambda.y(offset++) = ef.first;
    }

    return lambda;
}

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
denseEigenpairs(const Schrodinger<Scalar> *self, Eigen::Index eigenvalueCount) {
    using MatrixXs = typename Schrodinger<Scalar>::MatrixXs;
    using VectorXs = typename Schrodinger<Scalar>::VectorXs;

    Eigen::Index rows = self->intersections.size();
    Eigen::Index colsX = self->columns.x;
    Eigen::Index colsY = self->columns.y;
    PerDirection<VectorXs, 2> lambda = computeLambda(self);

    MatrixXs kernel, A, BK;
    PerDirection<MatrixXs, 2> beta = computeBeta(self);

    MatrixXs crossingsMatch(rows, colsX + colsY);
    crossingsMatch << beta.x, -beta.y;

    kernel = schrodinger::internal::rightKernel<MatrixXs>(crossingsMatch, 1e-6);
    // std::cout << "Dense: " << (crossingsMatch * kernel).cwiseAbs().rowwise().sum().maxCoeff() << std::endl;

    A = beta.x * lambda.x.asDiagonal() * kernel.topRows(colsX) +
        beta.y * lambda.y.asDiagonal() * kernel.bottomRows(colsY);

    BK = colsY < colsX // rows x kernelSize
         ? beta.y * kernel.bottomRows(colsY)
         : beta.x * kernel.topRows(colsX);

    RectangularPencil<withEigenfunctions, MatrixXs> pencil(A, BK, self->options.pencilThreshold);

    const auto &values = pencil.eigenvalues();
    if (eigenvalueCount < 0 || eigenvalueCount > values.size())
        eigenvalueCount = values.size();

    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + eigenvalueCount, indices.end(), [&values](int a, int b) {
        if (std::abs(values(b).imag()) > 1e-8) return true;
        if (std::abs(values(a).imag()) > 1e-8) return false;
        return values(a).real() < values(b).real();
    });
    indices.resize(eigenvalueCount);

    if constexpr (withEigenfunctions) {
        const auto &vectors = pencil.eigenvectors();

        typedef std::pair<Scalar, std::unique_ptr<typename Schrodinger<Scalar>::Eigenfunction>> Eigenpair;
        std::vector<Eigenpair> eigenfunctions;

        eigenfunctions.reserve(indices.size());
        for (int i: indices) {
            VectorXs coeffs = kernel * vectors.col(i).real();

            eigenfunctions.emplace_back(
                    values(i).real(),
                    std::make_unique<typename Schrodinger<Scalar>::Eigenfunction>(self, values(i).real(), coeffs)
            );
        }
        return eigenfunctions;
    } else {
        std::vector<Scalar> eigenvalues;
        eigenvalues.reserve(indices.size());
        for (int i: indices)
            eigenvalues.push_back(values(i).real());

        return eigenvalues;
    }
}

#define SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, withEigenfunctions) \
template \
std::vector<typename std::conditional_t<(withEigenfunctions), std::pair<Scalar, std::unique_ptr<typename Schrodinger<Scalar>::Eigenfunction>>, Scalar>> \
denseEigenpairs<Scalar, withEigenfunctions>(const Schrodinger<Scalar> *, Eigen::Index);

#define SCHRODINGER_INSTANTIATE(Scalar) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, false) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, true)

#include "instantiate.h"
