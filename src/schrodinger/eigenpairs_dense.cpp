#include "eigenpairs.h"

#include <numeric>
#include "../util/right_kernel.h"
#include "../util/rectangular_pencil.h"

using namespace strands;

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
denseEigenpairs(const Schrodinger<Scalar> *self, const EigensolverOptions &options) {
    using MatrixXs = typename Schrodinger<Scalar>::MatrixXs;
    using VectorXs = typename Schrodinger<Scalar>::VectorXs;

    Eigen::Index rows = self->intersections.size();
    Eigen::Index colsX = self->columns.x;
    Eigen::Index colsY = self->columns.y;
    PerDirection<VectorXs> lambda = self->Lambda();

    MatrixXs kernel, A, BK;
    PerDirection<MatrixXs> beta = self->Beta();

    MatrixXs crossingsMatch(rows, colsX + colsY);
    crossingsMatch << beta.x, -beta.y;

    kernel = strands::internal::rightKernel<MatrixXs>(crossingsMatch, 1e-6);
    // std::cout << "Dense: " << (crossingsMatch * kernel).cwiseAbs().rowwise().sum().maxCoeff() << std::endl;

    A = beta.x * lambda.x.asDiagonal() * kernel.topRows(colsX) +
        beta.y * lambda.y.asDiagonal() * kernel.bottomRows(colsY);

    BK = colsY < colsX // rows x kernelSize
         ? beta.y * kernel.bottomRows(colsY)
         : beta.x * kernel.topRows(colsX);

    strands::internal::RectangularPencil<withEigenfunctions, MatrixXs> pencil(A, BK, self->options.pencilThreshold);

    const auto &values = pencil.eigenvalues();
    validate_argument([&]() {
        return options.k > 0 && options.k < values.size();
    }, [&](auto &message) {
        message << "The number of eigenvalues k should be strictly positive and less than " << values.size() << ".";
    });

    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + options.k, indices.end(), [&values](int a, int b) {
        if (std::abs(values(b).imag()) > 1e-8) return true;
        if (std::abs(values(a).imag()) > 1e-8) return false;
        return values(a).real() < values(b).real();
    });
    indices.resize(options.k);

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
denseEigenpairs<Scalar, withEigenfunctions>(const Schrodinger<Scalar> *, const EigensolverOptions &);

#define STRANDS_INSTANTIATE(Scalar) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, false) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, true)

#include "instantiate.h"
