#ifndef STRANDS_EIGENPAIRS_H
#define STRANDS_EIGENPAIRS_H

#include <vector>
#include <type_traits>
#include <memory>
#include "../schrodinger.h"

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename strands::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
denseEigenpairs(const strands::Schrodinger<Scalar> *self, const strands::EigensolverOptions &solverOptions);

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename strands::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
sparseEigenpairs(const strands::Schrodinger<Scalar> *self, const strands::EigensolverOptions &solverOptions);

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename strands::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
eigenpairs(const strands::Schrodinger<Scalar> *self, const strands::EigensolverOptions &solverOptions) {
    if (solverOptions.sparse)
        return sparseEigenpairs<Scalar, withEigenfunctions>(self, solverOptions);
    else
        return denseEigenpairs<Scalar, withEigenfunctions>(self, solverOptions);
}

#endif //STRANDS_EIGENPAIRS_H
