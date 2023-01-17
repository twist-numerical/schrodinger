#ifndef SCHRODINGER_EIGENPAIRS_H
#define SCHRODINGER_EIGENPAIRS_H

#include <vector>
#include <type_traits>
#include <memory>
#include "../schrodinger.h"

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
denseEigenpairs(const schrodinger::Schrodinger<Scalar> *self, const schrodinger::EigensolverOptions &solverOptions);

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
sparseEigenpairs(const schrodinger::Schrodinger<Scalar> *self, const schrodinger::EigensolverOptions &solverOptions);

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
eigenpairs(const schrodinger::Schrodinger<Scalar> *self, const schrodinger::EigensolverOptions &solverOptions) {
    if (solverOptions.sparse)
        return sparseEigenpairs<Scalar, withEigenfunctions>(self, solverOptions);
    else
        return denseEigenpairs<Scalar, withEigenfunctions>(self, solverOptions);
}

#endif //SCHRODINGER_EIGENPAIRS_H
