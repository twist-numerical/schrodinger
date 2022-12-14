#ifndef SCHRODINGER_EIGENPAIRS_H
#define SCHRODINGER_EIGENPAIRS_H

#include <vector>
#include <type_traits>
#include <memory>
#include "../schrodinger.h"

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
denseEigenpairs(const schrodinger::Schrodinger<Scalar> *self, Eigen::Index eigenvalueCount);

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
sparseEigenpairs(const schrodinger::Schrodinger<Scalar> *self, Eigen::Index eigenvalueCount, bool shiftInvert);

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger<Scalar>::Eigenfunction>>, Scalar>>
eigenpairs(const schrodinger::Schrodinger<Scalar> *self, Eigen::Index eigenvalueCount) {
    if (self->options.sparse)
        return sparseEigenpairs<Scalar, withEigenfunctions>(self, eigenvalueCount, self->options.shiftInvert);
    else
        return denseEigenpairs<Scalar, withEigenfunctions>(self, eigenvalueCount);
}

#endif //SCHRODINGER_EIGENPAIRS_H
