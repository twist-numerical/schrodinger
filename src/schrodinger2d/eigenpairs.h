#ifndef SCHRODINGER2D_EIGENPAIRS_H
#define SCHRODINGER2D_EIGENPAIRS_H

#include <vector>
#include <type_traits>
#include <memory>
#include "../schrodinger2d.h"

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>>
denseEigenpairs(const schrodinger::Schrodinger2D<Scalar> *self, int eigenvalueCount);

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>>
sparseEigenpairs(const schrodinger::Schrodinger2D<Scalar> *self, int eigenvalueCount);

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename schrodinger::Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>>
eigenpairs(const schrodinger::Schrodinger2D<Scalar> *self, int eigenvalueCount) {
    if (self->options.sparse)
        return sparseEigenpairs<Scalar, withEigenfunctions>(self, eigenvalueCount);
    else
        return denseEigenpairs<Scalar, withEigenfunctions>(self, eigenvalueCount);
}

#endif //SCHRODINGER2D_EIGENPAIRS_H
