#ifndef SCHRODINGER2D_SCHRODINGER_H
#define SCHRODINGER2D_SCHRODINGER_H

#include "catch.hpp"
#include "../src/schrodinger2d.h"
#include <vector>

template<typename Scalar, typename T=Scalar>
inline void
checkEigenvalues(const std::vector<T> &expected,
                 const std::vector<Scalar> &found, Scalar tolerance = 1e-8) {
    REQUIRE(expected.size() <= found.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(Approx(Scalar(expected[i])).epsilon(tolerance) == found[i]);
    }
}

#endif //SCHRODINGER2D_SCHRODINGER_H
