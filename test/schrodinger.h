#ifndef SCHRODINGER2D_SCHRODINGER_H
#define SCHRODINGER2D_SCHRODINGER_H

#include "catch.hpp"
#include "../src/schrodinger2d.h"
#include <vector>

inline void checkEigenvalues(const std::vector<double> &expected, const std::vector<double> &found, double tolerance = 1e-8) {
    REQUIRE(expected.size() <= found.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(Approx(expected[i]).margin(tolerance) == found[i]);
    }
}

#endif //SCHRODINGER2D_SCHRODINGER_H
