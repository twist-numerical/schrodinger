#include "schrodinger.h"

using namespace Eigen;
using namespace schrodinger;
using namespace schrodinger::geometry;

const std::vector<double> expected{
        2, 4, 4, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10, 10, 10
};

TEST_CASE("Harmonic potential", "[harmonic]") {
    Schrodinger2D<double> s([](double x, double y) { return x * x + y * y; },
                            Rectangle<double, 2>{-8.0, 8.0, -8.0, 8.0},
                            Options{
                                    .gridSize={.x=26, .y=26},
                                    .maxBasisSize=14
                            });

    checkEigenvalues<double>(expected, s.eigenvalues(), 1e-4);
}


#ifdef SCHRODINGER_LONG_DOUBLE

TEST_CASE("Harmonic potential long double", "[harmonic][long double]") {
    typedef long double Scalar;
    Schrodinger2D<Scalar> s([](Scalar x, Scalar y) { return x * x + y * y; },
                            Rectangle<Scalar, 2>{-8.0, 8.0, -8.0, 8.0},
                            Options{
                                    .gridSize={.x=30, .y=30},
                                    .maxBasisSize=20
                            });

    checkEigenvalues<Scalar, double>(expected, s.eigenvalues(), 1e-8);
}

#endif
