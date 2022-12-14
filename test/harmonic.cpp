#include "schrodinger.h"

using namespace schrodinger;
using namespace schrodinger::geometry;

const std::vector<double> expected{
        2, 4, 4, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12
};

TEST_CASE("Harmonic potential", "[harmonic]") {
    Rectangle<double, 2> domain{-8.0, 8.0, -8.0, 8.0};
    Schrodinger<double> s([](double x, double y) { return x * x + y * y; },
                          domain,
                          Options{
                                    .gridSize={.x=48, .y=48},
                                    .maxBasisSize=20
                            });

    checkOrthogonality(domain, expected, s.eigenfunctions(expected.size()), 1e-4);
}

TEST_CASE("Sparse harmonic potential", "[harmonic][sparse]") {
    Rectangle<double, 2> domain{-8.0, 8.0, -8.0, 8.0};
    Schrodinger<double> s([](double x, double y) { return x * x + y * y; },
                          domain,
                          Options{
                                    .gridSize={.x=48, .y=48},
                                    .maxBasisSize=20,
                                    .sparse=true
                            });

    checkOrthogonality(domain, expected, s.eigenfunctions(expected.size()), 1e-4);
}

#ifdef SCHRODINGER_LONG_DOUBLE

TEST_CASE("Harmonic potential long double", "[harmonic][long double][sparse]") {
    typedef long double Scalar;
    Schrodinger<Scalar> s([](Scalar x, Scalar y) { return x * x + y * y; },
                            Rectangle<Scalar, 2>{-8.0, 8.0, -8.0, 8.0},
                            Options{
                                    .gridSize={.x=50, .y=50},
                                    .maxBasisSize=30,
                            });

    checkEigenvalues<Scalar, double>(expected, s.eigenvalues(expected.size()), 1e-8);
}

TEST_CASE("Sparse harmonic potential long double", "[harmonic][long double][sparse]") {
    typedef long double Scalar;
    Schrodinger<Scalar> s([](Scalar x, Scalar y) { return x * x + y * y; },
                            Rectangle<Scalar, 2>{-8.0, 8.0, -8.0, 8.0},
                            Options{
                                    .gridSize={.x=50, .y=50},
                                    .maxBasisSize=30,
                                    .sparse=true,
                                    .shiftInvert=false
                            });

    checkEigenvalues<Scalar, double>(expected, s.eigenvalues(expected.size()), 1e-8);
}

#endif
