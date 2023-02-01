#include "check_eigenvalues.h"

using namespace strands;
using namespace strands::geometry;

const std::vector<double> expected{
        2, 4, 4, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10, 10, 10, 12, 12,
        12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16,
        16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20,
};

TEST_CASE("Harmonic potential", "[harmonic]") {
    Rectangle<double, 2> domain{-8.0, 8.0, -8.0, 8.0};
    Schrodinger<double> s([](double x, double y) { return x * x + y * y; },
                          domain,
                          Options{
                                  .gridSize={.x=48, .y=48},
                                  .maxBasisSize=20
                          });

    checkOrthogonality(domain, expected, s.eigenfunctions(EigensolverOptions{
            .k = (Eigen::Index) expected.size()
    }), 1e-4);
}

TEST_CASE("Sparse harmonic potential", "[harmonic][sparse]") {
    Rectangle<double, 2> domain{-8.0, 8.0, -8.0, 8.0};
    Schrodinger<double> s([](double x, double y) { return x * x + y * y; },
                          domain,
                          Options{
                                  .gridSize={.x=48, .y=48},
                                  .maxBasisSize=20,
                          });

    checkOrthogonality(domain, expected, s.eigenfunctions(EigensolverOptions{
            .k = (Eigen::Index) expected.size(),
            .sparse = true
    }), 1e-4);
}

TEST_CASE("Sparse harmonic potential, large grid", "[harmonic][sparse]") {
    Rectangle<double, 2> domain{-9.5, 9.5, -9.5, 9.5};
    Schrodinger<double> s([](double x, double y) { return x * x + y * y; },
                          domain,
                          Options{
                                  .gridSize={.x=64, .y=64},
                                  .maxBasisSize=32,
                          });

    checkOrthogonality(domain, expected, s.eigenfunctions(EigensolverOptions{
            .k = (Eigen::Index) expected.size(),
            .ncv =  4 * (Eigen::Index) expected.size(),
            .sparse = true
    }), 1e-4);
}

TEST_CASE("Sparse harmonic potential, large grid, without shiftInvert", "[harmonic][sparse][no_invert]") {
    Rectangle<double, 2> domain{-9.5, 9.5, -9.5, 9.5};
    Schrodinger<double> s([](double x, double y) { return x * x + y * y; },
                          domain,
                          Options{
                                  .gridSize={.x=64, .y=64},
                                  .maxBasisSize=32,
                          });

    checkOrthogonality(domain, expected, s.eigenfunctions(EigensolverOptions{
            .k = (Eigen::Index) expected.size(),
            .ncv =  4 * (Eigen::Index) expected.size(),
            .sparse = true,
            .shiftInvert = false,
    }), 1e-4);
}

TEST_CASE("Sparse harmonic potential on disc", "[harmonic][sparse][disc]") {
    Sphere<double, 2> domain{9.5};
    Schrodinger<double> s([](double x, double y) { return x * x + y * y; },
                          domain,
                          Options{
                                  .gridSize={.x=64, .y=64},
                                  .maxBasisSize=48,
                          });

    checkOrthogonality(domain, expected, s.eigenfunctions(EigensolverOptions{
            .k = (Eigen::Index) expected.size(),
            .ncv =  4 * (Eigen::Index) expected.size(),
            .sparse = true,
            .shiftInvert = true,
    }), 1e-4);
}

#ifdef STRANDS_LONG_DOUBLE

TEST_CASE("Harmonic potential long double", "[harmonic][long double][sparse]") {
    typedef long double Scalar;
    Schrodinger<Scalar> s([](Scalar x, Scalar y) { return x * x + y * y; },
                          Rectangle<Scalar, 2>{-8.0, 8.0, -8.0, 8.0},
                          Options{
                                  .gridSize={.x=50, .y=50},
                                  .maxBasisSize=30,
                          });

    checkEigenvalues<Scalar, double>(expected, s.eigenvalues(EigensolverOptions{
            .k = (Eigen::Index) expected.size(),
    }), 1e-8);
}

TEST_CASE("Sparse harmonic potential long double", "[harmonic][long double][sparse]") {
    typedef long double Scalar;
    Schrodinger<Scalar> s([](Scalar x, Scalar y) { return x * x + y * y; },
                          Rectangle<Scalar, 2>{-8.0, 8.0, -8.0, 8.0},
                          Options{
                                  .gridSize={.x=50, .y=50},
                                  .maxBasisSize=30,
                          });

    checkEigenvalues<Scalar, double>(expected, s.eigenvalues(EigensolverOptions{
            .k = (Eigen::Index) expected.size(),
            .sparse=true,
            .shiftInvert=false
    }), 1e-8);
}

#endif
