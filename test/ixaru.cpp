#include "check_eigenvalues.h"

using namespace Eigen;
using namespace strands;
using namespace strands::geometry;

const std::vector<double> expected{
        3.1959180863164467, 5.526743874681427, 5.526743875458036,
        7.557803329778595, 8.031272341129998, 8.444581359170941,
        9.928061056175023, 9.928061059792322, 11.311817048862636,
        11.311817048862636, 12.103253580682654, 12.20117897122486,
        13.33233126944338, 14.348268533723068, 14.348268539359145,
        14.450478722829562, 14.580556316639049
};

TEST_CASE("Ixaru potential vertical", "[ixaru]") {
    Schrodinger<double> s([](double x, double y) { return (1 + x * x) * (1 + y * y); },
                          Rectangle<double, 2>(-5.5, 5.5, -5.5, 5.5),
                          Options{
                                  .gridSize={.x=28, .y=30},
                                  .maxBasisSize=18
                          });

    checkEigenvalues<double>(expected, s.eigenvalues(EigensolverOptions{
            .k = (Eigen::Index) expected.size()
    }), 1e-4);
}

TEST_CASE("Ixaru potential horizontal", "[ixaru]") {
    Schrodinger<double> s([](double x, double y) { return (1 + x * x) * (1 + y * y); },
                          Rectangle<double, 2>(-5.5, 5.5, -5.5, 5.5),
                          Options{
                                  .gridSize={.x=30, .y=28},
                                  .maxBasisSize=18
                          });

    checkEigenvalues<double>(expected, s.eigenvalues(EigensolverOptions{
            .k = (Eigen::Index) expected.size()
    }), 1e-4);
}

TEST_CASE("Ixaru potential diamond", "[ixaru]") {
    Schrodinger<double> s([](double x, double y) { return (1 + x * x) * (1 + y * y); },
                          DomainTransform<double, 2>(Rectangle<double, 2>(-5.5, 5.5, -5.5, 5.5),
                                                     Rotation2D<double>(M_PI / 2)),
                          Options{
                                  .gridSize={.x=30, .y=30},
                                  .maxBasisSize=18
                          });

    checkEigenvalues<double>(expected, s.eigenvalues(EigensolverOptions{
            .k = (Eigen::Index) expected.size()
    }), 1e-4);
}

TEST_CASE("Ixaru potential disc", "[ixaru]") {
    Schrodinger<double> s([](double x, double y) { return (1 + x * x) * (1 + y * y); },
                          Sphere<double, 2>(5.5),
                          Options{
                                  .gridSize={.x=30, .y=30},
                                  .maxBasisSize=18
                          });

    checkEigenvalues<double>(expected, s.eigenvalues(EigensolverOptions{
            .k = (Eigen::Index) expected.size()
    }), 1e-4);
}

#ifdef STRANDS_LONG_DOUBLE

TEST_CASE("Ixaru potential long double", "[ixaru][long double]") {
    typedef long double Scalar;
    Schrodinger<Scalar> s([](Scalar x, Scalar y) { return (1 + x * x) * (1 + y * y); },
                          Rectangle<Scalar, 2>(-5.5, 5.5, -5.5, 5.5),
                          Options{
                                  .gridSize={.x=35, .y=35},
                                  .maxBasisSize=20
                          });

    checkEigenvalues<Scalar, double>(expected, s.eigenvalues(EigensolverOptions{
            .k = (Eigen::Index) expected.size()
    }), 1e-8);
}

#endif
