#include "schrodinger.h"

using namespace Eigen;
using namespace schrodinger;
using namespace schrodinger::geometry;

const std::vector<double> expected{
        3.1959181, 5.5267439, 5.5267439, 7.5578033, 8.0312723,
        8.4445814, 9.9280611, 9.9280611, 11.3118171, 11.3118171,
        12.1032536, 12.2011790, 13.3323313
};

TEST_CASE("Ixaru potential vertical", "[ixaru]") {
    Schrodinger2D s([](double x, double y) { return (1 + x * x) * (1 + y * y); },
                    Rectangle<double, 2>(-5.5, 5.5, -5.5, 5.5),
                    Options{
                            .gridSize={.x=28, .y=30},
                            .maxBasisSize=14
                    });

    checkEigenvalues(expected, s.eigenvalues(), 1e-6);
}

TEST_CASE("Ixaru potential horizontal", "[ixaru]") {
    Schrodinger2D s([](double x, double y) { return (1 + x * x) * (1 + y * y); },
                    Rectangle<double, 2>(-5.5, 5.5, -5.5, 5.5),
                    Options{
                            .gridSize={.x=30, .y=28},
                            .maxBasisSize=16
                    });

    checkEigenvalues(expected, s.eigenvalues(), 1e-6);
}

TEST_CASE("Ixaru potential diamond", "[ixaru]") {
    Schrodinger2D s([](double x, double y) { return (1 + x * x) * (1 + y * y); },
                    Rectangle<double, 2>(-5.5, 5.5, -5.5, 5.5),
                    Options{
                            .gridSize={.x=30, .y=28},
                            .maxBasisSize=16
                    });

    checkEigenvalues(expected, s.eigenvalues(), 1e-6);
}