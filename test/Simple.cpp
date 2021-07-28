#include "catch.hpp"
#include "../src/Schrodinger2D.h"

using namespace Eigen;

TEST_CASE("On a disc", "[disc]") {
    Schrodinger2D s([](double, double) { return 0; },
                    Sphere<double, 2>({0., 0.}, 1),
                    Options{
                            .gridSize={.x=28, .y=30},
                            .maxBasisSize=12
                    });
    std::vector<std::pair<double, Schrodinger2D::Eigenfunction>> eigs = s.eigenfunctions();
    int count = 0;
    for (auto &ef : eigs) {
        ArrayXd xs = ArrayXd::LinSpaced(41, -1, 1);
        ArrayXd ys = ArrayXd::LinSpaced(29, -1, 1);
        for (Index i = 0; i < xs.size(); ++i)
            for (Index j = 0; j < ys.size(); ++j)
                ef.second(xs[i], ys[j]);

        if (++count > 5)
            break;
    }
}

TEST_CASE("On two disc", "[disc]") {
    Schrodinger2D s([](double, double) { return 0; },
                    Union<double, 2>{
                            Sphere<double, 2>({-3.5, 0.}, 4),
                            Sphere<double, 2>({3.5, 0.}, 4),
                    },
                    Options{
                            .gridSize={.x=60, .y=60},
                            .maxBasisSize=12
                    });
    std::vector<std::pair<double, Schrodinger2D::Eigenfunction>> eigs = s.eigenfunctions();
    int count = 0;
    for (auto &ef : eigs) {
        ArrayXd xs = ArrayXd::LinSpaced(41, -1, 1);
        ArrayXd ys = ArrayXd::LinSpaced(29, -1, 1);
        for (Index i = 0; i < xs.size(); ++i)
            for (Index j = 0; j < ys.size(); ++j)
                ef.second(xs[i], ys[j]);

        if (++count > 5)
            break;
    }
}

TEST_CASE("On a square dumbbell", "[dumbbell]") {
    double r = 5;
    Union<double, 2> domain{
            Rectangle<double, 2>(-r / 4, r, -r / 4, r), Rectangle<double, 2>(-r, r / 4, -r, r / 4)
    };
    Schrodinger2D s([](double x, double y) { return x * x + y * y; }, domain,
                    Options{
                            .gridSize={.x=50, .y=50},
                            .maxBasisSize=16
                    });

    s.eigenvalues();
}