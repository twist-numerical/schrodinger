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

TEST_CASE("Harmonic potential eigenfunction", "[harmonic]") {
    Schrodinger2D<double> s([](double x, double y) { return x * x + y * y; },
                            Rectangle<double, 2>{-8.0, 8.0, -8.0, 8.0},
                            Options{
                                    .gridSize={.x=40, .y=40},
                                    .maxBasisSize=15,
                                    .pencilMethod=-1,
                                    .interpolationMethod=7
                            });

    auto eigenfunctions = s.eigenfunctions();
    std::function<bool(std::pair<double, Schrodinger2D<double>::Eigenfunction>, std::pair<double, Schrodinger2D<double>::Eigenfunction>)> comp =
            [](auto a, auto b) {return a.first < b.first;};

    std::sort(eigenfunctions.begin(), eigenfunctions.end(), comp);

    printf("Eigenvalues:\n");
    for (int i = 0; i < (int)eigenfunctions.size() && i < (int)eigenfunctions.size(); i++) {
        printf("%d: %f\n", i, eigenfunctions[i].first);

        // 100x100 grid
        /*
        Schrodinger2D<double>::ArrayXs xs(100*100);
        Schrodinger2D<double>::ArrayXs ys(100*100);
        Array<double, Eigen::Dynamic, 1> grid = Array<double, Eigen::Dynamic, 1>::LinSpaced(102, -8, 8);
        for (int xi = 0; xi < 100; xi++) {
            for (int yi = 0; yi < 100; yi++) {
                xs(xi + 100*yi) = grid(xi+1);
                ys(xi + 100*yi) = grid(yi+1);
            }
        }
*/


        Schrodinger2D<double>::ArrayXs xs(4);
        Schrodinger2D<double>::ArrayXs ys(4);
        xs << 0.1, 0.1, 0.1, 0.1;
        ys << 0.2, 0.3, 0.4, 0.5;


        Schrodinger2D<double>::ArrayXs funValues = eigenfunctions[i].second(xs, ys);

        printf("Result:");
        //for (int j = 0; j < funValues.size(); j++) printf("%.3f; ", funValues(j));
        printf("\n");

    }

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
