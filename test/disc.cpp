#include "schrodinger.h"

using namespace Eigen;
using namespace schrodinger;
using namespace schrodinger::geometry;

const std::vector<double> expected{
        5.783185962946589, 14.681970642123728, 14.681970642123728, 26.374616427163247,
        26.374616427163247, 30.471262343662087, 40.706465818200314, 40.706465818200314,
        49.2184563216946, 49.2184563216946
};

TEST_CASE("Zero potential on a disc", "[disc]") {
    Schrodinger2D<double> s([](double, double) { return 0; },
                    Sphere<double, 2>({0, 0}, 1),
                    Options{
                            .gridSize={.x=40, .y=40},
                            .maxBasisSize=20
                    });

    checkEigenvalues<double>(expected, s.eigenvalues(), 1);
}

TEST_CASE("Zero potential eigenfunction", "[disc]") {
    Schrodinger2D<double> s([](double, double) { return 0; },
                            Sphere<double, 2>({0, 0}, 1),
                            Options{
                                    .gridSize={.x=40, .y=40},
                                    .maxBasisSize=20
                            });

    auto eigenfunctions = s.eigenfunctions();
    std::function<bool(std::pair<double, Schrodinger2D<double>::Eigenfunction>, std::pair<double, Schrodinger2D<double>::Eigenfunction>)> comp =
            [](auto a, auto b) {return a.first < b.first;};

    std::sort(eigenfunctions.begin(), eigenfunctions.end(), comp);

    /*
    printf("Eigenvalues:\n");
    for (int i = 0; i < (int)eigenfunctions.size() && i < (int)eigenfunctions.size(); i++) {
        printf("%d: %f\n", i, eigenfunctions[i].first);
        printf("Function value: %f\n", eigenfunctions[i].second(0.1, 0.2));
    }
     */

    printf("%f, %f\n", s.grid.x(10), s.grid.y(20));

}