//
// Created by Django on 003 03/12/2021.
//

#include "schrodinger.h"

using namespace Eigen;
using namespace schrodinger;
using namespace schrodinger::geometry;

TEST_CASE("Henon Heiles", "[henonheiles]") {

    std::vector<int> n_values = {10, 15, 20, 30, 40, 60, 80, 120};
    int k = 8;

    for (int n: n_values) {
        int N = 20;

        // Schrodinger2D<double> s([](double x, double y) { return (1 + x*x) * (1 + y*y); },
        //                         Rectangle<double, 2>{-5.5, 5.5, -5.5, 5.5},
        Schrodinger2D<double> s([](double x, double y) { return x*x + y*y + sqrt(5)/10 * (x*y*y - x*x*x / 3); },
                                Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
                                Options{
                                        .gridSize={.x=n, .y=n},
                                        .maxBasisSize=N
                                });

        // Write out eigenvalues
        std::vector<double> eigenvalues = s.eigenvalues();
        printf("[");
        for (int i = 0; i < k; i++) {
            printf("%.16f", eigenvalues[i]);
            if (i != k-1) printf(", ");
        }
        printf("],\n");
    }
}
