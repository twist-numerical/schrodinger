//
// Created by Django on 003 03/12/2021.
//

#include "schrodinger.h"

using namespace Eigen;
using namespace schrodinger;
using namespace schrodinger::geometry;

TEST_CASE("Henon Heiles", "[henonheiles][slow]") {
    int n = 40;
    int N = 15;

    Schrodinger2D<double> s([](double x, double y) { return x*x + y*y + sqrt(5)/10 * (x*y*y - x*x*x / 3); },
                            Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
                            Options{
                                    .gridSize={.x=n, .y=n},
                                    .maxBasisSize=N
                            });

    auto eigenfunctions = s.eigenfunctions();

    // Sort eigenfunctions
    std::function<bool(std::pair<double, Schrodinger2D<double>::Eigenfunction>, std::pair<double, Schrodinger2D<double>::Eigenfunction>)> comp =
            [](auto a, auto b) {return a.first < b.first;};
    std::sort(eigenfunctions.begin(), eigenfunctions.end(), comp);

    for (int i = 0; i < 4; i++) {
        double E = eigenfunctions[i].first;
        Schrodinger2D<double>::Eigenfunction phi = eigenfunctions[i].second;
        printf("Eigenvalue: %f\n", E);
    }
}

TEST_CASE("Henon Heiles interpolation", "[henonheilesinterpolation][slow]") {
    int n = 30;
    int N = 10;

    Schrodinger2D<double> s([](double x, double y) { return x*x + y*y + sqrt(5)/10 * (x*y*y - x*x*x / 3); },
                            Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
                            Options{
                                    .gridSize={.x=n, .y=n},
                                    .maxBasisSize=N
                            });

    Schrodinger2D<double> c([](double x, double y) { return x * x + y * y + sqrt(5) / 10 * (x * y * y - x * x * x / 3); },
                            Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
                            Options{
                                    .gridSize={.x=4*n+3, .y=4*n+3},
                                    .maxBasisSize=N
                            });

    // Single eigenvalues
    auto eigenfunctions = s.eigenfunctions();
    auto eigenfunctions2 = c.eigenfunctions();
    // Sort eigenvalues
    std::function<bool(std::pair<double, Schrodinger2D<double>::Eigenfunction>, std::pair<double, Schrodinger2D<double>::Eigenfunction>)> comp =
            [](auto a, auto b) {return a.first < b.first;};

    std::sort(eigenfunctions.begin(), eigenfunctions.end(), comp);
    std::sort(eigenfunctions2.begin(), eigenfunctions2.end(), comp);

    // Get function value on grid point
    double x = s.grid.x[5];
    double y = s.grid.y[17];

    printf("Grid point: %f, %f\n", x, y);

    std::vector<int> k_values = {0, 3, 8, 9};

    printf("Eigenvalues:\n");
    for (int i = 0; i < (int)eigenfunctions.size() && i < (int)eigenfunctions2.size(); i++) {
        printf("%d: %f; %f\n", i, eigenfunctions[i].first, eigenfunctions2[i].first);
    }

    for (int k : k_values) {
        auto& p = eigenfunctions[k];
        auto& p2 = eigenfunctions2[k];

        printf("Eigenvalue: %f, %f\n", p.first, p2.first);

        // Rel error of grid point
        double val1 = p.second(x, y);
        double val2 = p2.second(x, y);

        double sum1 = 0;
        for (auto& intersection : s.intersections) {
            // Calculate function value in intersection
            double functionVal = p.second(intersection.position.x, intersection.position.y);
            sum1 += functionVal*functionVal;
        }
        double sum2 = 0;
        for (auto& intersection : c.intersections) {
            // Calculate function value in intersection
            double functionVal = p2.second(intersection.position.x, intersection.position.y);
            sum2 += functionVal*functionVal;
        }

        // Just print out all the values
        for (int i = 0; i < s.grid.x.size(); i++) {
            double xp = s.grid.x(i);
            printf("[");
            for (int j = 0; j < s.grid.y.size(); j++) {
                double yp = s.grid.y(j);
                printf("%f", p.second(xp, yp));
                if (j != s.grid.y.size()-1) printf(",");
            }
            printf("],\n");
        }
        printf("\n");

        for (int i = 0; i < c.grid.x.size(); i+=4) {
            double xp = c.grid.x(i);
            printf("[");
            for (int j = 0; j < c.grid.y.size(); j+=4) {
                double yp = c.grid.y(j);
                printf("%f", p2.second(xp, yp));
                if (j < c.grid.y.size()-5) printf(",");
            }
            printf("],\n");
        }
        printf("\n");

        return;

        printf("Total sum 1: %f\n", sum1);
        printf("Total sum 2: %f\n", sum2);

        printf("Val1: %f, val2: %f\n", val1, val2);

        double err = abs((val1 - val2) / val2);

        printf("Rel err: %f\n", err);



    }

}