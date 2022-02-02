//
// Created by Django on 003 03/12/2021.
//

#include "schrodinger.h"

using namespace Eigen;
using namespace schrodinger;
using namespace schrodinger::geometry;

TEST_CASE("Henon Heiles", "[henonheiles][slow]") {
    //int n = 40;
    //int N = 15;


    int n = 18;
    int N = 15;
    //Schrodinger2D<double> s([](double x, double y) { return x*x + y*y + sqrt(5)/10 * (x*y*y - x*x*x / 3); },
    //                        Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
    Schrodinger2D<double> s([](double x, double y) { return (1+x*x)*(1+y*y); },
                            Rectangle<double, 2>{-5.5, 5.5, -5.5, 5.5},
                            Options{
                                    .gridSize={.x=n, .y=n},
                                    .maxBasisSize=N
                            });

    auto eigenfunctions = s.eigenfunctions();

    // Sort eigenfunctions
    std::function<bool(std::pair<double, Schrodinger2D<double>::Eigenfunction>, std::pair<double, Schrodinger2D<double>::Eigenfunction>)> comp =
            [](auto a, auto b) {return a.first < b.first;};
    std::sort(eigenfunctions.begin(), eigenfunctions.end(), comp);

    for (int i = 0; i < (int)eigenfunctions.size(); i++) {
        double E = eigenfunctions[i].first;
        Schrodinger2D<double>::Eigenfunction phi = eigenfunctions[i].second;
        printf("Eigenvalue: %f\n", E);
    }
}

TEST_CASE("test_pencil", "") {
    MatrixXd A = MatrixXd::Zero(14, 16);
    A(0, 3) = 1;
    A(1, 5) = 1; A(2, 6) = 1;
    A(4, 7) = 1; A(5, 8) = 1; A(6, 9) = 1;
    A(8, 10) = 1; A(9, 11) = 1; A(10, 12) = 1;
    A(11, 13) = 2; A(12, 14) = 3; A(13, 15) = 3;
    A(13, 14) = -1;

    MatrixXd B = MatrixXd::Zero(14, 16);
    B(0, 2) = 1;
    B(1, 4) = 1; B(2, 5) = 1;
    B(5, 7) = 1; B(6, 8) = 1; B(7, 9) = 1;
    B(10, 11) = 1;
    B(11, 13) = 1; B(12, 14) = 1; B(13, 15) = 1;

    MatrixXd temp;
    temp = A.adjoint(); A = temp;
    temp = B.adjoint(); B = temp;

    Eigen::ColPivHouseholderQR<MatrixXd> QR(A+20*B);
    int r = (int) QR.rank();
    printf("Rank %d\n", r);

    printf("A, B: %dx%d\n", (int)A.rows(), (int)A.cols());
    // int M = (int) std::min(A.rows(), A.cols());
    int M = 10;

    std::cout << A << std::endl;
    std::cout << std::endl;
    std::cout << B << std::endl;

    Eigen::BDCSVD<MatrixXd> svdA;
    svdA.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    MatrixXd A_Ut = svdA.matrixU().leftCols(M);
    MatrixXd A_St = svdA.singularValues().head(M).asDiagonal();
    MatrixXd A_Vt = svdA.matrixV().leftCols(M);

    Eigen::BDCSVD<MatrixXd> svdB;
    svdB.compute(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatrixXd B_Ut = svdB.matrixU().leftCols(M);
    MatrixXd B_St = svdB.singularValues().head(M);
    MatrixXd B_Vt = svdB.matrixV().leftCols(M);

    std::cout << "B sing values" << std::endl;
    std::cout << B_St << std::endl;

    MatrixXd G = B_Ut.adjoint() * A_Ut * A_St * A_Vt.adjoint() * B_Vt;
    G *= B_St.array().inverse().matrix().asDiagonal();

    Eigen::EigenSolver<MatrixXd> eigenSolver;
    eigenSolver.compute(G);
    std::cout << "Eigenvalues:" << std::endl;
    std::cout << eigenSolver.eigenvalues() << std::endl;

}

TEST_CASE("Henon Heiles interpolation", "[henonheilesinterpolation][slow]") {
    int n = 25;
    int N = 25;

    Schrodinger2D<double> s([](double x, double y) { return x*x + y*y + sqrt(5)/10 * (x*y*y - x*x*x / 3); },
                            Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
                            Options{
                                    .gridSize={.x=n, .y=n},
                                    .maxBasisSize=N
                            });

    Schrodinger2D<double> c([](double x, double y) { return x*x + y*y + sqrt(5)/10 * (x*y*y - x*x*x / 3); },
                            Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
                            Options{
                                    .gridSize={.x=3*n+2, .y=3*n+2},
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

    }

}