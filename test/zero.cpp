#include "check_eigenvalues.h"
#include <map>
#include <tuple>
#include <vector>

using namespace Eigen;
using namespace strands;
using namespace strands::geometry;

const std::vector<double> expected_on_disc{
        5.783185962946589, 14.681970642123728, 14.681970642123728, 26.374616427163247,
        26.374616427163247, 30.471262343662087, 40.706465818200314, 40.706465818200314,
        49.2184563216946, 49.2184563216946
};

TEST_CASE("Zero potential on a disc", "[zero][disc]") {
    Schrodinger<double> s([](double, double) { return 0; },
                          Sphere<double, 2>({0, 0}, 1),
                          Options{
                                  .gridSize={.x=40, .y=40},
                                  .maxBasisSize=20
                          });

    std::vector<double> eigenvalues = s.eigenvalues(EigensolverOptions{
            .k = 10
    });
    REQUIRE(eigenvalues.size() == 10);
    checkEigenvalues<double>(expected_on_disc, eigenvalues, 1);
}

TEST_CASE("Zero potential eigenfunction", "[zero]") {
    double pi = 3.14159265358;
    Rectangle<double, 2> domain{0., pi, 0., pi};
    Schrodinger<double> s([](double, double) { return 0; },
                          domain,
                          Options{
                                  .gridSize={.x=30, .y=30},
                                  .maxBasisSize=12
                          });

    int total = 0;
    std::map<int, std::vector<std::pair<int, int>>> sumOfSquares;
    for (int i = 1; i < 10; ++i)
        for (int j = 1; j < 10; ++j)
            if (i * i + j * j < 100) {
                sumOfSquares[i * i + j * j].emplace_back(i, j);
                ++total;
            }

    auto eigenfunctions = s.eigenfunctions(EigensolverOptions{
            .k = total
    });
    REQUIRE((int) eigenfunctions.size() >= total);
    std::sort(eigenfunctions.begin(), eigenfunctions.end(), [](auto &a, auto &b) { return a.first < b.first; });
    eigenfunctions.erase(eigenfunctions.begin() + total, eigenfunctions.end());

    std::vector<std::pair<double, std::vector<std::function<double(double, double)>>>> expected;
    for (auto &sij: sumOfSquares) {
        std::vector<std::function<double(double, double)>> functions;
        for (auto &ij: sij.second) {
            int i, j;
            std::tie(i, j) = ij;
            functions.emplace_back([=](double x, double y) {
                return std::sin(i * x) * std::sin(j * y);
            });
        }
        expected.emplace_back(sij.first, functions);
    }

    checkEigenpairs<double>(domain, expected, eigenfunctions, 1e-3);
}
