#include "check_eigenvalues.h"

using namespace Eigen;
using namespace strands;
using namespace strands::geometry;

inline std::vector<double> referenceHenonHeiles{
        2 * 0.998594690530479, 2 * 1.99007660445524, 2 * 1.99007660445524, 2 * 2.95624333869018,
        2 * 2.98532593386986, 2 * 2.98532593386986, 2 * 3.92596412795287, 2 * 3.92596412795287,
        2 * 3.98241882458866, 2 * 3.98575763690663, 2 * 4.87014557482289, 2 * 4.89864497284387, 2 * 4.89864497284387
};

template<typename T, typename... Args>
T timedConstructor([[maybe_unused]]  const std::string &profile_name, Args... args) {
    MATSLISE_SCOPED_TIMER(profile_name);
    return T(args...);
}

TEST_CASE("Profile sparse Henon Heiles", "[henonheiles][sparse][profile]") {
    MATSLISE_SCOPED_TIMER("Profile Hénon-Heiles");
    auto s = timedConstructor<Schrodinger<double>>(
            "construct Schrodinger",
            [](double x, double y) { return x * x + y * y + sqrt(5) / 10 * (x * y * y - x * x * x / 3); },
            Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
            Options{
                    .gridSize={.x=64, .y=64},
                    .maxBasisSize=48,
            });

    std::vector<double> eigenvalues;
    {
        MATSLISE_SCOPED_TIMER("calculate eigenvalues");
        eigenvalues = s.eigenvalues(EigensolverOptions{
                .k = 100,
                .sparse=true,
                .shiftInvert=false
        });
    }

    {
        MATSLISE_SCOPED_TIMER("check eigenvalues");
        checkEigenvalues<double>(referenceHenonHeiles, eigenvalues, 1e-4);
    }
}

TEST_CASE("Profile sparse Henon Heiles (shiftInvert)", "[henonheiles][sparse][profile]") {
    MATSLISE_SCOPED_TIMER("Profile Hénon-Heiles (shiftInvert)");
    auto s = timedConstructor<Schrodinger<double>>(
            "construct Schrodinger",
            [](double x, double y) { return x * x + y * y + sqrt(5) / 10 * (x * y * y - x * x * x / 3); },
            Rectangle<double, 2>{-6.0, 6.0, -6.0, 6.0},
            Options{
                    .gridSize={.x=64, .y=64},
                    .maxBasisSize=48,
            });

    std::vector<double> eigenvalues;
    {
        MATSLISE_SCOPED_TIMER("calculate eigenvalues");
        eigenvalues = s.eigenvalues(EigensolverOptions{
                .k = 100,
                .sparse=true,
                .shiftInvert=true
        });
    }

    {
        MATSLISE_SCOPED_TIMER("check eigenvalues");
        checkEigenvalues<double>(referenceHenonHeiles, eigenvalues, 1e-4);
    }
}
