#ifndef SCHRODINGER2D_SCHRODINGER_H
#define SCHRODINGER2D_SCHRODINGER_H

#include "catch.hpp"
#include "../src/schrodinger2d.h"
#include <vector>

template<typename Scalar, typename T=Scalar>
inline void
checkEigenvalues(const std::vector<T> &expected,
                 const std::vector<Scalar> &found, Scalar tolerance = 1e-8) {
    REQUIRE(expected.size() <= found.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(Approx(Scalar(expected[i])).epsilon(tolerance) == found[i]);
    }
}

template<typename Scalar, typename Eigenfunction>
inline void checkEigenpairs(
        const schrodinger::geometry::Domain<Scalar, 2> &domain,
        const std::vector<std::pair<Scalar, std::vector<std::function<Scalar(Scalar, Scalar)>>>> &expected,
        const std::vector<std::pair<Scalar, std::unique_ptr<Eigenfunction>>> &found, Scalar tolerance = 1e-8
) {
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> ArrayXs;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXXs;

    int n = 101;

    Scalar xmin, xmax, ymin, ymax;

    std::tie(xmin, xmax) = domain.bounds(Eigen::Matrix<Scalar, 2, 2>::Identity().col(0));
    std::tie(ymin, ymax) = domain.bounds(Eigen::Matrix<Scalar, 2, 2>::Identity().col(1));

    ArrayXs xs(n * n);
    ArrayXs ys(n * n);
    int k = 0;
    {
        ArrayXs xLinSpaced = ArrayXs::LinSpaced(n + 2, xmin, xmax);
        ArrayXs yLinSpaced = ArrayXs::LinSpaced(n + 2, ymin, ymax);

        Eigen::Matrix<Scalar, 2, 1> v;
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j) {
                v << xLinSpaced[i], yLinSpaced[j];
                if (domain.contains(v)) {
                    xs[k] = xLinSpaced[i];
                    ys[k] = yLinSpaced[j];
                    ++k;
                }
            }
    }
    xs.conservativeResize(k);
    ys.conservativeResize(k);

    MatrixXXs othogonalCheck(k, found.size());

    auto i = found.begin();
    for (auto &ef: expected) {
        const Scalar &e = ef.first;
        INFO("Eigenvalue " << e);

        MatrixXXs zs(k, ef.second.size());

        int l = 0;
        for (auto &f: ef.second) {
            for (int j = 0; j < k; ++j)
                zs(j, l) = f(xs[j], ys[j]);
            ++l;
        }

        for (size_t j = 0; j < ef.second.size(); ++j) {

            REQUIRE(i < found.end());
            REQUIRE(Approx(e).epsilon(tolerance) == i->first);

            ArrayXs v = (*i->second)(xs, ys);
            othogonalCheck.col(std::distance(found.begin(), i)) = v;

            ArrayXs x = (zs.transpose() * zs).ldlt().solve(zs.transpose() * v.matrix()).array();

            REQUIRE((zs * x.matrix() - v.matrix()).norm() < tolerance);

            ++i;
        }
    }

    for (; i < found.end(); ++i) {
        othogonalCheck.col(std::distance(found.begin(), i)) = (*i->second)(xs, ys);
    }

    MatrixXXs errors = ((xmax - xmin) * (ymax - ymin) / ((n + 1) * (n + 1) * n * n)) * othogonalCheck.transpose() *
                       othogonalCheck -
                       MatrixXXs::Identity(found.size(), found.size());
    // ignore diagonal
    errors.diagonal() *= 0;
    REQUIRE(errors.maxCoeff() < tolerance);
}

#endif //SCHRODINGER2D_SCHRODINGER_H
