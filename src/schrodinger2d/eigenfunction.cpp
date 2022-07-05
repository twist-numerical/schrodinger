#include "matslise/util/quadrature.h"
#include "../util/polymorphic_value.h"
#include "../schrodinger2d.h"
#include <numeric>
#include "../util/right_kernel.h"
#include "../util/rectangular_pencil.h"
#include <chrono>

using namespace std;
using namespace matslise;
using namespace Eigen;
using namespace isocpp_p0201;
using namespace schrodinger;

template<typename Scalar>
class Schrodinger2D<Scalar>::Eigenfunction::EigenfunctionTile {
public:
    EigenfunctionTile() {

    }
};

template<typename Scalar>
Schrodinger2D<Scalar>::Eigenfunction::Eigenfunction(const Schrodinger2D<Scalar> *problem, Scalar E, const VectorXs &c)
        : problem(problem), E(E), c(c) {

    // Initialize function values
    size_t numIntersections = problem->intersections.size();
    functionValues.x = VectorXs::Zero(numIntersections);
    functionValues.y = VectorXs::Zero(numIntersections);

    for (size_t i = 0; i < numIntersections; i++) {
        Intersection intersection = problem->intersections[i];

        const Thread *tx = intersection.thread.x;
        functionValues.x(i) = intersection.evaluation.x.matrix().dot(
                c.segment(tx->offset, tx->eigenpairs.size()));

        const Thread *ty = intersection.thread.y;
        functionValues.y(i) = intersection.evaluation.y.matrix().dot(
                c.segment(problem->columns.x + ty->offset, ty->eigenpairs.size()));
    }
}

template<typename Scalar>
Index highestLowerIndex(const Array<Scalar, Dynamic, 1> &range, Scalar value) {
    const Scalar *start = range.data();
    Index d = std::distance(start, std::lower_bound(start, start + range.size(), value));
    if (d >= range.size() || value < range[d])
        --d;
    return d;
}

template<typename Scalar, int Rows>
Array<Scalar, Rows, 1>
reconstructEigenfunction(const typename Schrodinger2D<Scalar>::Thread *t, const Ref<const Array<Scalar, Dynamic, 1>> &c,
                         const Array<Scalar, Rows, 1> &v) {
    Array<Scalar, Rows, 1> r = Array<Scalar, Rows, 1>::Zero();
    for (size_t i = 0; i < t->eigenpairs.size(); ++i)
        r += c(t->offset + i) * (*t->eigenpairs[i].second)(v).col(0);
    return r;
}

template<typename Scalar>
Scalar Schrodinger2D<Scalar>::Eigenfunction::operator()(Scalar x, Scalar y) const {

    ArrayXs xs = ArrayXs::Zero(1);
    xs(0) = x;
    ArrayXs ys = ArrayXs::Zero(1);
    ys(0) = y;

    return this->operator()(xs, ys)(0);
}


template<typename Scalar>
Eigen::Array<Scalar, Eigen::Dynamic, 1>
Schrodinger2D<Scalar>::Eigenfunction::operator()(ArrayXs xs, ArrayXs ys) const {
    assert(xs.size() == ys.size());

    // Enhanced Schrodinger method (finite difference method on a 3x3 grid)
    ArrayXs result = ArrayXs::Zero(xs.size());

    for (Index p = 0; p < xs.size(); p++) {
        Scalar x = xs(p);
        Scalar y = ys(p);

        Index ix = highestLowerIndex(problem->grid.x, x);
        Index iy = highestLowerIndex(problem->grid.y, y);

        if (ix < 0 || ix >= problem->options.gridSize.x || iy < 0 || iy >= problem->options.gridSize.y) continue;

        const Tile &tile = problem->tiles(ix + 1, iy + 1);

        Scalar xOffset = problem->grid.x[ix];
        Scalar yOffset = problem->grid.y[iy];
        Scalar hx = problem->grid.x[1] - problem->grid.x[0];
        Scalar hy = problem->grid.y[1] - problem->grid.y[0];

        Eigen::Vector<Scalar, 5> xpoints, ypoints;
        xpoints << xOffset, xOffset + 0.25 * hx, xOffset + 0.5 * hx, xOffset + 0.75 * hx, xOffset + hx;
        ypoints << yOffset, yOffset + 0.25 * hy, yOffset + 0.5 * hy, yOffset + 0.75 * hy, yOffset + hy;

        // get function values for 3 points on each of the 4 sides (boundary conditions)
        Matrix<Scalar, 5, 5> gridValues = Matrix<Scalar, 5, 5>::Zero();

        if (tile.intersections[0] == nullptr || tile.intersections[1] == nullptr
            || tile.intersections[2] == nullptr || tile.intersections[3] == nullptr)
            continue;

        Array<Scalar, 3, 1> input;
        Array<Scalar, 3, 1> output;
        input << ypoints.segment(1, 3);
        output = reconstructEigenfunction<Scalar>(
                tile.intersections[0]->thread.x, c.topRows(problem->columns.x), input);
        gridValues.row(0).template segment<3>(1) = output;

        output = reconstructEigenfunction<Scalar>(
                tile.intersections[1]->thread.x, c.topRows(problem->columns.x), input);
        gridValues.row(4).template segment<3>(1) = output;

        input << xpoints.segment(1, 3);
        output = reconstructEigenfunction<Scalar>(
                tile.intersections[0]->thread.y, c.bottomRows(problem->columns.y), input);
        gridValues.col(0).template segment<3>(1) = output;

        output = reconstructEigenfunction<Scalar>(
                tile.intersections[2]->thread.y, c.bottomRows(problem->columns.y), input);
        gridValues.col(4).template segment<3>(1) = output;

        // finite diff formula coefficients
        Matrix<Scalar, 3, 5> coeff;
        coeff <<
              11, -20, 6, 4, -1,
                -1, 16, -30, 16, -1,
                -1, 4, 6, -20, 11;
        coeff /= 12;

        Matrix<Scalar, 9, 9> A = Matrix<Scalar, 9, 9>::Zero();
        Eigen::Vector<Scalar, 9> B = Eigen::Vector<Scalar, 9>::Zero();

        for (int rx = 0; rx < 3; rx++) {
            for (int ry = 0; ry < 3; ry++) {
                // - D^2 psi + (V - E) * psi = 0

                A(rx + ry * 3, rx + ry * 3) =
                        problem->V(xOffset + (rx + 1) * hx / 4., yOffset + (ry + 1) * hy / 4.) - E;

                // Horizontal formula
                A(rx + ry * 3, 0 + ry * 3) -= 16. / (hx * hx) * coeff(rx, 1);
                A(rx + ry * 3, 1 + ry * 3) -= 16. / (hx * hx) * coeff(rx, 2);
                A(rx + ry * 3, 2 + ry * 3) -= 16. / (hx * hx) * coeff(rx, 3);
                B(rx + ry * 3) += 16. / (hx * hx) * coeff(rx, 0) * gridValues(0, ry + 1);
                B(rx + ry * 3) += 16. / (hx * hx) * coeff(rx, 4) * gridValues(4, ry + 1);

                // Vertical formula
                A(rx + ry * 3, rx + 0) -= 16. / (hy * hy) * coeff(ry, 1);
                A(rx + ry * 3, rx + 3) -= 16. / (hy * hy) * coeff(ry, 2);
                A(rx + ry * 3, rx + 6) -= 16. / (hy * hy) * coeff(ry, 3);
                B(rx + ry * 3) += 16. / (hy * hy) * coeff(ry, 0) * gridValues(rx + 1, 0);
                B(rx + ry * 3) += 16. / (hy * hy) * coeff(ry, 4) * gridValues(rx + 1, 4);
            }
        }

        // Solve system
        Eigen::Vector<Scalar, 9> sol = A.partialPivLu().solve(B);

        // 5x5 grid interpolation
        for (size_t j = 0; j < 4; j++) {
            if (tile.intersections[j] != nullptr)
                gridValues((j % 2) * 4, (j / 2) * 4) = functionValues.x[tile.intersections[j]->index];
        }

        gridValues.col(1).template segment<3>(1) = sol.template segment<3>(0);
        gridValues.col(2).template segment<3>(1) = sol.template segment<3>(3);
        gridValues.col(3).template segment<3>(1) = sol.template segment<3>(6);



        // horizontal Lagrange polynomials
        Eigen::Vector<Scalar, 5> lx;
        for (int i = 0; i < 5; i++) {
            lx(i) = 1;
            for (int j = 0; j < 5; j++)
                if (j != i)
                    lx(i) *= (x - xpoints(j)) / (xpoints(i) - xpoints(j));
        }

        // vertical Lagrange polynomials
        Eigen::Vector<Scalar, 5> ly;
        for (int i = 0; i < 5; i++) {
            ly(i) = 1;
            for (int j = 0; j < 5; j++)
                if (j != i)
                    ly(i) *= (y - ypoints(j)) / (ypoints(i) - ypoints(j));
        }

        for (int rx = 0; rx < 5; rx++) {
            for (int ry = 0; ry < 5; ry++) {
                result(p) += gridValues(rx, ry) * lx(rx) * ly(ry);
            }
        }
    }

    return result;
}

template<typename Scalar>
Schrodinger2D<Scalar>::Eigenfunction::~Eigenfunction() = default;

#include "instantiate.h"
