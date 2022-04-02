#include "matslise/util/quadrature.h"
#include "util/polymorphic_value.h"
#include "schrodinger2d.h"
#include <numeric>
#include "util/right_kernel.h"
#include "util/rectangular_pencil.h"
#include <chrono>

using namespace std;
using namespace matslise;
using namespace Eigen;
using namespace isocpp_p0201;
using namespace schrodinger;
using namespace schrodinger::geometry;

template<typename T>
struct IterateEigen {
private:
    T *t;

public:
    IterateEigen(T &_t) : t(&_t) {
    };

    auto begin() {
        return t->data();
    };

    auto end() {
        return t->data() + t->size();
    };
};

template<typename Scalar>
typename Schrodinger2D<Scalar>::Thread
computeThread(const std::function<Scalar(Scalar)> &V, Scalar min, Scalar max, size_t maxPairs,
              const Ref<const Array<Scalar, Dynamic, 1>> &grid, size_t &offset) {
    typedef IterateEigen<const Ref<const Array<Scalar, Dynamic, 1>>> Iterate;

    const Scalar *from = std::lower_bound(Iterate(grid).begin(), Iterate(grid).end(), min);
    if (*from == min)
        ++from;

    Index length = std::distance(from, std::lower_bound(from, Iterate(grid).end(), max));
    typename Schrodinger2D<Scalar>::Thread thread = {
            .value = 0,
            .valueIndex = 0,
            .offset = offset,
            .gridOffset = std::distance(Iterate(grid).begin(), from),
            .gridLength = length,
            .matslise = nullptr,
            .eigenpairs = std::vector<std::pair<Scalar, std::unique_ptr<typename matslise::Matslise<Scalar>::Eigenfunction>>>(),
    };
    if (length == 0) return thread;
    thread.matslise = make_unique<Matslise<Scalar>>(V, min, max, 1e-12);
    size_t pairs = std::min(maxPairs, (size_t) length);

    thread.eigenpairs.reserve(pairs);
    for (auto &iEf : thread.matslise->eigenpairsByIndex(0, pairs, Y<Scalar>::Dirichlet())) {
        thread.eigenpairs.emplace_back(get<1>(iEf), std::move(get<2>(iEf)));
    }
    offset += pairs;

    return thread;
}

template<typename Scalar>
Array<Scalar, Dynamic, 1>
internal_linspaced(Index size, const Domain<Scalar, 2> *domain, const Matrix<Scalar, 2, 1> &direction) {
    Scalar a, b;
    tie(a, b) = domain->bounds(direction);
    return Array<Scalar, Dynamic, 1>::LinSpaced(size + 2, a, b).segment(1, size);
}

template<typename Scalar>
Schrodinger2D<Scalar>::Schrodinger2D(const function<Scalar(Scalar, Scalar)> &V_,
                                     const Domain<Scalar, 2> &_domain,
                                     const Options &_options)
        : V(V_), domain(polymorphic_value<Domain<Scalar, 2>>(_domain.clone(), typename Domain<Scalar, 2>::copy{})),
          options(_options) {
    grid.x = internal_linspaced<Scalar>(options.gridSize.x, &*domain, Matrix<Scalar, 2, 1>::Unit(0));
    grid.y = internal_linspaced<Scalar>(options.gridSize.y, &*domain, Matrix<Scalar, 2, 1>::Unit(1));

    {
        columns.x = 0;
        Index i = 0;
// #pragma omp parallel for ordered schedule(dynamic, 1) collapse(2)
        for (Scalar &x : IterateEigen(grid.x)) {
            for (auto &dom : domain->intersections({{x, 0}, Matrix<Scalar, 2, 1>::Unit(1)})) {
                Thread thread = computeThread<Scalar>(
                        [this, x](Scalar y) -> Scalar { return V(x, y) / 2; },
                        dom.first, dom.second, (size_t) options.maxBasisSize, grid.y, columns.x);
// #pragma ordered
                thread.value = x;
                thread.valueIndex = i;
                if (thread.gridLength > 0)
                    threads.x.emplace_back(std::move(thread));
            }
            i++;
        }
    }

    size_t intersectionCount = 0;
    for (const Thread &t : threads.x)
        intersectionCount += t.gridLength;

    intersections.reserve(intersectionCount);

    for (const Thread &t : threads.x) {
        MatrixXs onGrid(t.gridLength, t.eigenpairs.size());
        ArrayXs subGrid = grid.y.segment(t.gridOffset, t.gridLength);
        Index j = 0;
        for (auto &Ef : t.eigenpairs)
            onGrid.col(j++) = (*get<1>(Ef))(subGrid).col(0);
        for (Index i = 0; i < t.gridLength; ++i) {
            intersections.emplace_back(Intersection{
                    .index=intersections.size(),
                    .position={.x=t.value, .y= grid.y[t.gridOffset + i]},
                    .thread={.x = &t, .y = nullptr},
                    .evaluation={.x = onGrid.row(i)},
            });
        }
    }

    {
        columns.y = 0;
        Index i = 0;
// #pragma omp parallel for schedule(dynamic, 1) collapse(2)
        for (Scalar &y : IterateEigen(grid.y)) {
            for (auto &dom : domain->intersections({{0, y}, Matrix<Scalar, 2, 1>::Unit(0)})) {
                Thread thread = computeThread<Scalar>(
                        [this, y](Scalar x) -> Scalar { return V(x, y) / 2; },
                        dom.first, dom.second, (size_t) options.maxBasisSize, grid.x, columns.y);
                thread.value = y;
                thread.valueIndex = i;
// #pragma ordered
                threads.y.emplace_back(std::move(thread));
            }
            i++;
        }
    }

    {
        std::vector<Intersection *> intersectionsByY;
        intersectionsByY.reserve(intersections.size());
        for (Intersection &i : intersections)
            intersectionsByY.push_back(&i);
        std::sort(intersectionsByY.begin(), intersectionsByY.end(),
                  [](const Intersection *a, const Intersection *b) {
                      if (a->position.y == b->position.y)
                          return a->position.x < b->position.x;
                      return a->position.y < b->position.y;
                  });
        auto intersection = intersectionsByY.begin();
        for (Thread &t : threads.y) {
            MatrixXs onGrid(t.gridLength, t.eigenpairs.size());
            ArrayXs subGrid = grid.x.segment(t.gridOffset, t.gridLength);
            Index j = 0;
            for (auto &Ef : t.eigenpairs)
                onGrid.col(j++) = (*get<1>(Ef))(subGrid).col(0);
            for (Index i = 0; i < t.gridLength; ++i) {
                assert((**intersection).position.x == grid.x[t.gridOffset + i] &&
                       (**intersection).position.y == t.value);
                (**intersection).thread.y = &t;
                (**intersection).evaluation.y = onGrid.row(i);
                ++intersection;
            }
        }
    }

    {
        tiles.resize(options.gridSize.x + 1, options.gridSize.y + 1);
        for (auto &intersection : intersections) {
            Index i = intersection.thread.x->valueIndex;
            Index j = intersection.thread.y->valueIndex;
            tiles(i + 1, j + 1).intersections[0] = &intersection;
            tiles(i, j + 1).intersections[1] = &intersection;
            tiles(i + 1, j).intersections[2] = &intersection;
            tiles(i, j).intersections[3] = &intersection;
        }
    }
}

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, typename Schrodinger2D<Scalar>::Eigenfunction>, Scalar>>
eigenpairs(const Schrodinger2D<Scalar> *self) {
    using ArrayXs = typename Schrodinger2D<Scalar>::ArrayXs;
    using MatrixXs = typename Schrodinger2D<Scalar>::MatrixXs;
    using VectorXs = typename Schrodinger2D<Scalar>::VectorXs;
    size_t rows = self->intersections.size();

    size_t colsX = self->columns.x;
    size_t colsY = self->columns.y;


    MatrixXs beta_x = MatrixXs::Zero(rows, colsX);
    MatrixXs beta_y = MatrixXs::Zero(rows, colsY);

    size_t row = 0;
    for (const typename Schrodinger2D<Scalar>::Intersection &intersection : self->intersections) {
        beta_x.row(row).segment(
                intersection.thread.x->offset, intersection.thread.x->eigenpairs.size()
        ) = intersection.evaluation.x;
        beta_y.row(row).segment(
                intersection.thread.y->offset, intersection.thread.y->eigenpairs.size()
        ) = intersection.evaluation.y;

        ++row;
    }
    assert(row == rows);

    VectorXs lambda_x(colsX);
    VectorXs lambda_y(colsY);

    for (const auto &x : self->threads.x) {
        Index offset = x.offset;
        for (auto &ef : x.eigenpairs)
            lambda_x(offset++) = get<0>(ef);
    }
    for (const auto &y : self->threads.y) {
        Index offset = y.offset;
        for (auto &ef : y.eigenpairs)
            lambda_y(offset++) = get<0>(ef);
    }

    MatrixXs crossingsMatch(rows, colsX + colsY);
    crossingsMatch << beta_x, -beta_y;
    MatrixXs kernel = schrodinger::internal::rightKernel<MatrixXs>(crossingsMatch, 1e-6);
    printf("Kernel: %dx%d -> %dx%d\n", (int)rows, (int)(colsX + colsY), (int)kernel.rows(), (int)kernel.cols());

    MatrixXs A(rows, colsX + colsY); // rows x (colsX + colsY)
    A << beta_x * lambda_x.asDiagonal(), beta_y * lambda_y.asDiagonal();

    MatrixXs BK = colsY < colsX // rows x kernelSize
                  ? beta_y * kernel.bottomRows(colsY)
                  : beta_x * kernel.topRows(colsX);

    RectangularPencil<withEigenfunctions, MatrixXs> pencil(A * kernel, BK, self->options.pencilMethod);


    if constexpr(withEigenfunctions) {
        const auto &values = pencil.eigenvalues();
        const auto &vectors = pencil.eigenvectors();

        typedef std::pair<Scalar, typename Schrodinger2D<Scalar>::Eigenfunction> Eigenpair;
        std::vector<Eigenpair> eigenfunctions;
        eigenfunctions.reserve(values.size());
        for (Index i = 0; i < values.size(); ++i) {
            // Normalize eigenfunction

            VectorXs coeffs = kernel * vectors.col(i).real();
            /*
            Scalar sum = 0;
            for (auto& intersection : self->intersections) {
                // Calculate function value in intersection
                Scalar functionVal = 0;
                ArrayXs vals = intersection.evaluation.x;
                for (int j = 0; j < vals.size(); j++) {
                    functionVal += vals(j) * coeffs(intersection.thread.x->offset + j);
                }
                sum += functionVal*functionVal;
            }


            Scalar factor = sqrt(self->intersections.size()) / sqrt(sum);
            // printf("Total sum: %f, num intersections: %d\n", (double)sum, (int)self->intersections.size());
             */

            eigenfunctions.emplace_back(
                    values[i].real(),
                    typename Schrodinger2D<Scalar>::Eigenfunction{
                            self, values[i].real(), coeffs}
            );
        }
        return eigenfunctions;
    } else {
        ArrayXs values = pencil.eigenvalues().array().real();
        std::sort(values.data(), values.data() + values.size());
        std::vector<Scalar> eigenvalues(values.size());
        std::copy(values.data(), values.data() + values.size(), eigenvalues.begin());

        return eigenvalues;
    }
}

template<typename Scalar>
std::vector<Scalar> Schrodinger2D<Scalar>::eigenvalues() const {
    return eigenpairs<Scalar, false>(this);
}

template<typename Scalar>
std::vector<std::pair<Scalar, typename Schrodinger2D<Scalar>::Eigenfunction>>
Schrodinger2D<Scalar>::eigenfunctions() const {
    return eigenpairs<Scalar, true>(this);
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

/*
    typedef Array<Scalar, 2, 1> Array2s;
    Array2s xs, ys;
    xs << xOffset + x1, xOffset + hx - x1;
    ys << yOffset + y1, yOffset + hy - y1;
    Array2s fy0 = reconstructEigenfunction<Scalar>(tile.intersections[0]->thread.y, c.bottomRows(problem->columns.y),
                                                   xs);
    Array2s fx0 = reconstructEigenfunction<Scalar>(tile.intersections[0]->thread.x, c.topRows(problem->columns.x), ys);
    Array2s fy3 = reconstructEigenfunction<Scalar>(tile.intersections[3]->thread.y, c.bottomRows(problem->columns.y),
                                                   xs);
    Array2s fx3 = reconstructEigenfunction<Scalar>(tile.intersections[3]->thread.x, c.topRows(problem->columns.x), ys);

    if (x == xOffset) {
        return fx0(0);
    }
    if (y == yOffset) {
        return fy0(0);
    }


    Matrix<Scalar, 2, 1> wx, wy;
    Scalar nx = 2 / (hx * x1 - x1 * x1);
    wx << (2 * hx - 3 * x1) * nx / hx, -2 * nx, nx, -(hx - 3 * x1) * nx / hx;
    Scalar ny = 2 / (hy * y1 - y1 * y1);
    wy << (2 * hy - 3 * y1) * ny / hy, -2 * ny, ny, -(hy - 3 * y1) * ny / hy;

    {
        Matrix<Scalar, 4, 4> w = Matrix<Scalar, 4, 4>::Zero();
        Matrix<Scalar, 4, 1> b;
        for (int i = 1; i <= 2; ++i)
            for (int j = 1; j <= 2; ++j) {
                b(2 * i + j - 3) =
                        wx(3 * i - 3) * fx0(j - 1) + wx(6 - 3 * i) * fx3(j - 1) + wy(3 * j - 3) * fy0(i - 1) +
                        wy(6 - 3 * j) * fy3(i - 1);

                w(2 * i + j - 3, 2 * i + j - 3) = problem->V(xs(i - 1), ys(j - 1)) - E - wx(1) - wy(1);
                w(2 * i + j - 3, 3 - 2 * i + j) = -wx(2);
                w(2 * i + j - 3, 2 * i - j) = -wy(2);
            }

        return (w.inverse() * b)(0);
    }

     */
}


template<typename Scalar>
Eigen::Array<Scalar, Eigen::Dynamic, 1> Schrodinger2D<Scalar>::Eigenfunction::operator()(ArrayXs xs, ArrayXs ys) const {
    int method = problem->options.interpolationMethod;
    if (method < 0) method = 0; // default

    assert(xs.size() == ys.size());

    // 4 corners interpolation / 8 points square interpolation / 16 points square interpolation
    if (method == 0 || method == 2 || method == 3) {
        // Sort the query points per tile
        size_t tileRows = problem->tiles.rows();
        size_t tileCols = problem->tiles.cols();

        Scalar grid_x0 = problem->grid.x(0);
        Scalar hx = problem->grid.x(1) - grid_x0;
        grid_x0 -= hx;
        Scalar grid_y0 = problem->grid.y(0);
        Scalar hy = problem->grid.y(1) - grid_y0;
        grid_y0 -= hy;

        std::vector<std::vector<Index>> points_per_tile(tileRows * tileCols, std::vector<Index>());

        for (Index i = 0; i < xs.rows(); i++) {
            size_t tileIndex_x = std::floor((xs(i) - grid_x0) / hx);
            size_t tileIndex_y = std::floor((ys(i) - grid_y0) / hy);
            points_per_tile[tileIndex_x + tileIndex_y * tileRows].emplace_back(i);
        }

        ArrayXs result = ArrayXs::Zero(xs.rows());

        // Calculate function values per tile
        for (size_t i = 0; i < points_per_tile.size(); i++) {
            if (points_per_tile[i].empty()) continue;

            size_t tileIndex_x = i % tileRows;
            size_t tileIndex_y = i / tileRows;
            Tile tile = problem->tiles(tileIndex_x, tileIndex_y);

            // Interpolation with 4 corners
            ArrayXs corners = Array<Scalar, 4, 1>::Zero();
            for (size_t j = 0; j < 4; j++) {
                if (tile.intersections[j] != nullptr) corners(j) = functionValues.x[tile.intersections[j]->index];
            }

            Scalar xOffset = grid_x0 + tileIndex_x * hx;
            Scalar yOffset = grid_y0 + tileIndex_y * hy;

            if (method == 0) {
                for (size_t j = 0; j < points_per_tile[i].size(); j++) {
                    Index point_index = points_per_tile[i][j];
                    Scalar x1 = (xs(point_index) - xOffset) / hx;
                    Scalar y1 = (ys(point_index) - yOffset) / hy;

                    assert(x1 >= 0 && x1 <= 1 && y1 >= 0 && y1 <= 1);

                    result(point_index) =
                            corners(0) * (1 - x1) * (1 - y1)
                            + corners(1) * x1 * (1 - y1)
                            + corners(2) * (1 - x1) * y1
                            + corners(3) * x1 * y1;
                }
            }
            // 8 points square interpolation
            else if (method == 2) {
                // get function values for 4 sides
                Vector<Scalar, 8> funValues = Vector<Scalar, 8>::Zero();
                funValues.topRows(4) = corners;

                Array<Scalar, 2, 1> fv = Array<Scalar, 2, 1>::Zero();
                Array<Scalar, 2, 1> input = Array<Scalar, 2, 1>::Zero();

                input << yOffset + 0.5*hy, yOffset + 0.5*hy;
                if (tile.intersections[0] == nullptr || tile.intersections[2] == nullptr) {
                    funValues(4) = 0;
                }
                else {
                    fv = reconstructEigenfunction<Scalar>(
                            tile.intersections[0]->thread.x,
                            c.bottomRows(problem->columns.x), input);
                    funValues(4) = fv(0);
                }
                if (tile.intersections[1] == nullptr || tile.intersections[3] == nullptr) {
                    funValues(5) = 0;
                }
                else {
                    fv = reconstructEigenfunction<Scalar>(
                            tile.intersections[3]->thread.x,
                            c.bottomRows(problem->columns.x), input);
                    funValues(5) = fv(0);
                }

                input << xOffset + 0.5*hx, xOffset + 0.5*hx;
                if (tile.intersections[0] == nullptr || tile.intersections[1] == nullptr) {
                    funValues(6) = 0;
                }
                else {
                    fv = reconstructEigenfunction<Scalar>(
                            tile.intersections[0]->thread.y,
                            c.topRows(problem->columns.y), input);
                    funValues(6) = fv(0);
                }
                if (tile.intersections[2] == nullptr || tile.intersections[3] == nullptr) {
                    funValues(7) = 0;
                }
                else {
                    fv = reconstructEigenfunction<Scalar>(
                            tile.intersections[3]->thread.y,
                            c.topRows(problem->columns.y), input);
                    funValues(7) = fv(0);
                }

                Matrix<Scalar, 8, 8> interpolationMat;
                interpolationMat <<
                         1,  0,  0,  0, -0,  0,  0,  0,
                        -3, -1,  0,  0, -0,  0,  4,  0,
                        -3,  0, -1, -0,  4,  0,  0,  0,
                         5, -1, -1, -3, -4,  4, -4,  4,
                         2,  2,  0,  0,  0, -0, -4, -0,
                         2,  0,  2,  0, -4,  0,  0, -0,
                        -2,  2, -2,  2,  4, -4,  0,  0,
                        -2, -2,  2,  2,  0,  0,  4, -4;

                Vector<Scalar, 8> coeffs = interpolationMat * funValues;
                /*
                printf("FunValues: ");
                for (int j = 0; j < 8; j++) printf("%.3e; ", funValues(j));
                printf("\n");
                 */

                for (Index point_index : points_per_tile[i]) {
                    Scalar x1 = (xs(point_index) - xOffset) / hx;
                    Scalar y1 = (ys(point_index) - yOffset) / hy;

                    assert(x1 >= 0 && x1 <= 1 && y1 >= 0 && y1 <= 1);

                    result(point_index) = coeffs(0)
                            + coeffs(1) * x1
                            + coeffs(2) * y1
                            + coeffs(3) * x1 * y1
                            + coeffs(4) * x1 * x1
                            + coeffs(5) * y1 * y1
                            + coeffs(6) * x1 * y1 * y1
                            + coeffs(7) * x1 * x1 * y1;

                    // printf("Result: %.3e\n", result(point_index));
                }
            }
            else {
                /*
                Matrix<Scalar, 16, 20> interpolationMat;
                interpolationMat <<
                         -3.62629758,  1.16262976,  1.16262976, -0.25605536, -7.0934256,  -7.09342561,   4.70934256,   4.70934256,   1.2733564 ,  -0.29065744,  -0.29065744,  -0.43598616,  -0.43598616,  -0.92733564,  -0.92733564,   0.60899654,   2,  2, -1, -1,
                          7.08650519, -3.5432526 , -1.88235294,  0.94117647, -1.7716263,  30.2283737 , -15.11418685,  -7.52941176,   0.44290657,   0.88581315,   0.47058824,  -0.23529412,   3.76470588,   1.88235294,  -0.22145329,  -0.94117647,  -0, -8,  0,  4,
                         -7.80622837,  3.90311419, -0.        ,  0.        , 13.9515570, -46.04844291,  23.02422145,  -0.        ,  -3.48788927,  -6.97577855,   0.        ,  -0.        ,  -0.        ,   0.        ,   1.74394464,   0.        ,  -0, 12, -0, -6,
                          7.08650519, -3.5432526 ,  1.88235294, -0.94117647, -1.7716263,  30.2283737 , -15.11418685,   7.52941176,   0.44290657,   0.88581315,  -0.47058824,   0.23529412,  -3.76470588,  -1.88235294,  -0.22145329,   0.94117647,   0, -8,  0,  4,
                         -3.62629758,  1.16262976, -1.16262976,  0.25605536, -7.0934256,  -7.09342561,   4.70934256,  -4.70934256,   1.2733564 ,  -0.29065744,   0.29065744,   0.43598616,   0.43598616,   0.92733564,  -0.92733564,  -0.60899654,   2,  2,  1, -1,
                          7.08650519, -1.88235294,  3.5432526 , -0.94117647, 30.2283737,  -1.7716263 ,  -7.52941176,  15.11418685,   0.44290657,   0.47058824,  -0.88581315,  -3.76470588,   0.23529412,   0.22145329,   1.88235294,   0.94117647,  -8,  0, -4,  0,
                         -7.80622837, -0.        , -3.90311419, -0.        ,-46.0484429,  13.95155709,   0.        , -23.02422145,  -3.48788927,   0.        ,   6.97577855,  -0.        ,   0.        ,  -1.74394464,  -0.        ,  -0.        ,  12, -0,  6,  0,
                          7.08650519,  1.88235294,  3.5432526 ,  0.94117647, 30.2283737,  -1.7716263 ,   7.52941176,  15.11418685,   0.44290657,  -0.47058824,  -0.88581315,   3.76470588,  -0.23529412,   0.22145329,  -1.88235294,  -0.94117647,  -8,  0, -4,  0,
                         -3.62629758, -1.16262976, -1.16262976, -0.25605536, -7.0934256,  -7.09342561,  -4.70934256,  -4.70934256,   1.2733564 ,   0.29065744,   0.29065744,  -0.43598616,  -0.43598616,   0.92733564,   0.92733564,   0.60899654,   2,  2,  1,  1,
                          7.08650519,  3.5432526 ,  1.88235294,  0.94117647, -1.7716263,  30.2283737 ,  15.11418685,   7.52941176,   0.44290657,  -0.88581315,  -0.47058824,  -0.23529412,   3.76470588,  -1.88235294,   0.22145329,  -0.94117647,  -0, -8, -0, -4,
                         -7.80622837, -3.90311419,  0.        ,  0.        , 13.9515570, -46.04844291, -23.02422145,  -0.        ,  -3.48788927,   6.97577855,  -0.        ,   0.        ,  -0.        ,   0.        ,  -1.74394464,  -0.        ,   0, 12,  0,  6,
                          7.08650519,  3.5432526 , -1.88235294, -0.94117647, -1.7716263,  30.2283737 ,  15.11418685,  -7.52941176,   0.44290657,  -0.88581315,   0.47058824,   0.23529412,  -3.76470588,   1.88235294,   0.22145329,   0.94117647,  -0, -8, -0, -4,
                         -3.62629758, -1.16262976,  1.16262976,  0.25605536, -7.0934256,  -7.09342561,  -4.70934256,   4.70934256,   1.2733564 ,   0.29065744,  -0.29065744,   0.43598616,   0.43598616,  -0.92733564,   0.92733564,  -0.60899654,   2,  2, -1,  1,
                          7.08650519,  1.88235294, -3.5432526 , -0.94117647, 30.2283737,  -1.7716263 ,   7.52941176, -15.11418685,   0.44290657,  -0.47058824,   0.88581315,  -3.76470588,   0.23529412,  -0.22145329,  -1.88235294,   0.94117647,  -8,  0,  4, -0,
                         -7.80622837, -0.        ,  3.90311419, -0.        ,-46.0484429,  13.95155709,  -0.        ,  23.02422145,  -3.48788927,   0.        ,  -6.97577855,   0.        ,   0.        ,   1.74394464,   0.        ,  -0.        ,  12, -0, -6,  0,
                          7.08650519, -1.88235294, -3.5432526 ,  0.94117647, 30.2283737,  -1.7716263 ,  -7.52941176, -15.11418685,   0.44290657,   0.47058824,   0.88581315,   3.76470588,  -0.23529412,  -0.22145329,   1.88235294,  -0.94117647,  -8,  0,  4,  0;

                interpolationMat /= 3*32;

                // get function values for 4 sides
                Vector<Scalar, 16> funValues = Vector<Scalar, 16>::Zero();

                Array<Scalar, 4, 1> input = Array<Scalar, 4, 1>::Zero();

                for (int side = 0; side < 4; side++) {
                    Intersection* intersection = tile.intersections[side == 0 ? 0 : (side%3) + 1]; // 0,2,3,1
                    if (intersection == nullptr) continue;  // function values near the border will be set to 0
                    const Thread* thread = side % 2 == 0 ? intersection->thread.x : intersection->thread.y;
                    if (side == 0)
                        input << yOffset, yOffset + 0.25*hy, yOffset + 0.5*hy, yOffset + 0.75*hy;
                    else if (side == 1)
                        input << xOffset, xOffset + 0.25*hx, xOffset + 0.5*hx, xOffset + 0.75*hx;
                    else if (side == 2)
                        input << yOffset + hy, yOffset + 0.75*hy, yOffset + 0.5*hy, yOffset + 0.25*hy;
                    else if (side == 3)
                        input << xOffset + hx, xOffset + 0.75*hx, xOffset + 0.5*hx, xOffset + 0.25*hx;

                    funValues.segment(side*4, 4) = reconstructEigenfunction<Scalar>(
                            thread, side % 2 == 0 ? c.bottomRows(problem->columns.x) : c.topRows(problem->columns.y), input);
                }

                Vector<Scalar, 20> coeffs = interpolationMat.transpose() * funValues;

                printf("Fun values: ");
                for (int j = 0; j < funValues.size(); j++) printf("%.3f; ", funValues(j));
                printf("\n");

                printf("Coefficients: ");
                for (int j = 0; j < coeffs.size(); j++) printf("%.8f, ", coeffs(j));
                printf("\n");


                for (Index point_index : points_per_tile[i]) {
                    // Coordinates in the square are in [-2, 2] x [-2, 2]
                    Scalar x1 = 4 * (xs(point_index) - xOffset) / hx - 2;
                    Scalar y1 = 4 * (ys(point_index) - yOffset) / hy - 2;

                    assert(x1 >= -2 && x1 <= 2 && y1 >= -2 && y1 <= 2);

                    Scalar xp2 = x1*x1;
                    Scalar xp3 = x1*x1*x1;
                    Scalar xp4 = x1*x1*x1*x1;

                    Scalar yp2 = y1*y1;
                    Scalar yp3 = y1*y1*y1;
                    Scalar yp4 = y1*y1*y1*y1;

                    result(point_index) = coeffs( 0)
                                        + coeffs( 1) * x1
                                        + coeffs( 2) * y1
                                        + coeffs( 3) * x1 * y1
                                        + coeffs( 4) * xp2
                                        + coeffs( 5) * yp2
                                        + coeffs( 6) * x1 * yp2
                                        + coeffs( 7) * xp2 * y1

                                        + coeffs( 8) * xp2 * yp2
                                        + coeffs( 9) * xp3
                                        + coeffs(10) * yp3
                                        + coeffs(11) * x1 * yp3
                                        + coeffs(12) * xp3 * y1
                                        + coeffs(13) * xp2 * yp3
                                        + coeffs(14) * xp3 * yp2
                                        + coeffs(15) * xp3 * yp3

                                        + coeffs(16) * xp4
                                        + coeffs(17) * yp4
                                        + coeffs(18) * xp4 * y1
                                        + coeffs(19) * x1 * yp4;
                }
                 */

                // get function values for 4 sides
                Vector<Scalar, 16> funValues = Vector<Scalar, 16>::Zero();

                Array<Scalar, 4, 1> input = Array<Scalar, 4, 1>::Zero();

                for (int side = 0; side < 4; side++) {
                    Intersection *intersection = tile.intersections[side == 0 ? 0 : (side % 3) + 1]; // 0,2,3,1
                    if (intersection == nullptr) continue;  // function values near the border will be set to 0
                    const Thread *thread = side % 2 == 0 ? intersection->thread.x : intersection->thread.y;
                    if (side == 0)
                        input << yOffset, yOffset + 0.25 * hy, yOffset + 0.5 * hy, yOffset + 0.75 * hy;
                    else if (side == 1)
                        input << xOffset, xOffset + 0.25 * hx, xOffset + 0.5 * hx, xOffset + 0.75 * hx;
                    else if (side == 2)
                        input << yOffset + hy, yOffset + 0.75 * hy, yOffset + 0.5 * hy, yOffset + 0.25 * hy;
                    else if (side == 3)
                        input << xOffset + hx, xOffset + 0.75 * hx, xOffset + 0.5 * hx, xOffset + 0.25 * hx;

                    funValues.segment(side * 4, 4) = reconstructEigenfunction<Scalar>(
                            thread, side % 2 == 0 ? c.bottomRows(problem->columns.x) : c.topRows(problem->columns.y),
                            input);
                }

                for (Index point_index: points_per_tile[i]) {
                    // Coordinates in the square are in [-2, 2] x [-2, 2]
                    Scalar x = 4 * (xs(point_index) - xOffset) / hx - 2;
                    Scalar y = 4 * (ys(point_index) - yOffset) / hy - 2;

                    Scalar x0 = x + 2, x1 = x + 1, x2 = x, x3 = x - 1, x4 = x - 2;
                    Scalar y0 = y + 2, y1 = y + 1, y2 = y, y3 = y - 1, y4 = y - 2;

                    // horizontal Lagrange polynomials
                    Scalar lx0 = x1 * x2 * x3 * x4 / (-1 * -2 * -3 * -4);
                    Scalar lx1 = x0 * x2 * x3 * x4 / (1 * -1 * -2 * -3);
                    Scalar lx2 = x0 * x1 * x3 * x4 / (2 * 1 * -1 * -2);
                    Scalar lx3 = x0 * x1 * x2 * x4 / (3 * 2 * 1 * -1);
                    Scalar lx4 = x0 * x1 * x2 * x3 / (4 * 3 * 2 * 1);

                    // vertical Lagrange polynomials
                    Scalar ly0 = y1 * y2 * y3 * y4 / (-1 * -2 * -3 * -4);
                    Scalar ly1 = y0 * y2 * y3 * y4 / (1 * -1 * -2 * -3);
                    Scalar ly2 = y0 * y1 * y3 * y4 / (2 * 1 * -1 * -2);
                    Scalar ly3 = y0 * y1 * y2 * y4 / (3 * 2 * 1 * -1);
                    Scalar ly4 = y0 * y1 * y2 * y3 / (4 * 3 * 2 * 1);

                    result(point_index) =
                              funValues(0) * lx0 * ly0
                            + funValues(1) * lx0 * ly1
                            + funValues(2) * lx0 * ly2
                            + funValues(3) * lx0 * ly3

                            + funValues(4) * lx0 * ly4
                            + funValues(5) * lx1 * ly4
                            + funValues(6) * lx2 * ly4
                            + funValues(7) * lx3 * ly4

                            + funValues(8) * lx4 * ly4
                            + funValues(9) * lx4 * ly3
                            + funValues(10) * lx4 * ly2
                            + funValues(11) * lx4 * ly1

                            + funValues(12) * lx4 * ly0
                            + funValues(13) * lx3 * ly0
                            + funValues(14) * lx2 * ly0
                            + funValues(15) * lx1 * ly0;
                }
            }
        }

        return result;
    }
    // Linear interpolation (4 points)
    else if (method == 1) {
        Index num_points = xs.size();

        assert(xs.size() == ys.size());

        ArrayXs result = ArrayXs::Zero(num_points);

        for (int direction = 0; direction < 2; direction++) {
            // Gather all points on the same horizontal/vertical line
            std::vector<Index> indices(num_points);
            for (Index i = 0; i < num_points; i++) indices[i] = i;

            auto &yxs = direction == 0 ? ys : xs;

            std::sort(indices.begin(), indices.end(),
                      [&yxs](Index i1, Index i2) { return yxs(i1) < yxs(i2); });

            // iterate over each line
            for (int i = 0; i < num_points; i++) {
                int start = i;
                while (i + 1 < num_points && yxs[indices[i]] == yxs[indices[i + 1]]) i++;

                // Calculate function values on the line
                Scalar yxVal = yxs[indices[start]];

                Index lineSize = direction == 0 ? problem->tiles.rows() : problem->tiles.cols();
                VectorXs funValues = ArrayXs::Zero(lineSize + 2);  // some zero values of padding left and right

                Index iyx;
                if (direction == 0)
                    iyx = std::floor((yxVal - problem->grid.y(0)) / (problem->grid.y(1) - problem->grid.y(0))) + 1;
                else
                    iyx = std::floor((yxVal - problem->grid.x(0)) / (problem->grid.x(1) - problem->grid.x(0))) + 1;

                for (int j = 0; j < lineSize; j++) {
                    const Tile &tile = direction == 0 ? problem->tiles(j, iyx) : problem->tiles(iyx, j);
                    if (tile.intersections[0] == nullptr) continue;  // leave function value as 0

                    Array<Scalar, 2, 1> yxInput;
                    yxInput << yxVal, yxVal;
                    Array<Scalar, 2, 1> fyx0 = reconstructEigenfunction<Scalar>(
                            direction == 0 ? tile.intersections[0]->thread.x : tile.intersections[0]->thread.y,
                            direction == 0 ? c.bottomRows(problem->columns.x) : c.topRows(problem->columns.y), yxInput);
                    funValues(j + 1) = fyx0(0);
                }

                /*
                printf("Fun values: ");
                for (int j = 0; j < funValues.size(); j++) printf("%.3f; ", funValues(j));
                printf("\n");
                 */

                // interpolate points
                Scalar xy0 = direction == 0 ? problem->grid.x(0) : problem->grid.y(0);
                Scalar hxy = direction == 0 ? problem->grid.x(1) - xy0 : problem->grid.y(1) - xy0;
                xy0 -= 2 * hxy;

                for (int j = start; j < i + 1; j++) {
                    Scalar xyVal = direction == 0 ? xs(indices[j]) : ys(indices[j]);
                    Index ixy = std::floor((xyVal - xy0) / hxy);
                    Scalar xy1 = (xyVal - xy0) / hxy - ixy;

                    assert(ixy >= 0 && ixy + 1 < funValues.size());
                    assert(xy1 >= 0 && xy1 <= 1);

                    result(indices[j]) += (funValues(ixy) * (1 - xy1) + funValues(ixy + 1) * xy1) / 2;
                }
            }
        }

        return result;
    }

    // Long interpolation (4 * 3 points)
    else if (method == 4) {
        Index num_points = xs.size();

        assert(xs.size() == ys.size());

        ArrayXs result = ArrayXs::Zero(num_points);

        for (int direction = 0; direction < 2; direction++) {
            // Gather all points on the same horizontal/vertical line
            std::vector<Index> indices(num_points);
            for (Index i = 0; i < num_points; i++) indices[i] = i;

            auto &yxs = direction == 0 ? ys : xs;

            std::sort(indices.begin(), indices.end(),
                      [&yxs](Index i1, Index i2) { return yxs(i1) < yxs(i2); });

            // iterate over each line
            for (int i = 0; i < num_points; i++) {
                int start = i;
                while (i + 1 < num_points && yxs[indices[i]] == yxs[indices[i + 1]]) i++;

                // Calculate function values on the line
                Scalar yxVal = yxs[indices[start]];

                Index lineSize = direction == 0 ? problem->tiles.rows() : problem->tiles.cols();
                VectorXs funValues = ArrayXs::Zero(lineSize + 6);  // some zero values of padding left and right

                Index iyx;
                if (direction == 0)
                    iyx = std::floor((yxVal - problem->grid.y(0)) / (problem->grid.y(1) - problem->grid.y(0))) + 1;
                else
                    iyx = std::floor((yxVal - problem->grid.x(0)) / (problem->grid.x(1) - problem->grid.x(0))) + 1;

                for (int j = 0; j < lineSize; j++) {
                    const Tile &tile = direction == 0 ? problem->tiles(j, iyx) : problem->tiles(iyx, j);
                    if (tile.intersections[0] == nullptr) continue;  // leave function value as 0

                    Array<Scalar, 2, 1> yxInput;
                    yxInput << yxVal, yxVal;
                    Array<Scalar, 2, 1> fyx0 = reconstructEigenfunction<Scalar>(
                            direction == 0 ? tile.intersections[0]->thread.x : tile.intersections[0]->thread.y,
                            direction == 0 ? c.bottomRows(problem->columns.x) : c.topRows(problem->columns.y), yxInput);
                    funValues(j + 3) = fyx0(0);
                }

                /*
                printf("Fun values: ");
                for (int j = 0; j < funValues.size(); j++) printf("%.3f; ", funValues(j));
                printf("\n");
                 */

                // Interpolate values
                Matrix<Scalar, 6, 6> interpolationMat;
                interpolationMat <<
                           0,    0,  120,    0,    0,    0,
                           6,  -60,  -40,  120,  -30,    4,
                          -5,   80, -150,   80,   -5,    0,
                          -5,   -5,   50,  -70,   35,   -5,
                           5,  -20,   30,  -20,    5,    0,
                          -1,    5,  -10,   10,   -5,    1;

                interpolationMat /= 2*120;

                // interpolate points
                Scalar xy0 = direction == 0 ? problem->grid.x(0) : problem->grid.y(0);
                Scalar hxy = direction == 0 ? problem->grid.x(1) - xy0 : problem->grid.y(1) - xy0;
                xy0 -= 4 * hxy;

                for (int j = start; j < i + 1; j++) {
                    Scalar xyVal = direction == 0 ? xs(indices[j]) : ys(indices[j]);
                    Index ixy = std::floor((xyVal - xy0) / hxy);
                    Scalar xy1 = (xyVal - xy0) / hxy - ixy;

                    assert(ixy - 2 >= 0 && ixy + 4 < funValues.size());
                    assert(xy1 >= 0 && xy1 <= 1);

                    VectorXs coeffs = interpolationMat * funValues.middleRows(ixy - 2, 6);

                    result(indices[j]) += coeffs(0) + xy1 * (coeffs(1) + xy1 * (coeffs(2) + xy1 * (
                            coeffs(3) + xy1 * (coeffs(4) + xy1 * coeffs(5)))));
                }
            }
        }

        return result;
    }
    // Solve Schrödinger inside the square with 4 points grid
    else if (method == 5) {
        ArrayXs result = ArrayXs::Zero(xs.size());

        for (Index p = 0; p < xs.size(); p++) {
            Scalar x = xs(p);
            Scalar y = ys(p);

            Index ix = highestLowerIndex(problem->grid.x, x);
            Index iy = highestLowerIndex(problem->grid.y, y);

            const Tile &tile = problem->tiles(ix + 1, iy + 1);

            Scalar xOffset = problem->grid.x[ix];
            Scalar yOffset = problem->grid.y[iy];
            Scalar x1 = x - xOffset;
            Scalar y1 = y - yOffset;
            Scalar hx = problem->grid.x[ix + 1] - xOffset;
            Scalar hy = problem->grid.y[iy + 1] - yOffset;

            typedef Array<Scalar, 2, 1> Array2s;
            Array2s vx, vy;
            vx << xOffset + x1, xOffset + hx - x1;
            vy << yOffset + y1, yOffset + hy - y1;
            Array2s fy0 = reconstructEigenfunction<Scalar>(tile.intersections[0]->thread.y,
                                                           c.bottomRows(problem->columns.y),
                                                           vx);
            Array2s fx0 = reconstructEigenfunction<Scalar>(tile.intersections[0]->thread.x,
                                                           c.topRows(problem->columns.x), vy);
            Array2s fy3 = reconstructEigenfunction<Scalar>(tile.intersections[3]->thread.y,
                                                           c.bottomRows(problem->columns.y),
                                                           vx);
            Array2s fx3 = reconstructEigenfunction<Scalar>(tile.intersections[3]->thread.x,
                                                           c.topRows(problem->columns.x), vy);

            if (x == xOffset) {
                result(p) = fx0(0);
                continue;
            }
            if (y == yOffset) {
                result(p) = fy0(0);
                continue;
            }

            Matrix<Scalar, 2, 1> wx, wy;
            Scalar nx = 2 / (hx * x1 - x1 * x1);
            wx << (2 * hx - 3 * x1) * nx / hx, -2 * nx, nx, -(hx - 3 * x1) * nx / hx;
            Scalar ny = 2 / (hy * y1 - y1 * y1);
            wy << (2 * hy - 3 * y1) * ny / hy, -2 * ny, ny, -(hy - 3 * y1) * ny / hy;

            {
                Matrix<Scalar, 4, 4> w = Matrix<Scalar, 4, 4>::Zero();
                Matrix<Scalar, 4, 1> b;
                for (int i = 1; i <= 2; ++i)
                    for (int j = 1; j <= 2; ++j) {
                        b(2 * i + j - 3) =
                                wx(3 * i - 3) * fx0(j - 1) + wx(6 - 3 * i) * fx3(j - 1) + wy(3 * j - 3) * fy0(i - 1) +
                                wy(6 - 3 * j) * fy3(i - 1);

                        w(2 * i + j - 3, 2 * i + j - 3) = problem->V(xs(i - 1), ys(j - 1)) - E - wx(1) - wy(1);
                        w(2 * i + j - 3, 3 - 2 * i + j) = -wx(2);
                        w(2 * i + j - 3, 2 * i - j) = -wy(2);
                    }

                result(p) = (w.inverse() * b)(0);
            }
        }

        return result;
    }

    return ArrayXs::Zero(1);
}

template
class schrodinger::Schrodinger2D<double>;

#ifdef SCHRODINGER_LONG_DOUBLE

template
class schrodinger::Schrodinger2D<long double>;

#endif
