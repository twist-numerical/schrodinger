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

            eigenfunctions.emplace_back(
                    values[i].real(),
                    typename Schrodinger2D<Scalar>::Eigenfunction{
                            self, values[i].real(), factor * coeffs}
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

template<typename Scalar>
Array<Scalar, 2, 1>
reconstructEigenfunction(const typename Schrodinger2D<Scalar>::Thread *t, const Ref<const Array<Scalar, Dynamic, 1>> &c,
                         const Array<Scalar, 2, 1> &v) {
    Array<Scalar, Dynamic, 1> x = v;
    Array<Scalar, 2, 1> r = Array<Scalar, 2, 1>::Zero();
    for (size_t i = 0; i < t->eigenpairs.size(); ++i)
        r += c(t->offset + i) * (*t->eigenpairs[i].second)(x).col(0);;
    return r;
}

template<typename Scalar>
Scalar Schrodinger2D<Scalar>::Eigenfunction::operator()(Scalar x, Scalar y) const {
    problem->domain->contains({x, y});

    Index ix = highestLowerIndex(problem->grid.x, x);
    Index iy = highestLowerIndex(problem->grid.y, y);

    const Tile &tile = problem->tiles(ix + 1, iy + 1);

    Scalar xOffset = problem->grid.x[ix];
    Scalar yOffset = problem->grid.y[iy];
    Scalar x1 = x - xOffset;
    Scalar y1 = y - yOffset;
    Scalar hx = problem->grid.x[ix + 1] - xOffset;
    Scalar hy = problem->grid.y[iy + 1] - yOffset;

    // Interpolation with 4 corners
    ArrayXs corners = ArrayXs::Zero(4);
    for (int i = 0; i < 4; i++) {
        if (tile.intersections[i] != nullptr) corners(i) = functionValues.x[tile.intersections[i]->index];
    }

    x1 /= hx;
    y1 /= hy;

    return corners(0) * (1-x1)*(1-y1) + corners(1) * x1*(1-y1) + corners(2) * (1-x1)*y1 + corners(3) * x1*y1;

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

    // 4 corners interpolation
    if (method == 0) {
        // Sort the query points per tile
        size_t tileRows = problem->tiles.rows();
        size_t tileCols = problem->tiles.cols();

        Scalar grid_x0 = problem->grid.x(0);
        Scalar hx = problem->grid.x(1) - grid_x0;
        grid_x0 -= hx;
        Scalar grid_y0 = problem->grid.y(0);
        Scalar hy = problem->grid.y(1) - grid_y0;
        grid_y0 -= hy;

        std::vector<std::vector<Index>> points_per_tile(tileRows*tileCols, std::vector<Index>());

        for (Index i = 0; i < xs.rows(); i++) {
            size_t tileIndex_x = std::floor((xs(i) - grid_x0) / hx);
            size_t tileIndex_y = std::floor((ys(i) - grid_y0) / hy);
            points_per_tile[tileIndex_x + tileIndex_y*tileRows].emplace_back(i);
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

            Scalar xOffset = problem->grid.x[tileIndex_x-1];
            Scalar yOffset = problem->grid.y[tileIndex_y-1];

            for (size_t j = 0; j < points_per_tile[i].size(); j++) {
                Index point_index = points_per_tile[i][j];
                Scalar x1 = (xs(point_index) - xOffset) / hx;
                Scalar y1 = (ys(point_index) - yOffset) / hy;

                assert(x1 >= 0 && x1 <= 1 && y1 >= 0 && y1 <= 1);

                result(point_index) =
                          corners(0) * (1-x1)*(1-y1)
                        + corners(1) * x1*(1-y1)
                        + corners(2) * (1-x1)*y1
                        + corners(3) * x1*y1;
            }
        }

        return result;
    }
    // Plus-shaped (4 points)
    else if (method == 1) {
        // Sort the query points per tile
        size_t tileRows = problem->tiles.rows();
        size_t tileCols = problem->tiles.cols();

        Scalar grid_x0 = problem->grid.x(0);
        Scalar hx = problem->grid.x(1) - grid_x0;
        grid_x0 -= hx;
        Scalar grid_y0 = problem->grid.y(0);
        Scalar hy = problem->grid.y(1) - grid_y0;
        grid_y0 -= hy;

        std::vector<std::vector<Index>> points_per_tile(tileRows*tileCols, std::vector<Index>());

        for (Index i = 0; i < xs.rows(); i++) {
            size_t tileIndex_x = std::floor((xs(i) - grid_x0) / hx);
            size_t tileIndex_y = std::floor((ys(i) - grid_y0) / hy);
            points_per_tile[tileIndex_x + tileIndex_y*tileRows].emplace_back(i);
        }

        ArrayXs result = ArrayXs::Zero(xs.rows());

        // Calculate function values per tile
        for (size_t i = 0; i < points_per_tile.size(); i++) {
            if (points_per_tile[i].empty()) continue;

            size_t tileIndex_x = i % tileRows;
            size_t tileIndex_y = i / tileRows;
            Tile tile = problem->tiles(tileIndex_x, tileIndex_y);

            // Sort points by x-coords


            // Interpolation with 4 corners
            ArrayXs corners = Array<Scalar, 4, 1>::Zero();
            for (size_t j = 0; j < 4; j++) {
                if (tile.intersections[j] != nullptr) corners(j) = functionValues.x[tile.intersections[j]->index];
            }

            Scalar xOffset = problem->grid.x[tileIndex_x-1];
            Scalar yOffset = problem->grid.y[tileIndex_y-1];

            for (size_t j = 0; j < points_per_tile[i].size(); j++) {
                Index point_index = points_per_tile[i][j];
                Scalar x1 = (xs(point_index) - xOffset) / hx;
                Scalar y1 = (ys(point_index) - yOffset) / hy;

                assert(x1 >= 0 && x1 <= 1 && y1 >= 0 && y1 <= 1);

                result(point_index) =
                        corners(0) * (1-x1)*(1-y1)
                        + corners(1) * x1*(1-y1)
                        + corners(2) * (1-x1)*y1
                        + corners(3) * x1*y1;
            }
        }
    }
    // Long interpolation (4 * 3 points)
    else if (method == 2) {
        Index num_points = xs.size();

        assert(xs.size() == ys.size());

        ArrayXs result = ArrayXs::Zero(num_points);

        for (int direction = 0; direction < 2; direction++) {
            // Gather all points on the same horizontal/vertical line
            std::vector<Index> indices(num_points);
            for (Index i = 0; i < num_points; i++) indices[i] = i;

            auto& yxs = direction == 0 ? ys : xs;

            std::sort(indices.begin(), indices.end(),
                      [&yxs](Index i1, Index i2) { return yxs(i1) < yxs(i2); });

            // iterate over each line
            for (int i = 0; i < num_points; i++) {
                int start = i;
                while (i+1 < num_points && yxs[indices[i]] == yxs[indices[i + 1]]) i++;
                i++;

                // Calculate function values on the line
                Scalar yxVal = yxs[indices[start]];

                Index lineSize = direction == 0 ? problem->tiles.rows() : problem->tiles.cols();
                VectorXs funValues = ArrayXs::Zero(lineSize + 8);  // some zero values of padding left and right

                Index iyx = highestLowerIndex(direction == 0 ? problem->grid.y : problem->grid.x, yxVal)+1;

                for (int j = 0; j < lineSize; j++) {
                    const Tile &tile = direction == 0 ? problem->tiles(j, iyx) : problem->tiles(iyx, j);
                    if (tile.intersections[0] == nullptr) continue;  // leave function value as 0

                    Array<Scalar, 2, 1> yxInput;
                    yxInput << yxVal, yxVal;
                    Array<Scalar, 2, 1> fyx0 = reconstructEigenfunction<Scalar>(
                            direction == 0 ? tile.intersections[0]->thread.x : tile.intersections[0]->thread.y,
                            direction == 0 ? c.bottomRows(problem->columns.x) : c.topRows(problem->columns.y), yxInput);
                    funValues(j + 4) = fyx0(0);
                }

                // Interpolate values
                Matrix<Scalar, 6, 6> interpolationMat;
                interpolationMat <<
                        0, 1728, -1440, -1440, 1440, -288,
                        0, -17280, 23040, -1440, -5760, 1440,
                        34560, -11520, -43200, 14400, 8640, -2880,
                        0, 34560, 23040, -20160, -5760, 2880,
                        0, -8640, -1440, 10080, 1440, -1440,
                        0, 1152, 0, -1440, 0, 288;

                interpolationMat /= 34560;

                // interpolate points
                Scalar xy0 = direction == 0 ? problem->grid.x(0) : problem->grid.y(0);
                Scalar hxy = direction == 0 ? problem->grid.x(1) - xy0 : problem->grid.y(1) - xy0;
                xy0 -= 4 * hxy;

                for (int j = start; j < i; j++) {
                    Scalar xVal = direction == 0 ? xs(indices[j]) : ys(indices[j]);
                    Index ixy = std::floor((xVal - xy0) / hxy);
                    Scalar x1 = (xVal - xy0) / hxy - ixy;

                    assert(ixy - 2 >= 0 && ixy + 4 < funValues.size());
                    assert(x1 >= 0 && x1 < 1);

                    VectorXs coeffs = interpolationMat * funValues.middleRows(ixy - 2, 6);

                    result(indices[j]) += coeffs(0) + x1 * (coeffs(1) + x1 * (coeffs(2) + x1 * (
                            coeffs(3) + x1 * (coeffs(4) + x1 * coeffs(5)))));
                }
            }
        }

        result /= 2;
        return result;
    }

    /*
    problem->domain->contains({x, y});

    Index ix = highestLowerIndex(problem->grid.x, x);
    Index iy = highestLowerIndex(problem->grid.y, y);

    const Tile &tile = problem->tiles(ix + 1, iy + 1);

    Scalar xOffset = problem->grid.x[ix];
    Scalar yOffset = problem->grid.y[iy];
    Scalar x1 = x - xOffset;
    Scalar y1 = y - yOffset;
    Scalar hx = problem->grid.x[ix + 1] - xOffset;
    Scalar hy = problem->grid.y[iy + 1] - yOffset;

    // Interpolation with 4 corners
    ArrayXs corners = ArrayXs::Zero(4);
    for (int i = 0; i < 4; i++) {
        if (tile.intersections[i] != nullptr) corners(i) = functionValues.x[tile.intersections[i]->index];
    }

    x1 /= hx;
    y1 /= hy;

    return corners(0) * (1-x1)*(1-y1) + corners(1) * x1*(1-y1) + corners(2) * (1-x1)*y1 + corners(3) * x1*y1;
     */

    return ArrayXs::Zero(1);
}

template
class schrodinger::Schrodinger2D<double>;

#ifdef SCHRODINGER_LONG_DOUBLE

template
class schrodinger::Schrodinger2D<long double>;

#endif
