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
    for (auto &iEf: thread.matslise->eigenpairsByIndex(0, pairs, Y<Scalar>::Dirichlet())) {
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
        for (Scalar &x: IterateEigen(grid.x)) {
            for (auto &dom: domain->intersections({{x, 0}, Matrix<Scalar, 2, 1>::Unit(1)})) {
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
    for (const Thread &t: threads.x)
        intersectionCount += t.gridLength;

    intersections.reserve(intersectionCount);

    for (const Thread &t: threads.x) {
        MatrixXs onGrid(t.gridLength, t.eigenpairs.size());
        ArrayXs subGrid = grid.y.segment(t.gridOffset, t.gridLength);
        Index j = 0;
        for (auto &Ef: t.eigenpairs)
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
        for (Scalar &y: IterateEigen(grid.y)) {
            for (auto &dom: domain->intersections({{0, y}, Matrix<Scalar, 2, 1>::Unit(0)})) {
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
        for (Intersection &i: intersections)
            intersectionsByY.push_back(&i);
        std::sort(intersectionsByY.begin(), intersectionsByY.end(),
                  [](const Intersection *a, const Intersection *b) {
                      if (a->position.y == b->position.y)
                          return a->position.x < b->position.x;
                      return a->position.y < b->position.y;
                  });
        auto intersection = intersectionsByY.begin();
        for (Thread &t: threads.y) {
            MatrixXs onGrid(t.gridLength, t.eigenpairs.size());
            ArrayXs subGrid = grid.x.segment(t.gridOffset, t.gridLength);
            Index j = 0;
            for (auto &Ef: t.eigenpairs)
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
        for (auto &intersection: intersections) {
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
    for (const typename Schrodinger2D<Scalar>::Intersection &intersection: self->intersections) {
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

    for (const auto &x: self->threads.x) {
        Index offset = x.offset;
        for (auto &ef: x.eigenpairs)
            lambda_x(offset++) = get<0>(ef);
    }
    for (const auto &y: self->threads.y) {
        Index offset = y.offset;
        for (auto &ef: y.eigenpairs)
            lambda_y(offset++) = get<0>(ef);
    }

    MatrixXs crossingsMatch(rows, colsX + colsY);
    crossingsMatch << beta_x, -beta_y;
    MatrixXs kernel = schrodinger::internal::rightKernel<MatrixXs>(crossingsMatch, 1e-6);

    MatrixXs A(rows, colsX + colsY); // rows x (colsX + colsY)
    A << beta_x * lambda_x.asDiagonal(), beta_y * lambda_y.asDiagonal();

    MatrixXs BK = colsY < colsX // rows x kernelSize
                  ? beta_y * kernel.bottomRows(colsY)
                  : beta_x * kernel.topRows(colsX);

    RectangularPencil<withEigenfunctions, MatrixXs> pencil(A * kernel, BK, self->options.pencilThreshold);


    if constexpr(withEigenfunctions) {
        const auto &values = pencil.eigenvalues();
        const auto &vectors = pencil.eigenvectors();

        typedef std::pair<Scalar, typename Schrodinger2D<Scalar>::Eigenfunction> Eigenpair;
        std::vector<Eigenpair> eigenfunctions;
        eigenfunctions.reserve(values.size());
        for (Index i = 0; i < values.size(); ++i) {
            // Normalize eigenfunction

            VectorXs coeffs = kernel * vectors.col(i).real();

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

        Vector<Scalar, 5> xpoints;
        xpoints << xOffset, xOffset + 0.25 * hx, xOffset + 0.5 * hx, xOffset + 0.75 * hx, xOffset + hx;

        Vector<Scalar, 5> ypoints;
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
        Vector<Scalar, 9> B = Vector<Scalar, 9>::Zero();

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
        Vector<Scalar, 9> sol = A.partialPivLu().solve(B);

        // 5x5 grid interpolation
        for (size_t j = 0; j < 4; j++) {
            if (tile.intersections[j] != nullptr)
                gridValues((j % 2) * 4, (j / 2) * 4) = functionValues.x[tile.intersections[j]->index];
        }

        gridValues.col(1).template segment<3>(1) = sol.template segment<3>(0);
        gridValues.col(2).template segment<3>(1) = sol.template segment<3>(3);
        gridValues.col(3).template segment<3>(1) = sol.template segment<3>(6);



        // horizontal Lagrange polynomials
        Vector<Scalar, 5> lx;
        for (int i = 0; i < 5; i++) {
            lx(i) = 1;
            for (int j = 0; j < 5; j++)
                if (j != i)
                    lx(i) *= (x - xpoints(j)) / (xpoints(i) - xpoints(j));
        }

        // vertical Lagrange polynomials
        Vector<Scalar, 5> ly;
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

template
class schrodinger::Schrodinger2D<double>;

#ifdef SCHRODINGER_LONG_DOUBLE

template
class schrodinger::Schrodinger2D<long double>;

#endif
