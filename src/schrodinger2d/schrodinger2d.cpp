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
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>>
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

        typedef std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>> Eigenpair;
        std::vector<Eigenpair> eigenfunctions;
        eigenfunctions.reserve(values.size());
        for (Index i = 0; i < values.size(); ++i) {
            // Normalize eigenfunction

            VectorXs coeffs = kernel * vectors.col(i).real();

            eigenfunctions.emplace_back(
                    values[i].real(),
                    std::make_unique<typename Schrodinger2D<Scalar>::Eigenfunction>(self, values[i].real(), coeffs)
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
std::vector<std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>>
Schrodinger2D<Scalar>::eigenfunctions() const {
    return eigenpairs<Scalar, true>(this);
}

#include "instantiate.h"
