#include "matslise/util/quadrature.h"
#include "../util/polymorphic_value.h"
#include "../schrodinger2d.h"
#include <numeric>
#include <map>
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
        std::map<std::pair<Index, Index>, Intersection *> intersectionsMap{};
        size_t maxCapacity = (1 + threads.x.size()) * (1 + threads.y.size());
        tiles.reserve(maxCapacity);

        auto getTile = [&](Index i, Index j, int dx, int dy, int t1, int t2, int t3, int xi, int yi) -> Tile * {
            auto f1 = intersectionsMap.find({i + dx, j});
            if (f1 != intersectionsMap.end() && f1->second->tiles[t1] != nullptr)
                return f1->second->tiles[t1];
            auto f2 = intersectionsMap.find({i + dx, j + dy});
            if (f2 != intersectionsMap.end() && f2->second->tiles[t2] != nullptr)
                return f2->second->tiles[t2];
            auto f3 = intersectionsMap.find({i, j + dy});
            if (f3 != intersectionsMap.end() && f3->second->tiles[t3] != nullptr)
                return f3->second->tiles[t3];
            tiles.emplace_back();
            tiles.back().index = {xi, yi};
            return &tiles.back();
        };

        for (auto &intersection: intersections) {
            Index i = intersection.thread.x->valueIndex;
            Index j = intersection.thread.y->valueIndex;

            intersection.tiles[0] = getTile(i, j, 1, 1, 1, 2, 3, i + 1, j + 1);
            intersection.tiles[1] = getTile(i, j, -1, 1, 0, 3, 2, i, j + 1);
            intersection.tiles[2] = getTile(i, j, -1, -1, 3, 0, 1, i, j);
            intersection.tiles[3] = getTile(i, j, 1, -1, 2, 1, 0, i + 1, j);

            for (int k = 0; k < 4; ++k) {
                assert(intersection.tiles[k]->intersections[k] == nullptr);
                intersection.tiles[k]->intersections[k] = &intersection;
            }

            intersectionsMap.emplace(std::make_pair(i, j), &intersection);
        }

        assert(maxCapacity == tiles.capacity());

        Scalar xmin, xmax, ymin, ymax;
        std::tie(xmin, xmax) = domain->bounds(Eigen::Matrix<Scalar, 2, 2>::Identity().col(0));
        std::tie(ymin, ymax) = domain->bounds(Eigen::Matrix<Scalar, 2, 2>::Identity().col(1));

        auto fx = [](const auto &arr, int a, int b, const Scalar &c) {
            return arr[a] != nullptr ? arr[a]->position.x : arr[b] != nullptr ? arr[b]->position.x : c;
        };
        auto fy = [](const auto &arr, int a, int b, const Scalar &c) {
            return arr[a] != nullptr ? arr[a]->position.y : arr[b] != nullptr ? arr[b]->position.y : c;
        };

        for (auto &tile: tiles) {
            tile.bounds.template min<0>() = fx(tile.intersections, 0, 3, xmin);
            tile.bounds.template max<0>() = fx(tile.intersections, 1, 2, xmax);
            tile.bounds.template min<1>() = fy(tile.intersections, 0, 1, ymin);
            tile.bounds.template max<1>() = fy(tile.intersections, 2, 3, ymax);

            tile.grid.x = Matrix<Scalar, interpolationGridSize, 1>::LinSpaced(
                    tile.bounds.template min<0>(),
                    tile.bounds.template max<0>());
            tile.grid.y = Matrix<Scalar, interpolationGridSize, 1>::LinSpaced(
                    tile.bounds.template min<1>(),
                    tile.bounds.template max<1>());
        }
    }
}

template<typename Scalar>
std::pair<typename Schrodinger2D<Scalar>::MatrixXs, typename Schrodinger2D<Scalar>::MatrixXs>
Schrodinger2D<Scalar>::Beta() const {
    size_t rows = intersections.size();

    MatrixXs beta_x = MatrixXs::Zero(rows, columns.x);
    MatrixXs beta_y = MatrixXs::Zero(rows, columns.y);

    size_t row = 0;
    for (const typename Schrodinger2D<Scalar>::Intersection &intersection: intersections) {
        beta_x.row(row).segment(
                intersection.thread.x->offset, intersection.thread.x->eigenpairs.size()
        ) = intersection.evaluation.x;
        beta_y.row(row).segment(
                intersection.thread.y->offset, intersection.thread.y->eigenpairs.size()
        ) = intersection.evaluation.y;

        ++row;
    }
    assert(row == rows);

    return {beta_x, beta_y};
}

template<typename Scalar>
std::pair<typename Schrodinger2D<Scalar>::VectorXs, typename Schrodinger2D<Scalar>::VectorXs>
Schrodinger2D<Scalar>::Lambda() const {
    VectorXs lambda_x(columns.x);
    VectorXs lambda_y(columns.y);

    for (const auto &x: threads.x) {
        Index offset = x.offset;
        for (auto &ef: x.eigenpairs)
            lambda_x(offset++) = get<0>(ef);
    }
    for (const auto &y: threads.y) {
        Index offset = y.offset;
        for (auto &ef: y.eigenpairs)
            lambda_y(offset++) = get<0>(ef);
    }

    return {lambda_x, lambda_y};
}


template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>>
eigenpairs(const Schrodinger2D<Scalar> *self, int eigenvalueCount) {
    using MatrixXs = typename Schrodinger2D<Scalar>::MatrixXs;
    using VectorXs = typename Schrodinger2D<Scalar>::VectorXs;

    Eigen::Index rows = self->intersections.size();
    Eigen::Index colsX = self->columns.x;
    Eigen::Index colsY = self->columns.y;
    MatrixXs beta_x, beta_y;
    VectorXs lambda_x, lambda_y;
    std::tie(beta_x, beta_y) = self->Beta();
    std::tie(lambda_x, lambda_y) = self->Lambda();

    MatrixXs crossingsMatch(rows, colsX + colsY);
    crossingsMatch << beta_x, -beta_y;

    MatrixXs kernel = schrodinger::internal::rightKernel<MatrixXs>(crossingsMatch, 1e-6);

    MatrixXs A(rows, colsX + colsY); // rows x (colsX + colsY)
    A << beta_x * lambda_x.asDiagonal(), beta_y * lambda_y.asDiagonal();

    MatrixXs BK = colsY < colsX // rows x kernelSize
                  ? beta_y * kernel.bottomRows(colsY)
                  : beta_x * kernel.topRows(colsX);

    RectangularPencil<withEigenfunctions, MatrixXs> pencil(A * kernel, BK, self->options.pencilThreshold);

    const auto &values = pencil.eigenvalues();
    if (eigenvalueCount < 0 || eigenvalueCount > values.size())
        eigenvalueCount = values.size();

    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + eigenvalueCount, indices.end(), [&values](int a, int b) {
        if (std::abs(values(b).imag()) > 1e-8) return true;
        if (std::abs(values(a).imag()) > 1e-8) return false;
        return values(a).real() < values(b).real();
    });
    indices.resize(eigenvalueCount);

    if constexpr(withEigenfunctions) {
        const auto &vectors = pencil.eigenvectors();

        typedef std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>> Eigenpair;
        std::vector<Eigenpair> eigenfunctions;

        eigenfunctions.reserve(indices.size());
        for (int i: indices) {
            VectorXs coeffs = kernel * vectors.col(i).real();

            eigenfunctions.emplace_back(
                    values(i).real(),
                    std::make_unique<typename Schrodinger2D<Scalar>::Eigenfunction>(self, values(i).real(), coeffs)
            );
        }
        return eigenfunctions;
    } else {
        std::vector<Scalar> eigenvalues;
        eigenvalues.reserve(indices.size());
        for (int i: indices)
            eigenvalues.push_back(values(i).real());

        return eigenvalues;
    }
}

template<typename Scalar>
std::vector<Scalar> Schrodinger2D<Scalar>::eigenvalues(int eigenvalueCount) const {
    return eigenpairs<Scalar, false>(this, eigenvalueCount);
}

template<typename Scalar>
std::vector<std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>>
Schrodinger2D<Scalar>::eigenfunctions(int eigenvalueCount) const {
    return eigenpairs<Scalar, true>(this, eigenvalueCount);
}

#include "instantiate.h"
