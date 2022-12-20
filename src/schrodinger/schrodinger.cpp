#include "../util/polymorphic_value.h"
#include "../schrodinger.h"
#include "./eigenpairs.h"
#include "../util/tensor_indexer.h"
#include <map>
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

template<typename Scalar, Index dimension>
std::unique_ptr<typename Schrodinger<Scalar, dimension>::Thread>
computeThread(const std::function<Scalar(Scalar)> &V, Scalar min, Scalar max, size_t maxPairs,
              const Ref<const Array<Scalar, Dynamic, 1>> &grid) {
    typedef typename Schrodinger<Scalar>::Thread Thread;
    typedef IterateEigen<const Ref<const Array<Scalar, Dynamic, 1>>> Iterate;

    const Scalar *from = std::lower_bound(Iterate(grid).begin(), Iterate(grid).end(), min);
    if (*from == min)
        ++from;

    Index length = std::distance(from, std::lower_bound(from, Iterate(grid).end(), max));
    auto thread = std::make_unique<Thread>(Thread{
            .gridOffset = std::distance(Iterate(grid).begin(), from),
            .gridLength = length,
            .matslise = nullptr,
            .eigenpairs = std::vector<std::pair<Scalar, std::unique_ptr<typename matslise::Matslise<Scalar>::Eigenfunction>>>(),
    });
    thread->intersections.reserve(length);
    if (length == 0) return thread;
    thread->matslise = make_unique<Matslise<Scalar>>(V, min, max, 1e-12);
    size_t pairs = std::min(maxPairs, (size_t) length);

    thread->eigenpairs.reserve(pairs);
    for (auto &iEf: thread->matslise->eigenpairsByIndex(0, pairs, Y<Scalar>::Dirichlet())) {
        thread->eigenpairs.emplace_back(get<1>(iEf), std::move(get<2>(iEf)));
    }


    return thread;
}

template<typename Scalar, Index dimension>
Array<Scalar, Dynamic, 1>
internal_linspaced(Index size, const Domain<Scalar, dimension> *domain, const Matrix<Scalar, dimension, 1> &direction) {
    Scalar a, b;
    tie(a, b) = domain->bounds(direction);
    return Array<Scalar, Dynamic, 1>::LinSpaced(size + 2, a, b).segment(1, size);
}

template<typename Scalar, Index dimension>
Schrodinger<Scalar, dimension>::Schrodinger(const function<Scalar(const geometry::Vector<Scalar, dimension> &)> &V_,
                                            const Domain<Scalar, 2> &domain_,
                                            const Options<Scalar, dimension> &options_)
        : V(V_), domain(polymorphic_value<Domain<Scalar, 2>>(domain_.clone(), typename Domain<Scalar, 2>::copy{})),
          options(options_) {
    TensorIndexer<dimension> indexer{options.gridSize};
    Index intersectionCount = indexer.totalSize();

    // Prepare all intersections
    intersections.resize(intersectionCount);
    for (auto &intersection: intersections)
        intersection = std::make_unique<Intersection>();

    // Set position of each intersection
    for (Index i = 0; i < dimension; ++i) {
        Index size = options.gridSize[i];
        grid[i] = internal_linspaced<Scalar, dimension>(size, &*domain, Matrix<Scalar, dimension, 1>::Unit(i));

        indexer.forEachDirectionStart(i, [&](auto multiIndex, Index size) {
            for (Index j = 0; j < size; ++j) {
                multiIndex[i] = j;
                intersections[indexer(multiIndex)]->position[i] = grid[i][j];
            }
        });
    }

    // Build threads in all directions
    for (Index i = 0; i < dimension; ++i) {
        Index unknownsOffset = 0;
        // forEachDirectionStart(i, [&](Index offset, Index stride, Index) {
        indexer.forEachDirectionStart(i, [&](auto multiIndex, Index) {
            geometry::Vector<Scalar, dimension> position = intersections[indexer(multiIndex)]->position;
            position[i] = 0;

            for (auto &dom: domain->intersections({position, Matrix<Scalar, dimension, 1>::Unit(i)})) {
                std::unique_ptr<Thread> new_thread = computeThread<Scalar, dimension>(
                        [position, i, this](Scalar x)mutable -> Scalar {
                            position[i] = x;
                            return V(position) / dimension;
                        },
                        dom.first, dom.second, (size_t) options.maxBasisSize, grid[i]);
                if (new_thread->gridLength > 0)
// #pragma omp critical
                {
                    Thread &thread = *threads[i].emplace_back(std::move(new_thread));
                    thread.offset = unknownsOffset;
                    unknownsOffset += thread.eigenpairs.size();
                    thread.intersections.reserve(thread.gridLength);

                    MatrixXs onGrid(thread.gridLength, thread.eigenpairs.size());
                    ArrayXs subGrid = grid.x.segment(thread.gridOffset, thread.gridLength);
                    Index eigenindex = 0;
                    for (auto &Ef: thread.eigenpairs)
                        onGrid.col(eigenindex++) = (*Ef.second)(subGrid).col(0);

                    for (Index j = thread.gridOffset; j < thread.gridLength; ++j) {
                        multiIndex[i] = j;
                        Intersection *intersection = intersections[indexer(multiIndex)].get();
                        thread.intersections.push_back(intersection);
                        assert(intersection->thread[i] == nullptr);
                        intersection->thread[i] = &thread;

                        intersection->evaluation[i] = onGrid.row(j);
                    }
                }
            }
        });
        columns[i] = unknownsOffset;
    }

    {
        TensorIndexer<dimension> tileIndexer{options.gridSize.toEigen() + 1};
        TensorIndexer<dimension> twoIndexer{Eigen::Matrix<Index, dimension, 1>::Constant(2)};

        geometry::Rectangle<Scalar, dimension> bounds;
        for (Index i = 0; i < dimension; ++i) {
            bounds.min(i) = grid[i][0];
            bounds.max(i) = grid[i][grid[i].size() - 1];
        }

        tiles.resize(tileIndexer.totalSize());
        tileIndexer.forEach([&](const auto &index) {
            Tile &tile = *tiles[tileIndexer(index)];
            tile.index = index;
            tile.bounds = bounds;
        });

        indexer.forEach([&](const auto &multiIndex) {
            Intersection *intersection = intersections[indexer(multiIndex)].get();
            twoIndexer.forEach([&](auto two_index) {
                Tile &tile = *tiles[tileIndexer(multiIndex.toEigen() + 1 - two_index.toEigen())];
                tile.intersections[twoIndexer(two_index.toEigen())] = intersection;

                for (Index i = 0; i < dimension; ++i) {
                    if (two_index[i] == 0)
                        tile.bounds.min(i) = intersection->position[i];
                    else
                        tile.bounds.max(i) = intersection->position[i];
                }
            });
        });

        for (auto &tile: tiles) {
            for (Index i = 0; i < dimension; ++i)
                tile->grid[i] = Eigen::Array<Scalar, interpolationGridSize, 1>::LinSpaced(
                        interpolationGridSize, tile->bounds.min(i), tile->bounds.max(i));
        }
    }


    {
        auto it = std::partition(intersections.begin(), intersections.end(), [](const auto &intersection) {
            bool is_valid = intersection->thread[0] != nullptr;
#ifndef NDEBUG
            for (Index i = 0; i < dimension; ++i)
                assert(is_valid == (intersection->thread[i] != nullptr));
#endif
            return is_valid;
        });
        intersections.erase(it, intersections.end());
        Index intersectionIndex;
        for (auto &intersection: intersections)
            intersection->index = intersectionIndex++;
    }
}


template<typename Scalar, Eigen::Index dimension>
std::vector<Scalar> Schrodinger<Scalar, dimension>::eigenvalues(int eigenvalueCount) const {
    return eigenpairs<Scalar, false>(this, eigenvalueCount);
}

template<typename Scalar, Eigen::Index dimension>
std::vector<std::pair<Scalar, std::unique_ptr<typename Schrodinger<Scalar, dimension>::Eigenfunction>>>
Schrodinger<Scalar, dimension>::eigenfunctions(int eigenvalueCount) const {
    return eigenpairs<Scalar, true>(this, eigenvalueCount);
}

#include "instantiate.h"
