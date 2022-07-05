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
const Eigen::Array<Scalar,
        Schrodinger2D<Scalar>::interpolationGridSize - 2, Schrodinger2D<Scalar>::interpolationGridSize - 2> &
getTilePotential(const Schrodinger2D<Scalar> &problem, const typename Schrodinger2D<Scalar>::Tile *tile) {
    constexpr int size = Schrodinger2D<Scalar>::interpolationGridSize;
    if (!tile->potential) {
        tile->potential.emplace();

        for (int i = 1; i < size - 1; ++i)
            for (int j = 1; j < size - 1; ++j)
                (*tile->potential)(i - 1, j - 1) = problem.V(Scalar(i) / size, Scalar(j) / size);
    }
    return *tile->potential;
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
class Schrodinger2D<Scalar>::Eigenfunction::EigenfunctionTile {
public:
    static constexpr int size = interpolationGridSize;

    const Tile *reference;
    bool initialized = false;
    Array<Scalar, size, size> eigenfunction;

    explicit EigenfunctionTile(const Tile *tile_) : reference(tile_) {};

    void initialize(const Eigenfunction &ef) {
        if (initialized) return;

        Matrix<Scalar, size, 1> xPoints = Matrix<Scalar, size, 1>::LinSpaced(
                reference->bounds.template min<0>(),
                reference->bounds.template max<0>());
        Matrix<Scalar, size - 2, 1> yPoints = Matrix<Scalar, size, 1>::LinSpaced(
                reference->bounds.template min<1>(),
                reference->bounds.template max<1>()).template segment<size - 2>(1);

        eigenfunction = Array<Scalar, size, size>::Zero(size, size);

        auto getXValues = [&](int i1, int i2) -> Array<Scalar, size, 1> {
            const Thread *thread = getThread<&PerDirection<const Thread *>::x>(i1, i2);
            if (thread == nullptr)
                return Array<Scalar, size, 1>::Zero();
            return reconstructEigenfunction<Scalar, size>(thread, ef.c.topRows(ef.problem->columns.x), xPoints);
        };
        auto getYValues = [&](int i1, int i2) -> Array<Scalar, size - 2, 1> {
            const Thread *thread = getThread<&PerDirection<const Thread *>::y>(i1, i2);
            if (thread == nullptr)
                return Array<Scalar, size - 2, 1>::Zero();
            return reconstructEigenfunction<Scalar, size - 2>(thread, ef.c.bottomRows(ef.problem->columns.x), yPoints);
        };

        eigenfunction.row(0) = getXValues(0, 1);

        eigenfunction.col(0).segment(1, size - 2) = getYValues(0, 3);

        eigenfunction.row(size - 1) = getXValues(2, 3);

        eigenfunction.col(size - 1).segment(1, size - 2) = getYValues(1, 2);

        static_assert(size >= 5);

        Matrix<Scalar, size - 2, size> coeff = Matrix<Scalar, size - 2, size>::Zero();
        coeff.row(0).segment(0, 5) << 11, -20, 6, 4, -1;
        for (int i = 1; i < size - 3; ++i) {
            coeff.row(i).segment(i - 1, 5) << -1, 16, -30, 16, -1;
        }
        coeff.row(size - 3).segment(size - 5, 5) << -1, 4, 6, -20, 11;
        coeff /= 12;

        Scalar dx = (reference->bounds.template max<0>() - reference->bounds.template min<0>()) / (size - 1);
        Scalar dy = (reference->bounds.template max<1>() - reference->bounds.template min<1>()) / (size - 1);

        constexpr int n = (size - 2) * (size - 2);

        const Eigen::Array<Scalar, size - 2, size - 2> &potential = getTilePotential(*ef.problem, reference);

        Matrix<Scalar, n, n> A = -1 / (dx * dx) * kroneckerProduct(Matrix<Scalar, size - 2, size - 2>::Identity(),
                                                                   coeff.template middleCols<size - 2>(1))
                                 - 1 / (dy * dy) * kroneckerProduct(coeff.template middleCols<size - 2>(1),
                                                                    Matrix<Scalar, size - 2, size - 2>::Identity());

        Matrix<Scalar, n, 1> b = Matrix<Scalar, n, 1>::Zero();

        for (int rx = 1; rx + 1 < size; rx++)
            for (int ry = 1; ry + 1 < size; ry++) {
                int k = rx - 1 + (ry - 1) * (size - 2);
                A(k, k) += potential(rx - 1, ry - 1) - ef.E;

                b(k) = (
                               coeff(rx - 1, 0) * eigenfunction(ry, 0)
                               + coeff(rx - 1, size - 1) * eigenfunction(ry, size - 1)
                       ) / (dx * dx)
                       + (
                                 coeff(ry - 1, 0) * eigenfunction(0, rx)
                                 + coeff(ry - 1, size - 1) * eigenfunction(size - 1, rx)
                         ) / (dy * dy);
            }

        Eigen::Matrix<Scalar, 9, 1> sol = A.partialPivLu().solve(b);
        for (int rx = 1; rx + 1 < size; rx++)
            for (int ry = 1; ry + 1 < size; ry++) {
                int k = rx - 1 + (ry - 1) * (size - 2);
                eigenfunction(rx, ry) = sol(k);
            }

        initialized = true;
    }

    Scalar operator()(Scalar x, Scalar y) {
        (void) x;
        (void) y;
        return eigenfunction(2, 2);
    }

    template<const Thread *PerDirection<const Thread *>::*x>
    const Thread *getThread(int i1, int i2) {
        Intersection *int1 = reference->intersections[i1];
        if (int1 != nullptr && int1->thread.*x != nullptr)
            return int1->thread.*x;
        Intersection *int2 = reference->intersections[i2];
        return int2 == nullptr ? nullptr : int2->thread.*x;
    }
};

template<typename Scalar>
Schrodinger2D<Scalar>::Eigenfunction::Eigenfunction(const Schrodinger2D<Scalar> *problem, Scalar E, const VectorXs &c)
        : problem(problem), E(E), c(c) {
    tiles.reserve(problem->tiles.size());
    for (const auto &reference: problem->tiles) {
        tiles.emplace_back(new EigenfunctionTile(&reference));
        tilesMap.emplace(reference.index, tiles.back().get());
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

        auto tileIt = tilesMap.find({ix, iy});
        if (tileIt == tilesMap.end()) throw std::runtime_error("No tile found.");

        EigenfunctionTile &tile = *tileIt->second;
        if (!tile.initialized)
            tile.initialize(*this);

        result(p) = tile(x, y);

/*
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
        */
    }

    return result;
}

template<typename Scalar>
Schrodinger2D<Scalar>::Eigenfunction::~Eigenfunction() = default;

#include "instantiate.h"
