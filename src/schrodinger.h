#ifndef SCHRODINGER2D_SCHRODINGER2D_H
#define SCHRODINGER2D_SCHRODINGER2D_H

#include <matslise/matslise.h>
#include <Eigen/Core>
#include <vector>
#include <map>
#include <optional>
#include "domain.h"
#include "util/polymorphic_value.h"
#include "util/per_direction.h"

namespace schrodinger {
    typedef Eigen::Index Index;

    template<typename Scalar=double, Index dimension = 2>
    struct Options {
        PerDirection<int, dimension> gridSize = PerDirection<int, dimension>::filled(23);
        int maxBasisSize = 22;
        double pencilThreshold = 1e-8;
        bool sparse = false;
        bool shiftInvert = true;
    };

    template<typename Scalar, Index dimension = 2>
    class Schrodinger {
    public:
        class Eigenfunction;

        static constexpr const int interpolationGridSize = 5;

        typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> ArrayXs;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

        struct Intersection;

        struct Thread {
            Index offset;
            Index gridOffset;
            Index gridLength;
            std::unique_ptr<matslise::Matslise<Scalar>> matslise;
            std::vector<std::pair<Scalar, std::unique_ptr<typename matslise::Matslise<Scalar>::Eigenfunction>>> eigenpairs;
            std::vector<Intersection *> intersections;
        };

        // struct Tile;

        struct Intersection {
            Index index;
            geometry::Vector<Scalar, dimension> position;
            PerDirection<const Thread *, dimension> thread;
            PerDirection<ArrayXs, dimension> evaluation;
            // [top-left, top-right, bottom-right, bottom-left]
            // std::array<Tile *, 4> tiles = {nullptr, nullptr, nullptr, nullptr};
        };

        struct Tile {
            PerDirection<Index, dimension> index;
            std::array<Intersection *, 1 << dimension> intersections;
            geometry::Rectangle<Scalar, dimension> bounds;

            mutable std::optional<Eigen::Array<Scalar, interpolationGridSize - 2, interpolationGridSize - 2>> potential;
            PerDirection<Eigen::Array<Scalar, interpolationGridSize, 1>, dimension> grid;
        };

        PerDirection<ArrayXs, dimension> grid;
        PerDirection<std::vector<std::unique_ptr<Thread>>, dimension> threads;
        std::vector<std::unique_ptr<Intersection>> intersections;
        std::vector<std::unique_ptr<Tile>> tiles;
        PerDirection<size_t, dimension> columns; // Total number of basisfunctions

        std::function<Scalar(const geometry::Vector<Scalar, dimension> &)> V;
        isocpp_p0201::polymorphic_value<geometry::Domain<Scalar, 2>> domain;
        Options<Scalar, dimension> options;

        Schrodinger(const Schrodinger &) = delete;

        Schrodinger &operator=(const Schrodinger &) = delete;

        template<bool enable = true, typename=std::enable_if_t<enable && dimension == 2>>
        Schrodinger(const std::function<Scalar(Scalar, Scalar)> &V,
                    const geometry::Domain<Scalar, 2> &domain,
                    const Options<Scalar, dimension> &options = Options<Scalar, dimension>()) :
                Schrodinger([&V](const geometry::Vector<Scalar, 2> &xy) { return V(xy[0], xy[1]); }, domain,
                            options) {};

        Schrodinger(const std::function<Scalar(const geometry::Vector<Scalar, dimension> &)> &V,
                    const geometry::Domain<Scalar, 2> &domain,
                    const Options<Scalar, dimension> &options = Options<Scalar, dimension>());

        std::vector<Scalar> eigenvalues(int eigenvaluesCount = -1) const;

        std::vector<std::pair<Scalar, std::unique_ptr<Eigenfunction>>> eigenfunctions(int eigenvaluesCount = -1) const;
    };

    template<typename Scalar, Index dimension>
    class Schrodinger<Scalar, dimension>::Eigenfunction {
        class EigenfunctionTile;

        const Schrodinger<Scalar> *problem;
        Scalar E;
        VectorXs c;

    public:
        std::vector<std::unique_ptr<EigenfunctionTile>> tiles;
        std::map<std::pair<int, int>, EigenfunctionTile *> tilesMap;

        Eigenfunction(const Schrodinger<Scalar> *problem, Scalar E, const PerDirection<VectorXs, dimension> &c);

        Scalar operator()(Scalar x, Scalar y) const;

        ArrayXs operator()(ArrayXs x, ArrayXs y) const;

        ~Eigenfunction();
    };
}

#endif //SCHRODINGER2D_SCHRODINGER2D_H
