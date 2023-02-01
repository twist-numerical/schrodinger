#ifndef STRANDS_SCHRODINGER_H
#define STRANDS_SCHRODINGER_H

#include <matslise/matslise.h>
#include <Eigen/Core>
#include <vector>
#include <map>
#include <stdexcept>
#include <sstream>
#include <optional>
#include "domain.h"

namespace strands {
    template<class Assertion, class Message>
    inline void validate_argument(Assertion toCheck, const Message &message) {
#ifndef STRANDS_NO_VALIDATE_ARGUMENTS
        if (!toCheck()) {
            if constexpr (std::is_invocable_v<Message, std::stringstream &>) {
                std::stringstream r;
                message(r);
                throw std::invalid_argument(r.str());
            } else {
                throw std::invalid_argument(message);
            }
        }
#endif
    }

    template<class T>
    struct PerDirection {
        T x;
        T y;
    };

    struct Options {
        PerDirection<int> gridSize = {.x=11, .y=11};
        int maxBasisSize = 22;
        double pencilThreshold = 1e-8;
    };

    struct EigensolverOptions {
        Eigen::Index k = 10;
        Eigen::Index ncv = -1;
        bool sparse = false;
        bool shiftInvert = true;
        double tolerance = 1e-10;
        Eigen::Index maxIterations = 1000;
    };

    template<typename Scalar>
    class Schrodinger {
    public:
        class Eigenfunction;

        static constexpr const int interpolationGridSize = 5;

        typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> ArrayXs;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

        struct Intersection;

        struct Thread {
            Scalar value;
            Eigen::Index valueIndex;
            size_t offset;
            Eigen::Index gridOffset;
            Eigen::Index gridLength;
            std::unique_ptr<matslise::Matslise<Scalar>> matslise;
            std::vector<std::pair<Scalar, std::unique_ptr<typename matslise::Matslise<Scalar>::Eigenfunction>>> eigenpairs;
            std::vector<Intersection *> intersections;
        };

        struct Tile;

        struct Intersection {
            size_t index; // index in the list of intersections
            PerDirection<Scalar> position;
            PerDirection<const Thread *> thread;
            PerDirection<ArrayXs> evaluation;
            // [top-left, top-right, bottom-right, bottom-left]
            std::array<Tile *, 4> tiles = {nullptr, nullptr, nullptr, nullptr};
        };

        struct Tile {
            std::pair<int, int> index;
            geometry::Rectangle<Scalar, 2> bounds;
            // [(xmax, ymin), (xmin,ymin), (xmin, ymax), (xmax, ymax)]
            std::array<Intersection *, 4> intersections = {nullptr, nullptr, nullptr, nullptr};
            mutable std::optional<Eigen::Array<Scalar, interpolationGridSize - 2, interpolationGridSize - 2>> potential;
            PerDirection<Eigen::Matrix<Scalar, interpolationGridSize, 1>> grid;
        };

        PerDirection<ArrayXs> grid;
        PerDirection<std::vector<Thread>> threads;
        std::vector<Intersection> intersections;
        std::vector<Tile> tiles;
        PerDirection<size_t> columns;

        std::function<Scalar(Scalar, Scalar)> V;
        std::shared_ptr<geometry::Domain<Scalar, 2>> domain;
        Options options;

        Schrodinger(const Schrodinger &) = delete;

        Schrodinger &operator=(const Schrodinger &) = delete;

        Schrodinger(std::function<Scalar(Scalar, Scalar)> V,
                    std::shared_ptr<geometry::Domain<Scalar, 2>> _domain,
                    Options options = Options());

        template<typename DomainType>
        Schrodinger(std::function<Scalar(Scalar, Scalar)> V,
                    DomainType _domain,
                    Options options = Options()) :
                Schrodinger(std::move(V), geometry::Domain<Scalar, 2>::as_ptr(_domain), std::move(options)) {}

        PerDirection<MatrixXs> Beta() const;

        PerDirection<VectorXs> Lambda() const;

        std::vector<Scalar> eigenvalues(const EigensolverOptions &) const;

        std::vector<std::pair<Scalar, std::unique_ptr<Eigenfunction>>>
        eigenfunctions(const EigensolverOptions &) const;
    };

    template<typename Scalar>
    class Schrodinger<Scalar>::Eigenfunction {
        class EigenfunctionTile;

        const Schrodinger<Scalar> *problem;
        Scalar E;
        VectorXs c;

    public:
        std::vector<std::unique_ptr<EigenfunctionTile>> tiles;
        std::map<std::pair<int, int>, EigenfunctionTile *> tilesMap;

        Eigenfunction(const Schrodinger<Scalar> *problem, Scalar E, const VectorXs &c);

        Scalar operator()(Scalar x, Scalar y) const;

        ArrayXs operator()(ArrayXs x, ArrayXs y) const;

        ~Eigenfunction();
    };
}

#endif //STRANDS_SCHRODINGER_H
