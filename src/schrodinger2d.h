//
// Created by toon on 4/8/21.
//

#ifndef SCHRODINGER2D_SCHRODINGER2D_H
#define SCHRODINGER2D_SCHRODINGER2D_H

#include <matslise/matslise.h>
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include "domain.h"
#include "util/polymorphic_value.h"

namespace schrodinger {
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

    template<typename Scalar>
    class Schrodinger2D {
    public:
        class Eigenfunction;

        typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> ArrayXs;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

        struct Thread {
            Scalar value;
            Eigen::Index valueIndex;
            size_t offset;
            Eigen::Index gridOffset;
            Eigen::Index gridLength;
            std::unique_ptr<matslise::Matslise<Scalar>> matslise;
            std::vector<std::pair<Scalar, std::unique_ptr<typename matslise::Matslise<Scalar>::Eigenfunction>>> eigenpairs;
        };

        struct Intersection {
            size_t index; // index in the list of intersections
            PerDirection<Scalar> position;
            PerDirection<const Thread *> thread;
            PerDirection<ArrayXs> evaluation;
        };

        struct Tile {
            std::array<Intersection *, 4> intersections = {nullptr, nullptr, nullptr,
                                                           nullptr}; // [(xmin,ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
        };

        PerDirection<ArrayXs> grid;
        PerDirection<std::vector<Thread>> threads;
        std::vector<Intersection> intersections;
        Eigen::Array<Tile, Eigen::Dynamic, Eigen::Dynamic> tiles;
        PerDirection<size_t> columns;

        std::function<Scalar(Scalar, Scalar)> V;
        isocpp_p0201::polymorphic_value<geometry::Domain<Scalar, 2>> domain;
        Options options;

        Schrodinger2D(const Schrodinger2D &) = delete;

        Schrodinger2D &operator=(const Schrodinger2D &) = delete;

        Schrodinger2D(const std::function<Scalar(Scalar, Scalar)> &V,
                      const geometry::Domain<Scalar, 2> &_domain,
                      const Options &options = Options());

        std::vector<Scalar> eigenvalues() const;

        std::vector<std::pair<Scalar, std::unique_ptr<Eigenfunction>>> eigenfunctions() const;
    };

    template<typename Scalar>
    class Schrodinger2D<Scalar>::Eigenfunction {
        class EigenfunctionTile;

        const Schrodinger2D<Scalar> *problem;
        Scalar E;
        VectorXs c;

    public:
        std::vector<std::unique_ptr<EigenfunctionTile>> tiles;
        // Function values evaluated in each intersection point
        PerDirection<VectorXs> functionValues;

        Eigenfunction(const Schrodinger2D<Scalar> *problem, Scalar E, const VectorXs &c);

        Scalar operator()(Scalar x, Scalar y) const;

        ArrayXs operator()(ArrayXs x, ArrayXs y) const;

        ~Eigenfunction();
    };
}

#endif //SCHRODINGER2D_SCHRODINGER2D_H
