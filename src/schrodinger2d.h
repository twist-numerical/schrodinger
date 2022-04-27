//
// Created by toon on 4/8/21.
//

#ifndef SCHRODINGER2D_SCHRODINGER2D_H
#define SCHRODINGER2D_SCHRODINGER2D_H

#include <matslise/matslise.h>
#include <Eigen/Dense>
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
        int pencilMethod = -1;
        int interpolationMethod = -1;
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

        std::vector<std::pair<Scalar, Eigenfunction>> eigenfunctions() const;
    };

    template<typename Scalar>
    class Schrodinger2D<Scalar>::Eigenfunction {
        const Schrodinger2D<Scalar> *problem;
        Scalar E;
        VectorXs c;

    public:
        // Function values evaluated in each intersection point
        PerDirection<VectorXs> functionValues;

        Eigenfunction(const Schrodinger2D *problem, Scalar E, const VectorXs &c)
                : problem(problem), E(E), c(c) {

            // Initialize function values
            size_t numIntersections = problem->intersections.size();
            functionValues.x = VectorXs::Zero(numIntersections);
            functionValues.y = VectorXs::Zero(numIntersections);

            for (size_t i = 0; i < numIntersections; i++) {
                Intersection intersection = problem->intersections[i];

                const Thread* tx = intersection.thread.x;
                functionValues.x(i) = intersection.evaluation.x.matrix().dot(
                        c.segment(tx->offset, tx->eigenpairs.size()));

                const Thread* ty = intersection.thread.y;
                functionValues.y(i) = intersection.evaluation.y.matrix().dot(
                        c.segment(problem->columns.x + ty->offset, ty->eigenpairs.size()));
            }
        }

        Scalar operator()(Scalar x, Scalar y, int interpolationMethod=-1) const;

        ArrayXs operator()(ArrayXs x, ArrayXs y, int interpolationMethod=-1) const;
    };
}

#endif //SCHRODINGER2D_SCHRODINGER2D_H
