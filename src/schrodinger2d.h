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
    };

    class Schrodinger2D {
    public:
        class Eigenfunction;

        struct Thread {
            double value;
            Eigen::Index valueIndex;
            size_t offset;
            Eigen::Index gridOffset;
            Eigen::Index gridLength;
            std::shared_ptr<matslise::Matslise<double>> matslise;
            std::vector<std::pair<double, matslise::Matslise<double>::Eigenfunction>> eigenpairs;
        };

        struct Intersection {
            PerDirection<double> position;
            PerDirection<const Thread *> thread;
            PerDirection<Eigen::ArrayXd> evaluation;
        };

        struct Tile {
            std::array<Intersection *, 4> intersections = {nullptr, nullptr, nullptr,
                                                           nullptr}; // [(xmin,ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
        };

        PerDirection<Eigen::ArrayXd> grid;
        PerDirection<std::vector<Thread>> threads;
        std::vector<Intersection> intersections;
        Eigen::Array<Tile, Eigen::Dynamic, Eigen::Dynamic> tiles;
        PerDirection<size_t> columns;

        std::function<double(double, double)> V;
        isocpp_p0201::polymorphic_value<geometry::Domain<double, 2>> domain;
        Options options;

        Schrodinger2D(const Schrodinger2D &) = delete;

        Schrodinger2D &operator=(const Schrodinger2D &) = delete;

        Schrodinger2D(const std::function<double(double, double)> &V,
                      const geometry::Domain<double, 2> &_domain,
                      const Options &options = Options());

        std::vector<double> eigenvalues() const;

        std::vector<std::pair<double, Eigenfunction>> eigenfunctions() const;

    private:
        Eigen::MatrixXd discreteProblem();
    };

    class Schrodinger2D::Eigenfunction {
        const Schrodinger2D *problem;
        double E;
        Eigen::VectorXd c;
    public:
        Eigenfunction(const Schrodinger2D *problem, double E, const Eigen::VectorXd &c)
                : problem(problem), E(E), c(c) {
        }

        double operator()(double x, double y) const;
    };
}

#endif //SCHRODINGER2D_SCHRODINGER2D_H
