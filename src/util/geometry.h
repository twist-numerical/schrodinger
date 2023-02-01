#ifndef STRANDS_GEOMETRY_H
#define STRANDS_GEOMETRY_H

#include <Eigen/Dense>

namespace strands::geometry {

    template<typename Scalar, int d>
    using Vector = Eigen::Matrix<Scalar, d, 1>;

    template<typename Scalar, int d>
    struct Hyperspace {
    public:
        Vector<Scalar, d> origin;
        Vector<Scalar, d> normal;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    template<typename Scalar, int d>
    struct Ray {
    public:
        Vector<Scalar, d> origin;
        Vector<Scalar, d> direction;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Scalar cast(const Hyperspace<Scalar, d> &space) const {
            return -space.normal.dot(origin - space.origin) / space.normal.dot(direction);
        }

        Vector<Scalar, d> operator()(Scalar v) const {
            return origin + v * direction;
        }
    };

}

#endif //STRANDS_GEOMETRY_H
