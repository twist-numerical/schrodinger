#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <utility>

#include "../schrodinger.h"

namespace py = pybind11;
using namespace strands;
using namespace strands::geometry;


template<typename T>
void perDirection(const py::handle &scope, const std::string &name) {
    py::class_<PerDirection<T>>(scope, name.c_str())
            .def_readonly("x", &PerDirection<T>::x)
            .def_readonly("y", &PerDirection<T>::y);
}

PYBIND11_MODULE(strands, m) {

    struct KeepAliveEigenfunction {
        std::unique_ptr<Schrodinger<double>::Eigenfunction> eigenfunction;
        std::shared_ptr<const Schrodinger<double>> problem;
    };

    py::class_<Domain<double, 2>, std::shared_ptr<Domain<double, 2>>>
            (m, "Domain2D")
            .def("contains", &Domain<double, 2>::contains)
            .def("intersections",
                 [](const Domain<double, 2> *self, const Eigen::Vector2d &origin, const Eigen::Vector2d &direction) {
                     return self->intersections({origin, direction});
                 })
            .def("bounds", &Domain<double, 2>::bounds);

    py::class_<Rectangle<double, 2>, Domain<double, 2>, std::shared_ptr<Rectangle<double, 2>>>(m, "Rectangle")
            .def(py::init<double, double, double, double>());

    py::class_<Sphere<double, 2>, Domain<double, 2>, std::shared_ptr<Sphere<double, 2>>>(m, "Circle")
            .def(py::init<double>())
            .def(py::init<Vector<double, 2>>())
            .def(py::init<Vector<double, 2>, double>());

    py::class_<Union<double, 2>, Domain<double, 2>, std::shared_ptr<Union<double, 2>>>(m, "Union")
            .def(py::init<const std::vector<std::shared_ptr<Domain<double, 2>>> &>());

    py::class_<DomainTransform<double, 2>, Domain<double, 2>, std::shared_ptr<DomainTransform<double, 2>>>(m,
                                                                                                           "DomainTransform")
            .def(py::init([](const std::shared_ptr<Domain<double, 2>> &domain, const Eigen::Matrix3d &transform) {
                Eigen::Affine2d t;
                t.matrix() = transform;
                return std::make_shared<DomainTransform<double, 2>>(domain, t);
            }), py::arg("domain"), py::arg("transformation"));

    py::class_<KeepAliveEigenfunction>(m, "Eigenfunction2D")
            .def("__call__", [](const KeepAliveEigenfunction &f, double x, double y) {
                return (*f.eigenfunction)(x, y);
            })
            .def("__call__", [](const KeepAliveEigenfunction &f,
                                Eigen::ArrayXd xs, Eigen::ArrayXd ys) -> Eigen::ArrayXd {
                validate_argument([&]() { return xs.size() == ys.size(); }, [&](auto &s) {
                    s << "Vectors should have the same length: " << xs.size() << " != " << ys.size();
                });
                return (*f.eigenfunction)(xs, ys);
            })
            .def("__call__", [](const KeepAliveEigenfunction &f,
                                Eigen::ArrayXXd xs, Eigen::ArrayXXd ys) -> Eigen::ArrayXXd {
                validate_argument([&]() { return xs.rows() == ys.rows() && xs.cols() == ys.cols(); }, [&](auto &s) {
                    s << "Matrices should have the same sizes: "
                      << xs.rows() << " x " << xs.cols() << " != " << ys.rows() << " x " << ys.cols();
                });
                return (*f.eigenfunction)(xs.reshaped(), ys.reshaped()).reshaped(xs.rows(), xs.cols());
            });

    perDirection<Schrodinger<double>::MatrixXs>(m, "PerDirectionMatrixXs");
    perDirection<Schrodinger<double>::VectorXs>(m, "PerDirectionVectorXs");

    py::class_<Schrodinger<double>, std::shared_ptr<Schrodinger<double>>>(m, "Schrodinger2D")
            .def(py::init(
                         [](const std::function<double(double, double)> &V,
                            const std::shared_ptr<Domain<double, 2>> &domain,
                            const std::array<int, 2> &gridSize, int maxBasisSize) {
                             py::gil_scoped_release release;
                             return std::make_shared<Schrodinger<double>>
                                     (V, domain, (Options) {
                                             .gridSize = {.x = gridSize[0], .y=gridSize[1]},
                                             .maxBasisSize=maxBasisSize,
                                     });
                         }), py::arg("V"), py::arg("domain"), py::arg("gridSize") = std::array<int, 2>{21, 21},
                 py::arg("maxBasisSize") = 16)
            .def("eigenvalues", [](
                         const std::shared_ptr<const Schrodinger<double>> &s,
                         Eigen::Index eigenvalueCount, bool sparse, bool shiftInvert, Eigen::Index ncv,
                         double tolerance, Eigen::Index maxIterations) {
                     py::gil_scoped_release release;
                     return s->eigenvalues(EigensolverOptions{
                             .k=eigenvalueCount,
                             .ncv=ncv,
                             .sparse=sparse,
                             .shiftInvert=shiftInvert,
                             .tolerance=tolerance,
                             .maxIterations=maxIterations
                     });
                 }, py::arg("k") = 10, py::arg("sparse") = true, py::arg("shiftInvert") = true, py::arg("ncv") = -1,
                 py::arg("tolerance") = 1e-10, py::arg("maxIterations") = 1000, R""""(\
Calculate the first k eigenvalues.

:param int k: minimal number of eigenvalues to find (default: 10).

:returns: a list of eigenvalues with length at least k.

>>> import numpy as np
>>> from math import pi
>>> r = Rectangle(0, pi, 0, pi)
>>> s = Schrodinger2D(lambda x, y: 0, r)
>>> np.array(s.eigenvalues(9)[:9]).round(6)
array([ 2.,  5.,  5.,  8., 10., 10., 13., 13., 17.])
)"""")
            .def("Beta", &Schrodinger<double>::Beta)
            .def("Lambda", &Schrodinger<double>::Lambda)
            .def("eigenfunctions", [](
                         const std::shared_ptr<const Schrodinger<double>> &s,
                         Eigen::Index eigenvalueCount, bool sparse, bool shiftInvert, Eigen::Index ncv,
                         double tolerance, Eigen::Index maxIterations) {
                     std::vector<std::pair<double, std::unique_ptr<Schrodinger<double>::Eigenfunction>>> eigs;
                     {
                         py::gil_scoped_release release;
                         eigs = s->eigenfunctions(EigensolverOptions{
                                 .k=eigenvalueCount,
                                 .ncv=ncv,
                                 .sparse=sparse,
                                 .shiftInvert=shiftInvert,
                                 .tolerance=tolerance,
                                 .maxIterations=maxIterations
                         });
                     }
                     py::list r;
                     for (auto &ef: eigs) {
                         py::tuple t(2);
                         t[0] = ef.first;
                         t[1] = (KeepAliveEigenfunction) {
                                 .eigenfunction = std::move(ef.second),
                                 .problem=s,
                         };
                         r.append(t);
                     }
                     return r;
                 }, py::arg("k") = 10, py::arg("sparse") = true, py::arg("shiftInvert") = true, py::arg("ncv") = -1,
                 py::arg("tolerance") = 1e-10, py::arg("maxIterations") = 1000, R""""(\
Calculate the first k eigenvalues with corresponding eigenfunctions.

:param int k: minimal number of eigenvalues (with eigenfunctions) to find (default: 10).

:returns: a list of pairs of eigenvalues and eigenfunctions with length at least k.

>>> import numpy as np
>>> from math import pi
>>> r = Rectangle(-9.5, 9.5, -9.5, 9.5)
>>> s = Schrodinger2D(lambda x, y: x * x + y * y, r, gridSize=(45,45), maxBasisSize=21)
>>> eigenvalues, eigenfunctions = zip(*s.eigenfunctions(9))
>>> np.array(eigenvalues[:9]).round(6)
array([2., 4., 4., 6., 6., 6., 8., 8., 8.])
)"""");
}
