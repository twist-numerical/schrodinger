#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <utility>

#include "../schrodinger.h"

namespace py = pybind11;
using namespace schrodinger;
using namespace schrodinger::geometry;


template<typename T>
void perDirection(const py::handle &scope, const std::string &name) {
    py::class_<PerDirection<T>>(scope, name.c_str())
            .def_readonly("x", &PerDirection<T>::x)
            .def_readonly("y", &PerDirection<T>::y);
}

PYBIND11_MODULE(schrodinger, m) {

    struct KeepAliveEigenfunction {
        std::unique_ptr<Schrodinger<double>::Eigenfunction> eigenfunction;
        std::shared_ptr<const Schrodinger<double>> problem;
    };

    py::class_<Domain<double, 2>>
            (m, "Domain2D")
            .def("contains", &Domain<double, 2>::contains)
            .def("intersections",
                 [](const Domain<double, 2> *self, const Eigen::Vector2d &origin, const Eigen::Vector2d &direction) {
                     return self->intersections({origin, direction});
                 })
            .def("bounds", &Domain<double, 2>::bounds);

    py::class_<Rectangle<double, 2>, Domain<double, 2 >>(m, "Rectangle")
            .def(py::init<double, double, double, double>());

    py::class_<Sphere<double, 2>, Domain<double, 2 >>(m, "Circle")
            .def(py::init<double>())
            .def(py::init<Vector<double, 2>>())
            .def(py::init<Vector<double, 2>, double>());

    py::class_<Union<double, 2>, Domain<double, 2 >>(m, "Union")
            .def(py::init<const std::vector<const Domain<double, 2> *> &>());

    py::class_<DomainTransform<double, 2>, Domain<double, 2 >>(m, "DomainTransform")
            .def(py::init([](const Domain<double, 2> &domain, const Eigen::Matrix3d &transform) {
                Eigen::Affine2d t;
                t.matrix() = transform;
                return new DomainTransform<double, 2>(domain, t);
            }), py::arg("domain"), py::arg("transformation"));

    py::class_<KeepAliveEigenfunction>(m, "Eigenfunction2D")
            .def("__call__", [](const KeepAliveEigenfunction &f, double x, double y) {
                return (*f.eigenfunction)(x, y);
            })
            .def("__call__", [](const KeepAliveEigenfunction &f, Schrodinger<double>::ArrayXs xs,
                                Schrodinger<double>::ArrayXs ys) {
                return (*f.eigenfunction)(xs, ys);
            });

    perDirection<Schrodinger<double>::MatrixXs>(m, "PerDirectionMatrixXs");
    perDirection<Schrodinger<double>::VectorXs>(m, "PerDirectionVectorXs");

    py::class_<Schrodinger<double>, std::shared_ptr<Schrodinger<double>>>(m, "Schrodinger2D")
            .def(py::init(
                         [](const std::function<double(double, double)> &V, const Domain<double, 2> &domain,
                            const std::array<int, 2> &gridSize, int maxBasisSize, bool sparse, bool shiftInvert) {
                             py::gil_scoped_release release;
                             return std::make_shared<Schrodinger<double>>(V, domain, (Options) {
                                     .gridSize = {.x = gridSize[0], .y=gridSize[1]},
                                     .maxBasisSize=maxBasisSize,
                                     .sparse=sparse,
                                     .shiftInvert=shiftInvert
                             });
                         }), py::arg("V"), py::arg("domain"), py::arg("gridSize") = std::array<int, 2>{21, 21},
                 py::arg("maxBasisSize") = 16, py::arg("sparse") = true, py::arg("shift_invert") = true)
            .def("eigenvalues", [](std::shared_ptr<const Schrodinger<double>> s, int eigenvalueCount) {
                py::gil_scoped_release release;
                return s->eigenvalues(eigenvalueCount);
            }, py::arg("k") = -1, R""""(\
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
            .def("eigenfunctions", [](std::shared_ptr<const Schrodinger<double>> s, int eigenvalueCount) {
                std::vector<std::pair<double, std::unique_ptr<Schrodinger<double>::Eigenfunction>>> eigs;
                {
                    py::gil_scoped_release release;
                    eigs = s->eigenfunctions(eigenvalueCount);
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
            }, py::arg("k") = -1, R""""(\
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
