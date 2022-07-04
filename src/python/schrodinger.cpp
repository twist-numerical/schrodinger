#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <utility>

#include "../schrodinger2d.h"

namespace py = pybind11;
using namespace schrodinger;
using namespace schrodinger::geometry;

PYBIND11_MODULE(schrodinger, m) {

    struct KeepAliveEigenfunction {
        Schrodinger2D<double>::Eigenfunction eigenfunction;
        std::shared_ptr<const Schrodinger2D<double>> problem;
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
                return f.eigenfunction(x, y);
            })
            .def("__call__", [](const KeepAliveEigenfunction &f, Schrodinger2D<double>::ArrayXs xs, Schrodinger2D<double>::ArrayXs ys) {
                return f.eigenfunction(xs, ys);
            });

    py::class_<Schrodinger2D<double>, std::shared_ptr<Schrodinger2D<double>>>(m, "Schrodinger2D")
            .def(py::init(
                    [](const std::function<double(double, double)> &V, const Domain<double, 2> &domain,
                       const std::array<int, 2> &gridSize, int maxBasisSize) {
                        return std::make_shared<Schrodinger2D<double>>(V, domain, (Options) {
                                .gridSize = {.x = gridSize[0], .y=gridSize[1]},
                                .maxBasisSize=maxBasisSize,
                        });
                    }), py::arg("V"), py::arg("domain"), py::arg("gridSize") = std::array<int, 2>{21, 21},
                 py::arg("maxBasisSize") = 16)
            .def("eigenvalues", &Schrodinger2D<double>::eigenvalues)
            .def("eigenfunctions", [](std::shared_ptr<const Schrodinger2D<double>> s) {
                std::vector<std::pair<double, Schrodinger2D<double>::Eigenfunction>> eigs = s->eigenfunctions();
                py::list r;
                for (const auto &ef : eigs) {
                    py::tuple t(2);
                    t[0] = ef.first;
                    t[1] = (KeepAliveEigenfunction) {
                            .eigenfunction = ef.second,
                            .problem=s,
                    };
                    r.append(t);
                }
                std::cout << "Python list: " << r.size() << std::endl;
                return r;
            });
}