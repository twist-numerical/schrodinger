#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "../schrodinger2d.h"

namespace py = pybind11;

PYBIND11_MODULE(schrodinger, m) {

    struct KeepAliveEigenfunction {
        Schrodinger2D::Eigenfunction eigenfunction;
        std::shared_ptr<const Schrodinger2D> problem;
    };

    py::class_<Domain<double, 2>>(m, "Domain2D")
            .def("contains", &Domain<double, 2>::contains)
            .def("intersections", &Domain<double, 2>::intersections)
            .def("min", &Domain<double, 2>::min)
            .def("max", &Domain<double, 2>::max);

    py::class_<Rectangle<double, 2>, Domain<double, 2>>(m, "Rectangle")
            .def(py::init<double, double, double, double>());

    py::class_<Circle<double>, Domain<double, 2>>(m, "Circle")
            .def(py::init<std::array<double, 2>, double>());

    py::class_<Union<double, 2>, Domain<double, 2>>(m, "Union")
            .def(py::init<const std::vector<const Domain<double, 2> *> &>());

    py::class_<KeepAliveEigenfunction>(m, "Eigenfunction2D")
            .def("__call__", [](const KeepAliveEigenfunction &f, double x, double y) {
                return f.eigenfunction(x, y);
            });

    py::class_<Schrodinger2D, std::shared_ptr<Schrodinger2D>>(m, "Schrodinger2D")
            .def(py::init(
                    [](const std::function<double(double, double)> &V, const Domain<double, 2> &domain,
                       const std::array<int, 2> &gridSize, int maxBasisSize) {
                        return new Schrodinger2D(V, domain, (Options) {
                                .gridSize = {.x = gridSize[0], .y=gridSize[1]},
                                .maxBasisSize=maxBasisSize
                        });
                    }), py::arg("V"), py::arg("domain"), py::arg("gridSize") = std::array<int, 2>{21, 21},
                 py::arg("maxBasisSize") = 16)
            .def("eigenvalues", &Schrodinger2D::eigenvalues)
            .def("eigenfunctions", [](std::shared_ptr<const Schrodinger2D> s) {
                std::vector<std::pair<double, Schrodinger2D::Eigenfunction>> eigs = s->eigenfunctions();
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