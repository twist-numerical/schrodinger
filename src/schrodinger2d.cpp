#include "matslise/util/quadrature.h"
#include "util/polymorphic_value.h"
#include "schrodinger2d.h"
#include <numeric>
#include "util/right_kernel.h"
#include <chrono>

void tic(int mode = 0) {
    static std::chrono::system_clock::time_point t_start;

    if (mode == 0)
        t_start = std::chrono::high_resolution_clock::now();
    else {
        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "Elapsed time is " << (t_end - t_start).count() * 1E-9 << " seconds\n";
    }
}

void toc() { tic(1); }

using namespace std;
using namespace matslise;
using namespace Eigen;
using namespace isocpp_p0201;
using namespace schrodinger;
using namespace schrodinger::geometry;

template<typename T>
struct IterateEigen {
private:
    T *t;

public:
    IterateEigen(T &_t) : t(&_t) {
    };

    auto begin() {
        return t->data();
    };

    auto end() {
        return t->data() + t->size();
    };
};

Schrodinger2D::Thread
computeThread(const std::function<double(double)> &V, double min, double max, size_t maxPairs,
              const Ref<const ArrayXd> &grid, size_t &offset) {
    const double *from = std::lower_bound(IterateEigen(grid).begin(), IterateEigen(grid).end(), min);
    if (*from == min)
        ++from;

    Index length = std::distance(from, std::lower_bound(from, IterateEigen(grid).end(), max));
    Schrodinger2D::Thread thread = {
            .offset = offset,
            .gridOffset = std::distance(IterateEigen(grid).begin(), from),
            .gridLength = length,
    };
    if (length == 0) return thread;
    thread.matslise = make_shared<Matslise<double>>(V, min, max);
    size_t pairs = std::min(maxPairs, (size_t) length);

    thread.eigenpairs.reserve(pairs);
    for (auto &iEf : thread.matslise->eigenpairsByIndex(0, pairs, Y<>::Dirichlet())) {
        auto &f = get<2>(iEf);
        thread.eigenpairs.emplace_back(get<1>(iEf), f);
    }
    offset += pairs;

    return thread;
}

Schrodinger2D::Schrodinger2D(const function<double(double, double)> &_V,
                             const Domain<double, 2> &_domain,
                             const Options &_options)
        : V(_V), domain(polymorphic_value<Domain < double, 2>>

(_domain.

clone(), Domain<double, 2>::copy()

)),
options(_options) {
        tic();


        grid.x = ArrayXd::LinSpaced(options.gridSize.x + 2, domain->min(0), domain->max(0))
        .segment(1, options.gridSize.x);
        grid.y = ArrayXd::LinSpaced(options.gridSize.y + 2, domain->min(1), domain->max(1))
        .segment(1, options.gridSize.y);

        {
            columns.x = 0;
            Index i = 0;
// #pragma omp parallel for ordered schedule(dynamic, 1) collapse(2)
            for (double &x : IterateEigen(grid.x)) {
                for (auto &dom : domain->intersections({{x, 0}, Vector<double, 2>::Unit(1)})) {
                    Thread thread = computeThread([this, x](double y) -> double { return V(x, y) / 2; },
                                                  dom.first, dom.second,
                                                  (size_t) options.maxBasisSize,
                                                  grid.y, columns.x);
// #pragma ordered
                    thread.value = x;
                    thread.valueIndex = i;
                    if (thread.gridLength > 0)
                        threads.x.push_back(thread);
                }
                i++;
            }
        }

        size_t intersectionCount = 0;
        for (const Thread &t : threads.x)
        intersectionCount += t.gridLength;

        intersections.reserve(intersectionCount);

        for (const Thread &t : threads.x) {
            MatrixXd onGrid(t.gridLength, t.eigenpairs.size());
            ArrayXd subGrid = grid.y.segment(t.gridOffset, t.gridLength);
            Index j = 0;
            for (auto &Ef : t.eigenpairs)
                onGrid.col(j++) = get<1>(Ef)(subGrid).unaryExpr([](const Y<> y) -> double { return y.y()[0]; });
            for (Index i = 0; i < t.gridLength; ++i) {
                intersections.emplace_back(Intersection{
                        .position={.x=t.value, .y= grid.y[t.gridOffset + i]},
                        .thread={.x = &t, .y = nullptr},
                        .evaluation={.x = onGrid.row(i)},
                });
            }
        }

        {
            columns.y = 0;
            Index i = 0;
// #pragma omp parallel for schedule(dynamic, 1) collapse(2)
            for (double &y : IterateEigen(grid.y)) {
                for (auto &dom : domain->intersections({{0, y}, Vector<double, 2>::Unit(0)})) {
                    Thread thread = computeThread([this, y](double x) -> double { return V(x, y) / 2; },
                                                  dom.first, dom.second,
                                                  (size_t) options.maxBasisSize,
                                                  grid.x, columns.y);
                    thread.value = y;
                    thread.valueIndex = i;
// #pragma ordered
                    threads.y.push_back(thread);
                }
                i++;
            }
        }

        {
            std::vector<Intersection *> intersectionsByY;
            intersectionsByY.reserve(intersections.size());
            for (Intersection &i : intersections)
                intersectionsByY.push_back(&i);
            std::sort(intersectionsByY.begin(), intersectionsByY.end(),
                      [](const Intersection *a, const Intersection *b) {
                          if (a->position.y == b->position.y)
                              return a->position.x < b->position.x;
                          return a->position.y < b->position.y;
                      });
            auto intersection = intersectionsByY.begin();
            for (Thread &t : threads.y) {
                MatrixXd onGrid(t.gridLength, t.eigenpairs.size());
                ArrayXd subGrid = grid.x.segment(t.gridOffset, t.gridLength);
                Index j = 0;
                for (auto &Ef : t.eigenpairs)
                    onGrid.col(j++) = get<1>(Ef)(subGrid).unaryExpr([](const Y<> y) -> double { return y.y()[0]; });
                for (Index i = 0; i < t.gridLength; ++i) {
                    assert((**intersection).position.x == grid.x[t.gridOffset + i] &&
                           (**intersection).position.y == t.value);
                    (**intersection).thread.y = &t;
                    (**intersection).evaluation.y = onGrid.row(i);
                    ++intersection;
                }
            }
        }

        {
            tiles.resize(options.gridSize.x + 1, options.gridSize.y + 1);
            for (auto &intersection : intersections) {
                Index i = intersection.thread.x->valueIndex;
                Index j = intersection.thread.y->valueIndex;
                tiles(i + 1, j + 1).intersections[0] = &intersection;
                tiles(i, j + 1).intersections[1] = &intersection;
                tiles(i + 1, j).intersections[2] = &intersection;
                tiles(i, j).intersections[3] = &intersection;
            }
        }

        toc();
        cout << "matslise done\n" << endl;
}

template<bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<double, Schrodinger2D::Eigenfunction>, double>>
eigenpairs(const Schrodinger2D *self) {
    size_t rows = self->intersections.size();

    size_t colsX = self->columns.x;
    size_t colsY = self->columns.y;


    MatrixXd beta_x = MatrixXd::Zero(rows, colsX);
    MatrixXd beta_y = MatrixXd::Zero(rows, colsY);

    tic();
    size_t row = 0;
    for (const Schrodinger2D::Intersection &intersection : self->intersections) {
        beta_x.row(row).segment(
                intersection.thread.x->offset, intersection.thread.x->eigenpairs.size()
        ) = intersection.evaluation.x;
        beta_y.row(row).segment(
                intersection.thread.y->offset, intersection.thread.y->eigenpairs.size()
        ) = intersection.evaluation.y;

        ++row;
    }
    assert(row == rows);

    VectorXd lambda_x(colsX);
    VectorXd lambda_y(colsY);

    for (const auto &x : self->threads.x) {
        Index offset = x.offset;
        for (auto &ef : x.eigenpairs)
            lambda_x(offset++) = get<0>(ef);
    }
    for (const auto &y : self->threads.y) {
        Index offset = y.offset;
        for (auto &ef : y.eigenpairs)
            lambda_y(offset++) = get<0>(ef);
    }
    toc();
    cout << "beta and lambda constructed\n" << endl;

    tic();
    MatrixXd crossingsMatch(rows, colsX + colsY);
    crossingsMatch << beta_x, -beta_y;
    MatrixXd kernel = schrodinger::internal::rightKernel<MatrixXd>(crossingsMatch, 1e-3);
    Index kernelSize = kernel.cols();
    toc();
    cout << "Kernel found of " << rows << "x" << (colsX + colsY) << " matrix. Kernel size: " << kernelSize << "\n"
         << endl;

    tic();
    MatrixXd A(rows, colsX + colsY); // rows x (colsX + colsY)
    A << beta_x * lambda_x.asDiagonal(), beta_y * lambda_y.asDiagonal();

    MatrixXd BK = colsY < colsX // rows x kernelSize
                  ? beta_y * kernel.bottomRows(colsY)
                  : beta_x * kernel.topRows(colsX);

    MatrixXd T = (BK.transpose() * BK).ldlt().solve(BK.transpose() * A * kernel);
    toc();
    cout << "A, BK and T constructed\n" << endl;


    if constexpr(withEigenfunctions) {
        tic();
        EigenSolver<MatrixXd> solver(T, true);
        const VectorXcd &values = solver.eigenvalues();
        MatrixXcd vectors = solver.eigenvectors();

        typedef std::pair<double, Schrodinger2D::Eigenfunction> Eigenpair;
        std::vector<Eigenpair> eigenfunctions;
        eigenfunctions.reserve(values.size());
        for (Index i = 0; i < values.size(); ++i)
            eigenfunctions.emplace_back(
                    values[i].real(),
                    Schrodinger2D::Eigenfunction(self, values[i].real(), kernel * vectors.col(i).real())
            );
        toc();
        cout << "Eigenfunctions: " << eigenfunctions.size() << "\n" << endl;
        return eigenfunctions;

    } else {
        tic();
        ArrayXd values = T.eigenvalues().array().real();
        std::sort(values.data(), values.data() + values.size());
        std::vector<double> eigenvalues(values.size());
        std::copy(values.data(), values.data() + values.size(), eigenvalues.begin());
        toc();
        cout << "Eigenvalues: " << values.transpose() << "\n" << endl;
        return eigenvalues;
    }
}

std::vector<double> Schrodinger2D::eigenvalues() const {
    return eigenpairs<false>(this);
}

std::vector<std::pair<double, Schrodinger2D::Eigenfunction>> Schrodinger2D::eigenfunctions() const {
    return eigenpairs<true>(this);
}

Index highestLowerIndex(const ArrayXd &range, double value) {
    const double *start = range.data();
    Index d = std::distance(start, std::lower_bound(start, start + range.size(), value));
    if (d >= range.size() || value < range[d])
        --d;
    return d;
}

Array2d reconstructEigenfunction(const Schrodinger2D::Thread *t, const Ref<const ArrayXd> &c, const Array2d &v) {
    ArrayXd x = v;
    Array2d r = Array2d::Zero();
    for (size_t i = 0; i < t->eigenpairs.size(); ++i)
        r += c(t->offset + i) * t->eigenpairs[i].second(x).unaryExpr([](const Y<> &y) { return y.y()[0]; });
    return r;
}

double Schrodinger2D::Eigenfunction::operator()(double x, double y) const {
    problem->domain->contains({x, y});

    Index ix = highestLowerIndex(problem->grid.x, x);
    Index iy = highestLowerIndex(problem->grid.y, y);

    const Tile &tile = problem->tiles(ix + 1, iy + 1);
    if (tile.intersections[0] == nullptr
        || tile.intersections[1] == nullptr
        || tile.intersections[2] == nullptr
        || tile.intersections[3] == nullptr)
        return 0;

    double xOffset = problem->grid.x[ix];
    double yOffset = problem->grid.y[iy];
    double x1 = x - xOffset;
    double y1 = y - yOffset;
    double hx = problem->grid.x[ix + 1] - xOffset;
    double hy = problem->grid.y[iy + 1] - yOffset;
    Vector4d wx, wy;
    double nx = 2 / (hx * x1 - x1 * x1);
    wx << (2 * hx - 3 * x1) * nx / hx, -2 * nx, nx, -(hx - 3 * x1) * nx / hx;
    double ny = 2 / (hy * y1 - y1 * y1);
    wy << (2 * hy - 3 * y1) * ny / hy, -2 * ny, ny, -(hy - 3 * y1) * ny / hy;

    Array2d xs, ys;
    xs << xOffset + x1, xOffset + hx - x1;
    ys << yOffset + y1, yOffset + hy - y1;
    Array2d fy0 = reconstructEigenfunction(tile.intersections[0]->thread.y, c.bottomRows(problem->columns.y), xs);
    Array2d fx0 = reconstructEigenfunction(tile.intersections[0]->thread.x, c.topRows(problem->columns.x), ys);
    Array2d fy3 = reconstructEigenfunction(tile.intersections[3]->thread.y, c.bottomRows(problem->columns.y), xs);
    Array2d fx3 = reconstructEigenfunction(tile.intersections[3]->thread.x, c.topRows(problem->columns.x), ys);

    {
        Matrix4d w = Matrix4d::Zero();
        Vector4d b;
        for (int i = 1; i <= 2; ++i)
            for (int j = 1; j <= 2; ++j) {
                b(2 * i + j - 3) =
                        wx(3 * i - 3) * fx0(j - 1) + wx(6 - 3 * i) * fx3(j - 1) + wy(3 * j - 3) * fy0(i - 1) +
                        wy(6 - 3 * j) * fy3(i - 1);

                w(2 * i + j - 3, 2 * i + j - 3) = problem->V(xs(i - 1), ys(j - 1)) - E - wx(1) - wy(1);
                w(2 * i + j - 3, 3 - 2 * i + j) = -wx(2);
                w(2 * i + j - 3, 2 * i - j) = -wy(2);
            }

        return (w.inverse() * b)(0);
    }
}