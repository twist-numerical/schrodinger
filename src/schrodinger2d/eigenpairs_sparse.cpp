#include "eigenpairs.h"
#include <Eigen/Sparse>
#include <algorithm>

using namespace schrodinger;

struct DirectionGetter {
    bool getX;

    template<typename T>
    constexpr const T &operator()(const PerDirection<T> &p) {
        return getX ? p.x : p.y;
    }
};

DirectionGetter xDirection{true};
DirectionGetter yDirection{false};

template<typename Scalar>
class PermutedBeta {
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

    Eigen::PermutationMatrix<Eigen::Dynamic> permutation;
    std::vector<MatrixXs> leastSquares;
    std::vector<MatrixXs> blocks;
    std::vector<MatrixXs> fullBlocks;
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;

    PermutedBeta(const Schrodinger2D<Scalar> *schrodinger, DirectionGetter direction) : permutation{
            Eigen::Index(schrodinger->intersections.size())} {
        auto &threads = direction(schrodinger->threads);
        leastSquares.reserve(threads.size());
        fullBlocks.reserve(threads.size());
        blocks.reserve(threads.size());
        Eigen::Index pIndex = 0;
        for (auto &thread: threads) {
            for (auto intersection: thread.intersections)
                permutation.indices()(intersection->index) = pIndex++;

            Eigen::Index m = thread.intersections.size();
            Eigen::Index n = thread.eigenpairs.size();
            rows += m;
            cols += n;
            MatrixXs B(m, n);
            {
                Eigen::Index i = 0;
                for (auto intersection: thread.intersections)
                    B.row(i++) = direction(intersection->evaluation);
            }
            VectorXs lambda(n);
            {
                Eigen::Index i = 0;
                for (auto &ef: thread.eigenpairs)
                    lambda(i++) = std::get<0>(ef);
            }

            leastSquares.push_back(B.colPivHouseholderQr().solve(MatrixXs::Identity(m, m)));
            fullBlocks.push_back(B * lambda.asDiagonal() * leastSquares.back());
            blocks.push_back(B);
        }
    }

    SparseMatrix lstsqMatrix() {
        SparseMatrix r{cols, rows};
        Eigen::VectorXi toReserve{rows};
        {
            Eigen::Index i = 0;
            for (auto &lstsq: leastSquares) {
                toReserve.middleRows(i, lstsq.rows()) = Eigen::VectorXi::Constant(lstsq.rows(), lstsq.cols());
                i += lstsq.rows();
            }
        }
        r.reserve(toReserve);
        {
            Eigen::Index i = 0;
            Eigen::Index j = 0;
            for (auto &lstsq: leastSquares) {
                for (int i1 = 0; i1 < lstsq.rows(); ++i1)
                    for (int j1 = 0; j1 < lstsq.cols(); ++j1)
                        r.insert(i + i1, j + j1) = lstsq(i1, j1);
                i += lstsq.rows();
                j += lstsq.cols();
            }
        }
        return r * permutation;
    }

    SparseMatrix fullMatrix() {
        SparseMatrix r{rows, rows};
        Eigen::VectorXi toReserve{rows};
        {
            Eigen::Index i = 0;
            for (auto &block: fullBlocks) {
                toReserve.middleRows(i, block.rows()) = Eigen::VectorXi::Constant(block.rows(), block.cols());
                i += block.rows();
            }
        }
        r.reserve(toReserve);
        {
            Eigen::Index i = 0;
            Eigen::Index j = 0;
            for (auto &block: fullBlocks) {
                for (int i1 = 0; i1 < block.rows(); ++i1)
                    for (int j1 = 0; j1 < block.cols(); ++j1)
                        r.insert(i + i1, j + j1) = block(i1, j1);
                i += block.rows();
                j += block.cols();
            }
        }
        return permutation.transpose() * r * permutation;
    }

    SparseMatrix asMatrix() {
        SparseMatrix r{rows, cols};
        Eigen::VectorXi toReserve{rows};
        {
            Eigen::Index i = 0;
            for (auto &block: blocks) {
                toReserve.middleRows(i, block.rows()) = Eigen::VectorXi::Constant(block.rows(), block.cols());
                i += block.rows();
            }
        }
        r.reserve(toReserve);
        {
            Eigen::Index i = 0;
            Eigen::Index j = 0;
            for (auto &block: blocks) {
                for (int i1 = 0; i1 < block.rows(); ++i1)
                    for (int j1 = 0; j1 < block.cols(); ++j1)
                        r.insert(i + i1, j + j1) = block(i1, j1);
                i += block.rows();
                j += block.cols();
            }
        }
        return permutation.transpose() * r;
    }
};


#ifdef SCHRODINGER_SLEPC

#include "../util/slepc.h"

#else

#include <Spectra/GenEigsSolver.h>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>

#endif

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>>
sparseEigenpairs(const Schrodinger2D<Scalar> *schrodinger, int) {
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
    typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

    PermutedBeta<Scalar> Bx{schrodinger, xDirection};
    PermutedBeta<Scalar> By{schrodinger, yDirection};

    Eigen::Index n = schrodinger->intersections.size();
    std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>> result;

    SparseMatrix lstsq_x = Bx.lstsqMatrix();
    SparseMatrix lstsq_y = By.lstsqMatrix();

#ifdef SCHRODINGER_SLEPC
    SLEPcMatrix A{Bx.fullMatrix() + By.fullMatrix()};
    EPS eps;
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, A.slepcMatrix, nullptr);
    EPSSetProblemType(eps, EPS_NHEP);

    SparseMatrix diff = lstsq_x - lstsq_y;
    std::vector<SLEPcVector> deflationSpace;
    deflationSpace.reserve(diff.rows());
    for (Eigen::Index i = 0; i < diff.rows(); ++i) {
        deflationSpace.emplace_back(diff.row(i).transpose().toDense());
    }

    // EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE);
    EPSSetTarget(eps, 8);
    // EPSSetInterval(eps, 0.1, 10);
    // EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    EPSSetDimensions(eps, 100, 150, 50);
    EPSSetDeflationSpace(eps, (PetscInt) deflationSpace.size(), (Vec *) deflationSpace.data());
    EPSSolve(eps);

    PetscInt nconv;
    EPSConvergedReason reason;
    PetscViewer viewer;
    PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO_DETAIL);
    EPSGetConvergedReason(eps, &reason);
    if (reason != EPS_CONVERGED_USER) {
        EPSConvergedReasonView(eps, viewer);
        EPSErrorView(eps, EPS_ERROR_RELATIVE, viewer);
    } else {
        EPSGetConverged(eps, &nconv);
        PetscViewerASCIIPrintf(viewer, "Eigensolve finished with %" PetscInt_FMT " converged eigenpairs; reason=%s\n",
                               nconv, EPSConvergedReasons[reason]);
    }
    PetscViewerPopFormat(viewer);

#else
    Eigen::Index nev = n - 2;
    Spectra::SparseGenMatProd<Scalar, Eigen::RowMajor> op(Bx.fullMatrix() + By.fullMatrix());
    Spectra::GenEigsSolver<decltype(op)> eigenSolver(op, nev, std::min(2 * nev + 1, n));
    eigenSolver.init();
    Eigen::Index nconv = eigenSolver.compute();
    std::cout << "Converged: " << nconv << std::endl;

    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> values = eigenSolver.eigenvalues();
    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> vectors = eigenSolver.eigenvectors();
    MatrixXs vx = lstsq_x * vectors.real();
    MatrixXs vy = lstsq_y * vectors.real();
    VectorXs norms = (Bx.asMatrix() * vx - By.asMatrix() * vy).colwise().norm();
    VectorXs norm2s = (Bx.asMatrix() * vx + By.asMatrix() * vy).colwise().norm();

    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (std::abs(values(i).imag()) < 1e-4 && norms(i) < 1e-1 && norm2s(i) > 1e-2) {
            if constexpr (withEigenfunctions) {
                VectorXs v(Bx.cols + By.cols);
                v.topRows(Bx.cols) = vx.col(i);
                v.bottomRows(By.cols) = vy.col(i);
                result.emplace_back(
                        values(i).real(),
                        std::make_unique<typename Schrodinger2D<Scalar>::Eigenfunction>(
                                schrodinger, values(i).real(), v)
                );
            } else {
                result.emplace_back(values(i).real());
            }
        }
    }

#endif

    std::sort(result.begin(), result.end(), [](auto &a, auto &b) {
        if constexpr (withEigenfunctions)
            return a.first < b.first;
        else
            return a < b;
    });

    return result;
}

#define SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, withEigenfunctions) \
template \
std::vector<typename std::conditional_t<(withEigenfunctions), std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>> \
sparseEigenpairs<Scalar, withEigenfunctions>(const Schrodinger2D<Scalar> *, int);

#define SCHRODINGER_INSTANTIATE(Scalar) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, false) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, true)

#include "instantiate.h"
