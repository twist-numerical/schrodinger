#include "eigenpairs.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <numeric>
#include "../util/right_kernel.h"

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

    SparseMatrix lstsqNullSpace() {
        std::vector<MatrixXs> nullBlocks;

        Eigen::VectorXi toReserve{rows};
        Eigen::Index i = 0;
        Eigen::Index nullCols = 0;
        for (auto &block: leastSquares) {
            nullBlocks.emplace_back(schrodinger::internal::rightKernel(block, 1e-8));
            nullCols += nullBlocks.back().cols();
            toReserve.middleRows(i, block.cols()) = Eigen::VectorXi::Constant(block.cols(), nullBlocks.back().cols());
            i += block.cols();
        }

        SparseMatrix r{rows, nullCols};
        r.reserve(toReserve);

        {
            Eigen::Index i = 0;
            Eigen::Index j = 0;
            for (auto &block: nullBlocks) {
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

// https://spectralib.org/doc/classspectra_1_1sparsegenrealshiftsolve
template<typename Scalar_>
class SparseDeflatedRealShiftSolve {
public:
    using Scalar = Scalar_;

private:
    using Index = Eigen::Index;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;

    const Eigen::Ref<const SparseMatrix> m_mat;
    const Index m_n;
    std::vector<SparseMatrix> deflations;

    Eigen::SparseLU<SparseMatrix> m_solver;
    // Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> m_solver;
    // Eigen::BiCGSTAB<SparseMatrix> m_solver;

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param mat An **Eigen** sparse matrix object, whose type can be
    /// `Eigen::SparseMatrix<Scalar, ...>` or its mapped version
    /// `Eigen::Map<Eigen::SparseMatrix<Scalar, ...> >`.
    ///
    template<typename Derived>
    explicit SparseDeflatedRealShiftSolve(const Eigen::SparseMatrixBase<Derived> &mat) :
            m_mat(mat), m_n(mat.rows()) {
        eigen_assert(mat.rows() == mat.cols());
    }

    ///
    /// Should be an orthonormal matrix with as columns a basis for a deflation space.
    /// Deflation spaces may intersect.
    ///
    void add_deflation(const SparseMatrix &basis) {
        eigen_assert(m_n == basis.rows());
        deflations.push_back(basis);
    }

    ///
    /// Return the number of rows of the underlying matrix.
    ///
    Index rows() const { return m_n; }

    ///
    /// Return the number of columns of the underlying matrix.
    ///
    Index cols() const { return m_n; }

    ///
    /// Set the real shift \f$\sigma\f$.
    ///
    void set_shift(const Scalar &sigma) {
        MATSLISE_SCOPED_TIMER("SPECTRA set_shift");
        SparseMatrix I(m_n, m_n);
        I.setIdentity();

        m_solver.compute(m_mat - sigma * I);
        if (m_solver.info() != Eigen::Success)
            throw std::invalid_argument("SparseGenRealShiftSolve: factorization failed with the given shift");
    }

    ///
    /// Perform the shift-solve operation \f$y=(A-\sigma I)^{-1}x\f$.
    ///
    /// \param x_in  Pointer to the \f$x\f$ vector.
    /// \param y_out Pointer to the \f$y\f$ vector.
    ///
    // y_out = inv(A - sigma * I) * x_in
    void perform_op(const Scalar *x_in, Scalar *y_out) const {
        MATSLISE_SCOPED_TIMER("SPECTRA perform_op");
        // https://web.stanford.edu/~lmackey/papers/deflation-nips08.pdf
        Vector x1, x2;
        x1 = MapConstVec{x_in, m_n};
        for (auto &U: deflations) {
            MATSLISE_SCOPED_TIMER("SPECTRA deflate forward");
            x2.noalias() = U.transpose() * x1;
            x1.noalias() -= U * x2;
        }
        MapVec y(y_out, m_n);
        y.noalias() = m_solver.solve(x1);
        for (auto it = deflations.rbegin(); it != deflations.rend(); ++it) {
            MATSLISE_SCOPED_TIMER("SPECTRA deflate backward");
            x2.noalias() = (*it).transpose() * y;
            y.noalias() -= (*it) * x2;
        }
    }
};

#endif

template<typename Scalar, bool withEigenfunctions>
std::vector<typename std::conditional_t<withEigenfunctions, std::pair<Scalar, std::unique_ptr<typename Schrodinger2D<Scalar>::Eigenfunction>>, Scalar>>
sparseEigenpairs(const Schrodinger2D<Scalar> *schrodinger, Eigen::Index nev) {
    MATSLISE_SCOPED_TIMER("Sparse eigenpairs");

    if (nev < 0)
        nev = 10;
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

    SparseMatrix yNullSpace = Bx.lstsqNullSpace();
    SparseMatrix xNullSpace = By.lstsqNullSpace();
    SparseMatrix diff{xNullSpace.rows(), 0}; // (lstsq_x - lstsq_y).transpose();
    // std::cout << diff.rows() << ", " << xNullSpace.rows() << ", " << yNullSpace.rows() << std::endl;

    assert(xNullSpace.rows() == yNullSpace.rows());
    SparseMatrix nullSpace{xNullSpace.rows(), diff.cols() + xNullSpace.cols() + yNullSpace.cols()};
    nullSpace.reserve(xNullSpace.nonZeros() + yNullSpace.nonZeros());
    for (Eigen::Index r = 0; r < nullSpace.rows(); ++r) {
        nullSpace.startVec(r);
        Eigen::Index col = 0;
        for (typename SparseMatrix::InnerIterator it_x(xNullSpace, r); it_x; ++it_x)
            nullSpace.insertBack(r, col + it_x.col()) = it_x.value();
        col += xNullSpace.cols();
        for (typename SparseMatrix::InnerIterator it_y(yNullSpace, r); it_y; ++it_y)
            nullSpace.insertBack(r, col + it_y.col()) = it_y.value();
        col += yNullSpace.cols();
        for (typename SparseMatrix::InnerIterator it_diff(diff, r); it_diff; ++it_diff)
            nullSpace.insertBack(r, col + it_diff.col()) = it_diff.value();
    }
    nullSpace.finalize();

    Eigen::SparseQR<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>> nullSpaceQR;
    {
        MATSLISE_SCOPED_TIMER("QR-decomposition");
        nullSpaceQR.setPivotThreshold(.1);
        nullSpaceQR.compute(nullSpace);

        std::cout << nullSpace.rows() << " x " << nullSpace.cols() << " -> " << nullSpaceQR.rank() << std::endl;
    }
    MatrixXs nullSpaceQ;

    {
        MATSLISE_SCOPED_TIMER("Q-Matrix");
        nullSpaceQ = nullSpaceQR.matrixQ() *
                     MatrixXs::Identity(nullSpace.rows(), nullSpace.rows()).leftCols(nullSpaceQR.rank());
        std::cout << "Q computed" << std::endl;
    }

    std::vector<SLEPcVector> deflationSpace;

    {
        MATSLISE_SCOPED_TIMER("Build deflation space");
        deflationSpace.reserve(diff.rows() + xNullSpace.cols() + yNullSpace.cols());
        for (Eigen::Index i = 0; i < nullSpaceQ.cols(); ++i) {
            deflationSpace.emplace_back(nullSpaceQ.col(i));
        }
    }

    // EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE);
    // EPSSetTarget(eps, 8);
    // EPSSetInterval(eps, 0.1, 10);
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    EPSSetDimensions(eps, nev, 3 * nev / 2, nev / 2);
    EPSSetDeflationSpace(eps, (PetscInt) deflationSpace.size(), (Vec *) deflationSpace.data());

    {
        MATSLISE_SCOPED_TIMER("EPSSolve");
        EPSSolve(eps);
    }

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


    {
        MATSLISE_SCOPED_TIMER(withEigenfunctions ? "Collect eigenfunctions" : "Collect eigenvalues");
        EPSGetConverged(eps, &nconv);
        result.reserve(nconv);
        for (int i = 0; i < nconv; ++i) {
            Scalar value;
            SLEPcVector vec;
            EPSGetEigenpair(eps, i, &value, nullptr, vec.slepcVector, nullptr);


            if constexpr (withEigenfunctions) {
                VectorXs v(Bx.cols + By.cols);
                v.topRows(Bx.cols) = lstsq_x * vec.toEigen();
                v.bottomRows(By.cols) = lstsq_y * vec.toEigen();
                result.emplace_back(
                        value,
                        std::make_unique<typename Schrodinger2D<Scalar>::Eigenfunction>(schrodinger, value, v)
                );
            } else {
                result.emplace_back(value);
            }
        }
    }
#else
    SparseDeflatedRealShiftSolve<Scalar> op(Bx.fullMatrix() + By.fullMatrix());
    op.add_deflation(Bx.lstsqNullSpace());
    op.add_deflation(By.lstsqNullSpace());
    // To do: make sure sigma is lower bound!
    Scalar sigma = -1;
    Spectra::GenEigsRealShiftSolver<decltype(op)> eigenSolver(
            op, nev, std::min(2 * nev + 1, n), sigma);
    {
        MATSLISE_SCOPED_TIMER("SPECTRA init");
        eigenSolver.init();
    }

    {
        MATSLISE_SCOPED_TIMER("SPECTRA compute");

        eigenSolver.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10,
                            Spectra::SortRule::SmallestReal);
    }
    if (eigenSolver.info() != Spectra::CompInfo::Successful)
        throw std::runtime_error("The eigensolver did not find success...");

    {
        MATSLISE_SCOPED_TIMER(withEigenfunctions ? "Collect eigenfunctions" : "Collect eigenvalues");
        Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> values = eigenSolver.eigenvalues();
        Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> vectors = eigenSolver.eigenvectors();
        MatrixXs vx = lstsq_x * vectors.real();
        MatrixXs vy = lstsq_y * vectors.real();

        /*
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
         */

        for (Eigen::Index i = 0; i < values.size(); ++i) {
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
sparseEigenpairs<Scalar, withEigenfunctions>(const Schrodinger2D<Scalar> *, Eigen::Index);

#define SCHRODINGER_INSTANTIATE(Scalar) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, false) \
SCHRODINGER_INSTANTIATE_EIGENPAIRS(Scalar, true)

#include "instantiate.h"
