#ifndef SCHRODINGER2D_SLEPC_H
#define SCHRODINGER2D_SLEPC_H

#include <slepcsvd.h>
#include <Eigen/Sparse>

struct SLEPcMatrix {
    static inline bool init = false;
    Mat slepcMatrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, PetscInt> eigenMatrix;


    explicit SLEPcMatrix(const Eigen::SparseMatrix<double, Eigen::RowMajor> &eigenMatrix_)
            : slepcMatrix{}, eigenMatrix(eigenMatrix_) {
        if (!init)
            SlepcInitialize(nullptr, nullptr, nullptr, nullptr);

        std::vector<PetscInt> nnz;

        eigenMatrix.makeCompressed();
        MatCreateSeqAIJWithArrays(
                PETSC_COMM_WORLD,
                (PetscInt) eigenMatrix.rows(), (PetscInt) eigenMatrix.cols(),
                eigenMatrix.outerIndexPtr(),
                eigenMatrix.innerIndexPtr(),
                eigenMatrix.valuePtr(),
                &slepcMatrix);
    }


    SLEPcMatrix(const SLEPcMatrix &) = delete;

    SLEPcMatrix(SLEPcMatrix &&) = delete;

    ~SLEPcMatrix() {
        MatDestroy(&slepcMatrix);
    }
};

#endif //SCHRODINGER2D_SLEPC_H
