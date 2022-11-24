#ifndef SCHRODINGER2D_SLEPC_H
#define SCHRODINGER2D_SLEPC_H

#include <slepcsvd.h>
#include <Eigen/Sparse>

inline void schrodingerInitSLEPc() {
    static bool init = false;
    if (!init)
        SlepcInitialize(nullptr, nullptr, nullptr, nullptr);
}

struct SLEPcMatrix {
    Mat slepcMatrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, PetscInt> eigenMatrix;


    explicit SLEPcMatrix(const Eigen::SparseMatrix<double, Eigen::RowMajor> &eigenMatrix_)
            : slepcMatrix{}, eigenMatrix(eigenMatrix_) {
        schrodingerInitSLEPc();

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

    SLEPcMatrix(SLEPcMatrix &&) = default;

    ~SLEPcMatrix() {
        MatDestroy(&slepcMatrix);
    }
};

struct SLEPcVector {
    Vec slepcVector;

    explicit SLEPcVector() : slepcVector{} {
        schrodingerInitSLEPc();
    }

    explicit SLEPcVector(const Eigen::Matrix<double, Eigen::Dynamic, 1> &v)
            : slepcVector{} {
        schrodingerInitSLEPc();

        VecCreateSeq(PETSC_COMM_WORLD, (PetscInt) v.rows(), &slepcVector);

        double *data;
        VecGetArray(slepcVector, &data);
        std::copy(v.data(), v.data() + v.rows(), data);
    }

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> toEigen() {
        PetscInt size;
        double *data;
        VecGetArray(slepcVector, &data);
        VecGetSize(slepcVector, &size);
        return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>{data, size};
    }

    SLEPcVector(const SLEPcVector &) = delete;

    SLEPcVector(SLEPcVector &&) = default;

    ~SLEPcVector() {
        VecDestroy(&slepcVector);
    }
};

#endif //SCHRODINGER2D_SLEPC_H
