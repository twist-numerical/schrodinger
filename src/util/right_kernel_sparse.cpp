#include "right_kernel.h"
#include "slepc.h"
#include <iostream>

inline PetscErrorCode kernelStoppingCondition(
        SVD svd, PetscInt its, PetscInt max_it, PetscInt nconv, PetscInt nsv,
        SVDConvergedReason *reason, void *) {
    SVDStoppingBasic(svd, its, max_it, nconv, nsv, reason, nullptr);
    std::cout << *reason << " " << nconv << std::endl;
    if (*reason == SVD_CONVERGED_TOL) {
        double r;
        for (int i = 0; i < nconv; ++i) {
            SVDGetSingularTriplet(svd, i, &r, nullptr, nullptr);
            std::cout << "  " << r;
        }
        std::cout << std::endl;
    }
    return 0;
};


template<>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
strands::internal::sparseRightKernel<double>(
        const Eigen::SparseMatrix<double, Eigen::RowMajor> &eigenA, double threshold) {
    SLEPcMatrix A{eigenA};
    ((void) threshold);

    MatView(A.slepcMatrix, nullptr);

    SVD svd;
    SVDCreate(PETSC_COMM_WORLD, &svd);
    SVDSetOperators(svd, A.slepcMatrix, nullptr);
    SVDSetProblemType(svd, SVD_STANDARD);
    SVDSetType(svd, SVDTRLANCZOS);
    // SVDSetTolerances(svd, 1e-8, 100);
    SVDSetWhichSingularTriplets(svd, SVD_SMALLEST);
    // SVDSetStoppingTestFunction(svd, &kernelStoppingCondition, &threshold, nullptr);
    SVDSetDimensions(svd, 20, PETSC_DEFAULT, PETSC_DEFAULT);
    SVDSetUp(svd);
    SVDView(svd, nullptr);
    SVDSolve(svd);

    int converged;
    SVDGetConverged(svd, &converged);
    std::cout << converged << " singular values found:" << std::endl;
    double r;
    for (int i = 0; i < converged; ++i) {
        SVDGetSingularTriplet(svd, i, &r, nullptr, nullptr);
        std::cout << "  " << r;
    }
    std::cout << std::endl;

    SVDDestroy(&svd);

    return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>{};
}
