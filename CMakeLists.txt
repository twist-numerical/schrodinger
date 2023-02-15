cmake_minimum_required(VERSION 3.13)
project(strands LANGUAGES CXX)
cmake_policy(SET CMP0077 NEW)

set(CMAKE_CXX_STANDARD 17)


Include(FetchContent)
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

if (MSVC)
    add_compile_options(/W4 /WX)
else ()
    add_compile_options(-Wall -Wextra -Wno-missing-field-initializers)
endif ()

SET(STRANDS_SRC src/schrodinger/eigenfunction.cpp src/schrodinger/schrodinger.cpp src/schrodinger/eigenpairs_dense.cpp src/schrodinger/eigenpairs_sparse.cpp)

add_library(strands ${STRANDS_SRC})

set(defines "")

OPTION(STRANDS_PROFILING "Compile with profiling information." OFF)
if (STRANDS_PROFILING)
    SET(MATSLISE_PROFILING ON)
ENDIF (STRANDS_PROFILING)
SET(MATSLISE_PROFILE ON)
SET(MATSLISE_STATIC ON)
SET(MATSLISE_MATSCS OFF)
SET(MATSLISE_2D OFF)
SET(MATSLISE_3D OFF)
SET(MATSLISE_PYTHON OFF)

OPTION(STRANDS_LONG_DOUBLE "Compile with support for long double" OFF)
if (STRANDS_LONG_DOUBLE)
    list(APPEND defines STRANDS_LONG_DOUBLE)
    message("-- with support for long double")
    set(MATSLISE_LONG_DOUBLE ON)
else ()
    set(MATSLISE_LONG_DOUBLE OFF)
    message("-- without support for long double")
endif ()

add_subdirectory(lib/matslise)
include_directories(lib/matslise)
set_property(TARGET matslise PROPERTY POSITION_INDEPENDENT_CODE ON)

message("Strands")

set(libraries matslise)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#set(ENV{PKG_CONFIG_PATH} "$ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{SLEPC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
#set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
#set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}:$ENV{PETSC_DIR}:$ENV{PKG_CONFIG_PATH}")
#find_package(PkgConfig REQUIRED)
#pkg_search_module(SLEPC IMPORTED_TARGET slepc)
#if (SLEPC_VERSION)
#    message("-- Using SLEPc: ${SLEPC_VERSION}")
#    set(libraries ${libraries} PkgConfig::SLEPC)
#    set(STRANDS_SRC ${STRANDS_SRC} src/util/right_kernel_sparse.cpp)
#    # add_definitions(-DSTRANDS_SLEPC)
#endif ()


include_directories(lib/spectra/include)

target_link_libraries(strands PUBLIC ${libraries})
target_compile_definitions(strands PUBLIC ${defines})

OPTION(STRANDS_TESTS "Also build the python bindings" ON)
if (STRANDS_TESTS)
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.3.1 # or a later release
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(strands_test
            test/zero.cpp
            test/harmonic.cpp
            test/ixaru.cpp
            test/quartic.cpp
            test/henon_heiles.cpp
            test/profile_henon_heiles.cpp
            )

    target_link_libraries(strands_test PRIVATE strands Catch2::Catch2WithMain)
    add_dependencies(strands_test strands)

    enable_testing()
    add_test(NAME strands_test COMMAND strands_test)
endif ()


OPTION(STRANDS_PYTHON "Also build the python bindings" ON)
if (STRANDS_PYTHON)
    if (IS_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/matslise/cmake/pybind11)
        add_subdirectory(lib/matslise/cmake/pybind11)
    else ()
        find_package(pybind11 QUIET)
    endif ()

    if (pybind11_FOUND)
        message("Adding strands_py")
        pybind11_add_module(strands_py MODULE src/python/strands.cpp ${STRANDS_SRC})
        target_link_libraries(strands_py PRIVATE ${libraries})
        target_compile_definitions(strands_py PUBLIC ${defines} _hypot=hypot)
        set_target_properties(strands_py PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/strands
                OUTPUT_NAME "strands")
        install(
                TARGETS strands_py
                COMPONENT strands_py
                LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}
                RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
        message("-- pybind: ${pybind11_VERSION}")
        message("-- python: ${PYTHON_EXECUTABLE}")

        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/build_strands_py")
        add_custom_target(install_strands_py
                COMMAND "${PYTHON_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/setup.py" bdist_wheel && pip uninstall --yes strands && pip install dist/*.whl
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/build_strands_py"
                )
    else ()
        message("To enable python bindings, make sure pybind11 is available in /lib/pybind11. Or disable the python bindings with -DSTRANDS_PYTHON=OFF")
    endif ()
endif ()
