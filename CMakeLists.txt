cmake_minimum_required(VERSION 3.13)
project(schrodinger2d LANGUAGES CXX)
cmake_policy(SET CMP0077 NEW)

set(SCHRODINGER_VERSION "0.0.1")

set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    add_compile_options(/W4 /WX)
else ()
    add_compile_options(-Wall -Wextra -Werror -Wno-missing-field-initializers)
endif ()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message("-- Eigen3 version ${Eigen3_VERSION}")
set(libraries Eigen3::Eigen)

SET(SCHRODINGER_SRC
        src/schrodinger2d.cpp)

add_library(schrodinger ${SCHRODINGER_SRC})

SET(MATSLISE_MATSCS OFF)
SET(MATSLISE_2D OFF)
SET(MATSLISE_3D OFF)
SET(MATSLISE_PYTHON OFF)

add_subdirectory(lib/matslise)
include_directories(lib/matslise)

message("Schrodinger")

set(libraries ${libraries} matslise)

target_link_libraries(schrodinger PUBLIC ${libraries})

add_executable(schrodinger_test
        test/ixaru.cpp
        test/disc.cpp
        test/main.cpp)

target_link_libraries(schrodinger_test PRIVATE schrodinger ${libraries})

enable_testing()
add_test(NAME schrodinger_test COMMAND schrodinger_test)


if (IS_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/pybind11)
    add_subdirectory(lib/pybind11)
else ()
    find_package(pybind11 QUIET)
endif ()

if (pybind11_FOUND)
    message("Adding schrodinger_py")
    pybind11_add_module(schrodinger_py MODULE src/python/schrodinger.cpp ${SCHRODINGER_SRC})
    target_link_libraries(schrodinger_py PRIVATE ${libraries})
    message("-- pybind: ${pybind11_VERSION}")
    message("-- python: ${PYTHON_EXECUTABLE}")

    configure_file(src/python/setup.py setup.py NEWLINE_STYLE LF)
    configure_file(src/python/description.md description.md NEWLINE_STYLE LF)

    add_custom_target(install_schrodinger_py
            COMMAND ${CMAKE_COMMAND} -E env SCHRODINGER_LIBRARY="$<TARGET_FILE:schrodinger_py>"
            ${PYTHON_EXECUTABLE} -m pip install --upgrade --ignore-installed .
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_dependencies(install_schrodinger_py schrodinger_py)
else ()
    message("-- pybind11 not found")
endif ()