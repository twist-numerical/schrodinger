cmake_minimum_required(VERSION 3.13)
project(schrodinger2d LANGUAGES CXX)
cmake_policy(SET CMP0077 NEW)

set(SCHRODINGER_VERSION "0.0.1")

set(CMAKE_CXX_STANDARD 17)

# -fPIC needed for pybind
set(BUILD_SHARED_LIBS ON)

if (MSVC)
    add_compile_options(/W4 /WX)
else ()
    add_compile_options(-Wall -Wextra -Werror -Wno-missing-field-initializers)
endif ()

SET(SCHRODINGER_SRC src/schrodinger2d.cpp)

add_library(schrodinger ${SCHRODINGER_SRC})

SET(MATSLISE_STATIC ON)
SET(MATSLISE_MATSCS OFF)
SET(MATSLISE_2D OFF)
SET(MATSLISE_3D OFF)
SET(MATSLISE_PYTHON OFF)

OPTION(SCHRODINGER_LONG_DOUBLE "Compile with support for long double" OFF)
if (SCHRODINGER_LONG_DOUBLE)
    add_definitions(-DSCHRODINGER_LONG_DOUBLE)
    message("-- with support for long double")
    set(LONG_DOUBLE ON)
else ()
    set(LONG_DOUBLE OFF)
    message("-- without support for long double")
endif ()

add_subdirectory(lib/matslise)
include_directories(lib/matslise)
set_property(TARGET matslise PROPERTY POSITION_INDEPENDENT_CODE ON)

message("Schrodinger")

set(libraries matslise)

target_link_libraries(schrodinger PUBLIC ${libraries})

add_executable(schrodinger_test
        test/ixaru.cpp
        test/disc.cpp
        test/main.cpp)

target_link_libraries(schrodinger_test PRIVATE schrodinger)
add_dependencies(schrodinger_test schrodinger)

enable_testing()
add_test(NAME schrodinger_test COMMAND schrodinger_test)


OPTION(SCHRODINGER_PYTHON "Also build the python bindings" ON)
if (SCHRODINGER_PYTHON)
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
        message("To enable python bindings, make sure pybind11 is available in /lib/pybind11. Or disable the python bindings with -DSCHRODINGER_PYTHON=OFF")
    endif ()
endif ()
