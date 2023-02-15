# Adapted from: https://github.com/diegoferigo/cmake-build-extension/blob/master/example/setup.py
import inspect
import os
import sys
from pathlib import Path

import cmake_build_extension
import setuptools

with open(os.path.join(os.path.dirname(__file__), "src", "python", "__init__.py")) as init_file:
    init_py = init_file.read()

# Extra options passed to the CI/CD pipeline that uses cibuildwheel
CIBW_CMAKE_OPTIONS = []
if "CIBUILDWHEEL" in os.environ and os.environ["CIBUILDWHEEL"] == "1":

    # The manylinux variant runs in Debian Stretch and it uses lib64 folder
    if sys.platform == "linux":
        CIBW_CMAKE_OPTIONS += ["-DCMAKE_INSTALL_LIBDIR=lib"]


# This example is compliant with PEP517 and PEP518. It uses the setup.cfg file to store
# most of the package metadata. However, build extensions are not supported and must be
# configured in the setup.py.
setuptools.setup(
    name="Strands",
    version="0.1.0",
    author="Toon Baeyens",
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="Strands",
            # Name of the resulting package name (import mymath_pybind11)
            install_prefix="strands",
            # Note: pybind11 is a build-system requirement specified in pyproject.toml,
            #       therefore pypa/pip or pypa/build will install it in the virtual
            #       environment created in /tmp during packaging.
            #       This cmake_depends_on option adds the pybind11 installation path
            #       to CMAKE_PREFIX_PATH so that the example finds the pybind11 targets
            #       even if it is not installed in the system.
            # cmake_depends_on=["pybind11"],
            # Writes the content to the top-level __init__.py
            write_top_level_init=init_py,
            # Selects the folder where the main CMakeLists.txt is stored
            # (it could be a subfolder)
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_configure_options=[
                # This option points CMake to the right Python interpreter, and helps
                # the logic of FindPython3.cmake to find the active version
                f"-DPYTHON_EXECUTABLE={Path(sys.executable)}",
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
                "-DSTRANDS_TESTS:BOOL=OFF",
            ]
            + CIBW_CMAKE_OPTIONS,
            cmake_component="strands_py"
        ),
    ],
    cmdclass=dict(
        # Enable the CMakeExtension entries defined above
        build_ext=cmake_build_extension.BuildExtension,
        # If the setup.py or setup.cfg are in a subfolder wrt the main CMakeLists.txt,
        # you can use the following custom command to create the source distribution.
        # sdist=cmake_build_extension.GitSdistFolder
    ),
)
