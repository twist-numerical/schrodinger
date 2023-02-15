# Strands

**St**anding waves app**r**oxim**a**tions for the **n-d**imensional **S**chrödinger problem (with n = 2).

*Strands* is a library to compute eigenvalues of two-dimensional time-independent Schrödinger equations.
$$
-\nabla \psi(x, y) + V(x, y) \psi(x, y) = E \psi(x, y)
$$
The library is written in C++ with Python-bindings.

## Installation

Installing it is as simple as
```sh
pip install strands
```

## Examples

### Harmonic oscillator on a circular domain

Consider the harmonic oscillator potential
$$
V(x, y) = x^2 + y^2
$$
on the circular domain around zero with radius $9.5$.

```python
from strands import Schrodinger2D , Circle

def V(x, y):
    return x * x + y * y

schrodinger = Schrodinger2D(V, Circle ((0, 0), 9.5), gridSize=(40, 40), maxBasisSize=30)
print(schrodinger.eigenvalues(10))
```

The values `gridSize` and `maxBasisSize` determine how accurate the used method has to be.
Eigenfunctions can be computed with:

```python
import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(-4, 4, 100)
ys = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(xs, ys)

for E, f in schrodinger.eigenfunctions(3):
    plt.pcolormesh(X, Y, f(X, Y))
    plt.show ()
```

## Development

This is developed in C++ with CMake. To get started, make a recursive clone:
```sh
git clone --recursive https://github.com/twist-numerical/strands.git
cd strands
```

To compile and run the tests the following can be used:
```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSTRANDS_PYTHON=OFF ..  # Build without python
cmake --build . --target strands_test
./strands_test --durations yes
```
