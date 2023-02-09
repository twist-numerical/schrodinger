import strands
from strands import Schrodinger2D, Rectangle
import sys
import doctest
from math import pi

s = Schrodinger2D(lambda x, y: 0, Rectangle(0,pi,0,pi))
eigs = s.eigenvalues(1)
assert abs(eigs[0] - 2) < 1e-6, "Wrong result from zero potential."


failures, test_count = doctest.testmod(strands.strands)

assert test_count > 0, "No tests were found"

print(f"{test_count} tests, {failures} failures")

sys.exit(1 if failures else 0)
