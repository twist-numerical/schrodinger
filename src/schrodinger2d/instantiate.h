#include "../schrodinger2d.h"

#ifndef SCHRODINGER_INSTANTIATE
#define SCHRODINGER_INSTANTIATE(Scalar)
#endif

template
class schrodinger::Schrodinger2D<double>;

SCHRODINGER_INSTANTIATE(double)

#ifdef SCHRODINGER_LONG_DOUBLE

template
class schrodinger::Schrodinger2D<long double>;
SCHRODINGER_INSTANTIATE(long double)
#endif
