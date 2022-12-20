#include "../schrodinger.h"

#ifndef SCHRODINGER_INSTANTIATE
#define SCHRODINGER_INSTANTIATE(Scalar, dimension)
#endif

template
class schrodinger::Schrodinger<double>;

SCHRODINGER_INSTANTIATE(double, 2)

#ifdef SCHRODINGER_LONG_DOUBLE

template
class schrodinger::Schrodinger<long double>;
SCHRODINGER_INSTANTIATE(long double, 2)
#endif
