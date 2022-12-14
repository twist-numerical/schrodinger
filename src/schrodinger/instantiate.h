#include "../schrodinger.h"

#ifndef SCHRODINGER_INSTANTIATE
#define SCHRODINGER_INSTANTIATE(Scalar)
#endif

template
class schrodinger::Schrodinger<double>;

SCHRODINGER_INSTANTIATE(double)

#ifdef SCHRODINGER_LONG_DOUBLE

template
class schrodinger::Schrodinger<long double>;
SCHRODINGER_INSTANTIATE(long double)
#endif
