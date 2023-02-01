#include "../schrodinger.h"

#ifndef STRANDS_INSTANTIATE
#define STRANDS_INSTANTIATE(Scalar)
#endif

template
class strands::Schrodinger<double>;

STRANDS_INSTANTIATE(double)

#ifdef STRANDS_LONG_DOUBLE

template
class strands::Schrodinger<long double>;

STRANDS_INSTANTIATE(long double)

#endif
