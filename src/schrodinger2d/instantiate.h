#include "../schrodinger2d.h"

template
class schrodinger::Schrodinger2D<double>;

#ifdef SCHRODINGER_LONG_DOUBLE

template
class schrodinger::Schrodinger2D<long double>;

#endif
