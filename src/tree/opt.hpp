/**
    opt.hpp
    Useful functions / structs for decision tree optimization

    @author Antoine Passemiers
    @version 1.0 10/07/2017
*/

#ifndef OPT_HPP
#define OPT_HPP

#include <stdlib.h>

namespace scythe {

#if true
    #include <omp.h>
    #define _OMP _OPENMP
#else
    typedef int omp_int_t;
    inline omp_int_t omp_get_thread_num() { return 0; }
    inline omp_int_t omp_get_max_threads() { return 1; }
#endif

#if defined(_POSIX_C_SOURCE) && defined(_OMP)
    #if _POSIX_C_SOURCE >= 200112L
        #define _MEM_ALIGN 1
    #endif
#endif

// Portability of restrict qualifier, as suggested by Intel
#if defined(__INTEL_COMPILER) && defined(USE_RESTRICT_OPTION)
    #define RESTRICT restrict
#elif defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN32__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT
#endif

// Clause 5.2.9 of the C++ international standard (see also : 4.12.1)
static_assert((8 > 4) == static_cast<size_t>(1), "Bool to int conversion issue");

} // namespace

#endif // OPT_HPP