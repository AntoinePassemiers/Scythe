/**
    opt.hpp
    Useful functions / structs for decision tree optimization

    @author Antoine Passemiers
    @version 1.0 10/07/2017
*/

#ifndef OPT_HPP
#define OPT_HPP

#include <stdlib.h>

#if defined(_OPENMP)
    #include <omp.h>
    #define _OMP _OPENMP
#else
    typedef int omp_int_t;
    inline omp_int_t omp_get_thread_num() { return 0; }
    inline omp_int_t omp_get_max_threads() { return 1; }
#endif

#ifdef defined(_POSIX_C_SOURCE) && defined(_OMP)
    #if _POSIX_C_SOURCE >= 200112L
        #define _MEM_ALIGN 1
    #endif
#endif

#endif // OPT_HPP