/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2013 the CP2K developers group                      *
 *****************************************************************************/
#ifndef CLSMM_COMMON_H
#define CLSMM_COMMON_H

#if defined (__ACC)

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/******************************************************************************
 * There is no nativ support for atomicAdd on doubles in OpenCL 1.1.          *
 ******************************************************************************/
void AtomicAdd(volatile __global double *address, const double val) {
    union {
        unsigned long long int intVal;
        double dblVal;
    } newVal;
    union {
        unsigned long long int intVal;
        double dblVal;
    } prevVal;
    do {
        prevVal.dblVal = *address;
        newVal.dblVal = prevVal.dblVal + val;
    } while (atomic_cmpxchg((volatile __global unsigned int *)address, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline void AtomicAddLocal(volatile __local double *address, const double val) {
    union {
        unsigned long long int intVal;
        double dblVal;
    } newVal;
    union {
        unsigned long long int intVal;
        double dblVal;
    } prevVal;
    do {
        prevVal.dblVal = *address;
        newVal.dblVal = prevVal.dblVal + val;
    } while (atomic_cmpxchg((volatile __global unsigned int *)address, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

#endif
#endif
