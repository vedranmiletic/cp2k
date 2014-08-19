/******************************************************************************
*  CP2K: A general program to perform molecular dynamics simulations
*  Copyright (C) 2000 - 2014 the CP2K developers group
*****************************************************************************/

#ifndef LIBCLSMM_H
#define LIBCLSMM_H

#if defined (__ACC) && defined (__OPENCL)

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int libclsmm_process_d (int *param_stack, int stack_size,
    void stream, int m, int n, int k,
    double * a_data, double * b_data, double * c_data);

int libclsmm_transpose_d (int *trs_stack, int offset, int nblks, double *buffer,
                         int m, int n, void *stream);

void libclsmm_list_blocksizes_d (const int **list, int *length);

#endif

#endif
//EOF
