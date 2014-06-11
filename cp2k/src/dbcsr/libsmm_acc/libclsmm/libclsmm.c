/******************************************************************************
*  CP2K: A general program to perform molecular dynamics simulations
*  Copyright (C) 2000 - 2013 the CP2K developers group
*****************************************************************************/

// dependencies
//#include "./kernels/clsmm_dnt_largeDB.h"
//#include "./kernels/clsmm_common.h"
//#include "./kernels/clsmm_transpose.h"
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>

#include "../include/libsmm_acc.h"


// global definitions
#define dbcsr_type_real_4     1
#define dbcsr_type_real_8     3
#define dbcsr_type_complex_4  5
#define dbcsr_type_complex_8  7

// debug flag
static const int verbose_print = 1;

// struct definitions
typedef struct {
   cl_platform_id   platform_id;
   cl_device_id     device_id;
   cl_context       ctx;
} acc_opencl_dev_type;

typedef struct {
   acc_opencl_dev_type  device;
   cl_command_queue     queue;
} acc_opencl_stream_type;


/****************************************************************************/
// Kernel launch
static int launch_clsmm_dnt_largeDB_16_23_23_12_23_96_2_3_12_10 (int *param_stack, int stack_size, cl_command_queue stream, int m_max, int n_max, int k_max, double *a_data, double *b_data, double *c_data){
  int shared_size = 0;
//{'name': 'clsmm_dnt_largeDB_16_23_23_12_23_96_2_3_12_10', 'tile_n': 3, 'tile_m': 2, 'm': 23, 'n': 23, 'threads': 96, 'w': 10, 'v': 12, 'minblocks': 12, 'k': 23, 'grouping': 16}
  int careful = (stack_size / 16);
  int nruns = stack_size - careful * 16;


  // CUDA code !!!!
  //clsmm_dnt_largeDB<23,23,23,2,3,10,12,96,16,12> <<< ((stack_size + 16 - 1) / 16), 96, shared_size, stream >>> (param_stack, careful, nruns, a_data, b_data, c_data);
  return 0;
}


/****************************************************************************/
// Kernel switch
int libclsmm_process_d (int *param_stack, int stack_size, cl_command_queue stream, int m, int n, int k, double *a_data, double *b_data, double *c_data){
  int idx = 0;
  int missing = 0; // false

  switch(m){
    case 23: idx = 0; break;
    default: missing = 1;
  }

  idx *= 1;
  switch(n){
    case 23: idx += 0; break;
    default: missing = 1;
  }

  idx *= 1;
  switch(k){
    case 23: idx += 0; break;
    default: missing = 1;
  }

  if (missing) return -1;
  switch(idx){
    case 0:
// m=23, n=23, k=23
    return launch_clsmm_dnt_largeDB_16_23_23_12_23_96_2_3_12_10(param_stack, stack_size, stream, 23, 23, 23, a_data, b_data, c_data);
fprintf(stdout,"calling process kernel ...\n");
  }

  return -1; // should never happen
}

/****************************************************************************/
// Transpose kernel switch and launch
int libclsmm_transpose_d (int *trs_stack, int offset, int nblks, double *buffer, int m, int n, cl_command_queue *stream){
  int idx = 0;
  int missing = 0; //false

  switch(m){
    case 23: idx = 0; break;
    default: missing = 1;
  }

  idx *= 1;
  switch(n){
    case 23: idx += 0; break;
    default: missing = 1;
  }

// If there is no kernel for these blocks, we don't need to transpose them.
  if(missing) return 0;

  switch(idx){
    case 0:
// m=23, n=23
// CUDA code
//    transpose_d<23,23> <<<nblks, 128, 0, *stream>>>(trs_stack+offset, nblks, buffer);
// OpenCL code
fprintf(stdout,"calling transpose kernel ...\n");

      return 0;
    break;
// If there is no kernel for these blocks, we don't need to transpose them.
    default: return 0;
  }

// CUDA code
  //return(cudaGetLastError());
}



/****************************************************************************/
// Helper routines
void libclsmm_list_blocksizes_d (const int **list, int *length){
  static const int blocksizes_d[] = { 23, 23, 23, };

  *list = blocksizes_d;
  *length = 1;
}



/****************************************************************************/
// Interface for Fortran side
#ifdef __cplusplus
extern "C" {
#endif
int libsmm_acc_process (int *param_stack, int stack_size, int nparams, int datatype, void *a_data, void *b_data, void *c_data, int m_max, int n_max, int k_max, int def_mnk, void *stream){
  // debug info
  if (verbose_print) fprintf(stdout,"entering libsmm_acc_process ...\n");

  // local queue pointer 
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  if (def_mnk != 1)
    return -1; // inhomogenous stacks not supported
  if (datatype == dbcsr_type_real_8)
    return libclsmm_process_d(param_stack, stack_size, (*clstream).queue, m_max, n_max, k_max,(double *) a_data, (double *) b_data, (double *) c_data);

  return -1; // datatype not supported
}
#ifdef __cplusplus
}
#endif

/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int libsmm_acc_transpose (int *trs_stack, int offset, int nblks, void *buffer,int datatype, int m, int n, void* stream){
  // debug info
  if (verbose_print) fprintf(stdout,"entering libsmm_acc_transpose ...\n");

  // local queue pointer 
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;
  if(datatype != dbcsr_type_real_8)
    return 0; //transpose not needed
  return libclsmm_transpose_d(trs_stack, offset, nblks, (double*) buffer, m, n, (*clstream).queue);

  return -1;
}
#ifdef __cplusplus
}
#endif


//EOF
