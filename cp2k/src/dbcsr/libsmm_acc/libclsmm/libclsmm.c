/******************************************************************************
*  CP2K: A general program to perform molecular dynamics simulations
*  Copyright (C) 2000 - 2013 the CP2K developers group
*****************************************************************************/

#if defined (__ACC) && defined (__OPENCL)
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
static const int verbose_print = 0;

// struct definitions (Ugly: is a copy from src/acc/opencl/*)
typedef struct {
   cl_platform_id   platform_id;
   cl_device_id     device_id;
   cl_context       ctx;
} acc_opencl_dev_type;

typedef struct {
   acc_opencl_dev_type  device;
   cl_command_queue     queue;
} acc_opencl_stream_type;

cl_int cl_error;


/****************************************************************************/
// Kernel launch
static int launch_clsmm_dnt_largeDB_16_23_23_12_23_96_2_3_12_10 (int *param_stack, int stack_size, cl_command_queue stream, int m_max, int n_max, int k_max, double *a_data, double *b_data, double *c_data){
  int shared_size = 0;
//{'name': 'clsmm_dnt_largeDB_16_23_23_12_23_96_2_3_12_10', 'tile_n': 3, 'tile_m': 2, 'm': 23, 'n': 23, 'threads': 96, 'w': 10, 'v': 12, 'minblocks': 12, 'k': 23, 'grouping': 16}
  int careful = (stack_size / 16);
  int nruns = stack_size - careful * 16;


  // CUDA code !!!!
  // clsmm_dnt_largeDB<23,23,23,2,3,10,12,96,16,12> <<< ((stack_size + 16 - 1) / 16), 96, shared_size, stream >>> (param_stack, careful, nruns, a_data, b_data, c_data);
  // return 0;
fprintf(stdout, "OpenCL Multiplication kernel not implemented ---> implement!\n");
  return -1;
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
  }

  return -1; // should never happen
}

/****************************************************************************/
// Transpose kernel switch and launch
int libclsmm_transpose_d (void *trs_stack, int offset, int nblks, void *buffer, int m, int n, void *stream){
  int idx = 0;
  int missing = 0; //false

  // local queue pointer and device + context value 
  acc_opencl_stream_type *opencl_stream = (acc_opencl_stream_type *) stream;
  acc_opencl_dev_type     opencl_device = (*opencl_stream).device;
  cl_context              opencl_ctx    = opencl_device.ctx;
  cl_device_id            opencl_dev    = opencl_device.device_id;

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
  if (missing) return 0;

  if (verbose_print) fprintf(stdout, "Transpose %d blocks.\n", nblks);

  switch(idx){
    case 0:
      if (verbose_print) fprintf(stdout,"reading transpose kernel ...\n");
      FILE *fIn = fopen("clsmm_transpose.cl", "r");
      fseek(fIn, 0L, SEEK_END);
      size_t sz = ftell(fIn); 
      rewind(fIn);
      char *file = (char*) malloc(sizeof(char) * sz + 1);
      fread(file, sizeof(char), sz, fIn);
      const char* cfile = (const char *) file;
      fclose(fIn);
  
      // get kernel code, build program and kernel
      if (verbose_print) fprintf(stdout,"building transpose kernel ...\n");
      cl_program opencl_program = clCreateProgramWithSource(opencl_ctx, 1, &cfile, &sz, &cl_error);
      if (cl_error != CL_SUCCESS) fprintf(stdout,"Error in: clCreateProgramWithSource %d\n", (int) cl_error);
      cl_error = clBuildProgram(opencl_program, 1, (const cl_device_id *) &opencl_dev, "-D__ACC", NULL, NULL); // hard coded -D__ACC
      if (cl_error != CL_SUCCESS) fprintf(stdout,"Error in: clBuildProgram %d\n", (int) cl_error);

      if (cl_error != CL_SUCCESS){
        size_t param_value_size_ret;
        cl_error = clGetProgramBuildInfo(opencl_program, opencl_dev, CL_PROGRAM_BUILD_LOG, (size_t) 0, NULL, &param_value_size_ret);
        if (cl_error != CL_SUCCESS) fprintf(stdout,"Error 1 %d\n", (int) cl_error);
        char *build_log = (char *) malloc(param_value_size_ret * sizeof(char));
        cl_error = clGetProgramBuildInfo(opencl_program, opencl_dev, CL_PROGRAM_BUILD_LOG, param_value_size_ret, (void *) build_log, NULL);
        if (cl_error != CL_SUCCESS) fprintf(stdout,"Error 2 %d\n", (int) cl_error);
        fprintf(stdout, "BUILD LOG:\n %s\n", build_log);
      }

      cl_kernel opencl_kernel = clCreateKernel(opencl_program, "transpose_d", &cl_error);
      if (cl_error != CL_SUCCESS) fprintf(stdout,"Error in: clCreateKernel %d\n", (int) cl_error);
  
      // set kernel parameters
      if (verbose_print) fprintf(stdout,"set transpose kernel parameters ...\n");
      cl_error = clSetKernelArg(opencl_kernel, 0, sizeof(cl_mem), (void *) trs_stack);
      if (cl_error != CL_SUCCESS) fprintf(stdout,"Error in: clSetKernelArg(0) %d\n", (int) cl_error);
      cl_error = clSetKernelArg(opencl_kernel, 1, sizeof(int), &offset);
      if (cl_error != CL_SUCCESS) fprintf(stdout,"Error in: clSetKernelArg(1) %d\n", (int) cl_error);
      cl_error = clSetKernelArg(opencl_kernel, 2, sizeof(cl_mem), (void *) buffer);
      if (cl_error != CL_SUCCESS) fprintf(stdout,"Error in: clSetKernelArg(2) %d\n", (int) cl_error);

      // submit kernel
      if (verbose_print) fprintf(stdout,"calling transpose kernel ...\n");
      size_t local_work_size[1] = {128};
      size_t global_work_size[1] = {nblks * local_work_size[0]};

      cl_error = clEnqueueNDRangeKernel(
                   (*opencl_stream).queue, // command_queue
                   opencl_kernel,          // kernel
                   1,                      // work_dim
                   NULL,                   // global_work_offset
                   global_work_size,       // global_work_size
                   local_work_size,        // local_work_size
                   0,                      // num_events_in_wait_list
                   NULL,                   // event_wait_list
                   NULL);                  // event
      if (cl_error != CL_SUCCESS) fprintf(stdout,"Error in: clEnqueueNDRangeKernel %d\n", (int) cl_error);
      cl_error = clFinish((*opencl_stream).queue);
      if (cl_error != CL_SUCCESS) fprintf(stdout,"Error in: clFinish %d\n", (int) cl_error);

      return 0;
    break;
    // If there is no kernel for these blocks, we don't need to transpose them.
    default: return 0;
  }

}



/****************************************************************************/
// Helper routines
void libclsmm_list_blocksizes_d (const int **list, int *length){
  static const int blocksizes_d[] = { 23, 23, 23, };

  *list = blocksizes_d;
  *length = 1;
}



/****************************************************************************/
// Kernel interface for Fortran side
#ifdef __cplusplus
extern "C" {
#endif
int libsmm_acc_process (void *param_stack, int stack_size, int nparams, int datatype, void *a_data, void *b_data, void *c_data, int m_max, int n_max, int k_max, int def_mnk, void *stream){
  // debug info
  if (verbose_print) fprintf(stdout,"entering libsmm_acc_process ...\n");

  // local queue pointer 
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  if (def_mnk != 1)
    return -1; // inhomogenous stacks not supported
  if (datatype == dbcsr_type_real_8)
    return libclsmm_process_d((int *) param_stack, stack_size, (*clstream).queue, m_max, n_max, k_max,(double *) a_data, (double *) b_data, (double *) c_data);

  return -1; // datatype not supported
}
#ifdef __cplusplus
}
#endif

/****************************************************************************/
// Transpose kernel interface for Fortran side
#ifdef __cplusplus
extern "C" {
#endif
int libsmm_acc_transpose (void *trs_stack, int offset, int nblks, void *buffer, int datatype, int m, int n, void *stream){
  // debug info
  if (verbose_print) fprintf(stdout,"entering libsmm_acc_transpose ...\n");

  if (datatype != dbcsr_type_real_8) return 0; //transpose not needed
  
  return libclsmm_transpose_d(trs_stack, offset, nblks, buffer, m, n, stream);

  return -1;
}
#ifdef __cplusplus
}
#endif

#endif
//EOF
