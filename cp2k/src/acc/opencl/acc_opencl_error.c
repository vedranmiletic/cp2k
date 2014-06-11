/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2014 the CP2K developers group                      *
 *****************************************************************************/

#include <CL/cl.h>
#include <stdio.h>

// defines error check functions and 'cl_error'
#include "acc_opencl_error.h"

/****************************************************************************/
int acc_opencl_error_check (cl_int cl_error, int line){
  int pid;

  if (cl_error != CL_SUCCESS) {
    pid = getpid();
    fprintf(stderr, "%d OPENCL RT Error line: %d\n", pid, line);
    return -1;
  }
  return 0;
}

//EOF
