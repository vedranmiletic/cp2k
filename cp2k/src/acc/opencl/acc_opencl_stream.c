/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2014 the CP2K developers group                      *
 *****************************************************************************/


/* 
 *
 * NOTE: In OpenCL streams are called queues and the related device.ctx and
 *       device.platform is used in combination with it. Therefore we need 
 *       a struct 'acc_opencl_queue' which combines this information.
 *
 *       For convenience the routine names are called 'xxx_stream_xxx' to
 *       match the ACC interface.
 */

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>

// defines error check functions and 'cl_error'
#include "acc_opencl_error.h"

// defines 'acc_opencl_my_device' and some default lenghts
#include "acc_opencl_dev.h"

// defines 'acc_opencl_queue' struct
#include "acc_opencl_stream.h"

// defines the ACC interface
#include "../include/acc.h"

static const int verbose_print = 0;


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_stream_priority_range (int* least, int* greatest){
  // NOTE: This functionality is not available in OpenCL.
  *least = -1;
  *greatest = -1;

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
// NOTE: 'priority is ignored.
int acc_stream_create (void** stream_p, char* name, int priority){
  // get memory on pointer
  *stream_p = malloc(sizeof(acc_opencl_stream_type));

  // local queue pointer 
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) *stream_p;
  (*clstream).device = *acc_opencl_my_device;

  // create a command queue
  cl_command_queue_properties queue_properties = 0;
  (*clstream).queue = clCreateCommandQueue((*acc_opencl_my_device).ctx, (*acc_opencl_my_device).device_id, queue_properties, &cl_error);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_stream_destroy (void* stream){
  // local queue pointer 
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // release the command queue
  cl_error = clReleaseCommandQueue((*clstream).queue);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;
  // free the struct acc_opencl_queue 'stream'
  free(clstream);

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif

/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_stream_sync (void* stream){
  // local queue pointer 
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // synchronize the command queue
  // A ' clEnqueueBarrier is probably enough
  cl_error = clFlush((*clstream).queue);
//ToDo: Flush sends all commands in a queue to the device but does not
//      guarantee that they will be processed while return to host.
//  cl_error = clFinish((*clstream).queue);

  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;

  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif

//EOF
