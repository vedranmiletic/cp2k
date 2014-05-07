/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2014 the CP2K developers group                      *
 *****************************************************************************/

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>

// defines error check functions and 'cl_error'
#include "acc_opencl_error.h"

// defines 'acc_opencl_my_device' and some default lenghts
#include "acc_opencl_dev.h"

// defines 'acc_opencl_stream_type'
#include "acc_opencl_stream.h"

// defines the ACC interface
#include "../include/acc.h"

static const int verbose_print = 0;

/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
int acc_event_create (void** event_p){
  // local event object pointer
  *event_p = malloc(sizeof(cl_event));
  cl_event *clevent = (cl_event *) *event_p;

  // get a device event object
  *clevent = clCreateUserEvent((*acc_opencl_my_device).ctx, &cl_error);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;

  // print event address
  if (verbose_print)
    printf("acc_event_create:  %p -> %d\n", *event_p, *clevent);

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
int acc_event_destroy (void* event){
  // local event object pointer
  cl_event *clevent = (cl_event *) event;

  // release event object
  cl_error = clReleaseEvent(*clevent);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;
  free(clevent);

  // print event address
  if (verbose_print)
    printf("acc_event_destroy called\n");

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
int acc_event_record (void* event, void* stream){
  // local event and queue pointers
  cl_event *clevent = (cl_event *) event;
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // ToDo: ????
//  cudaError_t cErr = cudaEventRecord (*cuevent, *custream);

  // print event address
  if (verbose_print)
    printf("acc_event_record: %p -> %d,  %p -> %d\n", clevent, *clevent,  clstream, *clstream);

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
int acc_event_query (void* event, int* has_occured){
  // local event pointer
  cl_event *clevent = (cl_event *) event;

  // print message
  if (verbose_print)
    printf("acc_event_query called\n");

  // ToDo: ????
//  cudaError_t cErr = cudaEventQuery(*cuevent);
//  cl_error = clGetEventInfo(*clevent, ..., &has_occured,...)

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
int acc_stream_wait_event (void* stream, void* event){
  // local event and queue pointers
  cl_event *clevent = (const cl_event *) event;
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // print message
  if (verbose_print)
    printf("acc_stream_wait_event called\n");

  // wait for an event on a stream
  cl_error = clEnqueueWaitForEvents((*clstream).queue,
               (cl_uint) 1, clevent);
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
int acc_event_synchronize (void* event){
  // local event and queue pointers
  cl_event *clevent = (const cl_event *) event;

  // print message
  if (verbose_print) printf("acc_event_synchronize called\n");

  // wait for an event ( !!! need to share the same ctx !!! )
  cl_error = clWaitForEvents((cl_uint) 1, clevent);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;


  // assign return value
  return 0;
}
#ifdef __cplusplus
}
#endif

//EOF
