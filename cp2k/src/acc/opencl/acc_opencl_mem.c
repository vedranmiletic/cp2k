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

/*  ALIGN functions work only for: size = 2^n  */
#define MEMORY_ALIGNMENT 128
#define ALIGN_UP(x,size)   ( ((size_t)x + (size-1))&(~(size-1)) )
#define ALIGN_DOWN(x,size) ( ((size_t)x - (size-1))&(~(size-1)) )
#define SHIFT_BY(x,size)   ( ((size_t)x + size) )

static const int verbose_print = 0;


/****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Create a device buffer object of 'cl_mem' type
 *
 * Note: The data can't be accessed directly.
 *       Use the buffer object instead.
 */
int acc_dev_mem_allocate (void **dev_mem, size_t n){
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_dev_mem_allocate.\n");

  // local memory object pointer 
  *dev_mem = malloc(sizeof(cl_mem));
  cl_mem *clmem = (cl_mem *) *dev_mem;

  // get a device buffer object
  *clmem = clCreateBuffer(
             (*acc_opencl_my_device).ctx,                 // device context
             (CL_MEM_READ_WRITE), // flags
             (size_t) n,                                  // number of bytes
             NULL,                                        // host pointer
             &cl_error);                                  // error
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "Device buffer allocation address %p, size %ld\n", *dev_mem, (long) n);
    fprintf(stdout, "Leaving: acc_dev_mem_allocate.\n");
  }

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
/*
 * Destroy a device buffer object of 'cl_mem' type.
 */
int acc_dev_mem_deallocate (void *dev_mem){
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_dev_mem_deallocate.\n");

  // local memory object pointer 
  cl_mem *clmem = (cl_mem *) dev_mem;

  // print buffer address
  if (verbose_print)
    printf("Device deallocation address %p\n", dev_mem);

  // release device buffer object
  cl_error = clReleaseMemObject(*clmem);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;
  free(clmem);

  // debug info
  if (verbose_print) fprintf(stdout, "Leaving: acc_dev_mem_deallocate.\n");

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
/*
 * Create a host memory pointer to memory of size 'n' bytes and an associated
 * host buffer object of 'cl_mem' type.
 *
 * Note: The allocation is prefaced by the buffer object, but only the
 *       pointer to the host_mem is given back.
 */
int acc_host_mem_allocate (void **host_mem, size_t n){

  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_host_mem_allocate.\n");

  // check
  if (MEMORY_ALIGNMENT < sizeof(cl_mem) + sizeof(uintptr_t)) return 1;

  // get addresses
//  uintptr_t *adr  = (uintptr_t *) malloc(n * sizeof(char));
//            *host_mem  = (uintptr_t *) ALIGN_UP(adr, MEMORY_ALIGNMENT);
            *host_mem  = (uintptr_t *) malloc(n * sizeof(char));

  // debug infos
  if (verbose_print){
    fprintf(stdout, "\n --- HOST MEMORY ALLOCATION --- \n");
//    fprintf(stdout, "HOST pointer address: HEX=%p INT=%ld --- VAL=%ld\n",  adr,  (uintptr_t) adr,  (uintptr_t) *adr);
    fprintf(stdout, "HOST memory address:  HEX=%p INT=%ld\n",             *host_mem,  (uintptr_t) *host_mem);
    fprintf(stdout, "Leaving: acc_host_mem_allocate.\n");
  }

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
int acc_host_mem_deallocate (void *host_mem){

  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_host_mem_deallocate.\n");

  // debug infos
  if (verbose_print){
    fprintf(stdout, "\n --- HOST MEMORY DEALLOCATION --- \n");
    fprintf(stdout, "HOST memory address:  HEX=%p INT=%ld\n", host_mem, (uintptr_t) host_mem);
  }

  // free the whole memory (address is stored on 'keep')
  free(host_mem);

  // debug info
  if (verbose_print) fprintf(stdout, "Leaving: acc_host_mem_deallocate.\n");

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
int acc_memcpy_h2d (const void *host_mem, void *dev_mem, size_t count, void* stream){
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_memcpy_h2d.\n");

  // device buffer
  cl_mem *buf  = (cl_mem *) dev_mem;

  // local stream object and memory object pointers
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // copy host memory to device buffer
  cl_error = clEnqueueWriteBuffer(
               (*clstream).queue, // stream
               *buf,              // device buffer
               CL_TRUE,           // blocking write
               (size_t) 0,        // offset
               (size_t) count,    // number of bytes
               host_mem,          // host pointer
               (cl_uint) 0,       // number of events in wait list
               NULL,              // event wait list
               NULL);             // event
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "Copying %d bytes from host address %p to device address %p \n",
      count, host_mem, dev_mem);
    fprintf(stdout, "Leaving: acc_memcpy_h2d.\n");
  }

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
int acc_memcpy_d2h (const void *dev_mem, void *host_mem, size_t count, void* stream){
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_memcpy_d2h.\n");

  // local stream object and memory object pointers
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;
  const cl_mem *clmem = (const cl_mem *) dev_mem;

  // copy host memory to device buffer
  cl_error = clEnqueueReadBuffer((*clstream).queue,
               *clmem, CL_TRUE, (size_t) 0, count, // For now: blocking read!!!
               host_mem, (cl_uint) 0, NULL, NULL);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "Copying %d bytes from device address %p to host address %p\n",
      count, dev_mem, host_mem);
    fprintf(stdout, "Leaving: acc_memcpy_d2h.\n");
  }

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
int acc_memcpy_d2d (const void *devmem_src, void *devmem_dst, size_t count, void* stream){
// ToDo: what happens for stream = NULL?
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_memcpy_d2d.\n");

  // local stream object and memory object pointers
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;
  cl_mem *clmemsrc = (cl_mem *) devmem_src;
  cl_mem *clmemdst = (cl_mem *) devmem_dst;

  // copy device buffers from src to dst
  cl_error = clEnqueueCopyBuffer((*clstream).queue,
               *clmemsrc, *clmemdst, (size_t) 0, (size_t) 0, (size_t) count,
               (cl_uint) 0, NULL, NULL);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "Coping %d bytes from device address %p to device address %p \n",
      count, devmem_src, devmem_dst);
    fprintf(stdout, "Leaving: acc_memcpy_d2d.\n");
  }

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
int acc_memset_zero (void *dev_mem, size_t offset, size_t length, void* stream){
// ToDo: what happens for stream = NULL?
//       OpenCL 1.1 has no build in function for that!!!
//       We use cl_uchar because it's 8Bit = 1Byte long.
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_memset_zero.\n");

  size_t i;

  // local stream object and memory object pointers
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;
  cl_mem *clmem = (cl_mem *) dev_mem;

  // zero the values starting from offset in dev_mem
#ifdef CL_VERSION_1_2
  const cl_uchar zero = (cl_uchar) 0;

  cl_error = clEnqueueFillBuffer((*clstream).queue,
               *clmem, &zero, (size_t) sizeof(cl_uchar), (size_t) offset, (size_t) length,
               (cl_uint) 0, NULL, NULL);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;
#else
  // create a array of size 'lenght' and zero it
  cl_uchar *host_mem = (cl_uchar *) malloc(length * sizeof(cl_uchar));
  for (i=0; i<length; i++)
    host_mem[i] = (cl_uchar) 0;

  // transfer the 'zero_mem' to device buffer
  cl_error = clEnqueueWriteBuffer((*clstream).queue,
               *clmem, CL_TRUE, (size_t) offset, (size_t) length, // For now: zeroing blocks!!!
               (const void *) host_mem, (cl_uint) 0, NULL, NULL);
  if (acc_opencl_error_check(cl_error, __LINE__))
    return -1;
#endif

  // debug info
  if (verbose_print) fprintf(stdout, "Leaving: acc_memset_zero.\n");

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
int acc_dev_mem_info (size_t* free, size_t* avail){
// Note: OpenCL 1.x has no build in function for that!!!
  *free = 5500000000; // 5.5GByte
  *avail = *free;     // = same

  // assign return value
  return 0;

}
#ifdef __cplusplus
}
#endif

//EOF
