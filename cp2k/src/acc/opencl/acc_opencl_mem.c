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
 */
int acc_dev_mem_allocate (void **dev_mem, size_t n){
  // debug info
  if (verbose_print){
    fprintf(stdout, "\n --- DEVICE MEMORY ALLOCATION --- \n");
    fprintf(stdout, " ---> Entering: acc_dev_mem_allocate.\n");
  }

  // create cl_mem buffer pointer
  *dev_mem = (void *) malloc(sizeof(cl_mem));
  cl_mem *buffer = (cl_mem *) *dev_mem;

  // get a device buffer object
  *buffer = (cl_mem) clCreateBuffer(
                       (*acc_opencl_my_device).ctx,                 // device context
                       (CL_MEM_READ_WRITE),                         // flags
                       (size_t) n,                                  // number of bytes
                       NULL,                                        // host pointer
                       &cl_error);                                  // error
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "      DEVICE buffer address:  HEX=%p INT=%ld\n", buffer, (uintptr_t) buffer);
    fprintf(stdout, " <--- Leaving: acc_dev_mem_allocate.\n");
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
  if (verbose_print){
    fprintf(stdout, "\n --- DEVICE MEMORY DEALLOCATION --- \n");
    fprintf(stdout, " ---> Entering: acc_dev_mem_deallocate.\n");
    fprintf(stdout, "      DEVICE buffer address:  HEX=%p INT=%ld\n", dev_mem, (uintptr_t) dev_mem);
  }

  // local buffer object pointer 
  cl_mem *buffer = (cl_mem *) dev_mem;

  // release device buffer object
  cl_error = clReleaseMemObject(*buffer);
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;
  free(buffer);
  buffer = NULL;

  // debug info
  if (verbose_print){
    fprintf(stdout, " <--- Leaving: acc_dev_mem_deallocate.\n");
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
 * Create a host memory pointer to memory of size 'n' bytes and an associated
 * host buffer object of 'cl_mem' type.
 *
 * Note: The allocation is prefaced by the buffer object, but only the
 *       pointer to the host_mem is given back.
 */
int acc_host_mem_allocate (void **host_mem, size_t n, void *stream){

  // debug info
  if (verbose_print){
    fprintf(stdout, "\n --- HOST MEMORY ALLOCATION --- \n");
    fprintf(stdout, " ---> Entering: acc_host_mem_allocate.\n");
  }

  cl_mem buffer = clCreateBuffer(
                   (*acc_opencl_my_device).ctx,                 // device context
                   (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR), // flags
                   (size_t) n,                                  // number of bytes
                   NULL,                                        // host pointer
                   &cl_error);                                  // error
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // local stream object and memory object pointers
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  *host_mem = (void *) clEnqueueMapBuffer(
                         (*clstream).queue,             // stream
                         buffer,                        // host buffer
                         CL_TRUE,                       // blocking
                         (CL_MAP_READ | CL_MAP_WRITE),  // flags
                         (size_t) 0,                    // offset
                         (size_t) n,                    // number of bytes
                         (cl_uint) 0,                   // num_evenets_in_wait_list
                         NULL,                          // event_wait_list
                         NULL,                          // event
                         &cl_error);                    // error
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug infos
  if (verbose_print){
    fprintf(stdout, "      HOST memory address:  HEX=%p INT=%ld\n", *host_mem, (uintptr_t) *host_mem);
    fprintf(stdout, " <--- Leaving: acc_host_mem_allocate.\n");
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
    fprintf(stdout, " ---> Entering: acc_host_mem_deallocate.\n");
    fprintf(stdout, "      HOST memory address:  HEX=%p INT=%ld\n", host_mem, (uintptr_t) host_mem);
  }

  // free the memory (Todo: we need queue to clUnmap the associated buffer)
//  free(host_mem);

  // debug info
  if (verbose_print){
    fprintf(stdout, " <--- Leaving: acc_host_mem_deallocate.\n");
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
int acc_memcpy_h2d (const void *host_mem, void *dev_mem, size_t count, void *stream){
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_memcpy_h2d.\n");

  // local buffer object pointer 
  cl_mem *buffer  = (cl_mem *) dev_mem;

  // local stream object and memory object pointers
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // copy host memory to device buffer
  cl_error = clEnqueueWriteBuffer(
               (*clstream).queue, // cl_command_queue command_queue
               *buffer,           // cl_mem           buffer
               CL_TRUE,           // cl_bool          blocking_write
               (size_t) 0,        // size_t           offset
               (size_t) count,    // size_t           cb
               host_mem,          // const            void *ptr
               (cl_uint) 0,       // cl_uint          num_evenets_in_wait_list
               NULL,              // cl_event         *event_wait_list
               NULL);             // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "Copying %d bytes from host address %p to device address %p \n",
      count, host_mem, buffer);
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
int acc_memcpy_d2h (const void *dev_mem, void *host_mem, size_t count, void *stream){
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_memcpy_d2h.\n");

  // local buffer object pointer 
  const cl_mem *buffer = (const cl_mem *) dev_mem;

  // local stream object and memory object pointers
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // copy host memory to device buffer
  cl_error = clEnqueueReadBuffer(
               (*clstream).queue, // cl_command_queue command_queue
               *buffer,           // cl_mem           buffer
               CL_TRUE,           // cl_bool          blocking_read
               (size_t) 0,        // size_t           offset
               (size_t) count,    // size_t           cb
               host_mem,          // void             *ptr
               (cl_uint) 0,       // cl_uint          num_evenets_in_wait_list
               NULL,              // cl_event         *event_wait_list
               NULL);             // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "Copying %d bytes from device address %p to host address %p\n",
      count, buffer, host_mem);
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
int acc_memcpy_d2d (const void *devmem_src, void *devmem_dst, size_t count, void *stream){
  // debug info
  if (verbose_print) fprintf(stdout, "Entering: acc_memcpy_d2d.\n");

  // local buffer object pointer 
  cl_mem *buffer_src = (cl_mem *) devmem_src;
  cl_mem *buffer_dst = (cl_mem *) devmem_dst;

  // local stream object and memory object pointers
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // copy device buffers from src to dst
  cl_error = clEnqueueCopyBuffer(
               (*clstream).queue, // cl_command_queue command_queue
               *buffer_src,       // cl_mem           src_buffer
               *buffer_dst,       // cl_mem           dst_buffer
               (size_t) 0,        // size_t           src_offset
               (size_t) 0,        // size_t           dst_offset
               (size_t) count,    // size_t           cb
               (cl_uint) 0,       // cl_uint          num_evenets_in_wait_list
               NULL,              // cl_event         *event_wait_list
               NULL);             // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;

  // debug info
  if (verbose_print){
    fprintf(stdout, "Coping %d bytes from device address %p to device address %p \n",
      count, buffer_src, buffer_dst);
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
int acc_memset_zero (void *dev_mem, size_t offset, size_t length, void *stream){
//       OpenCL 1.1 has no build in function for that!!!
//       We use cl_uchar because it's 8Bit = 1Byte long.
  // debug info
  if (verbose_print){
    fprintf(stdout, "\n --- ZERO DEVICE MEMORY --- \n");
    fprintf(stdout, " ---> Entering: acc_memset_zero.\n");
  }

  // local buffer object pointer 
  cl_mem *buffer = (cl_mem *) dev_mem;

  // local stream object pointer
  acc_opencl_stream_type *clstream = (acc_opencl_stream_type *) stream;

  // zero the values starting from offset in dev_mem
#ifdef CL_VERSION_1_2
  const cl_uchar zero = (cl_uchar) 0;

  cl_error = clEnqueueFillBuffer((*clstream).queue,
               *clmem, &zero, (size_t) sizeof(cl_uchar), (size_t) offset, (size_t) length,
               (cl_uint) 0, NULL, NULL);
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;
#else
  size_t i;
  // create a array of size 'lenght' and zero it
  cl_uchar *host_mem = (cl_uchar *) malloc(length * sizeof(cl_uchar));
  for (i = 0; i < length; i++){
    host_mem[i] = (cl_uchar) 0;
  }
  // transfer the 'zero_mem' to device buffer
  cl_error = clEnqueueWriteBuffer(
               (*clstream).queue,       // cl_command_queue command_queue
               *buffer,                 // cl_mem           buffer
               CL_TRUE,                 // cl_bool          blocking_write
               (size_t) offset,         // size_t           offset
               (size_t) length,         // size_t           cb
               (const void *) host_mem, // const            void *ptr
               (cl_uint) 0,             // cl_uint          num_event_in_wait_list
               NULL,                    // const cl_event   *event_wait_list
               NULL);                   // cl_event         *event
  if (acc_opencl_error_check(cl_error, __LINE__)) return -1;
  // free host array
  free(host_mem);
#endif

  // debug info
  if (verbose_print){
    fprintf(stdout, "     DEVICE buffer address:  HEX=%p INT=%ld\n", buffer, (uintptr_t) buffer);
    fprintf(stdout, " <-- Leaving: acc_memset_zero.\n");
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
int acc_dev_mem_info (size_t *free, size_t *avail){
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
