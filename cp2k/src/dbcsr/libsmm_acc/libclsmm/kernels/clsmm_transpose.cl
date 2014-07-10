#if defined (__ACC)

#if defined(cl_khr_fp64)    // NVIDIA, Intel, Khronos
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#if defined(cl_intel_printf)    // Intel
#pragma OPENCL EXTENSION cl_intel_printf : enable
#elif defined(cl_amd_printf)    // AMD
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

__kernel void transpose_23_23_d (__global int    *trs_stack,
                                          int    trs_offset,
                                 __global double *mat,
                                 __local  double *local_buffer){

  int m=23;
  int n=23;

  int offset = trs_stack[trs_offset + get_group_id(0)];
  for (int i = get_local_id(0); i < m * n; i += get_local_size(0)){
    local_buffer[i] = mat[offset + i]; 
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = get_local_id(0); i < m * n; i += get_local_size(0)){
    int r_out = i % n;
    int c_out = i / n;
    int idx = r_out * m + c_out;
    mat[offset + i] = local_buffer[idx];
  }
}
#endif
