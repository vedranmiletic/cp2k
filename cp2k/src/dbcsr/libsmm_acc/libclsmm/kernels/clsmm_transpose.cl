#if defined (__ACC)

#if defined(cl_khr_fp64)    // NVIDIA, Intel, Khronos
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void transpose_d (__global const int    *trs_stack,
                           __global       double *mat)
{
  const int m=23;
  const int n=23;
  double buf[23*23];
  int offset = trs_stack[get_group_id(0)];
  for (int i = get_local_id(0); i < m * n; i += get_local_size(0)){
    buf[i] = mat[offset + i]; 
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = get_local_id(0); i < m * n; i += get_local_size(0)){
    int r_out = i % n;
    int c_out = i / n;
    int idx = r_out * m + c_out;
    mat[offset + i] = buf[idx];
  }

}
#endif
