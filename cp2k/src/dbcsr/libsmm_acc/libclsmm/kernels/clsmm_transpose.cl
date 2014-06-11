/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2013 the CP2K developers group                      *
 *  Author: Andreas Gloess <andreas.gloess@chem.uzh.ch>                      *
 *****************************************************************************/
#if defined (__ACC)

// OpenCL code
__kernel void transpose_d (__global int *trs_stack,
                           __global int nblks,
                           __global double* mat) {

  __local double buf[m*n];

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

//EOF
