// DBCSR_KERNEL datatype=dbcsr_type_real_8, homogeneous_only=True, m=5, n=16, k=13
#include "dbcsr_kernel.h"
#include "dbcsr_generic_kernel.h"

int
launch_stack_mm_mnk_kepler_NxNd_5_16_13_2 (int *param_stack, int stack_size,
					   cudaStream_t stream, int m_max,
					   int n_max, int k_max,
					   double *a_data, double *b_data,
					   double *c_data)
{
  int shared_size = 0;
  int careful = (stack_size / GROUPING);
  int nruns = stack_size - careful * GROUPING;
  stack_mm_mnk_kepler_NxNd < 5, 16, 13,
    2 > <<<((stack_size + GROUPING - 1) / GROUPING), 192, shared_size,
    stream >>> (param_stack, careful, nruns, m_max, n_max, k_max, a_data,
		b_data, c_data);
  return (0);
}