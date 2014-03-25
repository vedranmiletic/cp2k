!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2012  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief performs global geometry optimization
!> \par History
!> \author Ole
! *****************************************************************************
MODULE paramopt_types

  USE kinds,                           ONLY: default_string_length,&
                                             dp
  USE parallel_rng_types,              ONLY: rng_stream_type
  USE cp_para_types,                   ONLY: cp_para_env_type
  USE powell,                          ONLY: opt_state_type  
   USE cma_es_types,                    ONLY: cmaes_opt_type   
#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

 PUBLIC :: lbfgs_opt_type
 PUBLIC :: gauss_opt_type
 PUBLIC :: paramopt_master_type
 PUBLIC :: ener_ref_type 

 TYPE lbfgs_opt_type
     CHARACTER (len=60) :: task
     CHARACTER (len=60) :: csave
     LOGICAL :: lsave (4)
     INTEGER :: m, print_every, master, max_f_per_iter, status, n_iter
     INTEGER :: ref_count, id_nr, lenwa
     INTEGER, DIMENSION(:), POINTER :: kind_of_bound, i_work_array, isave
     REAL(kind=dp),POINTER :: f=>null()
     REAL(kind=dp) :: wanted_relative_f_delta, wanted_projected_gradient,&
          last_f, projected_gradient, eold, emin
     REAL(kind=dp), DIMENSION(:), POINTER :: x,lower_bound,upper_bound,&
          gradient,dsave,work_array
 END TYPE lbfgs_opt_type
 
 TYPE gauss_opt_type
   INTEGER                                              :: n,nf
   REAL(KIND=dp),DIMENSION(:),ALLOCATABLE               :: eta,m,xmin
   REAL(KIND=dp),DIMENSION(:),POINTER                   :: x 
   REAL(KIND=dp),DIMENSION(:),POINTER                   :: l_bound,u_bound
   REAL(KIND=dp),DIMENSION(:,:),ALLOCATABLE             :: C,Q
   REAL(KIND=dp)                                        :: c_t,fmin,r
   REAL(KIND=dp),POINTER                                :: f=>null()
   REAL(KIND=dp)                                        :: wm,wc,wt,beta,p,fe,fc
   CHARACTER(len=default_string_length)                 :: task
   TYPE(rng_stream_type),POINTER                        :: rng_normal_stream
 END TYPE gauss_opt_type

 TYPE paramopt_master_type
   INTEGER                                               :: iw
   INTEGER                                               :: num_inp
   INTEGER                                               :: num_calc
   INTEGER                                               :: num_ref
   INTEGER                                               :: num_var
   INTEGER                                               :: num_obj
   INTEGER                                               :: num_para_set
   INTEGER                                               :: n_walkers
   INTEGER                                               :: inp_iter
   INTEGER                                               :: run_type_id
   INTEGER                                               :: input_seed
   INTEGER,ALLOCATABLE                                   :: i_recv(:)
   TYPE(cp_error_type),POINTER                           :: error
   TYPE(ener_ref_type),POINTER                           :: ener_ref(:)
   CHARACTER(len=default_string_length),POINTER          :: input_files(:)
   CHARACTER(len=default_string_length),POINTER          :: coord_files(:)
   INTEGER,POINTER                                       :: charges(:)
   INTEGER,ALLOCATABLE                                   :: ref_num_atoms(:)
   REAL(KIND=dp),ALLOCATABLE                             :: Epot(:,:)
   INTEGER,ALLOCATABLE                                   :: converged(:,:) 
   REAL(KIND=dp),ALLOCATABLE                             :: timings(:)
   REAL(KIND=dp),POINTER                                 :: E_res(:)
   REAL(KIND=dp),POINTER                                 :: res(:)
   REAL(KIND=dp),POINTER                                 :: rmsd(:,:)
   REAL(KIND=dp),POINTER                                 :: dipole(:)
   REAL(KIND=dp),POINTER                                 :: force(:,:)
   REAL(KIND=dp),POINTER                                 :: function_matrix(:,:)
   REAL(KIND=dp),POINTER                                 :: obj_f(:)
   REAL(KIND=dp),POINTER                                 :: grad_obj_f(:)
   REAL(KIND=dp),POINTER                                 :: x(:)
   REAL(KIND=dp),POINTER                                 :: init_x(:)
   REAL(KIND=dp),POINTER                                 :: x_temp(:)
   REAL(KIND=dp),POINTER                                 :: xtrans(:)
   REAL(KIND=dp),POINTER                                 :: xtransold(:)
   REAL(KIND=dp),POINTER                                 :: xold(:)
   REAL(KIND=dp),POINTER                                 :: u_bound(:)
   REAL(KIND=dp),POINTER                                 :: l_bound(:)
   REAL(KIND=dp)                                         :: pi=2*asin(1._dp), &
                                                            kcalmol=6.27509468713739E+02_dp, &
                                                            ev=27.2116_dp,&
                                                            ev_kcal=23.06035_dp
   REAL(kind=dp),POINTER                                 :: eps(:)
   REAL(kind=dp)                                         :: eps_factor=1e-8_dp
   REAL(kind=dp)                                         :: penalty_alpha
   TYPE(cp_para_env_type), POINTER                       :: para_env
   TYPE(rng_stream_type),POINTER                         :: rng_stream
   LOGICAL,POINTER                                       :: sampling
   LOGICAL                                               :: calc_deriv,benchmark
   LOGICAL                                               :: rand_inp
   LOGICAL                                               :: multi_obj
   LOGICAL                                               :: variance_sorting
   LOGICAL                                               :: additional_constrain
   LOGICAL                                               :: restart
   REAL(KIND=dp)                                         :: step_size
   REAL(KIND=dp),POINTER                                 :: weights(:)
   TYPE(opt_state_type),POINTER                          :: powell_opt
   TYPE(gauss_opt_type),POINTER                          :: gauss_opt
   TYPE(lbfgs_opt_type),POINTER                          :: lbfgs_opt
   TYPE(cmaes_opt_type),POINTER                          :: cmaes_opt
 END TYPE paramopt_master_type

 TYPE :: ener_ref_type
   INTEGER                                               :: id
   CHARACTER(len=default_string_length)                  :: mol
   CHARACTER(len=default_string_length)                  :: mol_ref1
   CHARACTER(len=default_string_length)                  :: mol_ref2
   CHARACTER(len=default_string_length)                  :: mol_ref3
   CHARACTER(len=default_string_length)                  :: mol_ref4
   INTEGER                                               :: stoch1
   INTEGER                                               :: stoch2
   INTEGER                                               :: stoch3
   INTEGER                                               :: stoch4
   INTEGER                                               :: stoch5
   REAL(KIND=dp)                                         :: ref  
 END TYPE ener_ref_type                                               


END MODULE paramopt_types


