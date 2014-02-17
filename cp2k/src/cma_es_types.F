!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2012  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief performs global geometry optimization
!> \par History
!> \author Ole
! *****************************************************************************
MODULE cma_es_types
  USE kinds,                           ONLY: dp
  USE parallel_rng_types,              ONLY: rng_stream_type, &
                                             create_rng_stream, &
                                             next_random_number, &
                                             reset_to_next_rng_substream, &
                                             GAUSSIAN, &
                                             UNIFORM
#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

  TYPE :: individual_type_p
    TYPE(individual_type),POINTER                  :: p
 END TYPE  
!================================================================================
 TYPE :: individual_type
   INTEGER                                         :: id
   INTEGER                                         :: lambda ! pop size
   REAL(KIND=dp),DIMENSION(:),ALLOCATABLE          :: xmean,xold,xmin,weights,zmean
   REAL(KIND=dp),DIMENSION(:),POINTER              :: multmp_vN
   REAL(KIND=dp),DIMENSION(:,:),POINTER            :: multmp_mN,multmp_mMU,multmp_mN2,mutmp
   REAL(KIND=dp),DIMENSION(:),POINTER              :: x 
   REAL(KIND=dp),DIMENSION(:),ALLOCATABLE          :: f
   REAL(KIND=dp)                                   :: sigma
   REAL(KIND=dp)                                   :: vecnorm
   REAL(KIND=dp)                                   :: vecsum
   !strategy parameter setting
   REAL(KIND=dp)                                   :: cc,cp,cs,ccov,ccovm,damps
   REAL(KIND=dp)                                   :: p_succ,p_target,p_tresh
   ! initialize dynamic (internal) startegy parameters and constants
   REAL(KIND=dp),ALLOCATABLE,DIMENSION(:)          :: pc,ps
   REAL(KIND=dp)                                   :: eigenval
   REAL(KIND=dp)                                   :: chiN
   REAL(KIND=dp)                                   :: trace
   REAL(KIND=dp)                                   :: det
   REAL(KIND=dp),DIMENSION(:,:),ALLOCATABLE        :: B,D,BD,C,triuC,arx,arz
   REAL(KIND=dp),DIMENSION(:,:),ALLOCATABLE        :: tempMat,colVec,rowVec
      !for general cma-es---------------------------------------
!   INTEGER                                         :: info
!   REAL(KIND=dp)                                   :: mucov
   INTEGER                                         :: mu ! number of parents for recomb
!   REAL(KIND=dp)                                   :: hsig
!   REAL(KIND=dp)                                   :: mueff
   REAL(KIND=dp),DIMENSION(:,:),ALLOCATABLE        :: weightsMat
   !--------------------non dominated sorting-------------------------------
   INTEGER                                         :: front
   INTEGER                                         :: d_count
   REAL(kind=dp)                                   :: I_dist
   REAL(kind=dp)                                   :: kernel_dist
   REAL(kind=dp)                                   :: hypervol
   TYPE(individual_type_p),ALLOCATABLE             :: s_i(:)
   INTEGER                                         :: s_i_count
   !--------------------------------------------------------------------------
   REAL(KIND=dp)                                   :: penalty
   REAL(KIND=dp)                                   :: rmsd
 END TYPE individual_type 
!================================================================================
 TYPE cmaes_opt_type
   INTEGER                                              :: n
   INTEGER                                              :: lambda_mo
   INTEGER                                              :: info
   INTEGER                                              :: generation
!   REAL(KIND=dp),DIMENSION(:),POINTER                   :: newsort
!   INTEGER,ALLOCATABLE                                  :: idxsel(:) 
   !--bounderies
   REAL(KIND=dp),DIMENSION(:),POINTER                   :: l_bound,u_bound,insigma
   REAL(KIND=dp),DIMENSION(:),POINTER                   :: fmax,fmin
   REAL(KIND=dp)                                        :: penalty_alpha
   !--stop criteria--------------
   REAL(KIND=dp)                                        :: stopfitness 
   REAL(KIND=dp)                                        :: stopeval
   TYPE(individual_type),ALLOCATABLE                    :: population(:)
   TYPE(individual_type_p),ALLOCATABLE                  :: population_p(:)
   TYPE(individual_type_p),ALLOCATABLE                  :: a_k(:)
   TYPE(individual_type_p),ALLOCATABLE                  :: a_k_parents(:)
   TYPE(rng_stream_type),POINTER                        :: rng_normal_stream
   TYPE(rng_stream_type),POINTER                        :: rng_uniform_stream
   TYPE(cp_error_type),POINTER                          :: error
   LOGICAL                                              :: additional_constrain
 END TYPE cmaes_opt_type
!================================================================================
 PUBLIC :: cmaes_opt_type,individual_type,individual_type_p 
END MODULE cma_es_types 
