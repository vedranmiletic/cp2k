!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2012  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief performs covariance matrix adaptation evolutionary strategy algorithm
!> \par History
!> \author Ole?
! *****************************************************************************
MODULE cma_es
  USE kinds,                           ONLY: dp
  USE cma_es_utils,                    ONLY: mrgrnk, &
                                             tool_symmatrix, &
                                             tool_eigendecomp, &
                                             a_k_copy, &
                                             write_x
  USE cma_es_types,                    ONLY: individual_type,individual_type_p,cmaes_opt_type
  USE parallel_rng_types,              ONLY: rng_stream_type, &
                                             create_rng_stream, &
                                             next_random_number, &
                                             reset_to_next_rng_substream, &
                                             GAUSSIAN, &
                                             UNIFORM
   USE paramopt_utils,                 ONLY: deviation, &
                                             plugin
#include "cp_common_uses.h"
    implicit none
    private 
    
    CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cma_es'

    public :: cmaes,gen_pop

    contains

!-----------------------CMA-ES -----------Algorithm-----------------------------
         subroutine cmaes(x,cmaes_opt)
     
         implicit none
         type(cmaes_opt_type),target                               :: cmaes_opt
         type(individual_type),POINTER                             :: a_k
         type(individual_type),POINTER                             :: a_k_parents
         CHARACTER(len=*), PARAMETER :: routineN = 'cmaes_run', &
           routineP = moduleN//':'//routineN
         real(kind=dp),pointer,intent(inout)                       :: x(:)
         real(kind=dp)                                             :: lambda_succ
         real(kind=dp)                                             :: p_succ
         real(kind=dp)                                             :: tmp
     !    TYPE(cp_error_type),POINTER                               :: error
         INTEGER                                                   :: i,j,k,n
         INTEGER                                                   :: info
         INTEGER                                                   :: accepted
         LOGICAL                                                   :: boolean
!----------------------------set pointers to type struct-----------------------
         n = cmaes_opt%n
         lambda_succ = 0._dp
         p_succ = 0._dp
         boolean = .false.
         accepted = 0 
         call init_selection(cmaes_opt)
     !    do k=1,cmaes_opt%lambda_mo
     !        print *,'k',k,'I_dist',cmaes_opt%a_k(k)%p%I_dist,'kernel_dist',cmaes_opt%a_k(k)%p%kernel_dist
     !    enddo
         do k=1,cmaes_opt%lambda_mo

              a_k => cmaes_opt%a_k(k)%p
              a_k_parents => cmaes_opt%a_k_parents(k)%p
              call dominates(a_k,a_k_parents,boolean)
              !----yannick watch out...f now is hardcoded to only have 2 obj!!!!
              if (boolean) then
                  p_succ = 1._dp
                  accepted = accepted + 1
              else
                  p_succ = 0._dp
              endif
              !p_succ = lambda_succ/real(a_k%lambda)
              !print *,'k',p_succ,boolean
          !-------------selection and recombination (calc new xmean)---------------------------------
          !    CALL new_mean(x,cmaes_opt)
          !----------------update evolution paths ( cumulation) --------------------------------------
          !    CALL update_path(cmaes_opt)
          !---------------------Adapt covariance matrix-----------------------------
          !    CALL adapt_cov(cmaes_opt)
          !----------------------adapt sigma----------------------------------------
               CALL update_stepsize(p_succ,a_k_parents)  
               CALL update_stepsize(p_succ,a_k)
          !-----------------------elite aproach------------------------------------
                  CALL new_mean(a_k) !only for the offspring this is adpoted
                  !-----update covariance---------
                      call update_path(a_k)   
                      call adapt_cov(a_k) 
          !---------Update B and D from C-------------------------------------------
                  !-----------------------------------------------------------------------
                  !  Enforce symmetry
                  !-----------------------------------------------------------------------
                  CALL tool_symmatrix(a_k%C,cmaes_opt%n,a_k%triuC) 
                         !-----------------------------------------------------------------------
                  !  Eigen decomposition, D=diagonal matrix of eigenvalues, 
                  !  B=normalized eigenvectors
                  !-----------------------------------------------------------------------
                  CALL tool_eigendecomp(a_k%triuC,cmaes_opt%n,a_k%D,a_k%B,cmaes_opt%info)
                          !if tool_eigendecomp returned Error go to the Restart Loop
                  IF (cmaes_opt%info .NE. 0) THEN
                      print *,'error in a_k',a_k%Id,'eigendecomp failed'
                        DO i=1,cmaes_opt%n
                         WRITE(*,*) (a_k%triuC(i,j), j=1,cmaes_opt%n)
                       END DO
                      STOP
                  END IF
                  !-----------------------------------------------------------------------
                  !  D contains standard ind_counter now
                  !----------------------ind_counter---------------------------------------
                       !normalize D such that det = 1 
                       ! yannick new e.g. not sure....
                       !tmp = 1._dp
                       !do i=1,n
                       !tmp = tmp*a_k%D(i,i)
                       !enddo
                       !a_k%D=a_k%D*1._dp/tmp**(1._dp/n)
                  !     !---
                    if (any(a_k%D .lt. 0.)) then
                        print *,'diagonal matrix wrong'
                        STOP
                    endif
                    a_k%D = sqrt(a_k%D)
                  CALL DGEMM('N','N',n,n,n,1.0d0,a_k%B,n,a_k%D,n,0.0d0,a_k%BD,n)
                  !---recover C???-----------
                  CALL DGEMM('N','T',n,n,n,1.0d0,a_k%BD,n,a_k%BD,n,0.0d0,a_k%C,n)
                  !-------get trace and determinant
                  a_k%trace = 0.0_dp
                  a_k%det = 1.0_dp
                  DO i = 1,n    
                      a_k%trace = a_k%trace+a_k%D(i,i)**2
                      a_k%det = a_k%det*a_k%D(i,i)**2
                  END DO    
                  !open(999,file='bd_matrix.out',position='append')
                  !write(999,*) cmaes_opt%BD
                  !close(999)
              !print *,'a_k',a_k%front,a_k%I_dist,a_k%sigma
              !print *,'a_k_parents',a_k_parents%front,a_k_parents%I_dist,a_k_parents%sigma
              nullify(a_k,a_k_parents)
         enddo 
         
         call selection(cmaes_opt)
         !-------------------------------------------------------------------------
         ! numerical error management
         !-------------------------------------------------------------------------
         !CALL err_mng(cmaes_opt%population,cmaes_opt%n)
!----------------------okay?----------------------
         do k = 1, 2*cmaes_opt%lambda_mo
         a_k => cmaes_opt%population(k)
         call cp_assert( a_k%id .ne. 0 ,cp_failure_level,cp_assertion_failed,&
                            routineP, cp_to_string(k)//" id is wrong...something is strange ", &
                            cmaes_opt%error) 
         enddo
!--------output---------------------------------------------------------------
         call write_x(cmaes_opt)
         print *,'write x done'
         !=============yannick I'm not sure if that should happen here....
!----for 1:lambda_mo generate new population----------------------------------
         !print *,"generating new population"
         DO k = 1, cmaes_opt%lambda_mo
             a_k => cmaes_opt%a_k(k)%p
             a_k_parents => cmaes_opt%a_k_parents(k)%p
             a_k_parents%id = k+cmaes_opt%lambda_mo
             call a_k_copy(a_k,a_k_parents)
             !cmaes_opt%a_k(k) = cmaes_opt%a_k_parents(k)
             !print *,'old f',cmaes_opt%a_k_parents(k)%f
             CALL gen_pop(a_k,cmaes_opt)
             a_k%id = k
             !----this is just a random x !!!----------
             x = a_k%arx(:,1)
             !print *,'x should not be the same!',a_k_parents%arx(1,1),a_k%arx(1,1)
         !print *,'selected',cmaes_opt%a_k(k)%f,cmaes_opt%a_k_parents(k)%f,cmaes_opt%a_k(k)%s_iu
         END DO
         nullify(a_k,a_k_parents)
         print *,'===========================MO-CMAES-SUMMARY============================'
         print *,'number of accepted offspring',accepted
     
     END SUBROUTINE
!===========================================================================
    subroutine new_mean(a_k)
    implicit none
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i
!-------------selection and recombination (calc new xmean)---------------------------------
    n = size(a_k%xmean)
    a_k%xold = a_k%xmean
!    DO i = 1, a_k%mu   ! Extract mu best columns of arx and save it to tempMat
!        a_k%tempMat(:,i) = a_k%arx(:,a_k%idxsel(i))
!    END DO
    !-------(1+1) selection-------------
    a_k%tempMat(:,1) = a_k%arx(:,1)
    CALL dgemv('N',n,a_k%mu,1.0d0,a_k%tempMat,n,a_k%weights,1,0.0d0,a_k%xmean,1)
    ! update x to the best individual in population
    !x = a_k%tempMat(:,1)
    !print *,'this is new mean'
    !print *,'=================================================================='
    !print *,cmaes_opt%xmean
    !print *,'x matrix'
    !print *,cmaes_opt%tempMat
    !print *,'-------------------------------------'
    ! do same for zmean/arz
!    DO i = 1, a_k%mu   ! Extract mu best columns of arx and save it to tempMat
!        a_k%tempMat(:,i) = a_k%arz(:,a_k%idxsel(i))
!    END DO
    a_k%tempMat(:,1) = a_k%arz(:,1)
    CALL dgemv('N',n,a_k%mu,1.0d0,a_k%tempMat,n,a_k%weights,1,0.0d0,a_k%zmean,1)
    if (any(a_k%arz(:,1) .ne. a_k%zmean)) stop "shit happends"
    end subroutine new_mean
!===========================================================================
    subroutine update_path(a_k)
    implicit none
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i
    n = size(a_k%xmean)
!    CALL dgemv('N',n,n,1.0d0,a_k%B,n,a_k%zmean,1,0.0d0,a_k%multmp_vN,1)
!    cmaes_opt%ps = (1._dp-cmaes_opt%cs)*cmaes_opt%ps + (sqrt(cmaes_opt%cs*(2._dp-cmaes_opt%cs)* &
!                    cmaes_opt%mueff)) * cmaes_opt%multmp_vN     ! Eq.40 
                                                                ! in Hansen:2007
!      IF(sqrt(sum(cmaes_opt%ps**2))/sqrt(1._dp-(1.-cmaes_opt%cs)**(2._dp*real((countIter-lastRestart))))/chiN < &
!        1.4_MK + 2._MK/real(input%N+1)) THEN
!        cmaes_opt%hsig = 1._dp
!      ELSE
!        hsig = 0._MK
!      END IF
       if (a_k%p_succ .lt. a_k%p_tresh) then
       a_k%pc = (1._dp - a_k%cc)*a_k%pc + sqrt(a_k%cc*(2_dp-a_k%cc))* &
                        (a_k%xmean-a_k%xold)/a_k%sigma
       else
       a_k%pc = (1._dp - a_k%cc)*a_k%pc
       endif
                                  
      !postcondition
      if (any(isnan(a_k%pc))) then
             print *,'pc is NaN'
      endif 
!       cmaes_opt%pc = (1-cmaes_opt%cc)*cmaes_opt%pc + cmaes_opt%hsig* &
!                     (sqrt(cmaes_opt%cc*(2._dp-cmaes_opt%cc)*cmaes_opt%mueff)/cmaes_opt%sigma)* &
!                     (cmaes_opt%xmean-cmaes_opt%xold) !hsig = 1 check above eq
    end subroutine 
!===========================================================================
    subroutine update_stepsize(p_succ,a_k)
    implicit none
    type(individual_type),POINTER                             :: a_k
    REAL(kind=dp)                                             :: p_succ
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i
    
    a_k%p_succ = (1._dp-a_k%cp)*a_k%p_succ + a_k%cp*p_succ 
    a_k%sigma = a_k%sigma*exp((a_k%p_succ - a_k%p_target)/ &
                         (a_k%damps*(1._dp-a_k%p_target)))
!    cmaes_opt%sigma =  cmaes_opt%sigma*exp((sqrt(sum(cmaes_opt%ps**2))/ &
!                         cmaes_opt%chiN-1._dp)*cmaes_opt%cs/cmaes_opt%damps)    ! Eq.41 

    end subroutine
!===========================================================================
    subroutine adapt_cov(a_k)
    implicit none
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i
    real(kind=dp),dimension(:),allocatable                    :: bdz

    n = size(a_k%xmean)
    allocate(bdz(n))
      DO i = 1,n 
        a_k%colVec(i,1) = a_k%pc(i)
        a_k%rowVec(1,i) = a_k%pc(i)
      END DO
      a_k%weightsMat = 0.
      DO i = 1, a_k%mu
          a_k%tempMat(:,i) = a_k%xold(:)
          a_k%weightsMat(i,i) = a_k%weights(i)
      END DO   
      !a_k%mutmp = (a_k%arx(:,int(a_k%idxsel(1:a_k%mu))) - a_k%tempMat) 
      a_k%mutmp = (a_k%arx - a_k%tempMat) 
      CALL dgemm('N','T',n,n,1,1.0d0,a_k%colVec,n,a_k%rowVec,n,0.0d0,a_k%multmp_mN,n) 
      !------------calculate BDz and matrix for active covariance control--------------------
      CALL dgemv('N',n,n,1.0d0,a_k%BD,n,a_k%arz(:,1),1,0.0d0,bdz,1)
      DO i = 1,n 
        a_k%colVec(i,1) = bdz(i)
        a_k%rowVec(1,i) = bdz(i)
      END DO
      CALL dgemm('N','T',n,n,1,1.0d0,a_k%colVec,n,a_k%rowVec,n,0.0d0,a_k%multmp_mN2,n) 
      !---------------------------------------------------------------------------------------
!      CALL dgemm('N','T',a_k%mu,n,a_k%mu,1.0d0,a_k%weightsMat, &
!                 a_k%mu,a_k%mutmp,n,0.0d0,a_k%multmp_mMU,a_k%mu)
!      CALL dgemm('N','N',n,n,a_k%mu,1.0d0,a_k%mutmp,n,&
!                 a_k%multmp_mMU,a_k%mu,0.0d0,a_k%multmp_mN2,n)
!      cmaes_opt%C = (1._dp-cmaes_opt%ccov+(1._dp-cmaes_opt%hsig)*cmaes_opt%ccov* &
!                     cmaes_opt%cc*(2._dp-cmaes_opt%cc)/cmaes_opt%mucov)*cmaes_opt%C & ! old matrix
!                   + cmaes_opt%ccov*(1._dp/cmaes_opt%mucov)*cmaes_opt%multmp_mN &  ! plus rank one update
!                   + cmaes_opt%ccov*(1._dp-1._dp/cmaes_opt%mucov) &  ! plus rank mu update
!                   *cmaes_opt%sigma**(-2) * cmaes_opt%multmp_mN2
      !--------------update matrix------------------------------------------------------------
      if (a_k%p_succ .lt. a_k%p_tresh) then
          a_k%C = (1._dp-a_k%ccov)*a_k%C & ! old matrix
                   + a_k%ccov*(a_k%multmp_mN) ! plus rank one update
      else
          a_k%C = (1._dp-a_k%ccov)*a_k%C & ! old matrix
                   + a_k%ccov*(a_k%multmp_mN &  ! plus rank one update
                   + a_k%ccov*(2._dp-a_k%cc)*a_k%cc*a_k%C)
      !!----------------------------------------------------------------------------------------
      ! yannick needs to be tested!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
      !!----------------this should be active covariance control---------------------------------
      !a_k%C = (1._dp-a_k%ccovm)*a_k%C & ! old matrix
      !             - a_k%ccovm*(a_k%multmp_mN2) ! plus rank one update
      endif

      !if(any(isNaN(a_k%C))) then
      !  print *,'yannick Covariance Matrix is NaN'
      !  STOP
      !endif
      deallocate(bdz)

    end subroutine
!===========================================================================
    SUBROUTINE pareto_front(input_set,nondominated_set)
         IMPLICIT NONE
         TYPE(individual_type_p),DIMENSION(:)  :: input_set
         TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE :: nondominated_Set
         TYPE(individual_type),POINTER                    :: a_k,a_k2
         INTEGER                                     :: nondominated_size
         INTEGER                                     :: popsize
         INTEGER                                     :: i,j,k,t 
         INTEGER                                     :: numrows,numcols
         numrows=size(input_set)
         numcols=size(input_set(1)%p%f) 
         ! calculate fron with dominated count d_count and sub_set dominated s_i 
         do i =1,numrows 
            a_k => input_set(i)%p
            k = 1
            do j = 1,numrows
                a_k2 => input_set(j)%p
                if (all(a_k%f .lt. a_k2%f)) then
                    a_k%s_i(k)%p => a_k2
                    a_k%s_i_count = a_k%s_i_count + 1
                    k = k +1
                else if (all(a_k%f .gt. a_k2%f)) then
                    a_k%d_count = a_k%d_count + 1
                endif

               ! if ((a_k%f(1) .lt. a_k2%f(1)) .and. &
               !     (a_k%f(2) .lt. a_k2%f(2)) ) then
               !     a_k%s_i(k)%p => a_k2
               !     a_k%s_i_count = a_k%s_i_count + 1
               !     k = k +1
               ! else if (a_k%f(1) .gt. a_k2%f(1) .and. &
               !          a_k%f(2) .gt. a_k2%f(2) ) then
               !    a_k%d_count = a_k%d_count + 1
               ! endif
            enddo 
         enddo
         nondominated_size = 0
         do i =1,numrows
            a_k => input_set(i)%p
            if (a_k%d_count .eq. 0) &
                nondominated_Size = nondominated_size + 1
            a_k%front = a_k%d_count + 1
            if (a_k%s_i_count .gt. numrows) STOP "s_i count not possible"
            if (a_k%d_count .gt. numrows) STOP "d count not possible"
        enddo
        allocate(nondominated_set(nondominated_size))
        k=1
        do i =1,numrows
            a_k => input_set(i)%p
            if (a_k%d_count .eq. 0) then 
            nondominated_set(k)%p => a_k             
            k = k+1
            endif
           ! print *,a_k%id,'I am in pareto',a_k%d_count + 1 
           ! print *,"----------------------------",a_k%s_i_count,a_k%f
           ! do j=1,a_k%s_i_count; print *,'front set',a_k%s_i(j)%p%id; enddo
        enddo
        print *,'done with pareto_front'
        END SUBROUTINE pareto_front
!===========================================================================
    subroutine init_selection(cmaes_opt)
    implicit none
    type(cmaes_opt_type),TARGET                          :: cmaes_opt
    !------------------------NASG II---------selection------------------------
    TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE     :: new_set 
    TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE     :: sel_set 
    TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE     :: tmp_set 
    INTEGER                                                   :: offspring
    INTEGER                                                   :: tmp_off
    INTEGER                                                   :: sel_target
    REAL(kind=dp),DIMENSION(:,:),ALLOCATABLE    :: newsort
    REAL(kind=dp),DIMENSION(:),ALLOCATABLE      :: tmp_array
    INTEGER,DIMENSION(:),ALLOCATABLE            :: idxsel
    INTEGER                                     :: i,j,k
!------------------------selection here---------------------------------------
    offspring = 0
    tmp_off = 0
    if (.not. allocated(tmp_set)) allocate(tmp_set(2*cmaes_opt%lambda_mo))
    do i = 1,2*cmaes_opt%lambda_mo
    ! reset all election varaibles
    cmaes_opt%population(i)%d_count=0
    cmaes_opt%population(i)%I_dist=0._dp
    cmaes_opt%population(i)%kernel_dist=0._dp
    cmaes_opt%population(i)%hypervol=0._dp
    do j =1,cmaes_opt%lambda_mo;cmaes_opt%population(i)%s_i(j)%p=>null();enddo
    cmaes_opt%population(i)%s_i_count=0
    cmaes_opt%population(i)%front=0
    !----------------------------------------
    !   if (i .le. cmaes_opt%lambda_mo) then
    !       tmp_set(i)%p => cmaes_opt%a_k(i)%p
    !   else
    !       tmp_set(i)%p => cmaes_opt%a_k_parents(i-cmaes_opt%lambda_mo)%p
    !   endif
    tmp_set(i)%p => cmaes_opt%population(i)
!       print *,'before pareto front',tmp_set(i)%p%id
    enddo
!    do i = 1,cmaes_opt%lambda_mo
!       print *,'a_k',cmaes_opt%a_k(i)%p%id,'a_k_parents',cmaes_opt%a_k_parents(i)%p%id
!    enddo
    !---------find pareto front and add to Q (tmp_set)----------------------------------
    CALL pareto_front(input_set=tmp_set,nondominated_set=new_set)
!    print *,'=========================summary========================='
!    do i=1,2*cmaes_opt%lambda_mo
!        print * ,'id',i,'fronts',cmaes_opt%population(i)%front,'s_i_count',cmaes_opt%population(i)%s_i_count
!        do j = 1,cmaes_opt%population(i)%s_i_count; print *,'s_i',cmaes_opt%population(i)%s_i(j)%p%id; enddo
!    enddo
    CALL crowding_distance(new_set,cmaes_opt%variance_sorting)
    !CALL kernel_distance(new_set,cmaes_opt%error)
          !====ad front to Q
          do i = 1,size(new_set)
             new_set(i)%p%front = 1
             !call a_k_copy(cmaes_opt%a_k_temp(i),new_set(i)%p)
          enddo
          tmp_off = tmp_off + size(new_set)
          offspring = offspring + size(new_set)
          !print *,'first front done',offspring,'offspring'
     !---------add more fronts to Q-----------------
     j = 1   !this is also the number of fronts! 
     do while (tmp_off .lt. 2*cmaes_opt%lambda_mo) 
        ! -------copy front to selection set 
        if (allocated(sel_set)) deallocate(sel_set)
        allocate(sel_set(size(new_set)))
        do i = 1, size(new_set)
            sel_set(i)%p => new_set(i)%p
        enddo
!        print *,'size',size(tmp_set),size(sel_set)
        CALL new_front(tmp_set,sel_set,new_set)
        CALL crowding_distance(new_set,cmaes_opt%variance_sorting)
        !CALL kernel_distance(new_set,cmaes_opt%error)
        !print *,'size new set', size(new_set)
        tmp_off = tmp_off + size(new_set)
             do i = 1,size(new_set)
             ! -----ad to Q
             !cmaes_opt%a_k_temp(offspring + i) = new_set(i)%p
             new_set(i)%p%front = j + 1
             !call a_k_copy(cmaes_opt%a_k_temp(offspring+i),new_set(i)%p)
             enddo
             j = j + 1
             offspring = offspring + size(new_set)
          !   print *,'offspring',offspring
         !print *,'front',j,'done'
     enddo

    print *,'init_selection done'
    !-------there should be everythin in a_k and a_k_parents
    end subroutine init_selection
!===========================================================================
    subroutine selection(cmaes_opt)
    implicit none
    type(cmaes_opt_type),TARGET                               :: cmaes_opt
    TYPE(individual_type_p),DIMENSION(:),POINTER              :: tmp_set 
    TYPE(individual_type_p),DIMENSION(:),POINTER              :: sel_set
    CHARACTER(len=*), PARAMETER :: routineN = 'paramopt_selection', &
      routineP = moduleN//':'//routineN
    REAL(kind=dp),DIMENSION(:),ALLOCATABLE                    :: newsort
    REAL(kind=dp)                                             :: rnum
    integer                                                   :: irand
    integer,dimension(:),ALLOCATABLE                          :: tabu_list
    INTEGER,DIMENSION(:),ALLOCATABLE                          :: idxsel 
    integer                                                   :: idx
    integer                                                   :: i,j,k,n
    integer                                                   :: fsize
    integer                                                   :: psize
    integer                                                   :: front
    type(individual_type),POINTER                             :: a_k
    !--------everything could be done more efficient with pointers!!!--------------------
    k=1
    front = 1
    psize = 0
    allocate(sel_set(cmaes_opt%lambda_mo))
    do while (k .le. cmaes_opt%lambda_mo)
        !-------get front size-------------
        fsize = 0 
        do i=1,2*cmaes_opt%lambda_mo
          if (front .eq. cmaes_opt%population(i)%front) then
          fsize = fsize + 1
          endif
        enddo
        !-------------front size lower of what needed?----------
        if (fsize+psize .le. cmaes_opt%lambda_mo) then
        ! add all individual from front to Q    
           do i=1,2*cmaes_opt%lambda_mo
             if (front .eq. cmaes_opt%population(i)%front) then
             sel_set(k)%p => cmaes_opt%population(i)
             k = k + 1
             endif
           enddo
           psize = psize + fsize
        else
    !----------------------------------------          
          ! check crowding distance and add n best
          allocate(tmp_set(fsize))
             !----add to temporary set
             allocate(newsort(fsize))
             allocate(idxsel(fsize))
             j = 1
             do i=1,2*cmaes_opt%lambda_mo
               if (front .eq. cmaes_opt%population(i)%front) then
               tmp_set(j)%p => cmaes_opt%population(i)
               newsort(j) = cmaes_opt%population(i)%I_dist
               j = j + 1
               endif
             enddo
           !----sort according to I_dist-----------
           CALL mrgrnk(newsort,idxsel)
           !----add to sel set Q------------
!                           ! add random points besides borders
!                           allocate(tabu_list(cmaes_opt%lambda_mo-psize))
!                           tabu_list = 0
!                           do j = 1,2 ! add borders
!                               idx = size(idxsel)+1-j
!                               tabu_list(j) = idxsel(j)
!                               sel_set(k)%p => tmp_set(idxsel(j))%p
!                               k = k + 1
!                           enddo
!                        ! add random points
!                           j = j + 1
!                           rnum = next_random_number(cmaes_opt%rng_uniform_stream, error=cmaes_opt%error)
!                           irand=int(rnum*fsize)+1
!                           do i = 1,cmaes_opt%lambda_mo-psize-2
!                               do while (any(irand .eq. tabu_list))
!                               rnum = next_random_number(cmaes_opt%rng_uniform_stream, error=cmaes_opt%error)
!                               irand=int(rnum*fsize)+1
!                               enddo
!                               tabu_list(j) = irand
!                               j = j + 1
!                               sel_set(k)%p => tmp_set(irand)%p
!                               k = k + 1
!                           enddo
!                           deallocate(tabu_list,tmp_set)
           !-------------------------------------------
           do i = 1,cmaes_opt%lambda_mo-psize
               !---yannick depend what sorting we want
               ! smallest is best  
               ! 1) kernel density 
               idx = i
               ! biggest is best
               ! 2) crowding distance
               !idx = size(idxsel)+1-i
               sel_set(k)%p => tmp_set(idxsel(idx))%p
               k = k + 1
           enddo
           psize = psize + i
           deallocate(tmp_set,newsort,idxsel)

    !----------------------------------------          
        endif 
        front = front + 1 
    enddo
    !----------------------set new pointers-----------------------
    k=1
    n=1
    population_loop: do i=1,size(cmaes_opt%population_p) !cycle over population
       sel_set_loop: do j=1,cmaes_opt%lambda_mo 
         if (associated(cmaes_opt%population_p(i)%p,sel_set(j)%p)) then
             cmaes_opt%a_k_parents(n)%p => sel_set(j)%p
             n=n+1
             cycle population_loop
         endif
       enddo sel_set_loop
       cmaes_opt%a_k(k)%p => cmaes_opt%population_p(i)%p
       k = k + 1
    enddo population_loop
    !----------------------was selection okay?----------------------
    call cp_assert( k-1 .eq. cmaes_opt%lambda_mo,cp_failure_level,cp_assertion_failed,&
                       routineP, cp_to_string(k)//" k ne lambda_mo "//&
                       cp_to_string(cmaes_opt%lambda_mo),cmaes_opt%error)
    call cp_assert( n-1 .eq. cmaes_opt%lambda_mo,cp_failure_level,cp_assertion_failed,&
                       routineP, cp_to_string(n)//" n ne lambda_mo "//&
                       cp_to_string(cmaes_opt%lambda_mo),cmaes_opt%error)
    deallocate(sel_set)
    print *,'selection done'
    end subroutine selection
!===========================================================================
    subroutine dominates(a_k,a_k_parents,boolean)
    type(individual_type)                             :: a_k
    type(individual_type)                             :: a_k_parents
    LOGICAL                                           :: boolean
    IF (a_k%front .lt. a_k_parents%front) then
        boolean = .true. 
    else if (a_k%front .eq. a_k_parents%front) then
            if (a_k%I_dist .lt. a_k_parents%I_dist) then
                boolean = .true.
            else
                boolean = .false.
            endif
    else
        boolean = .false.   
    endif 
    end subroutine dominates
!===========================================================================
    subroutine gen_pop(a_k,cmaes_opt)
    implicit none
    type(cmaes_opt_type),target                               :: cmaes_opt
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i,j
    REAL(kind=dp)                                             :: alpha
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp
    REAL(kind=dp)                                             :: tmp2

    ! input:
    ! n, lambda, BD, xmean, sigma
    ! output:
    ! arx,arz
    n = cmaes_opt%n
    allocate(tmp(n))
    tmp=0._dp
    alpha = cmaes_opt%penalty_alpha
    !CALL reset_to_next_rng_substream(cmaes_opt%rng_normal_stream, error=error)
    do j=1,a_k%lambda
        do i=1,n; a_k%arz(i,j) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
    CALL DGEMV('N',N,N,1.0d0,a_k%BD,N,a_k%arz(:,j),1,0.0d0,a_k%multmp_vN,1)
    a_k%arx(:,j) = a_k%xmean + a_k%sigma * a_k%multmp_vN
    if (cmaes_opt%generation .eq. 1) a_k%arx(:,j) = a_k%xmean
    ! check if x in boundaries...set to bound limit otherwiese
            do i=1,n
                    if (a_k%arx(i,j) .lt. cmaes_opt%l_bound(i)) then
                        tmp(i) = abs(a_k%arx(i,j) - cmaes_opt%l_bound(i))
                        a_k%arx(i,j) = cmaes_opt%l_bound(i)
                    endif
                    if (a_k%arx(i,j) .gt. cmaes_opt%u_bound(i)) then
                        tmp(i) = abs(a_k%arx(i,j) - cmaes_opt%u_bound(i))
                        a_k%arx(i,j) = cmaes_opt%u_bound(i)
                    endif
            enddo
            !-----------yannick----Gss and gpp sepcial conditions!-------------
            ! constrains on variables go here ---------------------------------
        if (cmaes_opt%additional_constrain) then 
          ! for scp-------------
          if (n .eq. 26) then
            ! Gss gibber then gsp
          !  if (a_k%arx(11,j) .lt. a_k%arx(12,j))  then
          !      tmp(1) = tmp(1) +  a_k%arx(12,j) - a_k%arx(11,j) 
          !      tmp2 = a_k%arx(11,j)
          !      a_k%arx(11,j) = a_k%arx(12,j) 
          !      a_k%arx(12,j) = tmp2
          !  endif
            ! Gpp bigger then gp2
            if (a_k%arx(13,j) .lt. a_k%arx(14,j))  then
                tmp(1) = tmp(1) +  a_k%arx(14,j) - a_k%arx(13,j) 
                tmp2 = a_k%arx(13,j)
                a_k%arx(13,j) = a_k%arx(14,j) 
                a_k%arx(14,j) = tmp2
            endif
            ! Uss lower then upp
            if (a_k%arx(17,j) .lt. a_k%arx(16,j))  then
                tmp(1) = tmp(1) +  a_k%arx(16,j) - a_k%arx(17,j) 
                tmp2 = a_k%arx(16,j)
                a_k%arx(16,j) = a_k%arx(17,j) 
                a_k%arx(17,j) = tmp2
            endif
            ! beta_s lower than beta_p
            if (a_k%arx(19,j) .lt. a_k%arx(18,j))  then
                tmp(1) = tmp(1) +  abs(a_k%arx(18,j) - a_k%arx(19,j))
                tmp2 = a_k%arx(18,j)
                a_k%arx(18,j) = a_k%arx(19,j) 
                a_k%arx(19,j) = tmp2
            endif
            ! sigma should be bigger than gss
         !   if (a_k%arx(7,j) .lt. a_k%arx(1,j))  then
         !       tmp(1) = tmp(1) +  a_k%arx(7,j) - a_k%arx(1,j) 
         !       tmp2 = a_k%arx(7,j)
         !       a_k%arx(7,j) = a_k%arx(1,j) 
         !       a_k%arx(1,j) = tmp2
         !   endif
            ! sigma should be bigger than gss
         !   if (a_k%arx(23,j) .lt. a_k%arx(11,j))  then
         !       tmp(1) = tmp(1) +  a_k%arx(23,j) - a_k%arx(11,j) 
         !       tmp2 = a_k%arx(23,j)
         !       a_k%arx(23,j) = a_k%arx(11,j) 
         !       a_k%arx(11,j) = tmp2
         !   endif

            ! exponents ??
            if (a_k%arx(21,j) .lt. a_k%arx(22,j))  then
                tmp(1) = tmp(1) +  a_k%arx(22,j) - a_k%arx(21,j) 
                tmp2 = a_k%arx(21,j)
                a_k%arx(21,j) = a_k%arx(22,j) 
                a_k%arx(22,j) = tmp2
            endif
            ! exponents ??
            if (a_k%arx(5,j) .lt. a_k%arx(6,j))  then
                tmp(1) = tmp(1) +  a_k%arx(6,j) - a_k%arx(5,j) 
                tmp2 = a_k%arx(5,j)
                a_k%arx(5,j) = a_k%arx(6,j) 
                a_k%arx(6,j) = tmp2
            endif
        ! without scp-------------------------------------------    
      else if (n .eq. 22) then
            ! Gss gibber then gsp
         !   if (a_k%arx( 9,j) .lt. a_k%arx(10,j))  then
         !       tmp(1) = tmp(1) +  a_k%arx(10,j) - a_k%arx( 9,j) 
         !       tmp2 = a_k%arx( 9,j)
         !       a_k%arx( 9,j) = a_k%arx(10,j) 
         !       a_k%arx(10,j) = tmp2
         !   endif
            ! Gpp bigger then gp2
            if (a_k%arx(11,j) .lt. a_k%arx(12,j))  then
                tmp(1) = tmp(1) +  a_k%arx(12,j) - a_k%arx(11,j) 
                tmp2 = a_k%arx(11,j)
                a_k%arx(11,j) = a_k%arx(12,j) 
                a_k%arx(12,j) = tmp2
            endif
            ! Uss lower then upp
            if (a_k%arx(15,j) .lt. a_k%arx(14,j))  then
                tmp(1) = tmp(1) +  a_k%arx(14,j) - a_k%arx(15,j) 
                tmp2 = a_k%arx(14,j)
                a_k%arx(14,j) = a_k%arx(15,j) 
                a_k%arx(15,j) = tmp2
            endif
            ! beta_s lower than beta_p
            if (a_k%arx(17,j) .lt. a_k%arx(16,j))  then
                tmp(1) = tmp(1) +  abs(a_k%arx(16,j) - a_k%arx(17,j))
                tmp2 = a_k%arx(16,j)
                a_k%arx(16,j) = a_k%arx(17,j) 
                a_k%arx(17,j) = tmp2
            endif
            ! sigma should be bigger than gss
         !   if (a_k%arx(7,j) .lt. a_k%arx(1,j))  then
         !       tmp(1) = tmp(1) +  a_k%arx(7,j) - a_k%arx(1,j) 
         !       tmp2 = a_k%arx(7,j)
         !       a_k%arx(7,j) = a_k%arx(1,j) 
         !       a_k%arx(1,j) = tmp2
         !   endif
         !   ! sigma should be bigger than gss
         !   if (a_k%arx(21,j) .lt. a_k%arx( 9,j))  then
         !       tmp(1) = tmp(1) +  a_k%arx(21,j) - a_k%arx( 9,j) 
         !       tmp2 = a_k%arx(21,j)
         !       a_k%arx(21,j) = a_k%arx( 9,j) 
         !       a_k%arx( 9,j) = tmp2
         !   endif

            ! exponents ??
            if (a_k%arx(19,j) .lt. a_k%arx(20,j))  then
                tmp(1) = tmp(1) +  a_k%arx(20,j) - a_k%arx(19,j) 
                tmp2 = a_k%arx(19,j)
                a_k%arx(19,j) = a_k%arx(20,j) 
                a_k%arx(20,j) = tmp2
            endif
            ! exponents ??
            if (a_k%arx(5,j) .lt. a_k%arx(6,j))  then
                tmp(1) = tmp(1) +  a_k%arx(6,j) - a_k%arx(5,j) 
                tmp2 = a_k%arx(5,j)
                a_k%arx(5,j) = a_k%arx(6,j) 
                a_k%arx(6,j) = tmp2
            endif
      else if (n .eq. 20) then
            ! Uss lower then upp
            if (a_k%arx(11,j) .lt. a_k%arx(10,j))  then
                tmp(1) = tmp(1) +  a_k%arx(10,j) - a_k%arx(11,j) 
                tmp2 = a_k%arx(10,j)
                a_k%arx(10,j) = a_k%arx(11,j) 
                a_k%arx(11,j) = tmp2
            endif
            ! beta_s lower than beta_p
            if (a_k%arx(13,j) .lt. a_k%arx(12,j))  then
                tmp(1) = tmp(1) +  abs(a_k%arx(12,j) - a_k%arx(13,j))
                tmp2 = a_k%arx(12,j)
                a_k%arx(12,j) = a_k%arx(13,j) 
                a_k%arx(13,j) = tmp2
            endif
            ! sigma should be bigger than gss
           ! if (a_k%arx(7,j) .lt. a_k%arx(1,j))  then
           !     tmp(1) = tmp(1) +  a_k%arx(7,j) - a_k%arx(1,j) 
           !     tmp2 = a_k%arx(7,j)
           !     a_k%arx(7,j) = a_k%arx(1,j) 
           !     a_k%arx(1,j) = tmp2
           ! endif
            ! sigma should be bigger than gss
           ! if (a_k%arx(21,j) .lt. a_k%arx( 9,j))  then
           !     tmp(1) = tmp(1) +  a_k%arx(21,j) - a_k%arx( 9,j) 
           !     tmp2 = a_k%arx(21,j)
           !     a_k%arx(21,j) = a_k%arx( 9,j) 
           !     a_k%arx( 9,j) = tmp2
           ! endif

            ! exponents ??
            if (a_k%arx(15,j) .lt. a_k%arx(16,j))  then
                tmp(1) = tmp(1) +  a_k%arx(16,j) - a_k%arx(15,j) 
                tmp2 = a_k%arx(15,j)
                a_k%arx(15,j) = a_k%arx(16,j) 
                a_k%arx(16,j) = tmp2
            endif
            ! exponents ??
            if (a_k%arx(4,j) .lt. a_k%arx(5,j))  then
                tmp(1) = tmp(1) +  a_k%arx(5,j) - a_k%arx(4,j) 
                tmp2 = a_k%arx(4,j)
                a_k%arx(4,j) = a_k%arx(5,j) 
                a_k%arx(5,j) = tmp2
            endif

           endif
         endif
    enddo
    !---yannick this is stupid
    !print *,'new generating x...'
    !do k=1,a_k%lambda
    !    print *,a_k%arx(:,k)
    !    print *,a_k%arz(:,k)
    !enddo
    a_k%penalty = alpha*norm2(tmp)**2
    deallocate(tmp)
    end subroutine gen_pop

     SUBROUTINE crowding_distance(input_set,variance_sorting)
         IMPLICIT NONE
         TYPE(individual_type_p),DIMENSION(:)        :: input_Set
         REAL(kind=dp),DIMENSION(:,:),ALLOCATABLE    :: newsort
         REAL(kind=dp),DIMENSION(:),POINTER          :: fmax,fmin
         REAL(kind=dp),DIMENSION(:),ALLOCATABLE      :: maxvals,minvals
         REAL(kind=dp),DIMENSION(:),ALLOCATABLE      :: tmp
         REAL(kind=dp),DIMENSION(:),ALLOCATABLE      :: mydeviation,myIdist,maxIdist
         REAL(kind=dp)                               :: mymax,mymin,mydev
         INTEGER,DIMENSION(:),ALLOCATABLE            :: idxsel
         INTEGER                                     :: nondominated_size
         INTEGER                                     :: numcols
         INTEGER                                     :: sel_target
         INTEGER                                     :: i,j,k,n 
         INTEGER                                     :: j_m,j_p
         LOGICAL                                     :: variance_sorting
         !--------kernel density estim----------
         REAL(kind=dp),DIMENSION(:),ALLOCATABLE :: x,f
         REAL(kind=dp)                          :: h
         INTEGER                                :: m
         nondominated_size=size(input_set)
         numcols=size(input_set(1)%p%f)
         do i = 1,nondominated_size
             input_set(i)%p%I_dist = 0._dp
         enddo
         !------------------------------------------------
         allocate(newsort(nondominated_size,numcols))
         allocate(idxsel(nondominated_size))
         allocate(x(nondominated_size))
         allocate(f(nondominated_size))
!         allocate(myIdist(nondominated_size))
!         allocate(maxIdist(nondominated_size))
         ! crowding-distance-assignment
         do i = 1,nondominated_size
             newsort(i,:) = input_set(i)%p%f(:)
         enddo
        
        do k = 1,numcols     ! num objectives 
           CALL mrgrnk(newsort(:,k),idxsel)

           !============calculate using crowding dist NAGA-II===========
           !input_set(idxsel(1))%p%I_dist = huge(1._dp) 
           !input_set(idxsel(nondominated_size))%p%I_dist = huge(1._dp) 
           !do i = 2,nondominated_size-1
           !    j = idxsel(i)
           !    j_p = idxsel(i+1)
           !    j_m = idxsel(i-1)
           !    mymax = maxval(newsort(:,numcols))
           !    mymin = minval(newsort(:,numcols))
           !    mydev = mymax - mymin
           !    if (mydev .gt. huge(1._dp)) mydev = 1._dp
           !    if (isnan(mydev)) mydev = 1._dp
           !    if (mydev .eq. 0_dp) mydev = 1._dp
           !    input_set(j)%p%I_dist = input_set(j)%p%I_dist * & 
           !                     (input_set(j_p)%p%f(numcols)- &
           !                      input_set(j_m)%p%f(numcols))/& 
           !                     mydev
           !                     !(fmax(numcols)-fmin(numcols)) !---yannick doesnt know what to put here
           !                     !(maxval(newsort(:,numcols))-minval(newsort(:,numcols)))
           !!-------------check I_dist for NaN and Inf----------------------------------------
           !if (isnan( input_set(j)%p%I_dist)) input_set(j)%p%I_dist = tiny(1._dp)
           !if (( input_set(j)%p%I_dist) .gt. huge(1._dp) ) input_set(j)%p%I_dist = tiny(1._dp)
           !enddo           
           !!============calculate using plugin density (kernel density estimation)
           input_set(idxsel(1))%p%I_dist = -huge(1._dp) 
           input_set(idxsel(nondominated_size))%p%I_dist = -huge(1._dp)
                x = newsort(:,k)
                m = size(x)
                call plugin(x,m,x,m,f,h)
                do i = 2,nondominated_size-1 
                   j = idxsel(i)
                   input_set(j)%p%I_dist = input_set(j)%p%I_dist  + f(i)
                enddo
        enddo
        deallocate(x,f)
        deallocate(newsort,idxsel) 
        

        if (variance_sorting) then
        !------use standard deviation as crowding distance---------------------
        n = size(input_set(1)%p%arx(:,1))
        allocate(tmp(nondominated_size))
        allocate(mydeviation(nondominated_size))
        mydeviation =0._dp
           do i = 1,n
               do k =1,nondominated_size
               tmp(k) = input_set(k)%p%arx(i,1)
               enddo
               call deviation(tmp)
               ! deviation is already normalized...it is a realtive standard error
               mydeviation(:) = mydeviation(:) + tmp(:)**2 
           enddo
        deallocate(tmp)
        !-------sqrt and normalize----------
        mydeviation(:) = sqrt(mydeviation(:))
        !mydeviation(:) = mydeviation(:)/maxval(mydeviation(:))
        do i = 1,nondominated_size
             input_set(i)%p%I_dist = -mydeviation(i)
        enddo
        

        deallocate(mydeviation)

       endif
!         do i = 1,nondominated_size

!         enddo
!        !---------this is a version where dipole error is used as criteria------
!         do i = 1,nondominated_size
!             if (input_set(i)%p%I_dist .eq. huge(1._dp)) then
!             else
!                 if (input_set(i)%p%dipole .eq. huge(1._dp)) then
!                    input_set(i)%p%I_dist =  tiny(1._dp)
!                 else 
!                    input_set(i)%p%I_dist =  1._dp/input_set(i)%p%dipole
!                 endif
!             endif
!         enddo
!
       END SUBROUTINE crowding_distance
!===========================================================================
    SUBROUTINE new_front(global_set,input_set,output_set)
       IMPLICIT NONE
       TYPE(individual_type_p),DIMENSION(:)             :: input_Set
       TYPE(individual_type_p),DIMENSION(:)             :: global_Set
       TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE :: output_Set
       TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE :: tmp_Set
       INTEGER                                          :: i,j,k,idx
       INTEGER                                          :: numrows
       TYPE(individual_type),POINTER                    :: a_k,a_k2
       allocate(tmp_set(size(global_set)))
       numrows = size(input_set)
       if (allocated(output_set)) deallocate(output_set)
!       do i = 1,numrows
!            a_k => input_set(i)%p
!            print *,'all stuff',a_k%id,a_k%s_i_count 
!            do j=1,a_k%s_i_count; print *,'idx',a_k%s_i(j)%p%id ; enddo
!       enddo
       k = 1
       do i = 1,numrows
           a_k => input_set(i)%p
           !print *,'new_front',a_k%f,a_k%s_i,a_k%s_i_count,a_k%d_count
           do j = 1,a_k%s_i_count
               !------ go through subset ------
               a_k2 => a_k%s_i(j)%p
               a_k2%d_count = a_k2%d_count - 1
               if (a_k2%d_count .eq. 0) then
                   tmp_set(k)%p => a_k2
                   k = k + 1
               endif
            !   idx = a_k%s_i(j)
            !   global_set(idx)%p%d_count = global_set(idx)%p%d_count - 1
            !   if (global_set(idx)%p%d_count .eq. 0) then
            !      tmp_set(k)= global_set(idx)
            !      k = k + 1
            !   endif

           enddo
       enddo
       allocate(output_set(k-1))
       do i=1,size(output_set)
           output_set(i)%p => tmp_set(i)%p
       enddo 
!       print *,size(output_set)
       deallocate(tmp_set)
    END SUBROUTINE

!===========================================================================
!    subroutine err_mng(population,n)
!    TYPE(individual_type_p),DIMENSION(:)              :: population
!    integer                                           :: n
!    real(kind=dp),allocatable(:)                      :: diag
!    TYPE(individual_type),POINTER                     :: a_k
!
!    allocate(diag(n))
!      
!    do i=1,asdfadfa  
!          a_k => population(i)%p
!
!      !-------------------------------------------------------------------------
!      !  NUMERICAL ERROR MANAGEMENT
!      !-------------------------------------------------------------------------	  
!      DO j = 1, n
!        diag(i) = a_k%C(j,j) ! Get diagonal vector of C
!      END DO
!      
!      !-------------------------------------------------------------------------
!      !  Adjust maximal coordinate axis deviations
!      !-------------------------------------------------------------------------
!      IF(any(sigma*sqrt(diag) .GT. maxdx)) sigma = minval(maxdx/sqrt(diag))
!      
!      !-------------------------------------------------------------------------
!      !  Adjust minimal coordinate axis deviations
!      !-------------------------------------------------------------------------
!      IF(any(sigma*sqrt(diag) .LT. mindx)) &
!        sigma = maxval(mindx/sqrt(diag)) * exp(0.05+cs/damps)
!      
!      !-------------------------------------------------------------------------
!      !  Adjust too low coordinate axis deviations
!      !-------------------------------------------------------------------------
!      mask = .FALSE.
!      tempMat2 = 0._MK
!      DO i = 1, input%N
!        IF(xmean(i) .EQ. xmean(i) + 0.2_MK*sigma*sqrt(diag(i))) &
!                                                        mask(i,i) = .TRUE.
!        tempMat2(i,i) = diag(i)
!      END DO
!      
!      IF(any(mask)) THEN
!        IF(options%stopOnWarnings) THEN
!            stopflag = 'warnnoeffectcoord'
!        ELSE
!            WRITE(*,*) 'Warning: Iteration ', countIter, &
!                            ': Coordinate axis std deviation too low'
!            WHERE(.NOT. mask)
!                tempMat2 = 0._MK
!            END WHERE
!            C = C + ccov*tempMat2
!            sigma = sigma*exp(0.05_MK+cs/damps)
!        END IF
!      END IF
!      
!      !-------------------------------------------------------------------------
!      !  Adjust step size in case of (numerical) precision problem
!      !-------------------------------------------------------------------------
!      IF(all(xmean .EQ. xmean + &
!        0.1_MK*sigma*BD(:,1+int(mod((countIter-lastRestart),input%N))))) THEN
!        i = 1+int(mod((countIter-lastRestart),input%N))
!        IF(options%stopOnWarnings) THEN
!            stopflag = 'warnnoeffectaxis'
!        ELSE
!            WRITE(*,*) 'Warning: Iteration ', countIter, &
!                            ': Main axis std deviation ', sigma*D(i,i), &
!                            ' has no effect'
!            sigma = sigma*exp(0.2_MK+cs/damps)
!        END IF
!            
!      END IF
!      
!      !-------------------------------------------------------------------------
!      !  Adjust step size in case of equal function values (flat fitness)
!      !-------------------------------------------------------------------------
!      temp = int(0.1_MK+lambda/4._MK)				! Little hack to round up
!      IF(real(temp) .NE. (0.1_MK+lambda/4._MK)) &
!        temp = temp + 1._MK
!      IF(fitness%sel(1) .EQ. fitness%sel(1+int(temp))) THEN
!        IF(options%WarnOnEqualFunctionValues .AND. options%stopOnWarnings) THEN
!            stopflag = 'warnequalfunvals'
!        ELSE
!            IF(options%WarnOnEqualFunctionValues) THEN
!            DO i = 1, input%N
!                diag(i) = D(i,i)
!            END DO
!            WRITE(*,*) &
!                'Warning: Iteration ', countIter, ': Equal function values f=',&
!                fitness%sel(1), ' at maximal main axis sigma ', &
!                sigma*maxval(diag)
!            END IF
!            sigma = sigma*exp(0.2_MK+cs/damps)
!        END IF
!      END IF
!      
!      !-------------------------------------------------------------------------
!      !  Adjust step size in case of equal function values
!      !-------------------------------------------------------------------------
!      temp = tool_myrange(fitness%hist,size(fitness%hist),fitness%sel(1),1,posInf)
!      IF((countIter - lastRestart .GT. 2) .AND. (temp .EQ. 0.)) THEN
!        IF(options%stopOnWarnings) THEN
!            stopflag = 'warnequalfunvalhist'
!        ELSE
!            DO i = 1, input%N
!                diag(i) = D(i,i)
!            END DO
!            WRITE(*,*) &
!                'Warning: Iteration ', countIter, ': Equal function values', &
!                'at maximal main axis sigma ', sigma*maxval(diag)	  		
!            sigma = sigma*exp(0.2_MK+cs/damps)
!        END IF
!      END IF
!      
!      deallocate(diag)
!    end subroutine err_mng
!
!===========================================================================

END MODULE cma_es
