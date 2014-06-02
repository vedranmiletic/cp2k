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
                                             gen_pop, &
                                             write_x
  USE cma_es_types,                    ONLY: individual_type,individual_type_p,cmaes_opt_type
  USE parallel_rng_types,              ONLY: rng_stream_type, &
                                             create_rng_stream, &
                                             next_random_number, &
                                             reset_to_next_rng_substream, &
                                             GAUSSIAN, &
                                             UNIFORM
   USE paramopt_utils,                 ONLY: deviation, &
                                             plugin,&
                                             det
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
                  !CALL new_mean(a_k) !only for the offspring this is adpoted
                  !-----update covariance---------
                      !call update_path(a_k)   
                      if (p_succ .gt. 0._dp) then
                          call update_cholesky(a_k)
                          a_k%C = matmul(a_k%A,transpose(a_k%A))
                          a_k%det = det(n,a_k%C)
                      endif
                      !          ! normalize??? ---> should be no neeed actualllly
                      ! if (any(isnan(a_k%A))) STOP "isnan A"
                      ! a_k%det = det(n,a_k%A)
                      ! if (isnan(a_k%det)) STOP "isnan det"
                      ! if (a_k%det .ne. 0._dp) a_k%A = a_k%det**(-1._dp/n)*a_k%A
                      ! a_k%det = det(n,a_k%A)
                      !if (p_succ .gt. 0._dp) call adapt_cov(a_k) 
          !---------Update B and D from C-------------------------------------------
                  !-----------------------------------------------------------------------
                  !  Enforce symmetry
                  !-----------------------------------------------------------------------
                  !CALL tool_symmatrix(a_k%C,n,a_k%triuC)
                  !     !normalize ???
                  !     if (any(isnan(a_k%C))) STOP "isnan C"
                  !     a_k%det = det(n,a_k%C)
                  !     if (isnan(a_k%det)) STOP "isnan det"
                  !     !if (a_k%det .ne. 1._dp) a_k%C = a_k%det**(-1._dp/n)*a_k%C
                  !     !a_k%det = det(n,a_k%C)
                  !!a_k%triuC = a_k%C
                  !call DPBTRF( 'L', n, 1, a_k%triuC, n, INFO )
                  !a_k%A = a_k%triuC                
                  !a_k%det = det(n,a_k%A)
                  !-----------------------------------------------------------------------
                  !  Eigen decomposition, D=diagonal matrix of eigenvalues, 
                  !  B=normalized eigenvectors
                  !-----------------------------------------------------------------------
                  !CALL tool_eigendecomp(a_k%triuC,cmaes_opt%n,a_k%D,a_k%B,cmaes_opt%info)
                          !if tool_eigendecomp returned Error go to the Restart Loop
                  !IF (cmaes_opt%info .NE. 0) THEN
                  !    print *,'error in a_k',a_k%Id,'eigendecomp failed'
                  !      DO i=1,cmaes_opt%n
                  !       WRITE(*,*) (a_k%triuC(i,j), j=1,cmaes_opt%n)
                  !     END DO
                  !    STOP
                  !END IF
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
                    !if (any(a_k%D .lt. 0.)) then
                    !    print *,'diagonal matrix wrong'
                    !    STOP
                    !endif
                    !a_k%D = sqrt(a_k%D)
                  !CALL DGEMM('N','N',n,n,n,1.0d0,a_k%B,n,a_k%D,n,0.0d0,a_k%BD,n)
                  !---recover C???-----------
                  !CALL DGEMM('N','T',n,n,n,1.0d0,a_k%BD,n,a_k%BD,n,0.0d0,a_k%C,n)
                  !-------get trace and determinant
                  a_k%trace = 0.0_dp
                  DO i = 1,n    
                      a_k%trace = a_k%trace+a_k%C(i,i)**2
                  END DO    
              !print *,'a_k',a_k%Id,a_k%front,a_k%I_dist,a_k%sigma
              !print *,'a_k%x',a_k%x
              !print *,'a_k_parents',a_k_parents%Id,a_k_parents%front,a_k_parents%I_dist,a_k_parents%sigma
              !print *,'a_k_parents%x',a_k_parents%x
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
             x = a_k%x
             !print *,'x should not be the same!',a_k_parents%arx(1,1),a_k%arx(1,1)
         !print *,'selected',cmaes_opt%a_k(k)%f,cmaes_opt%a_k_parents(k)%f,cmaes_opt%a_k(k)%s_iu
         END DO
         nullify(a_k,a_k_parents)
         print *,'===========================MO-CMAES-SUMMARY============================'
         print *,'number of accepted offspring',accepted
     
     END SUBROUTINE
!===========================================================================
!    subroutine new_mean(a_k)
!    implicit none
!    type(individual_type),POINTER                             :: a_k
!    TYPE(cp_error_type),POINTER                               :: error
!    INTEGER                                                   :: n,k,i
!!-------------selection and recombination (calc new xmean)---------------------------------
!    n = size(a_k%xmean)
!    a_k%xold = a_k%xmean
!    !-------(1+1) selection-------------
!    a_k%tempMat(:,1) = a_k%arx(:,1)
!    CALL dgemv('N',n,a_k%mu,1.0d0,a_k%tempMat,n,a_k%weights,1,0.0d0,a_k%xmean,1)
!    ! update x to the best individual in population
!    !x = a_k%tempMat(:,1)
!    !print *,'this is new mean'
!    !print *,'=================================================================='
!    !print *,cmaes_opt%xmean
!    !print *,'x matrix'
!    !print *,cmaes_opt%tempMat
!    !print *,'-------------------------------------'
!    ! do same for zmean/arz
!    a_k%tempMat(:,1) = a_k%arz(:,1)
!    CALL dgemv('N',n,a_k%mu,1.0d0,a_k%tempMat,n,a_k%weights,1,0.0d0,a_k%zmean,1)
!    if (any(a_k%arz(:,1) .ne. a_k%zmean)) stop "shit happends"
!    end subroutine new_mean
!===========================================================================
!    subroutine update_path(a_k)
!    implicit none
!    type(individual_type),POINTER                             :: a_k
!    TYPE(cp_error_type),POINTER                               :: error
!    INTEGER                                                   :: n,k,i
!    n = size(a_k%xmean)
!       if (a_k%p_succ .lt. a_k%p_tresh) then
!       a_k%pc = (1._dp - a_k%cc)*a_k%pc + sqrt(a_k%cc*(2_dp-a_k%cc))* &
!                        (a_k%xmean-a_k%xold)/a_k%sigma
!       else
!       a_k%pc = (1._dp - a_k%cc)*a_k%pc
!       endif
!                                  
!      !postcondition
!      if (any(isnan(a_k%pc))) then
!             print *,'pc is NaN'
!      endif 
!    end subroutine 
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
    subroutine update_cholesky(a_k)
    implicit none
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i,j
    real(kind=dp),dimension(:,:),allocatable                  :: tmp
    real(kind=dp),dimension(:,:),allocatable                  :: tmp3
    real(kind=dp),dimension(:),allocatable                    :: tmp2
    real(kind=dp)                                             :: alpha,beta

    n = size(a_k%x)
    allocate(tmp(n,n))
    allocate(tmp3(n,n))
    allocate(tmp2(n))
       if (a_k%p_succ .lt. a_k%p_tresh) then
         a_k%pc = (1._dp - a_k%cc)*a_k%pc + sqrt(a_k%cc*(2_dp-a_k%cc))* &
                           a_k%Az 
         alpha = (1._dp-a_k%ccov)
       else
         a_k%pc = (1._dp - a_k%cc)*a_k%pc
         alpha = (1._dp-a_k%ccov)+a_k%ccov*a_k%cc*(2._dp-a_k%cc)
       endif
      beta = a_k%ccov
      !print *,'this is Ainv ',a_k%Ainv(1,1),a_k%pc(1)
      if(any(isNaN(a_k%pc))) STOP "a_k%pc is NaN"
      CALL dgemm('N','N',n,1,n,1.0d0,a_k%Ainv,n,a_k%pc,n,0.0d0,a_k%w,n)
      ! hmmm...not so sure if that is okay
      where(isNaN(a_k%pc)) a_k%pc= 0._dp
      !CALL dgemv('N',n,n,1._dp,a_k%Ainv,n,a_k%pc,1,0._dp,a_k%w,1)
      !print *,'this is x',a_k%x
      !print *,'this is xold',a_k%xold
      !print *, 'this is w',a_k%w
      !print *, 'this is pc',a_k%pc
      !
      CALL dgemm('N','T',n,n,1,1.0d0,a_k%pc,n,a_k%w,n,0.0d0,tmp,n)
      !tmp = 0._dp
      !CALL dger(n,n,1._dp,a_k%pc,n,a_k%w,n,tmp,n)
      
      CALL dgemm('T','N',1,1,n,1.0d0,a_k%w,n,a_k%Ainv,n,0.0d0,tmp2,n)
      
      !CALL dgemm('N','T',n,n,1,1.0d0,a_k%w,n,tmp2,n,0.0d0,tmp3,n)
      tmp3 = 0._dp
      CALL dger(n,n,1._dp,a_k%w,n,tmp2,n,tmp3,n)
      
      !print *,'a und b',alpha,beta
       !          print *,'-----tmp------'
       !          DO i=1,n
       !              WRITE(*,*) (tmp(i,j), j=1,n)
       !          END DO
    !  print *,'tmp2',tmp2
    !  print *,'tmp3',tmp3
    !  print *,'w',a_k%w
    !  print *,'pc',a_k%pc
    !             print *,'-----a------'
    !             DO i=1,n
    !                 WRITE(*,*) (a_k%A(i,j), j=1,n)
    !             END DO
      !--------------update matrix------------------------------------------------------------
      if (norm2(a_k%w) .ne. 0._dp .and. norm2(a_k%w) .lt. 1.e10_dp) then
      a_k%A = sqrt(alpha)*a_k%A + sqrt(alpha)/norm2(a_k%w)**2._dp * &
          (sqrt(1._dp+beta/alpha*norm2(a_k%w)**2._dp)-1._dp) * &
          tmp 
      a_k%Ainv = 1._dp/sqrt(alpha) * a_k%Ainv - 1._dp/(sqrt(alpha)*norm2(a_k%w)**2._dp) * &
          (1._dp - 1._dp/sqrt(1._dp+beta/alpha*norm2(a_k%w)**2._dp)) * &
          tmp3
      endif
      a_k%det = det(n,a_k%a)
      !print *,'chlesk update'
      !do i=1,n;print *,a_k%A(i,:);enddo;print *,'det',det(n,a_k%A)
      !do i=1,n;print *,a_k%Ainv(i,:);enddo;print *,'det',det(n,a_k%Ainv)
      if (any(isNaN(a_k%A))) then
          print *,'alpha',alpha,'beta',beta
          print *,'w',a_k%w
          print *,'norm',norm2(a_k%w)
          print *,'pc',a_k%pc
          print *,'-----tmp------'
          DO i=1,n
              WRITE(*,*) (tmp(i,j), j=1,n)
          END DO
          print *,'-----A------'
          DO i=1,n
              WRITE(*,*) (a_k%A(i,j), j=1,n)
          END DO
         STOP 'NaN update_chlesky A'
      endif
      if (any(isNaN(a_k%Ainv))) then
          print *,'w',a_k%w
                    DO i=1,n
              WRITE(*,*) (a_k%Ainv(i,j), j=1,n)
          END DO
          STOP 'NaN update_chlesky Ainv'
      endif
  !    if (det(n,a_k%A) .gt. 1.01_dp .or. det(n,a_K%A) .lt. 0.99_dp) then
  !                  print *,'-----a------'
  !               DO i=1,n
  !                   WRITE(*,*) (a_k%A(i,j), j=1,n)
  !               END DO
  !        print *,det(n,a_k%A)
  !        STOP "det is wroing in update cholesk"
  !    endif
      deallocate(tmp,tmp2,tmp3)

    end subroutine update_cholesky
!===========================================================================
    subroutine adapt_cov(a_k)
    implicit none
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i
    !real(kind=dp),dimension(:),allocatable                    :: bdz
    real(kind=dp),dimension(:,:),allocatable                    :: colVec
    real(kind=dp),dimension(:,:),allocatable                    :: rowVec
    real(kind=dp),dimension(:,:),allocatable                    :: multmp_mN
    real(kind=dp),dimension(:,:),allocatable                    :: multmp_mN2

    n = size(a_k%x)
    !allocate(bdz(n))
    allocate(colVec(n,n))
    allocate(rowVec(n,n))
    allocate(multmp_mN(n,n))
    allocate(multmp_mN2(n,n))
    colVec=0._dp
    rowVec=0._dp
      DO i = 1,n 
        colVec(i,1) = a_k%pc(i)
        rowVec(1,i) = a_k%pc(i)
      END DO
      !a_k%weightsMat = 0.
      !DO i = 1, a_k%mu
      !    a_k%tempMat(:,i) = a_k%xold(:)
      !    a_k%weightsMat(i,i) = a_k%weights(i)
      !END DO   
      !a_k%mutmp = (a_k%arx(:,int(a_k%idxsel(1:a_k%mu))) - a_k%tempMat) 
      !a_k%mutmp = (a_k%arx - a_k%tempMat) 
      !CALL dgemm('N','T',n,n,1,1.0d0,colVec,n,rowVec,n,0.0d0,multmp_mN,n) 
      CALL dgemm('N','T',n,n,1,1.0d0,a_k%pc,n,a_k%pc,n,0.0d0,multmp_mN,n) 
      !------------calculate BDz and matrix for active covariance control--------------------
      !CALL dgemv('N',n,n,1.0d0,a_k%A,n,a_k%z,1,0.0d0,bdz,1)
      !DO i = 1,n 
      !  colVec(i,1) = bdz(i)
      !  rowVec(1,i) = bdz(i)
      !END DO
      !CALL dgemm('N','T',n,n,1,1.0d0,colVec,n,rowVec,n,0.0d0,multmp_mN2,n) 
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
         a_k%pc = (1._dp - a_k%cc)*a_k%pc + sqrt(a_k%cc*(2_dp-a_k%cc))* &
                           a_k%z
          a_k%C = (1._dp-a_k%ccov)*a_k%C & ! old matrix
                   + a_k%ccov*(multmp_mN) ! plus rank one update
      else
          a_k%pc = (1._dp - a_k%cc)*a_k%pc         
          a_k%C = (1._dp-a_k%ccov)*a_k%C & ! old matrix
                   + a_k%ccov*(multmp_mN &  ! plus rank one update[
                   + a_k%cc*(2._dp-a_k%cc)*a_k%C)
      !!----------------------------------------------------------------------------------------
      ! yannick needs to be tested!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
      !!----------------this should be active covariance control---------------------------------
      !a_k%C = (1._dp-a_k%ccovm)*a_k%C & ! old matrix
      !             - a_k%ccovm*(a_k%multmp_mN2) ! plus rank one update
      endif
      !print *,'yannick',multmp_mN
      !if(any(isNaN(a_k%C))) then
      !  print *,'yannick Covariance Matrix is NaN'
      !  STOP
      !endif
      deallocate(colVec,rowVec,multmp_mN,multmp_mN2)

    end subroutine adapt_cov
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
    CALL crowding_distance(new_set)
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
        CALL crowding_distance(new_set)
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
               !idx = i
               ! biggest is best
               ! 2) crowding distance
               idx = size(idxsel)+1-i
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
            !---depends on 
            if (a_k%I_dist .gt. a_k_parents%I_dist) then
                boolean = .true.
            else
                boolean = .false.
            endif
    else
        boolean = .false.   
    endif
    ! ugly hack 
    if (any(a_k%f .gt. 1.e100_dp) ) boolean = .false.
    end subroutine dominates
!===========================================================================
    !============================================================================================
!    allocate(tmp_mat3(n,n))
!    allocate(tmp_vec(n))
!    allocate(Aw(n))
! if (a_k%sigma .ne. 0._dp) then
!    do while  (any(tmp .ne. 0._dp))
!    tmp=0._dp
!      do i=1,n; a_k%z(i) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
!      CALL DGEMV('N',N,N,1.0d0,a_k%A,n,a_k%z,1,0.0d0,a_k%Az,1) 
!      tmp_x = a_k%x + a_k%sigma * a_k%Az 
!
!            do i=1,n
!                    if (tmp_x(i) .lt. cmaes_opt%l_bound(i)) then
!                        tmp(i) = abs(tmp_x(i) - cmaes_opt%l_bound(i))
!                        !scaling(i) = abs((cmaes_opt%l_bound(i)) - (a_k%x(i)))/abs((tmp_x(i)) - (a_k%x(i)))
!                        !tmp_x(i) = cmaes_opt%l_bound(i)
!                        rcount = rcount + 1._dp
!                    endif
!                    if (tmp_x(i) .gt. cmaes_opt%u_bound(i)) then
!                        tmp(i) = abs(tmp_x(i) - cmaes_opt%u_bound(i))
!                        !scaling(i) = (cmaes_opt%u_bound(i) - a_k%x(i))/(tmp_x(i) - a_k%x(i))
!                        !tmp_x(i) = cmaes_opt%u_bound(i)
!                        rcount = rcount + 1._dp
!                    endif
!            enddo
!       if (any(tmp .ne. 0._dp) .and. (1._dp-a_k%ccovm/(1._dp+a_k%ccovm)*norm2(a_k%z)**2) .gt. 0._dp) then
!         !a_k%ccovm = min(0.4_dp/(n**1.6_dp+1._dp),1._dp/(2._dp*norm2(a_k%z)**2-1._dp)) 
!         !print *,'ccovm',a_k%ccovm
!         !if (a_k%ccovm .lt. 0._dp) a_k%ccovm = 1._dp/(2._dp*norm2(a_k%z)**2 - 1._dp)
!         a_k%ccovm = 0.4_dp/(n**1.6_dp+1._dp)
!         alpha = sqrt(1._dp+a_k%ccovm)
!         beta  = alpha/norm2(a_k%z)**2 * (sqrt(1._dp-a_k%ccovm/(1._dp+a_k%ccovm)* &
!                norm2(a_k%z)**2)-1._dp)
!         !------------------a_k%A-------------------
!         CALL dgemv('N',n,n,1._dp,a_k%Ainv,n,a_k%z,1,0._dp,a_k%w,1)
!                print *,'w',a_k%w
!         CALL dgemv('N',n,n,1._dp,a_k%A,n,a_k%w,1,0._dp,Aw,1) 
!                print *,'Aw',Aw
!
!         CALL dgemm('N','T',n,n,1,1.0d0,Aw,n,a_k%w,n,0.0d0,tmp_mat3,n)
!         !tmp_mat3 = 0._dp
!         !CALL dger(n,n,1._dp,Aw,n,a_k%w,n,tmp_mat3,n)
!             !    print *,'-----tmp------'
!             !    DO i=1,n
!             !        WRITE(*,*) (tmp_mat3(i,j), j=1,n)
!             !    END DO
!             !    print *,'det tmp',det(n,tmp_mat3)
!             !    print *,'-----A old------'
!             !    DO i=1,n
!             !        WRITE(*,*) (a_k%A(i,j), j=1,n)
!             !    END DO
!         !--------------update matrix------------------------------------------------------------
!         a_k%A = alpha*a_k%A + beta*tmp_mat3
!             !    print *,'a;pha',alpha,'beta',beta
!             !    print *,'-----A new------'
!             !    DO i=1,n
!             !        WRITE(*,*) (a_k%A(i,j), j=1,n)
!             !    END DO
!
!         if(any(isNaN(a_k%A))) STOP "ohhh A is NaN"
!         !                 print *,'-----A new------'
!         !        DO i=1,n
!         !            WRITE(*,*) (a_k%A(i,j), j=1,n)
!         !        END DO
!         !print *,'det a',det(n,a_k%A),'alpha',alpha,'beta',beta
!         !------------------a_k%ainv----------------------
!         CALL dgemm('N','N',n,1,n,1.0d0,a_k%w,n,a_k%Ainv,n,0.0d0,tmp_vec,n)
!         !CALL dgemm('N','T',n,1,n,1.0d0,a_k%Ainv,n,a_k%w,n,0.0d0,tmp_vec,n)
!                
!          !===========================here=============================      
!          !      print *,'-----Ainv------'
!          !       DO i=1,n
!          !           WRITE(*,*) (a_k%Ainv(i,j), j=1,n)
!          !       END DO
!          !      print *,'w',a_k%w
!          !      print *,'tmp_vec',tmp_vec
!          !         tmp_mat3 = 0._dp
!         !CALL dger(n,n,1._dp,tmp_vec,n,a_k%w,n,tmp_mat3,n)
!         CALL dgemm('N','T',n,n,1,1.0d0,a_k%w,n,tmp_vec,n,0.0d0,tmp_mat3,n)
!          !                          print *,'-----tmp------'
!          !       DO i=1,n
!          !           WRITE(*,*) (tmp_mat3(i,j), j=1,n)
!          !       END DO
!         !--------------update matrix------------------------------------------------------------
!         a_k%Ainv = 1._dp/alpha - beta/(alpha**2 + alpha*beta*norm2(a_k%w)**2)*tmp_mat3
!         !-------------------------------------------------
!      !print *,'a und b',alpha,beta
!       !          print *,'x',tmp_x
!       !          print *,'y',a_k%y
!       !     if ((det(n,a_k%A) .gt. 1.01_dp) .or. (det(n,a_k%A) .lt. 0.99_dp)) then
!       !          print *,'det a',det(n,a_k%a)
!       !          STOP "det not correct"
!       !     endif
!       !     if ((det(n,a_k%Ainv) .gt. 1.01_dp) .or. (det(n,a_k%Ainv) .lt. 0.99_dp)) then
!       !          print *,'det ainv',det(n,a_k%ainv)
!       !          STOP "det not correct"
!       !      endif
!         else if (all(tmp .eq. 0._dp)) then
!             a_k%x = tmp_x
!       endif
!    enddo
!  endif
!    deallocate(tmp_mat3,tmp_vec,Aw)
!    !=========================try to adapted Covariance for generation============================
!     allocate(tmp_C(n,n))
!     allocate(triuC(n,n))
!     allocate(tmp_D(n,n))
!     allocate(tmp_B(n,n))
!     allocate(tmp_mat(n,n))
!     allocate(tmp_a(n,n))
!     allocate(tmp_ainv(n,n))
!     allocate(scaling(n))
!     allocate(tmp_z(n))
!     allocate(ipiv(n))
!     allocate(work(n)) 
!     !a_k%C = matmul(a_k%A,transpose(a_k%A))
!     !print *,'this is c'
!     !print *,a_k%C
!     !print *,'this is A'
!     !print *,a_k%A
!     tmp_A = a_k%A
!     tmp_C = a_k%C
!  !   print *,'A-before-'
!  !   print *,tmp_A
!     do while (any(tmp .ne. 0._dp))
!            tmp = 0._dp
!            scaling = 1._dp
!             do i=1,n; a_k%y(i) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
!            CALL DGEMV('N',N,N,1.0d0,tmp_A,n,a_k%y,1,0.0d0,tmp_z,1) 
!            tmp_x = a_k%x + a_k%sigma * tmp_z 
!     !       print *,'this is x',tmp_x
!            !print *,'norm2',norm2(tmp_x) 
!            do i=1,n
!                    if (tmp_x(i) .lt. cmaes_opt%l_bound(i)) then
!                        tmp(i) = abs(tmp_x(i) - cmaes_opt%l_bound(i))
!                        scaling(i) = abs((cmaes_opt%l_bound(i)) - (a_k%x(i)))/abs((tmp_x(i)) - (a_k%x(i)))
!                        !tmp_x(i) = cmaes_opt%l_bound(i)
!                        rcount = rcount + 1._dp
!                    endif
!                    if (tmp_x(i) .gt. cmaes_opt%u_bound(i)) then
!                        tmp(i) = abs(tmp_x(i) - cmaes_opt%u_bound(i))
!                        scaling(i) = (cmaes_opt%u_bound(i) - a_k%x(i))/(tmp_x(i) - a_k%x(i))
!                        !tmp_x(i) = cmaes_opt%u_bound(i)
!                        rcount = rcount + 1._dp
!                    endif
!            enddo
!         !CALL tool_symmatrix(a_k%C,n,a_k%triuC)
!         !CALL tool_eigendecomp(tmp_C,n,tmp_D,tmp_B,info)
!         !if tool_eigendecomp returned Error go to the Restart Loop
!         !         IF (info .NE. 0) THEN
!         !             print *,'error in a_k',a_k%Id,'eigendecomp failed'
!         !             STOP
!         !         END IF
!                   !---------
!          !print *,'d matrix'
!          tmp_D = 0._dp
!          do i=1,n
!             tmp_D(i,i) = (scaling(i))
!          enddo   
!       !   print *,'scaling',scaling 
!          !normalize D such that det = 1 
!          ! yannick new e.g. not sure....
!          tmpnorm = 1._dp
!          do i=1,n
!            tmpnorm = tmpnorm*scaling(i)
!          enddo
!          tmp_D=tmp_D*1._dp/tmpnorm**(1._dp/n) 
!          !tmp_D=sqrt(tmp_D)
!     !     print *,'D mat-----'
!     !     do i=1,n;print *,tmp_D(i,:);enddo
!     !         print *,'det',det(n,tmp_D)
!     !     print *,'A mat-----'
!     !     do i=1,n;print *,tmp_A(i,:);enddo
!     !         print *,'det',det(n,tmp_A)
!          !------normalize A---
!          !tmpnorm = 1._dp
!          !do i=1,n
!          !  tmpnorm = tmpnorm*tmp_a(i,i)
!          !enddo
!          !tmp_A=tmp_A*1._dp/tmpnorm**(1._dp/n) 
!          !tmp_A = matmul(tmp_A,tmp_D)
!          do i =1,n
!              tmp_A(:,i) = tmp_A(:,i)*tmp_D(i,i)
!          enddo
!     !     print *,'new A mat-----'
!     !     do i=1,n;print *,tmp_A(i,:);enddo
!          if (any(isNaN(tmp_A))) STOP "gen pop A is NaN"
!          !tmp_Ainv = matmul(tmp_D,tmp_Ainv)    
!          !     ! CALL dgemv('N',n,n,1._dp,tmp_A,n,scaling,1,0._dp,a_k%w,1)
!          !     !--update Ainv------------------------------------------
!                 tmp_Ainv = tmp_A
!                 call dGETRF(n,n,tmp_Ainv,n,IPIV,info)
!                  if(info .ne. 0) then
!                   write(*,*)"zgetrf failed",info
!                   STOP
!                  end if
!                 call dGETRI(n,tmp_Ainv,n,IPIV,WORK,n,info)
!                  if(info .ne. 0) then
!                   write(*,*)"zgetri failed"
!                   STOP
!                  end if
!
!          
!          !tmp_Ainv = matmul(tmp_D,transpose(tmp_B))
!          !tmp_C = matmul(tmp_A,(tmp_Ainv)
!          !CALL tool_symmatrix(tmp_C,n,triuC)
!          !call DPBTRF( 'L', n, 1,triuC, n, info )
!          !tmp_A = triuC
!          !a_k%det = det(n,tmp_A)
!     !     print *,'det',det(n,tmp_D),det(n,tmp_Ainv),det(n,tmp_A)
!          if ((det(n,tmp_A) .gt. 1.01_dp) .or. (det(n,tmp_A) .lt. 0.99_dp)) STOP "det not correct"
!          if ((det(n,tmp_Ainv) .gt. 1.01_dp) .or. (det(n,tmp_Ainv) .lt. 0.99_dp)) STOP "det not correct"
!          !print *,'new c',tmp_c
!          gen_count = gen_count + 1
!          if (gen_count .ge. 200) then
!                a_k%p_succ = (1._dp-a_k%cp)*a_k%p_succ 
!                a_k%sigma = a_k%sigma*exp((a_k%p_succ - a_k%p_target)/ &
!                         (a_k%damps*(1._dp-a_k%p_target)))
!                inner_count = inner_count + 1
!                gen_count = 0
!                if (inner_count .ge. 100) then
!                print *,'sigma probably to big :('
!                    STOP
!                endif
!          endif
!   !       if (gen_count .gt. 200) then
!   !       print *,'x',tmp_x
!   !       print *,'D mat-----'
!   !       do i=1,n;print *,tmp_D(i,:);enddo
!   !           print *,'det',det(n,tmp_D)
!   !       print *,'A mat-----'
!   !       do i=1,n;print *,tmp_A(i,:);enddo
!   !           print *,'det',det(n,tmp_A)
!   !       print *,'det',det(n,tmp_D),det(n,tmp_Ainv),det(n,tmp_A)
!   !            STOP "gen count to big"
!   !       endif
!     enddo
!    ! print *,'----A-----'
!    ! do i=1,n;print *,tmp_A(i,:);enddo
!    ! a_k%C = tmp_C
!    ! a_k%A = tmp_A
!    ! a_k%Ainv = tmp_Ainv
!     a_k%x = tmp_x
!     a_k%z = tmp_z
!    ! a_k%pc = 0._dp
!     deallocate(tmp_c,tmp_d,tmp_b,tmp_mat,tmp_a,tmp_ainv,tmp_z)
!    !=========================try to adapted cholesky for generation============================
!    allocate(ipiv(n))
!    allocate(work(n))
!    allocate(tmp_z(n))
!    allocate(v(n,n))
!    allocate(tmp_w(n))
!    allocate(tmp_vec(n))
!    allocate(tmp_a(n,n))
!    allocate(tmp_C(n,n))
!    allocate(tmp_ainv(n,n))
!    allocate(Aw(n))
!    allocate(wtA(n))
!    allocate(tmp_mat3(n,n))
!    allocate(tmp_mat(n,n))
!    tmp_A = a_k%A
!    tmp_Ainv = a_k%A
!    tmp_C = a_k%C
!    v = 0._dp
!    cc = 1._dp/(real(n)+2._dp)
!    do while (any(tmp .ne. 0._dp))
!        print *,'gen count',gen_count,'Id',a_k%id
!        tmp=0._dp
!        rcount = 0._dp
!    !    do i=1,n; a_k%arz(i,j) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
!        do i=1,n; a_k%z(i) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
!    CALL DGEMV('N',N,N,1.0d0,tmp_A,N,a_k%z,1,0.0d0,tmp_z,1)
!    !print *,'old x',a_k%x
!    tmp_x = a_k%x + a_k%sigma * tmp_z
!
!            do i=1,n
!                    if (tmp_x(i) .lt. cmaes_opt%l_bound(i)) then
!                        tmp(i) = abs(tmp_x(i) - cmaes_opt%l_bound(i))
!                        v(i,:) = (1._dp - cc)*v(i,:) + cc*tmp_z
!                        rcount = rcount + 1._dp
!                    endif
!                    if (tmp_x(i) .gt. cmaes_opt%u_bound(i)) then
!                        tmp(i) = abs(tmp_x(i) - cmaes_opt%u_bound(i))
!                        v(i,:) = (1._dp - cc)*v(i,:) + cc*tmp_z
!                        rcount = rcount + 1._dp
!                    endif
!            enddo
!            !if (cmaes_opt%additional_constrain) CALL param_const(a_k,tmp)
!            !---------------------------------
!            !a_k%ccovm = min(0.4_dp/(n**1.6_dp+1._dp),1._dp/(2._dp*norm2(a_k%y)**2-1._dp))
!            tmp_mat = 0._dp
!            do i=1,n
!                if (tmp(i) .ne. 0._dp) then
!                    !----update C
!                 !   tmp_mat3 = 0._dp
!                !    CALL dger(n,n,1._dp,v(i,:),n,v(i,:),n,tmp_mat3,n)
!                    !CALL dger(n,n,1._dp,tmp_z,n,tmp_z,n,tmp_mat3,n)
!                !    tmp_C = (1._dp+a_k%ccovm)*tmp_C - a_k%ccovm*tmp_mat3
!                !    CALL tool_symmatrix(tmp_C,n,a_k%triuC)
!                !    a_k%triuC = tmp_C
!                !    call DPBTRF( 'L', n, 1, a_k%triuC, n, INFO )
!                !    tmp_A = a_k%triuC  
!                !--------------
!        !        !set parameter
!        !        alpha = sqrt(1._dp+a_k%ccovm)
!        !        beta  = alpha/norm2(a_k%y)**2 * (sqrt(1._dp-a_k%ccovm/(1._dp+a_k%ccovm)* &
!        !                norm2(a_k%y)**2)-1._dp)
!                 CALL dgemv('N',n,n,1._dp,tmp_Ainv,n,v(i,:),1,0._dp,tmp_w,1)
!                 if (any(isNaN(tmp_w))) STOP "w is NaN" 
!                 dotpro =  dot_product(tmp_w,tmp_w)
!                 !tmp_mat3 = 0._dp
!                 !CALL dger(n,n,1._dp,v(i,:),n,tmp_w,n,tmp_mat3,n)
!                 CALL dgemm('N','T',n,n,1,1.0d0,v(i,:),n,tmp_w,n,0.0d0,tmp_mat3,n)
!                 !if (minval(tmp_mat3) .gt. tiny(1._dp) .and. dotpro .gt. 1e-16_dp) then
!                   tmp_mat = tmp_mat + 1._dp/(dotpro)*tmp_mat3
!                 !endif
!       !              print *,'v',v(i,:)
!       !              print *,i,'w',tmp_w
!
!       !                           print *,'-----tmp_ma3t------'
!       !          DO k=1,n
!       !              WRITE(*,*) (tmp_mat3(k,j), j=1,n)
!       !          END DO
!        !         if (any(isNaN(tmp_mat))) then
!        !             print *,'Ainv',tmp_Ainv(1:2,1:2)
!        !             print *,'v',v(i,:)
!        !             print *,i,'w',tmp_w
!        !             print *,i,'dotpro',dotpro
!        !             print *,'dotpro',dotpro
!        !             STOP "tmp_mat is NaN"
!        !         endif
!        !        tmp_w = v(i,:) 
!        !         tmp_A = sqrt(alpha)*tmp_A + sqrt(alpha)/norm2(tmp_w)**2._dp * &
!        !             (sqrt(1._dp+beta/alpha*norm2(tmp_w)**2._dp)-1._dp) * &
!        !             tmp_mat3 
!        !         !tmp_Ainv = 1._dp/sqrt(alpha) * tmp_Ainv - 1._dp/(sqrt(alpha)*norm2(tmp_w)**2._dp) * &
!        !         !    (1._dp - 1._dp/sqrt(1._dp+beta/alpha*norm2(tmp_w)**2._dp)) * &
!                 endif 
!            enddo
!            !-------------
!            if (All(tmp .eq. 0._dp)) then
!               ! print *,'should be ok',det(n,tmp_A)
!               ! print *,'should be ok ainv',det(n,tmp_Ainv)
!               a_k%A = tmp_A 
!               a_k%x = tmp_x
!               if (a_k%sigma .ne. 0._dp) a_k%Az = tmp_z
!               !print *,a_k%xold,a_k%sigma,'a_k%z',a_k%z
!               a_k%Ainv = tmp_Ainv
!               a_k%det = det(n,a_k%A)
!            else
!               !a_k%triuC = tmp_C
!               !call DPBTRF( 'L', n, 1, a_k%triuC, n, INFO )
!               !tmp_A = a_k%triuC
!               !print *,'mod',modulo(gen_count,10)
!               
!               if (modulo(gen_count,5) .eq. 0) then
!               a_k%p_succ = (1._dp-a_k%cp)*a_k%p_succ 
!               a_k%sigma = a_k%sigma*exp((a_k%p_succ - a_k%p_target)/ &
!                           (a_k%damps*(1._dp-a_k%p_target))) 
!               endif
!
!               !a_k%pc = (1._dp - a_k%cc)*a_k%pc + sqrt(a_k%cc*(2_dp-a_k%cc))* &
!               !            tmp_z  
!               !if (minval(tmp_mat) .gt. tiny(1._dp)) then
!               beta = 0.1_dp/(real(n)+2._dp)
!               tmp_a = tmp_a - beta/rcount*tmp_mat
!               !------normalize A---
!               !tmpnorm = 1._dp
!               !do i=1,n
!               !  tmpnorm = tmpnorm*tmp_a(i,i)
!               !enddo
!               !print *,'tmpnorm'
!               !print *,tmpnorm
!               !tmp_A=tmp_A*1._dp/tmpnorm**(1._dp/n) 
!               
!               !endif
!               !a_k%det = det(n,tmp_A)
!               !!if (a_k%det .ne. 0._dp) tmp_A = a_k%det**(-1._dp/n)*tmp_A 
!               if (any(isNaN(tmp_A))) then
!                   DO k=1,n
!                     WRITE(*,*) (tmp_a(k,j), j=1,n)
!                   END DO
!                   print *,'det is',a_k%det,det(n,tmp_mat)
!                   print *,'beta & rcount',beta,rcount
!                   STOP "A is NaN"
!               endif
!                
!               !--update Ainv------------------------------------------
!                 tmp_Ainv = tmp_A
!                 call dGETRF(n,n,tmp_Ainv,n,IPIV,info)
!                  if(info .ne. 0) then
!                   write(*,*)"zgetrf failed",info
!                 DO i=1,n
!                     WRITE(*,*) (tmp_a(i,j), j=1,n)
!                 END DO
!                   STOP
!                  end if
!                 call dGETRI(n,tmp_Ainv,n,IPIV,WORK,n,info)
!                  if(info .ne. 0) then
!                   write(*,*)"zgetri failed"
!                   STOP
!                  end if
!           !      print *,'x'
!           !      print *,tmp_x
!           !      print *,'-----tmp_mat------'
!           !      DO i=1,n
!           !          WRITE(*,*) (tmp_mat(i,j), j=1,n)
!           !      END DO
!           !      print *,'-----tmp_a------'
!           !      DO i=1,n
!           !          WRITE(*,*) (tmp_a(i,j), j=1,n)
!           !      END DO
!           !      print *,'-----tmp_Ainv------'
!           !      DO i=1,n
!           !          WRITE(*,*) (tmp_ainv(i,j), j=1,n)
!           !      END DO
!            endif
!            !print *,'tmp_z',tmp_z
!            !print *,'tmp_W',tmp_w
!            !print *,'some stuff',beta,dotpro
!            !print *,'yannick',gen_count,det(n,tmp_A),det(n,tmp_Ainv)
!            gen_count = gen_count + 1
!            if (gen_count .ge. 100000) then 
!                print *,tmp_x,a_k%det
!                STOP "generation not possible"
!            endif
!            !print *,'iteration',gen_count
!            !print *, 'tmp x',tmp_x
!       enddo
!       deallocate(tmp,ipiv,work,tmp_z,tmp_x,v,tmp_w,tmp_vec,tmp_a,tmp_c, &
!                  tmp_ainv,aw,wta,tmp_mat3,tmp_mat)
!
!    end subroutine gen_pop
!=====================================================================================
     SUBROUTINE crowding_distance(input_set)
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
         !------depends which crowding distance is used
         !  a) crowding dist 1.0_Dp
         !  b) kernel dist 0._dp
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
           x = newsort(:,k)
           !print *,'this is old x',x 
           CALL mrgrnk(x,idxsel)
           f = x
           do i=1,size(x)
               x(i)=f(idxsel(i))
           enddo
           !============calculate using crowding dist NAGA-II===========
           input_set(idxsel(1))%p%I_dist = huge(1._dp) 
           input_set(idxsel(nondominated_size))%p%I_dist = huge(1._dp) 
           mymax = maxval(x)
           mymin = minval(x)
           mydev = mymax - mymin 
           do i = 2,nondominated_size-1
               j = idxsel(i)
               j_p = idxsel(i+1)
               j_m = idxsel(i-1)
               if (mydev .gt. huge(1._dp)) mydev = 1._dp
               !if (isnan(mydev)) mydev = 1._dp
               !if (mydev .eq. 0_dp) mydev = 1._dp
               input_set(j)%p%I_dist = input_set(j)%p%I_dist + & 
                                (input_set(j_p)%p%f(k)- &
                                 input_set(j_m)%p%f(k))/& 
                                mydev
            enddo    
           enddo 
           deallocate(x,f,newsort,idxsel)  
           !============calculate using plugin density (kernel density estimation)
           !input_set(idxsel(1))%p%I_dist = huge(1._dp) 
           !input_set(idxsel(nondominated_size))%p%I_dist = huge(1._dp)
           !     m = size(x)
           !     if (m .gt. 2) then
           !     call plugin(x,m,x,m,f,h)
           !     else 
           !         f = huge(1._dp)
           !     endif
           !     !print *,'this is f',f
           !     do i = 2,nondominated_size-1 
           !        j = idxsel(i)
           !        if (f(i) .ne. 0._dp ) then
           !        !print *,'this is f and h',f(i),1._dp/(f(i)),h
           !             if( .not. isnan(f(i))) then
           !             input_set(j)%p%I_dist = input_set(j)%p%I_dist + (f(i))
           !             endif
           !        endif
           !        
           !     !print *,j,'I_dist',input_set(j)%p%I_dist
           !     enddo
           !     !print *,'this is newsort',newsort(:,k)
           !     !print *,'this is x',x
           !     !print *,'this is idxsel',idxsel
           !     !print *,'this is f',f 
           !     !print *,'-----tmp_mat------'
           !     !DO i=1,nondominated_size
           !     !    WRITE(*,*) input_set(i)%p%I_dist 
           !     !END DO
           !enddo
           ! do i=1,nondominated_size
           !     if (input_set(i)%p%I_dist .ne. huge(1._dp)) then
           !      input_set(i)%p%I_dist = 1._dp/input_set(i)%p%I_dist
           !     endif
           ! enddo
           !deallocate(x,f)
           !deallocate(newsort,idxsel) 
        
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
    SUBROUTINE param_const(a_k,tmp)
       IMPLICIT NONE
       REAL(Kind=dp),DIMENSION(:)                       :: tmp
       REAL(kind=dp)                                    :: tmp2
       TYPE(individual_type),POINTER                    :: a_k,a_k2
     !========================================================================================== 
!            !-----------yannick----Gss and gpp sepcial conditions!-------------
!            ! constrains on variables go here ---------------------------------
!        if (cmaes_opt%additional_constrain) then
         if (size(a_k%x) .eq. 20) then
            ! Uss lower then upp
            if (a_k%x(11) .lt. a_k%x(10))  then
                tmp(1) = tmp(1) +  a_k%x(10) - a_k%x(11) 
                tmp2 = a_k%x(10)
                a_k%x(10) = a_k%x(11) 
                a_k%x(11) = tmp2
            endif
            ! beta_s lower than beta_p
            if (a_k%x(13) .lt. a_k%x(12))  then
                tmp(1) = tmp(1) +  abs(a_k%x(12) - a_k%x(13))
                tmp2 = a_k%x(12)
                a_k%x(12) = a_k%x(13) 
                a_k%x(13) = tmp2
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
            if (a_k%x(15) .lt. a_k%x(16))  then
                tmp(1) = tmp(1) +  a_k%x(16) - a_k%x(15) 
                tmp2 = a_k%x(15)
                a_k%x(15) = a_k%x(16) 
                a_k%x(16) = tmp2
            endif
            ! exponents ??
            if (a_k%x(4) .lt. a_k%x(5))  then
                tmp(1) = tmp(1) +  a_k%x(5) - a_k%x(4) 
                tmp2 = a_k%x(4)
                a_k%x(4) = a_k%x(5) 
                a_k%x(5) = tmp2
            endif

         endif 
!          ! for scp-------------
!          if (n .eq. 26) then
!            ! Gss gibber then gsp
!          !  if (a_k%arx(11,j) .lt. a_k%arx(12,j))  then
!          !      tmp(1) = tmp(1) +  a_k%arx(12,j) - a_k%arx(11,j) 
!          !      tmp2 = a_k%arx(11,j)
!          !      a_k%arx(11,j) = a_k%arx(12,j) 
!          !      a_k%arx(12,j) = tmp2
!          !  endif
!            ! Gpp bigger then gp2
!            if (a_k%arx(13,j) .lt. a_k%arx(14,j))  then
!                tmp(1) = tmp(1) +  a_k%arx(14,j) - a_k%arx(13,j) 
!                tmp2 = a_k%arx(13,j)
!                a_k%arx(13,j) = a_k%arx(14,j) 
!                a_k%arx(14,j) = tmp2
!            endif
!            ! Uss lower then upp
!            if (a_k%arx(17,j) .lt. a_k%arx(16,j))  then
!                tmp(1) = tmp(1) +  a_k%arx(16,j) - a_k%arx(17,j) 
!                tmp2 = a_k%arx(16,j)
!                a_k%arx(16,j) = a_k%arx(17,j) 
!                a_k%arx(17,j) = tmp2
!            endif
!            ! beta_s lower than beta_p
!            if (a_k%arx(19,j) .lt. a_k%arx(18,j))  then
!                tmp(1) = tmp(1) +  abs(a_k%arx(18,j) - a_k%arx(19,j))
!                tmp2 = a_k%arx(18,j)
!                a_k%arx(18,j) = a_k%arx(19,j) 
!                a_k%arx(19,j) = tmp2
!            endif
!            ! sigma should be bigger than gss
!         !   if (a_k%arx(7,j) .lt. a_k%arx(1,j))  then
!         !       tmp(1) = tmp(1) +  a_k%arx(7,j) - a_k%arx(1,j) 
!         !       tmp2 = a_k%arx(7,j)
!         !       a_k%arx(7,j) = a_k%arx(1,j) 
!         !       a_k%arx(1,j) = tmp2
!         !   endif
!            ! sigma should be bigger than gss
!         !   if (a_k%arx(23,j) .lt. a_k%arx(11,j))  then
!         !       tmp(1) = tmp(1) +  a_k%arx(23,j) - a_k%arx(11,j) 
!         !       tmp2 = a_k%arx(23,j)
!         !       a_k%arx(23,j) = a_k%arx(11,j) 
!         !       a_k%arx(11,j) = tmp2
!         !   endif
!
!            ! exponents ??
!            if (a_k%arx(21,j) .lt. a_k%arx(22,j))  then
!                tmp(1) = tmp(1) +  a_k%arx(22,j) - a_k%arx(21,j) 
!                tmp2 = a_k%arx(21,j)
!                a_k%arx(21,j) = a_k%arx(22,j) 
!                a_k%arx(22,j) = tmp2
!            endif
!            ! exponents ??
!            if (a_k%arx(5,j) .lt. a_k%arx(6,j))  then
!                tmp(1) = tmp(1) +  a_k%arx(6,j) - a_k%arx(5,j) 
!                tmp2 = a_k%arx(5,j)
!                a_k%arx(5,j) = a_k%arx(6,j) 
!                a_k%arx(6,j) = tmp2
!            endif
!        ! without scp-------------------------------------------    
!      else if (n .eq. 22) then
!            ! Gss gibber then gsp
!         !   if (a_k%arx( 9,j) .lt. a_k%arx(10,j))  then
!         !       tmp(1) = tmp(1) +  a_k%arx(10,j) - a_k%arx( 9,j) 
!         !       tmp2 = a_k%arx( 9,j)
!         !       a_k%arx( 9,j) = a_k%arx(10,j) 
!         !       a_k%arx(10,j) = tmp2
!         !   endif
!            ! Gpp bigger then gp2
!            if (a_k%arx(11,j) .lt. a_k%arx(12,j))  then
!                tmp(1) = tmp(1) +  a_k%arx(12,j) - a_k%arx(11,j) 
!                tmp2 = a_k%arx(11,j)
!                a_k%arx(11,j) = a_k%arx(12,j) 
!                a_k%arx(12,j) = tmp2
!            endif
!            ! Uss lower then upp
!            if (a_k%arx(15,j) .lt. a_k%arx(14,j))  then
!                tmp(1) = tmp(1) +  a_k%arx(14,j) - a_k%arx(15,j) 
!                tmp2 = a_k%arx(14,j)
!                a_k%arx(14,j) = a_k%arx(15,j) 
!                a_k%arx(15,j) = tmp2
!            endif
!            ! beta_s lower than beta_p
!            if (a_k%arx(17,j) .lt. a_k%arx(16,j))  then
!                tmp(1) = tmp(1) +  abs(a_k%arx(16,j) - a_k%arx(17,j))
!                tmp2 = a_k%arx(16,j)
!                a_k%arx(16,j) = a_k%arx(17,j) 
!                a_k%arx(17,j) = tmp2
!            endif
!            ! sigma should be bigger than gss
!         !   if (a_k%arx(7,j) .lt. a_k%arx(1,j))  then
!         !       tmp(1) = tmp(1) +  a_k%arx(7,j) - a_k%arx(1,j) 
!         !       tmp2 = a_k%arx(7,j)
!         !       a_k%arx(7,j) = a_k%arx(1,j) 
!         !       a_k%arx(1,j) = tmp2
!         !   endif
!         !   ! sigma should be bigger than gss
!         !   if (a_k%arx(21,j) .lt. a_k%arx( 9,j))  then
!         !       tmp(1) = tmp(1) +  a_k%arx(21,j) - a_k%arx( 9,j) 
!         !       tmp2 = a_k%arx(21,j)
!         !       a_k%arx(21,j) = a_k%arx( 9,j) 
!         !       a_k%arx( 9,j) = tmp2
!         !   endif
!
!            ! exponents ??
!            if (a_k%arx(19,j) .lt. a_k%arx(20,j))  then
!                tmp(1) = tmp(1) +  a_k%arx(20,j) - a_k%arx(19,j) 
!                tmp2 = a_k%arx(19,j)
!                a_k%arx(19,j) = a_k%arx(20,j) 
!                a_k%arx(20,j) = tmp2
!            endif
!            ! exponents ??
!            if (a_k%arx(5,j) .lt. a_k%arx(6,j))  then
!                tmp(1) = tmp(1) +  a_k%arx(6,j) - a_k%arx(5,j) 
!                tmp2 = a_k%arx(5,j)
!                a_k%arx(5,j) = a_k%arx(6,j) 
!                a_k%arx(6,j) = tmp2
!            endif
    !enddo
    END SUBROUTINE
END MODULE cma_es
