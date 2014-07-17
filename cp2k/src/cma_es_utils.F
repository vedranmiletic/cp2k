!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2012  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief performs global geometry optimization
!> \par History
!> \author Ole
! *****************************************************************************
MODULE cma_es_utils
  USE kinds,                           ONLY: dp
  USE cma_es_types,                    ONLY: individual_type,individual_type_p,cmaes_opt_type
  USE paramopt_types,                  ONLY: paramopt_master_type
  USE parallel_rng_types,              ONLY: rng_stream_type, &
                                             create_rng_stream, &
                                             next_random_number, &
                                             reset_to_next_rng_substream, &
                                             GAUSSIAN, &
                                             UNIFORM
  USE paramopt_utils,                   ONLY: var_trans,&
                                             var_back_trans, &
                                             det
    USE cp_files,                        ONLY: open_file,&
                                             close_file
#include "cp_common_uses.h"
    implicit none
    private 
    
    CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cma_es_utils'
    public :: mrgrnk,tool_symmatrix,tool_eigendecomp,a_k_copy
    public :: gen_pop
    public :: gen_init_x
    public :: a_k_init
    public :: write_x
    public :: read_x
    public :: str !convert integer to string
!	  private :: R_mrgrnk, I_mrgrnk, D_mrgrnk
      interface mrgrnk
         module procedure D_mrgrnk, R_mrgrnk, I_mrgrnk
      end interface mrgrnk
      interface gen_pop
         module procedure gen_pop1
      end interface
      contains
!===========================================================================
    subroutine gen_init_x(a_k,cmaes_opt)
        type(cmaes_opt_type),target                               :: cmaes_opt
        type(individual_type),POINTER                             :: a_k
        TYPE(cp_error_type),POINTER                               :: error
         REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                   :: tmp_x,tmp
         integer                                                  :: gen_count,i
         allocate(tmp_x(size(a_k%xtrans)))
         allocate(tmp(size(a_k%xtrans)))
       !  gen_count = 0
       ! print *,'generating random x gen_init_x'
       ! a_k%penalty = 1._dp
        !do while (a_k%penalty .ne. 0._dp)
        !    tmp = 0._dp
            do i=1,cmaes_opt%n; a_k%xtrans(i) = next_random_number(cmaes_opt%rng_uniform_stream,error=error); enddo           
            call var_back_trans(a_k%xtrans,tmp_x,cmaes_opt%l_bound,cmaes_opt%u_bound)
        !    gen_count = gen_count + 1
            if (cmaes_opt%additional_constrain) then
                !print *,'additional const'
                CALL param_const(tmp_x,tmp)
                !a_k%penalty = norm2(tmp)
                !if (a_k%penalty .ne. 0._dp) then 
                !!    print *,'add const not fullfilled',gen_count,tmp_x
                !else
                !endif 
            endif
            a_k%x = tmp_x
        !enddo    
        deallocate(tmp_x) 
    end subroutine gen_init_x
!===========================================================================
    subroutine gen_pop4(a_k,cmaes_opt)
    implicit none
    type(cmaes_opt_type),target                               :: cmaes_opt
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i,j
    INTEGER                                                   :: gen_count,info,inner_count
    REAL(kind=dp)                                             :: alpha
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: scaling
    INTEGER,ALLOCATABLE,DIMENSION(:)                          :: ipiv
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: work
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: v
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_w 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_x 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_z 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp2  
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_vec
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_ainv 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_C    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: triuC    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_A    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_B    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_D    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: Aw
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: wtA
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat3
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat2
    REAL(kind=dp)                                             :: beta,dotpro,cc
    REAL(kind=dp)                                             :: rcount,tmpnor
    !    ! input:
    ! n, lambda, BD, xmean, sigma
    ! output:
    ! arx,arz
    n = cmaes_opt%n
    a_k%xold = a_k%x
    gen_count = 0
    inner_count = 0
    allocate(tmp(n))
    allocate(tmp_x(n))
    rcount = 0._dp
    a_k%penalty = 0._dp
    alpha = cmaes_opt%penalty_alpha
    cc = 1._dp/(n+2._dp)
    beta = 0.1_dp/(n+2._dp)
    !=========================old generation=================================================== 
    ! resample till in region !!!!
    a_k%penalty = 1._dp
    do while (a_k%penalty .ne. 0._dp)
         tmp=0._dp
         do i=1,n; a_k%z(i) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
         CALL DGEMV('N',N,N,1.0d0,a_k%A,n,a_k%z,1,0.0d0,a_k%Az,1) 
         tmp_x = a_k%x + a_k%sigma * a_k%Az  
            gen_count = gen_count + 1
         ! check if x in boundaries...set to bound limit otherwiese
            if (cmaes_opt%additional_constrain) then
                CALL param_const(tmp_x,tmp)
                a_k%Az = (tmp_x - a_k%x)/a_k%sigma
                !if (norm2(tmp) .ne. 0._dp) then 
                !   print *,'add const not fullfilled',gen_count,tmp_x
                !   CYCLE
                !endif
                ! the eq. constr are handled and positions swaped
                ! set tmp to zero 
                tmp = 0._dp
            endif

            do i=1,n
                    if (tmp_x(i) .lt. cmaes_opt%l_bound(i)) then
                        tmp(i) = abs(tmp_x(i) - cmaes_opt%l_bound(i))
                    endif
                    if (tmp_x(i) .gt. cmaes_opt%u_bound(i)) then
                        tmp(i) = abs(tmp_x(i) - cmaes_opt%u_bound(i))
                    endif
            enddo
            !a_k%penalty = alpha*norm2(tmp)**2
            a_k%penalty = norm2(tmp)
            if (gen_count .gt. 200) then
                gen_count = 0
                a_k%p_succ = (1._dp-a_k%cp)*a_k%p_succ 
                a_k%sigma = a_k%sigma*exp((a_k%p_succ - a_k%p_target)/ &
                         (a_k%damps*(1._dp-a_k%p_target)))
                inner_count = inner_count + 1
                if (inner_count .ge. 1000) then
                print *,'sigma probably to big :('
                    STOP
                endif
            endif
            if (a_k%penalty .eq. 0._dp) then
                a_k%x = tmp_x
            endif

            print *,'iter gen',gen_count,a_k%penalty,a_k%sigma,tmp_x
    enddo
   ! print *,'x is',a_k%x
    a_k%conv = norm2(a_k%x - a_k%xold)
    deallocate(tmp,tmp_x)  
   end subroutine gen_pop4
!===========================================================================
    subroutine gen_pop3(a_k,cmaes_opt)
    implicit none
    type(cmaes_opt_type),target                               :: cmaes_opt
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i,j
    INTEGER                                                   :: gen_count,info,inner_count
    REAL(kind=dp)                                             :: alpha
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: scaling
    INTEGER,ALLOCATABLE,DIMENSION(:)                          :: ipiv
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: work
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: v
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_w 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_x 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_z 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp2  
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_vec
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_ainv 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_C    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: triuC    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_A    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_B    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_D    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: Aw
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: wtA
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat3
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat2
    REAL(kind=dp)                                             :: beta,dotpro,cc
    REAL(kind=dp)                                             :: rcount,tmpnor
    !    ! input:
    ! n, lambda, BD, xmean, sigma
    ! output:
    ! arx,arz
    n = cmaes_opt%n
    a_k%xold = a_k%x
    gen_count = 0
    inner_count = 0
    allocate(tmp(n))
    allocate(tmp_x(n))
    rcount = 0._dp
    a_k%penalty = 0._dp
    alpha = cmaes_opt%penalty_alpha
    cc = 1._dp/(n+2._dp)
    beta = 0.1_dp/(n+2._dp)
    !=========================old generation=================================================== 
    ! resample till in region !!!!
    a_k%penalty = 1._dp
    do while (a_k%penalty .ne. 0._dp)
    tmp=0._dp
    do i=1,n; a_k%z(i) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
    CALL DGEMV('N',N,N,1.0d0,a_k%A,n,a_k%z,1,0.0d0,a_k%Az,1) 
    tmp_x = a_k%x + a_k%sigma * a_k%Az  
    ! check if x in boundaries...set to bound limit otherwiese
            do i=1,n
                    if (tmp_x(i) .lt. cmaes_opt%l_bound(i)) then
                        tmp(i) = abs(tmp_x(i) - cmaes_opt%l_bound(i))
                    endif
                    if (tmp_x(i) .gt. cmaes_opt%u_bound(i)) then
                        tmp(i) = abs(tmp_x(i) - cmaes_opt%u_bound(i))
                    endif
            enddo
            if (a_k%sigma .eq. 0._dp) a_k%Az = 0._dp
            a_k%penalty = alpha*norm2(tmp)**2
            gen_count = gen_count + 1
            !if (gen_count .ge. 200) then
            !    a_k%p_succ = (1._dp-a_k%cp)*a_k%p_succ 
            !    a_k%sigma = a_k%sigma*exp((a_k%p_succ - a_k%p_target)/ &
            !             (a_k%damps*(1._dp-a_k%p_target)))
            !    inner_count = inner_count + 1
            !    if (inner_count .ge. 1000) then
            !    print *,'sigma probably to big :('
            !        STOP
            !    endif
            !endif
            if (a_k%penalty .eq. 0._dp) then
                a_k%x = tmp_x
            endif

            print *,'iter gen',gen_count,a_k%sigma,a_k%id
    enddo
    deallocate(tmp,tmp_x)  
   end subroutine gen_pop3
!===========================================================================
    subroutine gen_pop2(a_k,cmaes_opt)
    implicit none
    type(cmaes_opt_type),target                               :: cmaes_opt
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i,j
    INTEGER                                                   :: gen_count,info,inner_count
    REAL(kind=dp)                                             :: alpha
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: scaling
    INTEGER,ALLOCATABLE,DIMENSION(:)                          :: ipiv
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: work
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: v
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_w 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_x 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_z 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp2  
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_vec
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_ainv 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_C    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: triuC    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_A    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_B    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_D    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: Aw
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: wtA
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat3
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat2
    REAL(kind=dp)                                             :: beta,dotpro,cc
    REAL(kind=dp)                                             :: rcount,tmpnor
    !    ! input:
    ! n, lambda, BD, xmean, sigma
    ! output:
    ! arx,arz
    n = cmaes_opt%n
    a_k%xold = a_k%x
    gen_count = 0
    inner_count = 0
    allocate(tmp(n))
    allocate(tmp_x(n))
    rcount = 0._dp
    a_k%penalty = 0._dp
    alpha = cmaes_opt%penalty_alpha
    cc = 1._dp/(n+2._dp)
    beta = 0.1_dp/(n+2._dp)
    a_k%xold = a_k%x
    !=========================old generation=================================================== 
    ! resample till in region !!!!
    a_k%penalty = 1._dp
    do while (a_k%penalty .ne. 0._dp)
    tmp=0._dp
    do i=1,n; a_k%z(i) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
    CALL DGEMV('N',N,N,1.0d0,a_k%A,n,a_k%z,1,0.0d0,a_k%Az,1) 
    tmp_x = a_k%x + a_k%sigma * a_k%Az  
    ! check if x in boundaries...set to bound limit otherwiese
            do i=1,n
                    if (tmp_x(i) .lt. cmaes_opt%l_bound(i)) then
                        tmp(i) = abs(tmp_x(i) - cmaes_opt%l_bound(i))
                    endif
                    if (tmp_x(i) .gt. cmaes_opt%u_bound(i)) then
                        tmp(i) = abs(tmp_x(i) - cmaes_opt%u_bound(i))
                    endif
            enddo
            if (a_k%sigma .eq. 0._dp) a_k%Az = 0._dp
            a_k%penalty = alpha*norm2(tmp)**2
            gen_count = gen_count + 1
            if (gen_count .ge. 20) then
                a_k%p_succ = (1._dp-a_k%cp)*a_k%p_succ 
                a_k%sigma = a_k%sigma*exp((a_k%p_succ - a_k%p_target)/ &
                         (a_k%damps*(1._dp-a_k%p_target)))
                inner_count = inner_count + 1
                if (inner_count .ge. 1000) then
                print *,'sigma probably to big :('
                    STOP
                endif
            endif
            if (a_k%penalty .eq. 0._dp) then
                a_k%x = tmp_x
            endif

            print *,'iter gen',gen_count,a_k%sigma
    enddo
    a_k%conv = norm2(a_k%x - a_k%xold)
    deallocate(tmp,tmp_x)  
   end subroutine gen_pop2
!===========================================================================
    subroutine gen_pop1(a_k,cmaes_opt)
    implicit none
    type(cmaes_opt_type),target                               :: cmaes_opt
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i,j
    INTEGER                                                   :: gen_count,info,inner_count
    REAL(kind=dp)                                             :: alpha
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: scaling
    INTEGER,ALLOCATABLE,DIMENSION(:)                          :: ipiv
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: work
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: v
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_w 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_x 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_z 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp2  
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: tmp_vec
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_ainv 
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_C    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: triuC    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_A    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_B    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_D    
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: Aw
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:)                    :: wtA
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat3
    REAL(Kind=dp),ALLOCATABLE,DIMENSION(:,:)                  :: tmp_mat2
    REAL(kind=dp)                                             :: beta,dotpro,cc
    REAL(kind=dp)                                             :: rcount,tmpnor
    n = cmaes_opt%n
    a_k%xold = a_k%x
    gen_count = 0
    inner_count = 0
    allocate(tmp(n))
    allocate(tmp_x(n))
    rcount = 0._dp
    a_k%penalty = 0._dp
    alpha = cmaes_opt%penalty_alpha
    cc = 1._dp/(n+2._dp)
    beta = 0.1_dp/(n+2._dp)
!    !=========================try to adapted cholesky for generation============================
    allocate(ipiv(n))
    allocate(work(n))
    allocate(tmp_z(n))
    allocate(v(n,n))
    allocate(tmp_w(n))
    allocate(tmp_vec(n))
    allocate(tmp_a(n,n))
    allocate(tmp_C(n,n))
    allocate(tmp_ainv(n,n))
    allocate(Aw(n))
    allocate(wtA(n))
    allocate(tmp_mat3(n,n))
    allocate(tmp_mat(n,n))
    tmp_A = a_k%A
    tmp_Ainv = a_k%A
    tmp_C = a_k%C
    a_k%xold = a_k%x
    v = 0._dp
    cc = 1._dp/(real(n)+2._dp)
    do while (any(tmp .ne. 0._dp))
        print *,'gen count',gen_count,'Id',a_k%id
        tmp=0._dp
        rcount = 0._dp
    !    do i=1,n; a_k%arz(i,j) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
        do i=1,n; a_k%z(i) = next_random_number(cmaes_opt%rng_normal_stream,error=error); enddo
    CALL DGEMV('N',N,N,1.0d0,tmp_A,N,a_k%z,1,0.0d0,tmp_z,1)
    !print *,'old x',a_k%x
    tmp_x = a_k%x + a_k%sigma * tmp_z
    !   if (cmaes_opt%additional_constrain) CALL param_const(tmp_x,tmp)
            do i=1,n
                    if (tmp_x(i) .lt. cmaes_opt%l_bound(i)) then
                        print *,i,tmp_x(i),"is lower then",cmaes_opt%l_bound(i),'sigma',a_k%sigma
                        tmp(i) = abs(tmp_x(i) - cmaes_opt%l_bound(i))
                        rcount = rcount + 1._dp
                    endif
                    if (tmp_x(i) .gt. cmaes_opt%u_bound(i)) then
                        print *,i,tmp_x(i),"is greater then",cmaes_opt%u_bound(i),'sigma',a_k%sigma
                        tmp(i) = abs(tmp_x(i) - cmaes_opt%u_bound(i))
                        rcount = rcount + 1._dp
                    endif

                    if (tmp(i) .ne. 0._dp) v(i,:) = (1._dp - cc)*v(i,:) + cc*tmp_z 
            enddo
     
            !---------------------------------
            !a_k%ccovm = min(0.4_dp/(n**1.6_dp+1._dp),1._dp/(2._dp*norm2(a_k%y)**2-1._dp))
            tmp_mat = 0._dp
            do i=1,n
                if (tmp(i) .ne. 0._dp) then
                    !----update C
                 !   tmp_mat3 = 0._dp
                !    CALL dger(n,n,1._dp,v(i,:),n,v(i,:),n,tmp_mat3,n)
                    !CALL dger(n,n,1._dp,tmp_z,n,tmp_z,n,tmp_mat3,n)
                !    tmp_C = (1._dp+a_k%ccovm)*tmp_C - a_k%ccovm*tmp_mat3
                !    CALL tool_symmatrix(tmp_C,n,a_k%triuC)
                !    a_k%triuC = tmp_C
                !    call DPBTRF( 'L', n, 1, a_k%triuC, n, INFO )
                !    tmp_A = a_k%triuC  
                !--------------
        !        !set parameter
        !        alpha = sqrt(1._dp+a_k%ccovm)
        !        beta  = alpha/norm2(a_k%y)**2 * (sqrt(1._dp-a_k%ccovm/(1._dp+a_k%ccovm)* &
        !                norm2(a_k%y)**2)-1._dp)
                 CALL dgemv('N',n,n,1._dp,tmp_Ainv,n,v(i,:),1,0._dp,tmp_w,1)
                 if (any(isNaN(tmp_w))) STOP "w is NaN" 
                 dotpro =  dot_product(tmp_w,tmp_w)
                 if (isNaN(dotpro)) STOP "dotpro is NaN"
                 if (dotpro .lt. 1e-16_dp) then 
                     print *, dotpro,"dotpro too small" ,tmp_w
                     print *, "v vector",v(i,:) 
                     STOP
                 endif 
                 !tmp_mat3 = 0._dp
                 !CALL dger(n,n,1._dp,v(i,:),n,tmp_w,n,tmp_mat3,n)
                 CALL dgemm('N','T',n,n,1,1.0d0,v(i,:),n,tmp_w,n,0.0d0,tmp_mat3,n)
                 if (any(isNaN(tmp_mat3))) then
                   DO k=1,n
                     WRITE(*,*) (tmp_mat3(k,j), j=1,n)
                   END DO
                   STOP "tmp_mat3 is NaN"
                endif
                 !if (minval(tmp_mat3) .gt. tiny(1._dp) .and. dotpro .gt. 1e-16_dp) then
                   tmp_mat = tmp_mat + 1._dp/(dotpro)*tmp_mat3
                 !endif
       !              print *,'v',v(i,:)
       !              print *,i,'w',tmp_w

       !                           print *,'-----tmp_ma3t------'
       !          DO k=1,n
       !              WRITE(*,*) (tmp_mat3(k,j), j=1,n)
       !          END DO
        !         if (any(isNaN(tmp_mat))) then
        !             print *,'Ainv',tmp_Ainv(1:2,1:2)
        !             print *,'v',v(i,:)
        !             print *,i,'w',tmp_w
        !             print *,i,'dotpro',dotpro
        !             print *,'dotpro',dotpro
        !             STOP "tmp_mat is NaN"
        !         endif
        !        tmp_w = v(i,:) 
        !         tmp_A = sqrt(alpha)*tmp_A + sqrt(alpha)/norm2(tmp_w)**2._dp * &
        !             (sqrt(1._dp+beta/alpha*norm2(tmp_w)**2._dp)-1._dp) * &
        !             tmp_mat3 
        !         !tmp_Ainv = 1._dp/sqrt(alpha) * tmp_Ainv - 1._dp/(sqrt(alpha)*norm2(tmp_w)**2._dp) * &
        !         !    (1._dp - 1._dp/sqrt(1._dp+beta/alpha*norm2(tmp_w)**2._dp)) * &
                 endif 
            enddo
            !-------------
            if (All(tmp .eq. 0._dp)) then
               ! print *,'should be ok',det(n,tmp_A)
               ! print *,'should be ok ainv',det(n,tmp_Ainv)
               a_k%A = tmp_A 
               a_k%x = tmp_x
               if (a_k%sigma .ne. 0._dp) a_k%Az = tmp_z
               !print *,a_k%xold,a_k%sigma,'a_k%z',a_k%z
               a_k%Ainv = tmp_Ainv
               a_k%det = det(n,a_k%A)
            else
               !a_k%triuC = tmp_C
               !call DPBTRF( 'L', n, 1, a_k%triuC, n, INFO )
               !tmp_A = a_k%triuC
               !print *,'mod',modulo(gen_count,10)
               !
               !if (modulo(gen_count,5) .eq. 0) then
               !a_k%p_succ = (1._dp-a_k%cp)*a_k%p_succ 
               !a_k%sigma = a_k%sigma*exp((a_k%p_succ - a_k%p_target)/ &
               !            (a_k%damps*(1._dp-a_k%p_target))) 
               !endif

               !a_k%pc = (1._dp - a_k%cc)*a_k%pc + sqrt(a_k%cc*(2_dp-a_k%cc))* &
               !            tmp_z  
               !if (minval(tmp_mat) .gt. tiny(1._dp)) then
               beta = 0.1_dp/(real(n)+2._dp)
               tmp_a = tmp_a - beta/rcount*tmp_mat
               !------normalize A---
               !tmpnorm = 1._dp
               !do i=1,n
               !  tmpnorm = tmpnorm*tmp_a(i,i)
               !enddo
               !print *,'tmpnorm'
               !print *,tmpnorm
               !tmp_A=tmp_A*1._dp/tmpnorm**(1._dp/n) 
               
               !endif
               !a_k%det = det(n,tmp_A)
               !!if (a_k%det .ne. 0._dp) tmp_A = a_k%det**(-1._dp/n)*tmp_A 
               if (any(isNaN(tmp_A))) then
                   DO k=1,n
                     WRITE(*,*) (tmp_a(k,j), j=1,n)
                   END DO
                   print *,'det is',a_k%det,det(n,tmp_mat)
                   print *,'beta & rcount',beta,rcount
                   STOP "A is NaN"
               endif
                
               !--update Ainv------------------------------------------
                 tmp_Ainv = tmp_A
                 call dGETRF(n,n,tmp_Ainv,n,IPIV,info)
                  if(info .ne. 0) then
                   write(*,*)"zgetrf failed",info
                 DO i=1,n
                     WRITE(*,*) (tmp_a(i,j), j=1,n)
                 END DO
                   STOP
                  end if
                 call dGETRI(n,tmp_Ainv,n,IPIV,WORK,n,info)
                  if(info .ne. 0) then
                   write(*,*)"zgetri failed"
                   STOP
                  end if
           !      print *,'x'
           !      print *,tmp_x
           !      print *,'-----tmp_mat------'
           !      DO i=1,n
           !          WRITE(*,*) (tmp_mat(i,j), j=1,n)
           !      END DO
           !      print *,'-----tmp_a------'
           !      DO i=1,n
           !          WRITE(*,*) (tmp_a(i,j), j=1,n)
           !      END DO
           !      print *,'-----tmp_Ainv------'
           !      DO i=1,n
           !          WRITE(*,*) (tmp_ainv(i,j), j=1,n)
           !      END DO
            endif
            !print *,'tmp_z',tmp_z
            !print *,'tmp_W',tmp_w
            !print *,'some stuff',beta,dotpro
            !print *,'yannick',gen_count,det(n,tmp_A),det(n,tmp_Ainv)
            gen_count = gen_count + 1

            ! decrease sigma ?---------------------------
            if (gen_count .ge. 20) then
                gen_count = 0
                a_k%p_succ = (1._dp-a_k%cp)*a_k%p_succ 
                a_k%sigma = a_k%sigma*exp((a_k%p_succ - a_k%p_target)/ &
                         (a_k%damps*(1._dp-a_k%p_target)))
                inner_count = inner_count + 1
                if (inner_count .ge. 1000) then
                print *,'sigma probably to big :('
                    STOP
                endif
            endif
            if (gen_count .ge. 100000000) then 
                print *,tmp_x,a_k%det
                STOP "generation not possible"
            endif
            !print *,'iteration',gen_count
            !print *, 'tmp x',tmp_x
       enddo
       a_k%conv = norm2(a_k%x - a_k%xold)
       deallocate(tmp,ipiv,work,tmp_z,tmp_x,v,tmp_w,tmp_vec,tmp_a,tmp_c, &
                  tmp_ainv,aw,wta,tmp_mat3,tmp_mat)

    end subroutine gen_pop1
!===========================================================================
    SUBROUTINE param_const(tmpx,tmp)
       IMPLICIT NONE
       REAL(Kind=dp),DIMENSION(:)                       :: tmp
       REAL(Kind=dp),DIMENSION(:)                       :: tmpx
       REAL(kind=dp)                                    :: tmp2
!       TYPE(individual_type),POINTER                    :: a_k,a_k2
     !========================================================================================== 
!            !-----------yannick----Gss and gpp sepcial conditions!-------------
!            ! constrains on variables go here ---------------------------------
!        if (cmaes_opt%additional_constrain) then
!         if (size(a_k%x) .eq. 20) then
            ! Uss lower then upp
            if (tmpx(11) .lt. tmpx(10))  then
                tmp(11) = tmp(11) +  tmpx(10) - tmpx(11) 
                tmp2 = tmpx(10)
                tmpx(10) = tmpx(11) 
                tmpx(11) = tmp2
            endif
            ! beta_s lower than beta_p
            if (tmpx(13) .lt. tmpx(12))  then
                tmp(13) = tmp(13) +  abs(tmpx(12) - tmpx(13))
                tmp2 = tmpx(12)
                tmpx(12) = tmpx(13) 
                tmpx(13) = tmp2
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
            if (tmpx(15) .lt. tmpx(16))  then
                tmp(15) = tmp(15) +  tmpx(16) - tmpx(15) 
                tmp2 = tmpx(15)
                tmpx(15) = tmpx(16) 
                tmpx(16) = tmp2
            endif
            ! exponents ??
            if (tmpx(4) .lt. tmpx(5))  then
                tmp(4) = tmp(4) +  tmpx(5) - tmpx(4) 
                tmp2 = tmpx(4)
                tmpx(4) = tmpx(5) 
                tmpx(5) = tmp2
            endif

!         endif 
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
!=============================================================================
 SUBROUTINE a_k_init(a_k,lambda_mo,num_obj,step_size,insigma,this,k,x)
    IMPLICIT NONE
    TYPE(paramopt_master_type)               :: this
    type(individual_type),POINTER            :: a_k
    REAL(kind=dp),DIMENSION(:)               :: insigma
    REAL(kind=dp),DIMENSION(:),OPTIONAL      :: x
    REAL(kind=dp)                            :: step_size
    REAL(kind=dp)                            :: tmp
    INTEGER                                  :: n,l
    INTEGER                                  :: k
    INTEGER                                  :: i,j
    INTEGER                                  :: lambda_mo
    INTEGER                                  :: num_obj
    INTEGER                                  :: allocstat
    INTEGER                                  :: unit_number
    n = size(this%x)
    !random x
    if (this%rand_inp) then
    do i=1,n; this%xtrans(i) = next_random_number(this%rng_stream,error=this%error); enddo 
        call var_back_trans(this%xtrans,this%x,this%l_bound,this%u_bound)
    endif
    if (present(x)) then
        this%x = x
    endif
      ! initialize a_k
               IF (.NOT. ALLOCATED(a_k%Ainv)) THEN
                     ALLOCATE(a_k%Ainv(n,n),stat=allocStat)
                     IF(allocStat .NE. 0) STOP 'Error allocating Ainv'
               END IF
               IF (.NOT. ALLOCATED(a_k%A)) THEN
                     ALLOCATE(a_k%A(n,n),stat=allocStat)
                     IF(allocStat .NE. 0) STOP 'Error allocating A'
               END IF
               a_k%Ainv = 0._dp
               a_k%A = 0._dp
               !   do i=1,n
               !      a_k%A(i,i) = 1._dp          
               !   enddo
               !a_k%Ainv = A_k%A
               !   tmp = 1._dp
               !   do i=1,n
               !      tmp = tmp*insigma(i)
               !   enddo
               !   do i=1,n
               !      a_k%A(i,i)=sqrt(insigma(i)*1._dp/tmp**(1._dp/n))
               !      a_k%A(i,i)=1._dp/a_k%A(i,i)
               !   enddo
                  do i=1,n
                     a_k%A(i,i)=insigma(i)
                  enddo
                

                 !  DO i=1,n
                 !   WRITE(*,*) (a_k%A(i,j), j=1,n)
                 !  END DO
               a_k%Ainv = a_k%A
               IF (.NOT. ALLOCATED(a_k%c)) THEN
                     ALLOCATE(a_k%C(n,n),stat=allocStat)
                     IF(allocStat .NE. 0) STOP 'Error allocating B'
               END IF
               a_k%C = matmul(a_k%A,transpose(a_k%A))
               !     DO i=1,n
               !      WRITE(*,*) (a_k%c(i,j), j=1,n)
               !     END DO
               !a_k%C = 0._dp
               !DO i = 1,n
               !  a_k%C(i,i) = 1._dp
               !ENDDO
               IF (.NOT. ALLOCATED(a_k%triuC)) THEN
                     ALLOCATE(a_k%triuC(n,n),stat=allocStat)
                     IF(allocStat .NE. 0) STOP 'Error allocating B'
               END IF
               !IF (.NOT. ALLOCATED(a_k%B)) THEN
               !      ALLOCATE(a_k%B(n,n),stat=allocStat)
               !      IF(allocStat .NE. 0) STOP 'Error allocating B'
               !END IF
               !a_k%B = 0._dp
               !DO i = 1,n
               ! a_k%B(i,i) = 1._dp
               !END DO  
               !IF (.NOT. ALLOCATED(a_k%D)) THEN
               !    ALLOCATE(a_k%D(n,n),stat=allocStat)
               !    IF(allocStat .NE. 0) STOP 'Error allocating D'
               !END IF
               !a_k%D = 0._dp
               !DO i = 1, n
               !   a_k%D(i,i) = sqrt(insigma(i)*1_dp/product(insigma)**(1._dp/size(insigma)))
               !   !d * 1/nsqrt(det) so that det ~ 1
               !   ! and sqrt(d) so that BD*BD^t = C
               !   !a_k%D(i,i) = 1._dp 
               !END DO
               !IF (.NOT. ALLOCATED(a_k%BD)) THEN 
               !    ALLOCATE(a_k%BD(n,n),stat=allocStat)
               !    IF(allocStat .NE. 0) STOP 'Error allocating internal Matrix BD'
               !END IF
         !     ! a_k%BD = matmul(a_k%B,a_k%D) 
               !CALL DGEMM('N','N',n,n,n,1.0d0,a_k%B,n,a_k%D,n,0.0d0,a_k%BD,n) 
               !IF (.NOT. ALLOCATED(a_k%C)) THEN
               !    ALLOCATE(a_k%C(n,n),stat=allocStat)
               !    IF(allocStat .NE. 0) STOP 'Error allocating internal Matrix C'
               !END IF
          !    ! a_k%C = matmul(a_k%BD,transpose(a_k%BD))
               !CALL DGEMM('N','T',n,n,n,1.0d0,a_k%BD,n,a_k%BD,n,0.0d0,a_k%C,n)
               !allocate(a_k%multmp_vN(n))
             !----------(1+1) CMA-ES--------------------------
             !a_k%lambda = 1
             !a_k%mu = 1
             ALLOCATE(a_k%s_i(2*lambda_mo))
             !ALLOCATE(a_k%triuC(n,n),stat=allocStat)
             !ALLOCATE(a_k%arx(n,a_k%lambda),stat=allocStat)
             ALLOCATE(a_k%xold(n),stat=allocStat)
             ALLOCATE(a_k%x(n),stat=allocStat)
             ALLOCATE(a_k%xtrans(n),stat=allocStat)
             ALLOCATE(a_k%Az(n),stat=allocStat)
             ALLOCATE(a_k%z(n),stat=allocStat)
             a_k%xold=0._dp
             a_k%x=0._dp
             a_k%z=0._dp
             a_k%Az=0._dp
             ALLOCATE(a_k%w(n),stat=allocStat)
             a_k%w=0._dp
             !ALLOCATE(a_k%xmean(n))
             !ALLOCATE(a_k%zmean(n))
             !ALLOCATE(a_k%xold(n))
             ALLOCATE(a_k%f(num_obj))
             !ALLOCATE(a_k%tempMat(n,a_k%mu))
             !ALLOCATE(a_k%multmp_mN(n,n))
             !ALLOCATE(a_k%multmp_mN2(n,n))
             !ALLOCATE(a_k%multmp_mMU(a_k%mu,n))
             !ALLOCATE(a_k%mutmp(n,a_k%mu))
             !ALLOCATE(a_k%weightsMat(a_k%mu,a_k%mu))
             !ALLOCATE(a_k%weights(a_k%mu))
             !ALLOCATE(a_k%ps(n))
             ALLOCATE(a_k%pc(n))
             !ALLOCATE(a_k%colVec(n,1))
             !ALLOCATE(a_k%rowVec(1,n))
             do i=1,lambda_mo; a_k%s_i(i)%p=>null(); enddo
             a_k%front = 0
             a_k%penalty = 0._dp
             !a_k%weights = 1._dp !equal weights
             ! normalize weights
             !a_k%weights = a_k%weights/sum(a_k%weights)
                   !  mueff
             !a_k%vecsum = sum(a_k%weights)
             !a_k%vecnorm = sqrt(sum(a_k%weights**2))
         !    this%cmaes_opt%mueff = this%cmaes_opt%vecsum**2 / this%cmaes_opt%vecnorm**2
               !  Strategy internal parameter setting: Adaptation
         !    this%cmaes_opt%a_k%cc = 4.0_dp/(REAL(this%num_var)+4.0_dp)
         !    this%cmaes_opt%a_k%cs = REAL((this%cmaes_opt%mueff+2))/REAL((this%num_var+this%cmaes_opt%mueff+3))
         !    this%cmaes_opt%a_k%mucov = this%cmaes_opt%mueff
         !    this%cmaes_opt%a_k%ccov = (1._dp/this%cmaes_opt%mucov) * 2._dp/(REAL(this%num_var)+1.41)**2 + &
         !                          (1._dp - 1._dp/this%cmaes_opt%mucov) * min(1.0_dp,(2.0_dp*this%cmaes_opt%mueff-1._dp) &
         !                          /((REAL(this%num_var)+2._dp)**2 + this%cmaes_opt%mueff))
         !    this%cmaes_opt%a_k%damps = 1._dp + 2._dp*max(0._dp, sqrt((this%cmaes_opt%mueff-1.0_dp)/ &
         !                           real(this%num_var+1))-1.0_dp)+this%cmaes_opt%cs
             !----------------------default for (1+lambda) strategy----------------------------------
             !a_k%damps = 1._dp + real(n)/(2._dp*a_k%lambda)
             a_k%damps = 1._dp + real(n)/(2._dp)
             !a_k%p_target = 1_dp/(5._dp + sqrt(real(a_k%lambda)/2._dp))
             !a_k%p_target = 1_dp/(5._dp + sqrt(real(1._dp/2._dp)))
             a_k%p_target = 2._dp/11._dp
             !a_k%cp = a_k%p_target*a_k%lambda/ &
             !     (2._dp+a_k%p_target*a_k%lambda)
             !a_k%cp = a_k%p_target/(2._dp+a_k%p_target)
             a_k%cp = 1._dp/12._dp
             a_k%cc = 2._dp/(real(n) + 2._dp)
             a_k%ccov = 2._dp/(real(n)**2 + 6._dp)
             !a_k%ccov = 0.05_dp
             a_k%ccovm = 0.4_dp/(real(n)**1.6_dp + 1._dp)
             a_k%p_tresh = 0.44_dp
             a_k%p_succ = a_k%p_target
             !-----------------------------------------------------------
             !a_k%arx(:,1) = this%x
             !a_k%xmean = this%x
             a_k%sigma = step_size
             a_k%pc = 0._dp
             !a_k%ps = 0._dp
         !    this%cmaes_opt%a_k%hsig = 1._dp
             !a_k%chiN = sqrt(REAL(n))*((1.0_dp-1.0_dp/(4.0_dp*REAL(n))) &
             !                      +1.0_dp/(21.0_dp*REAL(n*n)))
             a_k%d_count = 0
             a_k%s_i_count = 0
             a_k%I_dist = 0._dp
             a_k%kernel_dist = 0._dp
             a_k%hypervol = 0._dp
             a_k%trace = 0._dp
             a_k%det = 0._dp
             a_k%Id = k
             a_k%rmsd = huge(1._dp)
             a_k%dipole = huge(1._dp)
             a_k%f = huge(1._dp) 

 END SUBROUTINE
!===========================================================================
    subroutine write_x(cmaes_opt)
    implicit none
    type(cmaes_opt_type),TARGET                               :: cmaes_opt
    integer                                                   :: i,j,k,l,n,t
    integer                                                   :: unit_number1
    integer                                                   :: unit_number2
    integer                                                   :: unit_number3
    integer                                                   :: unit_number4
    integer                                                   :: unit_number5
           ! do not print file if trace is not valid
           call system('cp fort.13 fort.13.bkp')
           n = cmaes_opt%n
           !do j=1,3*cmaes_opt%lambda_mo
           !  if (cmaes_opt%population(j)%trace /= cmaes_opt%population(j)%trace) then
           !        print *,'trace is NaN yannick'
           !        STOP
           !  endif
           !  if (isNaN(cmaes_opt%population(j)%trace)) then
           !        print *,'trace is NaN yannick'
           !        STOP
           !  endif
           !enddo
           !open(12,position='append')
           call open_file(file_name="fort.13",unit_number=unit_number1, &
                          file_action='write',file_status='replace')
           call open_file(file_name="fort.1",unit_number=unit_number5, &
                          file_action='write',file_status='unknown',file_position='append')

           do i = 1,2*cmaes_opt%lambda_mo
             write(unit_number1,*) &
                          cmaes_opt%population(i)%id, &
                          cmaes_opt%population(i)%f, &
                          cmaes_opt%population(i)%sigma, &
                          cmaes_opt%population(i)%p_succ, &
                          cmaes_opt%generation, &
                          cmaes_opt%population(i)%x, &
                          cmaes_opt%population(i)%penalty, &
                          cmaes_opt%population(i)%front, &
                          cmaes_opt%population(i)%I_dist, &
                          cmaes_opt%population(i)%trace, &
                          cmaes_opt%population(i)%det, &
                          cmaes_opt%population(i)%conv
              !--------------------------------
              if (i .gt. cmaes_opt%lambda_mo) then
              write(unit_number5,*) &
                          cmaes_opt%population(i)%id, &
                          cmaes_opt%population(i)%f, &
                          cmaes_opt%population(i)%sigma, &
                          cmaes_opt%population(i)%p_succ, &
                          cmaes_opt%generation, &
                          cmaes_opt%population(i)%x, &
                          cmaes_opt%population(i)%penalty, &
                          cmaes_opt%population(i)%front, &
                          cmaes_opt%population(i)%I_dist, &
                          cmaes_opt%population(i)%trace, &
                          cmaes_opt%population(i)%det, &
                          cmaes_opt%population(i)%conv
              endif
              t = cmaes_opt%population(i)%id
              call open_file(file_name="A_"//trim(str(t)),unit_number=unit_number2, &
              file_action='write',file_status='replace')
              !----print matrix
              DO k=1,n
                WRITE(unit_number2,*) (cmaes_opt%population(i)%A(k,l), l=1,n)
              END DO
              call close_file(unit_number2)

              call open_file(file_name="Ainv_"//trim(str(t)),unit_number=unit_number3, &
              file_action='write',file_status='replace')
              !----print matrix
              DO k=1,n
                WRITE(unit_number3,*) (cmaes_opt%population(i)%Ainv(k,l), l=1,n)
              END DO
              call close_file(unit_number3)
              call open_file(file_name="pc_"//trim(str(t)),unit_number=unit_number4, &
                        file_action='write',file_status='replace')
              !----print matrix
              DO k=1,n
                WRITE(unit_number4,*) cmaes_opt%population(i)%pc(k)
              END DO
              call close_file(unit_number4)
                   
                !================================
        !        call open_file(file_name="cov_"//trim(str(t)),unit_number=unit_number, &
        !                  file_action='write',file_status='replace')
        !        !----print matrix
        !        DO k=1,n
        !          WRITE(unit_number,*) (cmaes_opt%population(i)%C(k,l), l=1,n)
        !        END DO
        !        call close_file(unit_number)
                !================================
           enddo
           call close_file(unit_number1)
           call close_file(unit_number5)

    end subroutine write_x
!===========================================================================
    subroutine read_x(cmaes_opt)
    implicit none
    type(cmaes_opt_type),TARGET                               :: cmaes_opt
    integer                                                   :: i,j,k,l,n,t
    integer                                                   :: unit_number1
    integer                                                   :: unit_number2
    integer                                                   :: unit_number3
    integer                                                   :: unit_number4
           !open(12,position='append')
           call open_file(file_name="fort.13",unit_number=unit_number1, &
                          file_action='read')
           n = cmaes_opt%n
           do i = 1,2*cmaes_opt%lambda_mo
              read(unit_number1,*) &
                          cmaes_opt%population(i)%id, &
                          cmaes_opt%population(i)%f, &
                          cmaes_opt%population(i)%sigma, &
                          cmaes_opt%population(i)%p_succ, &
                          cmaes_opt%generation, &
                          cmaes_opt%population(i)%x, &
                          cmaes_opt%population(i)%penalty, &
                          cmaes_opt%population(i)%front, &
                          cmaes_opt%population(i)%I_dist, &
                          cmaes_opt%population(i)%trace, &
                          cmaes_opt%population(i)%det, &
                          cmaes_opt%population(i)%conv
                         
              t = cmaes_opt%population(i)%id
              call open_file(file_name="A_"//trim(str(t)),unit_number=unit_number2, &
              file_action='read')
              !----print matrix
              DO k=1,n
                READ(unit_number2,*) (cmaes_opt%population(i)%A(k,l), l=1,n)
              END DO
              call close_file(unit_number2)

              call open_file(file_name="Ainv_"//trim(str(t)),unit_number=unit_number3, &
              file_action='read')
              !----print matrix
              DO k=1,n
                READ(unit_number3,*) (cmaes_opt%population(i)%Ainv(k,l), l=1,n)
              END DO
              call close_file(unit_number3)
              call open_file(file_name="pc_"//trim(str(t)),unit_number=unit_number4, &
                        file_action='read')
              !----print matrix
              DO k=1,n
                READ(unit_number4,*) cmaes_opt%population(i)%pc(k)
              END DO
              call close_file(unit_number4)
              
           enddo
           call close_file(unit_number1)

    end subroutine read_x
 !=============================================================================
     character(len=20) function str(k)
!   "Convert an integer to string."
    integer, intent(in) :: k
    write (str, *) k
    str = adjustl(str)
    end function str
! *****************************************************************************
 SUBROUTINE a_k_copy(a_k2,a_k1)
    IMPLICIT NONE
    type(individual_type)            :: a_k1
    type(individual_type)            :: a_k2
    INTEGER                          :: i,n
             a_k2%A         = a_k1%A 
             a_k2%Ainv      = a_k1%Ainv
             !a_k2%BD        = a_k1%BD
             !a_k2%C         = a_k1%C
             a_k2%lambda    = a_k1%lambda
             !a_k2%mu        = a_k1%mu
             !a_k2%triuC     = a_k1%triuC   
             a_k2%trace     = a_k1%trace       
             a_k2%det       = a_k1%det       
             a_k2%x         = a_k1%x
             a_k2%Az        = a_k1%Az
             a_k2%z         = a_k1%z
             !a_k2%xmean     = a_k1%xmean                
             !a_k2%zmean     = a_k1%zmean                  
             !a_k2%xold      = a_k1%xold                
             !a_k2%tempMat   = a_k1%tempMat                  
             !a_k2%multmp_mN = a_k1%multmp_mN                   
             !a_k2%multmp_mN2= a_k1%multmp_mN2                      
             !a_k2%multmp_mMU= a_k1%multmp_mMU                   
             !a_k2%mutmp     = a_k1%mutmp                    
             !a_k2%weightsMat= a_k1%weightsMat                  
             !a_k2%weights   = a_k1%weights                     
             !a_k2%ps        = a_k1%ps                
             a_k2%pc        = a_k1%pc                    
             !a_k2%colVec    = a_k1%colVec             
             !a_k2%rowVec    = a_k1%rowVec             
             !a_k2%weights   = a_k1%weights                  
             !a_k2%vecsum    = a_k1%vecsum                   
             !a_k2%vecnorm   = a_k1%vecnorm             
             !----------------------default for (1+lambda) strategy----------------------------------
             a_k2%damps     = a_k1%damps 
             a_k2%p_target  = a_k1%p_target             
             a_k2%cp        = a_k1%cp                          
             a_k2%cc        = a_k1%cc                               
             a_k2%ccov      = a_k1%ccov                     
             !a_k2%ccovm     = a_k1%ccovm                     
             a_k2%p_tresh   = a_k1%p_tresh         
             a_k2%p_succ    = a_k1%p_succ           
             !----------------------------------
             a_k2%sigma     = a_k1%sigma      
             a_k2%pc        = a_k1%pc
             !a_k2%ps        = a_k1%ps
             !a_k2%chiN      = a_k1%chiN
             a_k2%f         = a_k1%f
             !-----------------------------------
             a_k2%d_count    = a_k1%d_count
             a_k2%I_dist     = a_k1%I_dist
             a_k2%kernel_dist= a_k1%kernel_dist
             a_k2%hypervol   = a_k1%hypervol
             a_k2%s_i        = a_k1%s_i
             a_k2%s_i_count  = a_k1%s_i_count
             a_k2%id         = a_k1%id
             a_k2%front      = a_k1%front
             !-------------------------------------
             a_k2%penalty    = a_k1%penalty
 END SUBROUTINE
!========================================================================================================
     SUBROUTINE tool_symmatrix(C,n,triuC)
      IMPLICIT NONE
      
      !-------------------------------------------------------------------------
      !  Parameters
      !-------------------------------------------------------------------------
      INTEGER, INTENT(in)							:: n
      REAL(dp),DIMENSION(n,n),INTENT(inout)			:: C
      REAL(dp),DIMENSION(n,n),INTENT(inout)			:: triuC

      !-------------------------------------------------------------------------
      !  Local Variables
      !-------------------------------------------------------------------------
      REAL(dp),DIMENSION(n,n)						:: triuC1,triuC2	
      LOGICAL,DIMENSION(n,n)						:: mask
      INTEGER										:: i,k
      
      	!-----------------------------------------------------------------------
        !  Enforce symmetry
      	!-----------------------------------------------------------------------
	  	triuC1 = 0.
	  	triuC2 = 0.
	  	mask = .FALSE.
	  	DO k = 1, n			! Create mask for upper triangular matrix
	  		DO i = 1, n
	  			IF(k .GE. i) THEN
	  				mask(i,k) = .TRUE.
	  			ELSE
	  				mask(i,k) = .FALSE.
	  			END IF
                if (isNaN(C(k,i))) then
                    print *,'matrix has NaN entries'
                    STOP
                endif
	  		END DO
	  	END DO

	  	WHERE(mask)
	  		triuC1 = C			! Get upper triangular matrix of C
	  	END WHERE				! See Matlab: triu(X)
	  	
	  	DO i = 1, n		! set new mask, see triu(X,k), here: k=1
	  		mask(i,i) = .FALSE.
	  	END DO
	  	
	  	WHERE(mask)
	  		triuC2 = C			! Get second triangular matrix
	  	END WHERE
	  	
	  	C = triuC1 + transpose(triuC2)	! Enforce symmetry
        triuC = triuC1


      RETURN
      END SUBROUTINE
!==============================================================================      
      SUBROUTINE tool_eigendecomp(A,n,eVals,eVecs,info)
      IMPLICIT NONE
      
      !-------------------------------------------------------------------------
      !  Parameters
      !-------------------------------------------------------------------------
      INTEGER, INTENT(in)							:: n
      REAL(dp),DIMENSION(n,n),INTENT(in)			:: A
      REAL(dp),DIMENSION(n,n),INTENT(out)			:: eVals
      REAL(dp),DIMENSION(n,n),OPTIONAL,INTENT(out)	:: eVecs
      INTEGER,INTENT(OUT)                           :: info
      !-------------------------------------------------------------------------
      !  Externals
      !-------------------------------------------------------------------------
	  EXTERNAL 									:: dsyev 
      !-------------------------------------------------------------------------
      !  Local Variables
      !-------------------------------------------------------------------------
      CHARACTER(len=1)							:: job
      INTEGER									:: lWork
      REAL(dp),ALLOCATABLE,DIMENSION(:)			:: work
      INTEGER									:: i
      REAL(dp),DIMENSION(n)						:: eValWork
      INTEGER                                   :: allocStat
     
      !-------------------------------------------------------------------------
      !  Specify job (only compute Eigenvalues('N') or also Eigenvectors('V') )
      !-------------------------------------------------------------------------
      IF(present(eVecs)) THEN
      	job = 'V'
      ELSE
      	job = 'N'
      END IF
      
      !-------------------------------------------------------------------------
      !  Make a copy of A (so it isnt changed unintentionally)
      !-------------------------------------------------------------------------
      eVecs = A
      
      !-------------------------------------------------------------------------
      !  Set length of work array
      !-------------------------------------------------------------------------
      !lWork = 2*n*n+6*n+1
      lWork = -1
      ALLOCATE(work(1),stat=allocStat)
      IF(allocStat .NE. 0) STOP 'Error allocating work(1)'
      CALL DSYEV(job,'U',n,eVecs,n,eValWork,work,-1,info)
      i = work(1)
      DEALLOCATE(work)
      
      ALLOCATE(work(i),stat=allocStat)
      IF(allocStat .NE. 0) STOP 'Error allocating work'
      !-------------------------------------------------------------------------
      !  Call LAPACK Routine
      !-------------------------------------------------------------------------
      CALL DSYEV(job,'U',n,eVecs,n,eValWork,work,i,info)
      !-------------------------------------------------------------------------
      !  Error management
      !-------------------------------------------------------------------------
      IF(info .GT. 0) WRITE(*,*) 'EVD failed to converge'
      IF(info .LT. 0) WRITE(*,*) 'EVD: Illegal value'
      
      !-------------------------------------------------------------------------
      !  Set output variables
      !-------------------------------------------------------------------------
      DO i = 1,n
      	eVals(i,i) = eValWork(i) ! Convert Vector to Matrix
      END DO


      RETURN
      END SUBROUTINE
! *****************************************************************************
	  Subroutine D_mrgrnk (XDONT, IRNGT)
! __________________________________________________________
!   MRGRNK = Merge-sort ranking of an array
!   For performance reasons, the first 2 passes are taken
!   out of the standard loop, and use dedicated coding.
! __________________________________________________________
! __________________________________________________________
      Real (kind=dp), Dimension (:), Intent (In) :: XDONT
      Integer, Dimension (:), Intent (Out) :: IRNGT
! __________________________________________________________
      Real (kind=dp) :: XVALA, XVALB
!
      Integer, Dimension (SIZE(IRNGT)) :: JWRKT
      Integer :: LMTNA, LMTNC, IRNG1, IRNG2
      Integer :: NVAL, IIND, IWRKD, IWRK, IWRKF, JINDA, IINDA, IINDB
!
      NVAL = Min (SIZE(XDONT), SIZE(IRNGT))
      Select Case (NVAL)
      Case (:0)
         Return
      Case (1)
         IRNGT (1) = 1
         Return
      Case Default
         Continue
      End Select
!
!  Fill-in the index array, creating ordered couples
!
      Do IIND = 2, NVAL, 2
         If (XDONT(IIND-1) <= XDONT(IIND)) Then
            IRNGT (IIND-1) = IIND - 1
            IRNGT (IIND) = IIND
         Else
            IRNGT (IIND-1) = IIND
            IRNGT (IIND) = IIND - 1
         End If
      End Do
      If (Modulo(NVAL, 2) /= 0) Then
         IRNGT (NVAL) = NVAL
      End If
!
!  We will now have ordered subsets A - B - A - B - ...
!  and merge A and B couples into     C   -   C   - ...
!
      LMTNA = 2
      LMTNC = 4
!
!  First iteration. The length of the ordered subsets goes from 2 to 4
!
      Do
         If (NVAL <= 2) Exit
!
!   Loop on merges of A and B into C
!
         Do IWRKD = 0, NVAL - 1, 4
            If ((IWRKD+4) > NVAL) Then
               If ((IWRKD+2) >= NVAL) Exit
!
!   1 2 3
!
               If (XDONT(IRNGT(IWRKD+2)) <= XDONT(IRNGT(IWRKD+3))) Exit
!
!   1 3 2
!
               If (XDONT(IRNGT(IWRKD+1)) <= XDONT(IRNGT(IWRKD+3))) Then
                  IRNG2 = IRNGT (IWRKD+2)
                  IRNGT (IWRKD+2) = IRNGT (IWRKD+3)
                  IRNGT (IWRKD+3) = IRNG2
!
!   3 1 2
!
               Else
                  IRNG1 = IRNGT (IWRKD+1)
                  IRNGT (IWRKD+1) = IRNGT (IWRKD+3)
                  IRNGT (IWRKD+3) = IRNGT (IWRKD+2)
                  IRNGT (IWRKD+2) = IRNG1
               End If
               Exit
            End If
!
!   1 2 3 4
!
            If (XDONT(IRNGT(IWRKD+2)) <= XDONT(IRNGT(IWRKD+3))) Cycle
!
!   1 3 x x
!
            If (XDONT(IRNGT(IWRKD+1)) <= XDONT(IRNGT(IWRKD+3))) Then
               IRNG2 = IRNGT (IWRKD+2)
               IRNGT (IWRKD+2) = IRNGT (IWRKD+3)
               If (XDONT(IRNG2) <= XDONT(IRNGT(IWRKD+4))) Then
!   1 3 2 4
                  IRNGT (IWRKD+3) = IRNG2
               Else
!   1 3 4 2
                  IRNGT (IWRKD+3) = IRNGT (IWRKD+4)
                  IRNGT (IWRKD+4) = IRNG2
               End If
!
!   3 x x x
!
            Else
               IRNG1 = IRNGT (IWRKD+1)
               IRNG2 = IRNGT (IWRKD+2)
               IRNGT (IWRKD+1) = IRNGT (IWRKD+3)
               If (XDONT(IRNG1) <= XDONT(IRNGT(IWRKD+4))) Then
                  IRNGT (IWRKD+2) = IRNG1
                  If (XDONT(IRNG2) <= XDONT(IRNGT(IWRKD+4))) Then
!   3 1 2 4
                     IRNGT (IWRKD+3) = IRNG2
                  Else
!   3 1 4 2
                     IRNGT (IWRKD+3) = IRNGT (IWRKD+4)
                     IRNGT (IWRKD+4) = IRNG2
                  End If
               Else
!   3 4 1 2
                  IRNGT (IWRKD+2) = IRNGT (IWRKD+4)
                  IRNGT (IWRKD+3) = IRNG1
                  IRNGT (IWRKD+4) = IRNG2
               End If
            End If
         End Do
!
!  The Cs become As and Bs
!
         LMTNA = 4
         Exit
      End Do
!
!  Iteration loop. Each time, the length of the ordered subsets
!  is doubled.
!
      Do
         If (LMTNA >= NVAL) Exit
         IWRKF = 0
         LMTNC = 2 * LMTNC
!
!   Loop on merges of A and B into C
!
         Do
            IWRK = IWRKF
            IWRKD = IWRKF + 1
            JINDA = IWRKF + LMTNA
            IWRKF = IWRKF + LMTNC
            If (IWRKF >= NVAL) Then
               If (JINDA >= NVAL) Exit
               IWRKF = NVAL
            End If
            IINDA = 1
            IINDB = JINDA + 1
!
!   Shortcut for the case when the max of A is smaller
!   than the min of B. This line may be activated when the
!   initial set is already close to sorted.
!
!          IF (XDONT(IRNGT(JINDA)) <= XDONT(IRNGT(IINDB))) CYCLE
!
!  One steps in the C subset, that we build in the final rank array
!
!  Make a copy of the rank array for the merge iteration
!
            JWRKT (1:LMTNA) = IRNGT (IWRKD:JINDA)
!
            XVALA = XDONT (JWRKT(IINDA))
            XVALB = XDONT (IRNGT(IINDB))
!
            Do
               IWRK = IWRK + 1
!
!  We still have unprocessed values in both A and B
!
               If (XVALA > XVALB) Then
                  IRNGT (IWRK) = IRNGT (IINDB)
                  IINDB = IINDB + 1
                  If (IINDB > IWRKF) Then
!  Only A still with unprocessed values
                     IRNGT (IWRK+1:IWRKF) = JWRKT (IINDA:LMTNA)
                     Exit
                  End If
                  XVALB = XDONT (IRNGT(IINDB))
               Else
                  IRNGT (IWRK) = JWRKT (IINDA)
                  IINDA = IINDA + 1
                  If (IINDA > LMTNA) Exit! Only B still with unprocessed values
                  XVALA = XDONT (JWRKT(IINDA))
               End If
!
            End Do
         End Do
!
!  The Cs become As and Bs
!
         LMTNA = 2 * LMTNA
      End Do
!
      Return
!
	  End Subroutine D_mrgrnk

	  Subroutine R_mrgrnk (XDONT, IRNGT)
! __________________________________________________________
!   MRGRNK = Merge-sort ranking of an array
!   For performance reasons, the first 2 passes are taken
!   out of the standard loop, and use dedicated coding.
! __________________________________________________________
! _________________________________________________________
      Real, Dimension (:), Intent (In) :: XDONT
      Integer, Dimension (:), Intent (Out) :: IRNGT
! __________________________________________________________
      Real :: XVALA, XVALB
!
      Integer, Dimension (SIZE(IRNGT)) :: JWRKT
      Integer :: LMTNA, LMTNC, IRNG1, IRNG2
      Integer :: NVAL, IIND, IWRKD, IWRK, IWRKF, JINDA, IINDA, IINDB
!
      NVAL = Min (SIZE(XDONT), SIZE(IRNGT))
      Select Case (NVAL)
      Case (:0)
         Return
      Case (1)
         IRNGT (1) = 1
         Return
      Case Default
         Continue
      End Select
!
!  Fill-in the index array, creating ordered couples
!
      Do IIND = 2, NVAL, 2
         If (XDONT(IIND-1) <= XDONT(IIND)) Then
            IRNGT (IIND-1) = IIND - 1
            IRNGT (IIND) = IIND
         Else
            IRNGT (IIND-1) = IIND
            IRNGT (IIND) = IIND - 1
         End If
      End Do
      If (Modulo(NVAL, 2) /= 0) Then
         IRNGT (NVAL) = NVAL
      End If
!
!  We will now have ordered subsets A - B - A - B - ...
!  and merge A and B couples into     C   -   C   - ...
!
      LMTNA = 2
      LMTNC = 4
!
!  First iteration. The length of the ordered subsets goes from 2 to 4
!
      Do
         If (NVAL <= 2) Exit
!
!   Loop on merges of A and B into C
!
         Do IWRKD = 0, NVAL - 1, 4
            If ((IWRKD+4) > NVAL) Then
               If ((IWRKD+2) >= NVAL) Exit
!
!   1 2 3
!
               If (XDONT(IRNGT(IWRKD+2)) <= XDONT(IRNGT(IWRKD+3))) Exit
!
!   1 3 2
!
               If (XDONT(IRNGT(IWRKD+1)) <= XDONT(IRNGT(IWRKD+3))) Then
                  IRNG2 = IRNGT (IWRKD+2)
                  IRNGT (IWRKD+2) = IRNGT (IWRKD+3)
                  IRNGT (IWRKD+3) = IRNG2
!
!   3 1 2
!
               Else
                  IRNG1 = IRNGT (IWRKD+1)
                  IRNGT (IWRKD+1) = IRNGT (IWRKD+3)
                  IRNGT (IWRKD+3) = IRNGT (IWRKD+2)
                  IRNGT (IWRKD+2) = IRNG1
               End If
               Exit
            End If
!
!   1 2 3 4
!
            If (XDONT(IRNGT(IWRKD+2)) <= XDONT(IRNGT(IWRKD+3))) Cycle
!
!   1 3 x x
!
            If (XDONT(IRNGT(IWRKD+1)) <= XDONT(IRNGT(IWRKD+3))) Then
               IRNG2 = IRNGT (IWRKD+2)
               IRNGT (IWRKD+2) = IRNGT (IWRKD+3)
               If (XDONT(IRNG2) <= XDONT(IRNGT(IWRKD+4))) Then
!   1 3 2 4
                  IRNGT (IWRKD+3) = IRNG2
               Else
!   1 3 4 2
                  IRNGT (IWRKD+3) = IRNGT (IWRKD+4)
                  IRNGT (IWRKD+4) = IRNG2
               End If
!
!   3 x x x
!
            Else
               IRNG1 = IRNGT (IWRKD+1)
               IRNG2 = IRNGT (IWRKD+2)
               IRNGT (IWRKD+1) = IRNGT (IWRKD+3)
               If (XDONT(IRNG1) <= XDONT(IRNGT(IWRKD+4))) Then
                  IRNGT (IWRKD+2) = IRNG1
                  If (XDONT(IRNG2) <= XDONT(IRNGT(IWRKD+4))) Then
!   3 1 2 4
                     IRNGT (IWRKD+3) = IRNG2
                  Else
!   3 1 4 2
                     IRNGT (IWRKD+3) = IRNGT (IWRKD+4)
                     IRNGT (IWRKD+4) = IRNG2
                  End If
               Else
!   3 4 1 2
                  IRNGT (IWRKD+2) = IRNGT (IWRKD+4)
                  IRNGT (IWRKD+3) = IRNG1
                  IRNGT (IWRKD+4) = IRNG2
               End If
            End If
         End Do
!
!  The Cs become As and Bs
!
         LMTNA = 4
         Exit
      End Do
!
!  Iteration loop. Each time, the length of the ordered subsets
!  is doubled.
!
      Do
         If (LMTNA >= NVAL) Exit
         IWRKF = 0
         LMTNC = 2 * LMTNC
!
!   Loop on merges of A and B into C
!
         Do
            IWRK = IWRKF
            IWRKD = IWRKF + 1
            JINDA = IWRKF + LMTNA
            IWRKF = IWRKF + LMTNC
            If (IWRKF >= NVAL) Then
               If (JINDA >= NVAL) Exit
               IWRKF = NVAL
            End If
            IINDA = 1
            IINDB = JINDA + 1
!
!   Shortcut for the case when the max of A is smaller
!   than the min of B. This line may be activated when the
!   initial set is already close to sorted.
!
!          IF (XDONT(IRNGT(JINDA)) <= XDONT(IRNGT(IINDB))) CYCLE
!
!  One steps in the C subset, that we build in the final rank array
!
!  Make a copy of the rank array for the merge iteration
!
            JWRKT (1:LMTNA) = IRNGT (IWRKD:JINDA)
!
            XVALA = XDONT (JWRKT(IINDA))
            XVALB = XDONT (IRNGT(IINDB))
!
            Do
               IWRK = IWRK + 1
!
!  We still have unprocessed values in both A and B
!
               If (XVALA > XVALB) Then
                  IRNGT (IWRK) = IRNGT (IINDB)
                  IINDB = IINDB + 1
                  If (IINDB > IWRKF) Then
!  Only A still with unprocessed values
                     IRNGT (IWRK+1:IWRKF) = JWRKT (IINDA:LMTNA)
                     Exit
                  End If
                  XVALB = XDONT (IRNGT(IINDB))
               Else
                  IRNGT (IWRK) = JWRKT (IINDA)
                  IINDA = IINDA + 1
                  If (IINDA > LMTNA) Exit! Only B still with unprocessed values
                  XVALA = XDONT (JWRKT(IINDA))
               End If
!
            End Do
         End Do
!
!  The Cs become As and Bs
!
         LMTNA = 2 * LMTNA
      End Do
!
      Return
!
	  End Subroutine R_mrgrnk
	  Subroutine I_mrgrnk (XDONT, IRNGT)
! __________________________________________________________
!   MRGRNK = Merge-sort ranking of an array
!   For performance reasons, the first 2 passes are taken
!   out of the standard loop, and use dedicated coding.
! __________________________________________________________
! __________________________________________________________
      Integer, Dimension (:), Intent (In)  :: XDONT
      Integer, Dimension (:), Intent (Out) :: IRNGT
! __________________________________________________________
      Integer :: XVALA, XVALB
!
      Integer, Dimension (SIZE(IRNGT)) :: JWRKT
      Integer :: LMTNA, LMTNC, IRNG1, IRNG2
      Integer :: NVAL, IIND, IWRKD, IWRK, IWRKF, JINDA, IINDA, IINDB
!
      NVAL = Min (SIZE(XDONT), SIZE(IRNGT))
      Select Case (NVAL)
      Case (:0)
         Return
      Case (1)
         IRNGT (1) = 1
         Return
      Case Default
         Continue
      End Select
!
!  Fill-in the index array, creating ordered couples
!
      Do IIND = 2, NVAL, 2
         If (XDONT(IIND-1) <= XDONT(IIND)) Then
            IRNGT (IIND-1) = IIND - 1
            IRNGT (IIND) = IIND
         Else
            IRNGT (IIND-1) = IIND
            IRNGT (IIND) = IIND - 1
         End If
      End Do
      If (Modulo(NVAL, 2) /= 0) Then
         IRNGT (NVAL) = NVAL
      End If
!
!  We will now have ordered subsets A - B - A - B - ...
!  and merge A and B couples into     C   -   C   - ...
!
      LMTNA = 2
      LMTNC = 4
!
!  First iteration. The length of the ordered subsets goes from 2 to 4
!
      Do
         If (NVAL <= 2) Exit
!
!   Loop on merges of A and B into C
!
         Do IWRKD = 0, NVAL - 1, 4
            If ((IWRKD+4) > NVAL) Then
               If ((IWRKD+2) >= NVAL) Exit
!
!   1 2 3
!
               If (XDONT(IRNGT(IWRKD+2)) <= XDONT(IRNGT(IWRKD+3))) Exit
!
!   1 3 2
!
               If (XDONT(IRNGT(IWRKD+1)) <= XDONT(IRNGT(IWRKD+3))) Then
                  IRNG2 = IRNGT (IWRKD+2)
                  IRNGT (IWRKD+2) = IRNGT (IWRKD+3)
                  IRNGT (IWRKD+3) = IRNG2
!
!   3 1 2
!
               Else
                  IRNG1 = IRNGT (IWRKD+1)
                  IRNGT (IWRKD+1) = IRNGT (IWRKD+3)
                  IRNGT (IWRKD+3) = IRNGT (IWRKD+2)
                  IRNGT (IWRKD+2) = IRNG1
               End If
               Exit
            End If
!
!   1 2 3 4
!
            If (XDONT(IRNGT(IWRKD+2)) <= XDONT(IRNGT(IWRKD+3))) Cycle
!
!   1 3 x x
!
            If (XDONT(IRNGT(IWRKD+1)) <= XDONT(IRNGT(IWRKD+3))) Then
               IRNG2 = IRNGT (IWRKD+2)
               IRNGT (IWRKD+2) = IRNGT (IWRKD+3)
               If (XDONT(IRNG2) <= XDONT(IRNGT(IWRKD+4))) Then
!   1 3 2 4
                  IRNGT (IWRKD+3) = IRNG2
               Else
!   1 3 4 2
                  IRNGT (IWRKD+3) = IRNGT (IWRKD+4)
                  IRNGT (IWRKD+4) = IRNG2
               End If
!
!   3 x x x
!
            Else
               IRNG1 = IRNGT (IWRKD+1)
               IRNG2 = IRNGT (IWRKD+2)
               IRNGT (IWRKD+1) = IRNGT (IWRKD+3)
               If (XDONT(IRNG1) <= XDONT(IRNGT(IWRKD+4))) Then
                  IRNGT (IWRKD+2) = IRNG1
                  If (XDONT(IRNG2) <= XDONT(IRNGT(IWRKD+4))) Then
!   3 1 2 4
                     IRNGT (IWRKD+3) = IRNG2
                  Else
!   3 1 4 2
                     IRNGT (IWRKD+3) = IRNGT (IWRKD+4)
                     IRNGT (IWRKD+4) = IRNG2
                  End If
               Else
!   3 4 1 2
                  IRNGT (IWRKD+2) = IRNGT (IWRKD+4)
                  IRNGT (IWRKD+3) = IRNG1
                  IRNGT (IWRKD+4) = IRNG2
               End If
            End If
         End Do
!
!  The Cs become As and Bs
!
         LMTNA = 4
         Exit
      End Do
!
!  Iteration loop. Each time, the length of the ordered subsets
!  is doubled.
!
      Do
         If (LMTNA >= NVAL) Exit
         IWRKF = 0
         LMTNC = 2 * LMTNC
!
!   Loop on merges of A and B into C
!
         Do
            IWRK = IWRKF
            IWRKD = IWRKF + 1
            JINDA = IWRKF + LMTNA
            IWRKF = IWRKF + LMTNC
            If (IWRKF >= NVAL) Then
               If (JINDA >= NVAL) Exit
               IWRKF = NVAL
            End If
            IINDA = 1
            IINDB = JINDA + 1
!
!   Shortcut for the case when the max of A is smaller
!   than the min of B. This line may be activated when the
!   initial set is already close to sorted.
!
!          IF (XDONT(IRNGT(JINDA)) <= XDONT(IRNGT(IINDB))) CYCLE
!
!  One steps in the C subset, that we build in the final rank array
!
!  Make a copy of the rank array for the merge iteration
!
            JWRKT (1:LMTNA) = IRNGT (IWRKD:JINDA)
!
            XVALA = XDONT (JWRKT(IINDA))
            XVALB = XDONT (IRNGT(IINDB))
!
            Do
               IWRK = IWRK + 1
!
!  We still have unprocessed values in both A and B
!
               If (XVALA > XVALB) Then
                  IRNGT (IWRK) = IRNGT (IINDB)
                  IINDB = IINDB + 1
                  If (IINDB > IWRKF) Then
!  Only A still with unprocessed values
                     IRNGT (IWRK+1:IWRKF) = JWRKT (IINDA:LMTNA)
                     Exit
                  End If
                  XVALB = XDONT (IRNGT(IINDB))
               Else
                  IRNGT (IWRK) = JWRKT (IINDA)
                  IINDA = IINDA + 1
                  If (IINDA > LMTNA) Exit! Only B still with unprocessed values
                  XVALA = XDONT (JWRKT(IINDA))
               End If
!
            End Do
         End Do
!
!  The Cs become As and Bs
!
         LMTNA = 2 * LMTNA
      End Do
!
      Return
!
	  End Subroutine I_mrgrnk

END MODULE cma_es_utils
