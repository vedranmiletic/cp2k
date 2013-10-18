!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2012  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief performs global geometry optimization
!> \par History
!> \author Ole
! *****************************************************************************
MODULE paramopt_master
  USE cp_lbfgs,                        ONLY: setulb

  USE swarm_message,                   ONLY: swarm_message_type

  USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                             section_vals_type,&
                                             section_vals_val_get
  USE input_constants,                 ONLY: paramopt_no_run,&
                                             paramopt_sampling_run,&
                                             paramopt_powell_run,&
                                             paramopt_gaussian_run, &
                                             paramopt_lbfgs_run, &
                                             paramopt_cmaes_run
  USE kinds,                           ONLY: default_string_length,&
                                             dp
  USE message_passing,                 ONLY: mp_bcast
  USE parallel_rng_types,              ONLY: rng_stream_type, &
                                             create_rng_stream, &
                                             next_random_number, &
                                             reset_to_next_rng_substream, &
                                             GAUSSIAN, &
                                             UNIFORM
  USE physcon,                         ONLY: angstrom,&
                                             kelvin
  USE powell,                          ONLY: powell_optimize, &
                                             opt_state_type
  USE swarm_message,                   ONLY: swarm_message_type,&
                                             swarm_message_add,&
                                             swarm_message_get,&
                                             swarm_message_haskey,&
                                             swarm_message_length
  USE cp_para_types,                   ONLY: cp_para_env_type
#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'paramopt'


 PUBLIC :: paramopt_master_type
 PUBLIC :: paramopt_master_init, paramopt_master_finalize
 PUBLIC :: paramopt_master_steer
  
 TYPE :: individual_type
   PRIVATE
   INTEGER                                         :: id
   INTEGER                                         :: sort_idx
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
   REAL(KIND=dp)                                   :: cc,cp,cs,ccov,damps
   REAL(KIND=dp)                                   :: p_succ,p_target,p_tresh
   ! initialize dynamic (internal) startegy parameters and constants
   REAL(KIND=dp),ALLOCATABLE,DIMENSION(:)          :: pc,ps
   REAL(KIND=dp)                                   :: eigenval
   REAL(KIND=dp)                                   :: chiN
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
   INTEGER,DIMENSION(:),ALLOCATABLE                :: s_i
   INTEGER                                         :: s_i_count
   !--------------------------------------------------------------------------
   REAL(KIND=dp)                                   :: penalty
 END TYPE individual_type  
  
 TYPE :: individual_type_p
    TYPE(individual_type),POINTER                  :: p
 END TYPE

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

 TYPE cmaes_opt_type
   INTEGER                                              :: n
   INTEGER                                              :: lambda_mo
   INTEGER                                              :: info
   INTEGER                                              :: generation
!   REAL(KIND=dp),DIMENSION(:),POINTER                   :: newsort
!   INTEGER,ALLOCATABLE                                  :: idxsel(:) 
   !--bounderies
   REAL(KIND=dp),DIMENSION(:),POINTER                   :: l_bound,u_bound
   !--stop criteria--------------
   REAL(KIND=dp)                                        :: stopfitness 
   REAL(KIND=dp)                                        :: stopeval
   TYPE(individual_type),ALLOCATABLE                    :: a_k_temp(:)
   TYPE(individual_type),ALLOCATABLE                    :: a_k(:)
   TYPE(individual_type),ALLOCATABLE                    :: a_k_parents(:)
   TYPE(rng_stream_type),POINTER                        :: rng_normal_stream
 END TYPE cmaes_opt_type

 TYPE paramopt_master_type
   PRIVATE
   INTEGER                                               :: iw
   INTEGER                                               :: num_inp
   INTEGER                                               :: num_calc
   INTEGER                                               :: num_ref
   INTEGER                                               :: num_var
   INTEGER                                               :: num_obj
   INTEGER                                               :: num_para_set
   INTEGER                                               :: n_walkers
   INTEGER                                               :: inp_iter
   INTEGER                                               :: run_iter
   INTEGER                                               :: run_type_id
   INTEGER                                               :: input_seed
   INTEGER,ALLOCATABLE                                   :: i_recv(:)
   TYPE(cp_error_type),POINTER                           :: error
   TYPE(ener_ref_type),POINTER                           :: ener_ref(:)
   CHARACTER(len=default_string_length),POINTER          :: input_files(:)
   CHARACTER(len=default_string_length),POINTER          :: coord_files(:)
   INTEGER,POINTER                                       :: charges(:)
   REAL(KIND=dp),ALLOCATABLE                             :: Epot(:),timings(:)
   REAL(KIND=dp),POINTER                                 :: res(:)
   REAL(KIND=dp),POINTER                                 :: E_res(:)
   REAL(KIND=dp),POINTER                                 :: rmsd(:)
   REAL(KIND=dp),POINTER                                 :: function_matrix(:,:)
   REAL(KIND=dp),POINTER                                 :: obj_f(:)
   REAL(KIND=dp),POINTER                                 :: grad_obj_f(:)
   REAL(KIND=dp),POINTER                                 :: x(:)
   REAL(KIND=dp),POINTER                                 :: x_temp(:)
   REAL(KIND=dp),POINTER                                 :: penalty
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
   TYPE(cp_para_env_type), POINTER                       :: para_env
   TYPE(rng_stream_type),POINTER                         :: rng_stream
   LOGICAL,POINTER                                       :: sampling
   LOGICAL                                               :: calc_deriv,benchmark
   LOGICAL                                               :: multi_obj
   REAL(KIND=dp)                                         :: step_size
   REAL(KIND=dp),POINTER                                 :: weights(:)
   TYPE(opt_state_type),POINTER                          :: powell_opt
   TYPE(gauss_opt_type),POINTER                          :: gauss_opt
   TYPE(lbfgs_opt_type),POINTER                          :: lbfgs_opt
   TYPE(cmaes_opt_type),POINTER                          :: cmaes_opt
 END TYPE paramopt_master_type

 TYPE :: ener_ref_type
   PRIVATE
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

 CONTAINS

! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_init(this, para_env, swarm_section, input_path, n_walkers, iw, error)
    TYPE(paramopt_master_type)               :: this
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: swarm_section
    CHARACTER(LEN=*), INTENT(IN)             :: input_path
    CHARACTER(len=default_string_length)     :: output_path, str
    INTEGER, INTENT(IN)                      :: n_walkers, iw
    TYPE(cp_error_type), INTENT(inout)       :: error
    TYPE(section_vals_type), POINTER         :: paramopt_section => null()
    TYPE(section_vals_type), POINTER         :: cmaes_section => null()
    INTEGER                                  :: i,charge
    INTEGER                                  :: time(3)
    LOGICAL                                  :: restart
    REAL(kind=dp),DIMENSION(3,2)             :: seed

    type(individual_type),POINTER            :: a_k
    type(individual_type),POINTER            :: a_k_parents
    INTEGER                                  :: k
    !---------------------
    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init called"
    !====================num para set======================== 
    this%num_para_set = 1
    !-------------------
    paramopt_section => section_vals_get_subs_vals(swarm_section, "PARAM_OPT", error=error)
    CALL section_vals_val_get(paramopt_section,"RUN_TYPE",i_val=this%run_type_id,error=error)
    if (this%run_type_id .eq. paramopt_cmaes_run) then
      cmaes_section => section_vals_get_subs_vals(paramopt_section, "CMAES_OPT", error=error)
      CALL section_vals_val_get(cmaes_section,"LAMBDA_MO",i_val=this%num_para_set,error=error)
    endif
    !------------------
    CALL section_vals_val_get(paramopt_section,"NUM_VAR", i_val=this%num_var, error=error)
    CALL section_vals_val_get(paramopt_section,"SEED", i_val=this%input_seed, error=error)
    CALL section_vals_val_get(paramopt_section,"BENCHMARK", l_val=this%benchmark, error=error)
    !------------------
    this%num_obj = 2
    this%multi_obj = .true.
    !-------------------
    this%num_inp = 31
    if (this%benchmark) this%num_inp = 2 !is basically the number of objectives
    !-------------------
    this%num_calc = 31
    !--------------------
    this%num_ref = 30
    !-------------------
    this%calc_deriv = .false.
    if (this%run_type_id .eq. paramopt_lbfgs_run) this%calc_deriv = .true.
    if (this%calc_deriv) this%num_para_set = this%num_var+1
    !--------------------
    this%iw = iw
    !------------------------
    this%para_env => para_env
    this%n_walkers = n_walkers
    !-------------------
    nullify(this%powell_opt,this%gauss_opt,this%sampling, &
            this%lbfgs_opt,this%cmaes_opt)
    !-------------------
    allocate(this%E_res(this%num_para_set))
    allocate(this%rmsd(this%num_para_set))
    allocate(this%ener_ref(this%num_ref))
    allocate(this%Epot(this%num_ref))

    allocate(this%x(this%num_var))
    allocate(this%eps(this%num_var))
    allocate(this%x_temp(this%num_var))
    allocate(this%xtrans(this%num_var))
    allocate(this%xtransold(this%num_var))
    allocate(this%xold(this%num_var))
    allocate(this%obj_f(this%num_para_set))
    if (this%calc_deriv) allocate(this%grad_obj_f(this%num_var))
    allocate(this%l_bound(this%num_var))
    allocate(this%u_bound(this%num_var))
    ALLOCATE(this%input_files(this%num_inp))
    ALLOCATE(this%coord_files(this%num_inp))
    ALLOCATE(this%charges(this%num_inp))
    allocate(this%res(this%num_inp))
    allocate(this%rng_stream)

    allocate(this%timings(this%num_inp))
    allocate(this%weights(this%num_obj))
    allocate(this%i_recv(this%num_para_set))
    allocate(this%function_matrix(this%num_inp,this%num_para_set))

    CALL section_vals_val_get(paramopt_section,"STEP_SIZE", r_val=this%step_size, error=error)
    CALL section_vals_val_get(paramopt_section,"UPPER_BOUNDS", r_vals=this%u_bound, error=error)
    CALL section_vals_val_get(paramopt_section,"LOWER_BOUNDS", r_vals=this%l_bound, error=error)
    CALL section_vals_val_get(paramopt_section,"WEIGHTS", r_vals=this%weights, error=error)
    CALL section_vals_val_get(paramopt_section,"RUN_TYPE",i_val=this%run_type_id,error=error)
    print *,'num var',this%num_var
    print *,'run_type',this%run_type_id,'step size',this%step_size
    print *,'upper bounds',this%u_bound
    print *,'lower bounds',this%l_bound
    
    !-------------------
    this%inp_iter = this%num_inp*this%num_para_set
    this%run_iter = 1
    this%i_recv = 0
    this%Epot = 0._dp
    !allocate(this%penalty)
    !this%penalty = 0._dp
    if (this%input_seed .gt. 0) then
        !deterministic seed
        seed(:,:) = &
            RESHAPE( (/this%input_seed*42.0_dp, &
                       this%input_seed*54.0_dp, &
                       this%input_seed*63.0_dp, &
                       this%input_seed*98.0_dp, &
                       this%input_seed*10.0_dp, &
                       this%input_seed*2.0_dp/),&
                     (/ 3, 2 /) )        
    else
        !random seed from itime
        call itime(time)
        seed(:,1) = REAL(time)
        seed(:,2) = (/ 1._dp, 2._dp, 3._dp /)
    endif
    CALL create_rng_stream(this%rng_stream,'normal',distribution_type=UNIFORM,seed=seed,error=this%error)
    !---------------------
    this%res = 0._dp
    this%E_res = 0._dp
    !------------------------
    ! chose optim method
    SELECT CASE (this%run_type_id)
        CASE (paramopt_no_run)
        ! do nothing
        CASE (paramopt_sampling_run)
            allocate(this%sampling)
        CASE (paramopt_powell_run)
            allocate(this%powell_opt)
        CASE (paramopt_gaussian_run)
            allocate(this%gauss_opt)
        CASE (paramopt_lbfgs_run)
            allocate(this%lbfgs_opt)
        CASE (paramopt_cmaes_run)
            allocate(this%cmaes_opt)
        CASE default
       ! CPAssert(.FALSE.,cp_failure_level,routineP,error,failure)
    END SELECT
    !-----read x --------
    !random x
        do i=1,this%num_var; this%xtrans(i) = next_random_number(this%rng_stream,error=this%error); enddo           
        call var_back_trans(this%xtrans,this%x,this%l_bound,this%u_bound)
        !end random x
    if (.not. this%benchmark) then
      !CALL read4file(this%num_var,this%x,restart)
      CALL para2file(this%num_var,this%x,"para.xml","para.xml.out")
    endif
      print *,'this is inital x'
      print *, this%x
    !----------------set pointers for bfgs------------------------
    this%eps = this%eps_factor*abs(this%x)
        !----check lowest eps....eps should not be zero!
        do i=1,this%num_var
        if (this%eps(i) .eq. 0._dp) this%eps(i) = this%eps_factor
        enddo
    !----------------set pointers for sampling------------------------
    if (associated(this%sampling)) then
        this%sampling = .true.
        allocate(this%obj_f(this%num_para_set))
    endif
    !----------------set pointers gauss adapt------------------------
    if (associated(this%gauss_opt)) then
    CALL section_vals_val_get(paramopt_section,"C_T", r_val=this%gauss_opt%c_t, error=error)
    allocate(this%gauss_opt%eta(this%num_var),this%gauss_opt%m(this%num_var), &
             this%gauss_opt%xmin(this%num_var))
    allocate(this%gauss_opt%x(this%num_var),this%gauss_opt%Q(this%num_var,this%num_var), &
             this%gauss_opt%C(this%num_var,this%num_var))
    allocate(this%gauss_opt%rng_normal_stream)
    CALL create_rng_stream(this%gauss_opt%rng_normal_stream,'gaussian',seed=seed,distribution_type=GAUSSIAN,error=this%error)
    this%gauss_opt%task = 'startx'
    this%gauss_opt%l_bound => this%l_bound
    this%gauss_opt%u_bound => this%u_bound
    this%gauss_opt%r = this%step_size
    this%gauss_opt%x = this%x
    this%gauss_opt%f => this%obj_f(1)
    CALL GaussAdapt(this%num_var,this%xtrans,this%gauss_opt)
    endif
    !----gauss------------
    if (associated(this%powell_opt)) then
    this%powell_opt%state = 0
    this%powell_opt%maxfun = 500 !----change that to imput files value
    this%powell_opt%rhobeg = this%step_size
    this%powell_opt%rhoend = 0.000001_dp
    this%powell_opt%iprint = 1
    this%powell_opt%nvar = this%num_var
    this%powell_opt%f => this%obj_f(1)
    CALL powell_optimize(this%num_var,this%xtrans,this%powell_opt)
    endif
    !------bfgs--------------
    if (associated(this%lbfgs_opt)) then
    this%lbfgs_opt%lower_bound => this%l_bound
    this%lbfgs_opt%upper_bound => this%u_bound
    this%lbfgs_opt%x => this%x
    this%lbfgs_opt%gradient => this%grad_obj_f
    this%lbfgs_opt%f => this%obj_f(1)
    this%lbfgs_opt%m = this%num_var ! m for the limited memory bfgs
    this%lbfgs_opt%ref_count = 0
    this%lbfgs_opt%id_nr = 0
    this%lbfgs_opt%print_every = 200
    allocate(this%lbfgs_opt%kind_of_bound(this%num_var))
    allocate(this%lbfgs_opt%i_work_array(3*this%num_var))
    allocate(this%lbfgs_opt%isave(44))
    allocate(this%lbfgs_opt%dsave(30))
    this%lbfgs_opt%dsave(30) = 0.5_dp ! trust radius for lbfgs
    this%lbfgs_opt%lenwa = 2*this%lbfgs_opt%m*this%num_var + 5*this%num_var &
                           + 11*this%lbfgs_opt%m*this%lbfgs_opt%m + 8*this%lbfgs_opt%m
    allocate(this%lbfgs_opt%work_array(this%lbfgs_opt%lenwa))
    this%lbfgs_opt%kind_of_bound = 2 !upper and lower bound
    this%lbfgs_opt%wanted_relative_f_delta = 1e-04_dp
    this%lbfgs_opt%wanted_projected_gradient = 1e-04_dp
    this%lbfgs_opt%task='START'
    CALL setulb(this%num_var,this%lbfgs_opt%m, this%x, &
       this%lbfgs_opt%lower_bound, this%lbfgs_opt%upper_bound, &
       this%lbfgs_opt%kind_of_bound, this%lbfgs_opt%f, this%lbfgs_opt%gradient, &
       this%lbfgs_opt%wanted_relative_f_delta, &
       this%lbfgs_opt%wanted_projected_gradient, &
       this%lbfgs_opt%work_array, &
       this%lbfgs_opt%i_work_array, this%lbfgs_opt%task, & 
       this%lbfgs_opt%print_every, this%lbfgs_opt%csave, &
       this%lbfgs_opt%lsave, this%lbfgs_opt%isave, & 
       this%lbfgs_opt%dsave) 
    endif
    !------------cma-es------------------------
    if (associated(this%cmaes_opt)) then
      this%cmaes_opt%lambda_mo = this%num_para_set
      allocate(this%cmaes_opt%a_k(this%cmaes_opt%lambda_mo))
      allocate(this%cmaes_opt%a_k_temp(2*this%cmaes_opt%lambda_mo))
      allocate(this%cmaes_opt%a_k_parents(this%cmaes_opt%lambda_mo))
!      ALLOCATE(this%cmaes_opt%newsort(2*this%cmaes_opt%lambda_mo))
!      ALLOCATE(this%cmaes_opt%idxsel(2*this%cmaes_opt%lambda_mo))
      this%cmaes_opt%n = this%num_var
      this%cmaes_opt%l_bound => this%l_bound
      this%cmaes_opt%u_bound => this%u_bound
      allocate(this%cmaes_opt%rng_normal_stream)
      CALL create_rng_stream(this%cmaes_opt%rng_normal_stream,'gaussian',seed=seed, &
                             distribution_type=GAUSSIAN,error=this%error)
      this%cmaes_opt%generation = 1
     do k=1,this%cmaes_opt%lambda_mo
      a_k => this%cmaes_opt%a_k(k)
      CALL a_k_init(a_k,this%num_var,this%cmaes_opt%lambda_mo)
      ALLOCATE(a_k%f(this%num_obj))
      a_k%f = 0._dp
      a_k%xmean = this%x
      a_k%arx(:,1) = this%x 
!      a_k%f => this%obj_f(k)
      a_k%sigma = this%step_size
      a_k%id = k
      CALL gen_pop(a_k,this%cmaes_opt)
      !----init parents too---------------
      a_k => this%cmaes_opt%a_k_parents(k)
      CALL a_k_init(a_k,this%num_var,this%cmaes_opt%lambda_mo)
      a_k%sigma = this%step_size
      a_k%xmean = this%x
      a_k%arx(:,1) = this%x
      ALLOCATE(a_k%f(this%num_obj))
      a_k%f = huge(1._dp)
      a_k%id = k
      !----init tmp too---------------
      a_k => this%cmaes_opt%a_k_temp(k)
      CALL a_k_init(a_k,this%num_var,this%cmaes_opt%lambda_mo)
      a_k%sigma = this%step_size
      a_k%xmean = this%x
      a_k%arx(:,1) = this%x
      ALLOCATE(a_k%f(this%num_obj))
      a_k%f = huge(1._dp)
       !----init tmp too---------------
      a_k => this%cmaes_opt%a_k_temp(k+this%cmaes_opt%lambda_mo)
      CALL a_k_init(a_k,this%num_var,this%cmaes_opt%lambda_mo)
      a_k%sigma = this%step_size
      a_k%xmean = this%x
      a_k%arx(:,1) = this%x
      ALLOCATE(a_k%f(this%num_obj))
      a_k%f = huge(1._dp)
     enddo
     nullify(a_k) 
    endif
!================================================================================
    if (.not. this%benchmark) then
    !---------------------
    OPEN(88,file='WATER27structures/mol_list.txt')
    DO i=1,this%num_ref !-5
          !-----watch out num_inp 30 or 35 !!
          READ(88,*) str,charge
          this%charges(i) = charge
          this%input_files(i)= TRIM(str) // ".inp"
          this%coord_files(i)= "WATER27structures/" // trim(str) // ".xyz"
    ENDDO
    CLOSE(88)
!    this%input_files(this%num_inp-4) = 'eaip_H2O.inp'
!    this%input_files(this%num_inp-3) = 'eaip_H2Op.inp'
!    this%input_files(this%num_inp-2) = 'eaip_H2On.inp'
!    this%input_files(this%num_inp-1) = 'dipole.inp'
    this%input_files(this%num_ref+1) = 'geo_opt.inp' 
    this%charges(this%num_ref+1) = 0
    this%coord_files(this%num_ref+1) = "WATER27structures/H2O.xyz" 
    !---------------------
    open(99,file='WATER27.dat')
      ! only the first 30 mol have ref data...so 35 - 5 ;/
      do i=1,this%num_ref !-5
          !---watch out num_inp 30 or 35 !!!!!
          read(99,*) this%ener_ref(i)%id, &
          this%ener_ref(i)%mol, &
          this%ener_ref(i)%mol_ref1, &
          this%ener_ref(i)%mol_ref2, &
          this%ener_ref(i)%mol_ref3, &
          this%ener_ref(i)%mol_ref4, &
          this%ener_ref(i)%stoch1, &
          this%ener_ref(i)%stoch2, &
          this%ener_ref(i)%stoch3, &
          this%ener_ref(i)%stoch4, &
          this%ener_ref(i)%stoch5, &
          this%ener_ref(i)%ref
      enddo
    close(99)
    endif
    !------------------------
!    allocate(this%function_matrix(this%num_para_set,this%num_inp))
    !------------------------
    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init done"
 END SUBROUTINE paramopt_master_init

 SUBROUTINE a_k_init(a_k,n,lambda_mo)
    IMPLICIT NONE
    type(individual_type),POINTER            :: a_k
    INTEGER                                  :: i,n,lambda_mo
    INTEGER                                  :: allocstat
      ! initialize a_k
               IF (.NOT. ALLOCATED(a_k%B)) THEN
                     ALLOCATE(a_k%B(n,n),stat=allocStat)
                     IF(allocStat .NE. 0) STOP 'Error allocating B'
               END IF
               a_k%B = 0._dp
               DO i = 1,n
                a_k%B(i,i) = 1._dp
               END DO  
               IF (.NOT. ALLOCATED(a_k%D)) THEN
                   ALLOCATE(a_k%D(n,n),stat=allocStat)
                   IF(allocStat .NE. 0) STOP 'Error allocating D'
               END IF
               a_k%D = 0._dp
               DO i = 1, n
                  !D(i,i) = input%insigma(i)/maxval(input%insigma)
                  a_k%D(i,i) = 1._dp 
               END DO
               IF (.NOT. ALLOCATED(a_k%BD)) THEN 
                   ALLOCATE(a_k%BD(n,n),stat=allocStat)
                   IF(allocStat .NE. 0) STOP 'Error allocating internal Matrix BD'
               END IF
               a_k%BD = matmul(a_k%B,a_k%D) 
         !      CALL DGEMM('N','N',n,n,n,1.0d0,a_k%B,n,a_k%D,n,0.0d0,a_k%BD,n) 
               IF (.NOT. ALLOCATED(a_k%C)) THEN
                   ALLOCATE(a_k%C(n,n),stat=allocStat)
                   IF(allocStat .NE. 0) STOP 'Error allocating internal Matrix C'
               END IF
               a_k%C = matmul(a_k%BD,transpose(a_k%BD))
             allocate(a_k%multmp_vN(n))
             !----------(1+1) CMA-ES--------------------------
             a_k%lambda = 1
             a_k%mu = 1
             ALLOCATE(a_k%s_i(2*lambda_mo))
             a_k%s_i = 0
             ALLOCATE(a_k%triuC(n,n),stat=allocStat)
             ALLOCATE(a_k%arx(n,a_k%lambda),stat=allocStat)
             ALLOCATE(a_k%arz(n,a_k%lambda),stat=allocStat)
             ALLOCATE(a_k%xmean(n))
             ALLOCATE(a_k%zmean(n))
             ALLOCATE(a_k%xold(n))
             !ALLOCATE(a_k%f(a_k%lambda))
             !ALLOCATE(a_k%f_old)
             ALLOCATE(a_k%tempMat(n,a_k%mu))
             ALLOCATE(a_k%multmp_mN(n,n))
             ALLOCATE(a_k%multmp_mN2(n,n))
             ALLOCATE(a_k%multmp_mMU(a_k%mu,n))
             ALLOCATE(a_k%mutmp(n,a_k%mu))
             ALLOCATE(a_k%weightsMat(a_k%mu,a_k%mu))
             ALLOCATE(a_k%weights(a_k%mu))
             ALLOCATE(a_k%ps(n))
             ALLOCATE(a_k%pc(n))
             ALLOCATE(a_k%colVec(n,1))
             ALLOCATE(a_k%rowVec(1,n))
             a_k%front = 0
             a_k%sort_idx = 0
             a_k%penalty = 0._dp
             a_k%weights = 1._dp !equal weights
             ! normalize weights
             a_k%weights = a_k%weights/sum(a_k%weights)
                   !  mueff
             a_k%vecsum = sum(a_k%weights)
             a_k%vecnorm = sqrt(sum(a_k%weights**2))
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
             a_k%damps = 1._dp + real(n)/(2._dp*a_k%lambda)
             a_k%p_target = 1_dp/(5._dp + sqrt(real(a_k%lambda))/2._dp)
             a_k%cp = a_k%p_target*a_k%lambda/ &
                  (2._dp+a_k%p_target*a_k%lambda)
             a_k%cc = 2._dp/(n + 2._dp)
             a_k%ccov = 2._dp/(real(n)**2 + 6._dp)
             a_k%p_tresh = 0.44_dp
             a_k%p_succ = a_k%p_target
             !-----------------------------------------------------------
          !   a_k%sigma = 1._dp
             a_k%pc = 0._dp
             a_k%ps = 0._dp
         !    this%cmaes_opt%a_k%hsig = 1._dp
             a_k%ps = 0._dp
         !    a_k%f_old = 1e16_dp !should be inifinty
             a_k%chiN = sqrt(REAL(n))*((1.0_dp-1.0_dp/(4.0_dp*REAL(n))) &
                                   +1.0_dp/(21.0_dp*REAL(n*n)))
             a_k%d_count = 0
             a_k%s_i_count = 0
             a_k%I_dist = 0._dp
 END SUBROUTINE
! *****************************************************************************
 SUBROUTINE a_k_copy(a_k2,a_k1)
    IMPLICIT NONE
    type(individual_type)            :: a_k1
    type(individual_type)            :: a_k2
    INTEGER                                  :: i,n
             a_k2%B         = a_k1%B 
             a_k2%D         = a_k1%D
             a_k2%BD        = a_k1%BD
             a_k2%C         = a_k1%C
             a_k2%lambda    = a_k1%lambda
             a_k2%mu        = a_k1%mu
             a_k2%triuC     = a_k1%triuC          
             a_k2%arx       = a_k1%arx             
             a_k2%arz       = a_k1%arz             
             a_k2%xmean     = a_k1%xmean                
             a_k2%zmean     = a_k1%zmean                  
             a_k2%xold      = a_k1%xold                
             a_k2%tempMat   = a_k1%tempMat                  
             a_k2%multmp_mN = a_k1%multmp_mN                   
             a_k2%multmp_mN2= a_k1%multmp_mN2                      
             a_k2%multmp_mMU= a_k1%multmp_mMU                   
             a_k2%mutmp     = a_k1%mutmp                    
             a_k2%weightsMat= a_k1%weightsMat                  
             a_k2%weights   = a_k1%weights                     
             a_k2%ps        = a_k1%ps                
             a_k2%pc        = a_k1%pc                    
             a_k2%colVec    = a_k1%colVec             
             a_k2%rowVec    = a_k1%rowVec             
             a_k2%weights   = a_k1%weights                  
             a_k2%vecsum    = a_k1%vecsum                   
             a_k2%vecnorm   = a_k1%vecnorm             
             !----------------------default for (1+lambda) strategy----------------------------------
             a_k2%damps     = a_k1%damps 
             a_k2%p_target  = a_k1%p_target             
             a_k2%cp        = a_k1%cp                          
             a_k2%cc        = a_k1%cc                               
             a_k2%ccov      = a_k1%ccov                     
             a_k2%p_tresh   = a_k1%p_tresh         
             a_k2%p_succ    = a_k1%p_succ           
             !----------------------------------
             a_k2%sigma     = a_k1%sigma      
             a_k2%pc        = a_k1%pc
             a_k2%ps        = a_k1%ps
             !a_k2%f_old     = a_k1%f_old
             a_k2%chiN      = a_k1%chiN
             a_k2%f         = a_k1%f
             !-----------------------------------
             a_k2%d_count    = a_k1%d_count
             a_k2%I_dist     = a_k1%I_dist
             a_k2%s_i        = a_k1%s_i
             a_k2%s_i_count  = a_k1%s_i_count
             a_k2%id         = a_k1%id
             a_k2%sort_idx   = a_k1%sort_idx
             a_k2%front      = a_k1%front
             !-------------------------------------
             a_k2%penalty      = a_k1%penalty
 END SUBROUTINE
! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_steer(this, report, cmd, should_stop)
    TYPE(paramopt_master_type)               :: this
    TYPE(swarm_message_type), INTENT(IN)     :: report
    TYPE(swarm_message_type), INTENT(INOUT)  :: cmd
    LOGICAL, INTENT(INOUT)                   :: should_stop
    INTEGER                                  :: i,j,k,i_inp,r_iter
    INTEGER                                  :: report_worker_id
    integer,parameter    :: h2o_id=1,h3o_id=16,ohm_id=22,h2o8s4_id=11,h2o_eaip_id=31 
    real(kind=dp) :: h2o8s4_ref,h2o_eaip_ref,h2o_ref,h3op_ref,ohm_ref
    REAL(KIND=dp)                            :: energy
    REAL(KIND=dp)                            :: report_rmsd
    REAL(KIND=dp)                            :: E_res_temp
    CHARACTER(len=default_string_length)     :: report_status

    r_iter = 1
    CALL swarm_message_get(report, "worker_id", report_worker_id)
    CALL swarm_message_get(report, "status",report_status)
    
    IF (report_status == "ene_calc") THEN
    !---------we got a new message, deal with it :)---------------
    CALL swarm_message_get(report, "num_inp", i_inp)
    CALL swarm_message_get(report, "iter", r_iter)
    j = mod(r_iter,this%num_para_set) + 1
    i = ceiling(r_iter/real(this%num_para_set))
       CALL swarm_message_get(report, "Epot", this%function_matrix(i,j))
       !------------change status---------------------------------
       report_status = "wait_done"
       !------------deal with condition---------------------------
       this%i_recv(j) = this%i_recv(j) + 1
       !---------------print------------------------------------
!         IF(this%iw>0) THEN
!            WRITE (this%iw,'(A,20X,I8,I8,I8)')&
!                   " PRMOPT| Reporting worker ", report_worker_id, i_inp, r_iter
!            WRITE (this%iw,'(A,20X,E15.8)')&
!                   " PRMOPT| Reported potential Energy [Hartree] ", this%Epot(i_inp)
!         END IF
!       print *,'receved',i,j,this%function_matrix(i,j)-this%function_matrix(i,1),this%function_matrix(i,j)
!       print *,'wid',report_worker_id
!       print *,trim(this%input_files(i))
    ELSE IF (report_status == 'rmsd_calc') THEN
    !---------we got a new message, deal with it :)---------------
    CALL swarm_message_get(report, "num_inp", i_inp)
    CALL swarm_message_get(report, "iter", r_iter)
         ! map this%inp_iter to task and send out
         ! k = inp_iter, m = num_para_set
         j = mod(r_iter,this%num_para_set) + 1
         i = ceiling(real(r_iter)/real(this%num_para_set)) 
    
       CALL swarm_message_get(report, "rmsd", this%rmsd(j))
       this%function_matrix(i,j) = this%rmsd(j)
       !------------change status---------------------------------
       report_status = "wait_done"
       !------------deal with condition---------------------------
       this%i_recv(j) = this%i_recv(j) + 1
       !---------------print------------------------------------
!         IF(this%iw>0) THEN
!            WRITE (this%iw,'(A,20X,I8,I8,I8)')&
!                   " PRMOPT| Reporting worker ", report_worker_id, i_inp, r_iter
!            WRITE (this%iw,'(A,20X,E15.8)')&
!                   " PRMOPT| Reported rmsd ", this%rmsd
!         END IF
!      print *,'receved',i,j,this%rmsd
    ELSE IF (report_status == "bench_calc") THEN
      CALL swarm_message_get(report, "num_inp", i_inp)
      CALL swarm_message_get(report, "iter", r_iter)
         ! map this%inp_iter to task and send out
         ! k = inp_iter, m = num_para_set
         j = mod(r_iter,this%num_para_set) + 1
         i = ceiling(real(r_iter)/real(this%num_para_set))
      CALL swarm_message_get(report, "Epot", this%function_matrix(1,j))
      CALL swarm_message_get(report, "rmsd", this%function_matrix(2,j))
       !------------change status---------------------------------
       report_status = "wait_done"
       !------------deal with condition---------------------------
       this%i_recv(j) = this%i_recv(j) + 1


    ENDIF
    
    IF (sum(this%i_recv) .eq. this%num_para_set*this%num_inp) THEN
    !--------------we are done with the iteration/condition-------- 
        do j=1,this%num_para_set 
          if (.not. this%benchmark) then
             call calc_obj_f(this, this%function_matrix(:,j), this%E_res(j))
             this%obj_f(j) = this%weights(1)*this%E_res(j) + &
                          this%weights(2)*this%function_matrix(this%num_inp,j) + &
                          this%penalty
          else
             if (associated(this%cmaes_opt)) then
             !    print *,'penalty of a_k',j,'is',this%cmaes_opt%a_k(j)%penalty
             this%obj_f(j) = this%function_matrix(1,j) !+ this%penalty(j)**2
             this%cmaes_opt%a_k(j)%f(1) = this%obj_f(j) + this%cmaes_opt%a_k(j)%penalty
             this%cmaes_opt%a_k(j)%f(2) = this%function_matrix(2,j) + this%cmaes_opt%a_k(j)%penalty
           !  if (this%cmaes_opt%a_k(j)%f(1) .lt. 1e-6) this%cmaes_opt%a_k(j)%f(1) = 0._dp
           !  if (this%cmaes_opt%a_k(j)%f(2) .lt. 1e-6) this%cmaes_opt%a_k(j)%f(2) = 0._dp
             endif
          endif
        enddo
        
        if (this%calc_deriv) then
         do j=2,this%num_para_set
            this%grad_obj_f(j-1) = ( this%obj_f(j) - this%obj_f(1)) / this%eps(j-1)
         enddo
        endif 
       
        if (this%calc_deriv) print *,'gradient'
        if (this%calc_deriv) print *,this%grad_obj_f
        
        IF(this%iw>0) THEN
          !print *,'-------Rmsd-------'
          !print *,this%function_matrix(this%num_inp,1)
          !print *,'-------Epot-------'
          !print *,this%Epot
     !     print *,'-------obj_f-------'
     !     print *,this%obj_f
     !     !open(999,file='xmean.out',position='append')
     !     !write(999,*) this%cmaes_opt%xmean
     !     !close(999)
     !     open(998,file='obj_f.out',position='append')
     !     !write(998,*) this%x,minval(this%obj_f),this%cmaes_opt%sigma !,this%E_res,this%rmsd
     !     write(998,*) this%x,this%obj_f,this%cmaes_opt%a_k%sigma,this%penalty**2 !,this%E_res,this%rmsd
     !     close(998)
         
        ENDIF
        !--------------------new x to file?--------------------
        if (.not. this%benchmark) then
        CALL para2file(this%num_var,this%x,"para.xml","para.xml.old.out") 
        else
        !   !open(12,position='append')
        !   open(12)
        !   do i = 1,this%cmaes_opt%lambda_mo
        !      write(12,*) this%cmaes_opt%a_k_parents(i)%f,this%cmaes_opt%a_k_parents(i)%id
        !      write(12,*) this%cmaes_opt%a_k(i)%f,this%cmaes_opt%a_k(i)%id
        !   enddo
        !   close(12)
        endif
        !----new x-------------------
        !------------------------optimizer-----------------------------
          CALL var_trans(this%x,this%xtrans,this%l_bound,this%u_bound)
          if (associated(this%powell_opt)) CALL powell_optimize(this%num_var, &
              this%xtrans,this%powell_opt)
          if (associated(this%gauss_opt)) CALL GaussAdapt(this%num_var, &
              this%xtrans,this%gauss_opt)
          if (associated(this%sampling)) CALL sample(this,this%num_var, &
              this%xtrans,this%l_bound,this%u_bound)
          if (associated(this%lbfgs_opt)) CALL setulb(this%num_var,this%lbfgs_opt%m, this%x, &
                                         this%lbfgs_opt%lower_bound, this%lbfgs_opt%upper_bound, &
                                         this%lbfgs_opt%kind_of_bound, this%lbfgs_opt%f, &
                                         this%lbfgs_opt%gradient, &
                                         this%lbfgs_opt%wanted_relative_f_delta, &
                                         this%lbfgs_opt%wanted_projected_gradient, &
                                         this%lbfgs_opt%work_array, &
                                         this%lbfgs_opt%i_work_array, this%lbfgs_opt%task, & 
                                         this%lbfgs_opt%print_every, this%lbfgs_opt%csave, &
                                         this%lbfgs_opt%lsave, this%lbfgs_opt%isave, & 
                                         this%lbfgs_opt%dsave)
         if (associated(this%cmaes_opt)) then
          !   do i=1,this%cmaes_opt%lambda_mo
          !      if (this%cmaes_opt%a_k(i)%sigma .lt. 1e-9_dp) STOP
          !   enddo
             this%cmaes_opt%generation = this%cmaes_opt%generation + 1
            ! if (this%cmaes_opt%generation .eq. 2000) STOP
             CALL cmaes(this%x,this%cmaes_opt)
         endif
         if (associated(this%sampling) .or. associated(this%gauss_opt) .or. associated(this%powell_opt)) then
          !------------------check constrains-------------------------
          do i=1,this%num_var 
              if (this%xtrans(i) .lt. 0._dp) this%xtrans(i) = 0._dp
              if (this%xtrans(i) .gt. 1._dp) this%xtrans(i) = 1._dp
          enddo
          !condition on outer H_core???
          !if (this%xtrans(13) .lt. this%xtrans(11)) this%xtrans(13) = this%xtrans(11)
          CALL var_back_trans(this%xtrans,this%x,this%l_bound,this%u_bound)
         else
           !print *,'task',this%lbfgs_opt%task
           CALL var_trans(this%x,this%xtrans,this%l_bound,this%u_bound)
         endif
         !---------reset inp_iter--------
         this%inp_iter = this%num_inp*this%num_para_set
         this%i_recv = 0
         !--------------------new x to file?--------------------
          if (.not. this%benchmark) then
          CALL para2file(this%num_var,this%x,"para.xml","para.xml.out")
          else
           !open(12,position='append')
           open(13)
           do i = 1,2*this%cmaes_opt%lambda_mo
              write(13,*) this%cmaes_opt%a_k_temp(i)%f, &
                          this%cmaes_opt%a_k_temp(i)%id, &
                          this%cmaes_opt%a_k_temp(i)%front, &
                          this%cmaes_opt%a_k_temp(i)%I_dist, &
                          this%cmaes_opt%a_k_temp(i)%sigma, &
                          this%cmaes_opt%a_k_temp(i)%arx(:,1)
           enddo
           close(13)
          endif
          !-----------------x for diff-------------------------
          if (this%calc_deriv) this%eps = this%eps_factor*this%x
             !----check lowest eps....eps should not be zero!
             do i=1,this%num_var
                 if (this%eps(i) .eq. 0._dp) this%eps(i) = this%eps_factor
             enddo
          !this%x_matrix(:,2) = this%x + this%eps
          !this%x_matrix(:,3) = this%x - this%eps 
    ENDIF
    
    !--------------master have a task to distr---------
    IF (this%inp_iter .gt. 0 .and. report_status == "wait_done" ) THEN
         ! send new task to idle worker
         ! map this%inp_iter to task and send out
         ! k = inp_iter, m = num_para_set
         j = mod(this%inp_iter,this%num_para_set) + 1 !x(j)
         i = ceiling(real(this%inp_iter)/real(this%num_para_set))
         this%x_temp = this%x
         if (.not. this%benchmark) then
            if (i .lt. this%num_inp) then
               CALL swarm_message_add(cmd, "command", "energy")
            else if (i .eq. this%num_inp) then
               CALL swarm_message_add(cmd, "command", "geo_opt")
            endif
            CALL swarm_message_add(cmd, "input", this%input_files(i))
            CALL swarm_message_add(cmd, "coord", this%coord_files(i))
            CALL swarm_message_add(cmd, "charge", this%charges(i))
            CALL swarm_message_add(cmd, "num_inp",i)
            CALL swarm_message_add(cmd, "iter", this%inp_iter)
            if (associated(this%cmaes_opt)) then
              this%x_temp = this%cmaes_opt%a_k(j)%arx(:,1)
            endif
            if (this%calc_deriv) then
              if (j .gt. 1) then
              this%x_temp(j-1) = this%x_temp(j-1) + this%eps(j-1)
              !print *,j,this%x_temp(j-1),this%x(j-1),this%eps(j-1)
              endif
            endif
          else if ( this%benchmark) then
            CALL swarm_message_add(cmd, "command", "benchmark")
            CALL swarm_message_add(cmd, "num_inp",i)
            CALL swarm_message_add(cmd, "iter", this%inp_iter)
            !print *,this%cmaes_opt%a_k(j)%arx
            this%x_temp = this%cmaes_opt%a_k(j)%arx(:,1)
          endif
         CALL swarm_message_add(cmd, "x", this%x_temp)
       IF(this%iw>0) THEN
          WRITE (this%iw,'(A,20X,I8,I8,I8,I8,I8)')&
                 " PRMOPT| cmd send worker ", report_worker_id,  &
                 this%inp_iter, i, j,this%num_para_set
        !  print *,this%x_temp-this%x
       END IF
         this%inp_iter = this%inp_iter - 1
    ELSE
    !-----else wait cmd----------------
         CALL swarm_message_add(cmd, "command", "wait")

    ENDIF

 END SUBROUTINE paramopt_master_steer


! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_finalize(this)
    TYPE(paramopt_master_type)                 :: this

    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_finalize called"
 END SUBROUTINE paramopt_master_finalize
 !-----------------------CMA-ES -----------Algorithm-----------------------------
    subroutine cmaes(x,cmaes_opt)

    implicit none
    type(cmaes_opt_type),target                               :: cmaes_opt
    type(individual_type),POINTER                             :: a_k
    type(individual_type),POINTER                             :: a_k_parents
    real(kind=dp),pointer,intent(inout)                       :: x(:)
    real(kind=dp)                                             :: lambda_succ
    real(kind=dp)                                             :: p_succ
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: i,j,k,n
    INTEGER                                                   :: info
    LOGICAL                                                   :: boolean
!----------------------------set pointers to type struct-----------------------
    n = cmaes_opt%n
    lambda_succ = 0._dp
    p_succ = 0._dp
    boolean = .false.
    call init_selection(cmaes_opt)

    do k=1,cmaes_opt%lambda_mo
         a_k => cmaes_opt%a_k(k)
         a_k_parents => cmaes_opt%a_k_parents(k)
         !print *,'both f',a_k%front,a_k_parents%front,a_k%I_dist,a_k_parents%I_dist,a_k_parents%sigma
         call dominates(a_k,a_k_parents,boolean)
         !----yannick watch out...f now is hardcoded to only have 2 obj!!!!
     !    if ((a_k%f(1) .le. a_k_parents%f(1)) .and. &
     !               (a_k%f(2) .le. a_k_parents%f(2)) )
     !        then
     !        lambda_succ = 1._dp
     !    else
     !        lambda_succ = 0._dp
     !    endif
         if (boolean) then
             p_succ = 1._dp
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
             CALL new_mean(a_k)
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
                 STOP
             END IF 
             !-----------------------------------------------------------------------
             !  D contains standard ind_counter now
             !----------------------ind_counter---------------------------------------
                 a_k%D = sqrt(a_k%D)
             CALL DGEMM('N','N',n,n,n,1.0d0,a_k%B,n,a_k%D,n,0.0d0,a_k%BD,n)  
             !open(999,file='bd_matrix.out',position='append')
             !write(999,*) cmaes_opt%BD
             !close(999)
         nullify(a_k,a_k_parents)
    enddo 
    
    call selection(cmaes_opt)
!----for 1:lambda_mo generate new population----------------------------------
    !print *,"generating new population"
    DO k = 1, cmaes_opt%lambda_mo
        a_k => cmaes_opt%a_k(k)
        a_k_parents => cmaes_opt%a_k_parents(k)
        cmaes_opt%a_k_temp(k)%id = k
        call a_k_copy(a_k_parents,cmaes_opt%a_k_temp(k))
        !cmaes_opt%a_k_parents(k) = cmaes_opt%a_k_temp(k)
        call a_k_copy(a_k,cmaes_opt%a_k_temp(k))
        !cmaes_opt%a_k(k) = cmaes_opt%a_k_parents(k)
        !print *,'old f',cmaes_opt%a_k_parents(k)%f

        CALL gen_pop(a_k,cmaes_opt)
        a_k_parents%id = a_k%id + cmaes_opt%lambda_mo
        !----this is just a random x !!!----------
        x = a_k%arx(:,1)
        !print *,'x should not be the same!',a_k_parents%arx(1,1),a_k%arx(1,1)
    !print *,'selected',cmaes_opt%a_k(k)%f,cmaes_opt%a_k_parents(k)%f,cmaes_opt%a_k(k)%s_i
    END DO
    nullify(a_k,a_k_parents)
END SUBROUTINE
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
    do i = 1,cmaes_opt%lambda_mo
    ! reset all election varaibles
    cmaes_opt%a_k(i)%d_count=0
    cmaes_opt%a_k(i)%I_dist=0._dp
    cmaes_opt%a_k(i)%s_i=0
    cmaes_opt%a_k(i)%s_i_count=0
    cmaes_opt%a_k(i)%front=0
    cmaes_opt%a_k_parents(i)%d_count=0
    cmaes_opt%a_k_parents(i)%I_dist=0._dp
    cmaes_opt%a_k_parents(i)%s_i=0
    cmaes_opt%a_k_parents(i)%s_i_count=0
    cmaes_opt%a_k_parents(i)%front=0
    tmp_set(i)%p => cmaes_opt%a_k(i)
    tmp_set(i)%p => cmaes_opt%a_k(i)
    tmp_set(i+cmaes_opt%lambda_mo)%p => cmaes_opt%a_k_parents(i)
    enddo
    !do i = 1,2*cmaes_opt%lambda_mo
    !    print *,'before',tmp_set(i)%p%f,tmp_set(i)%p%sigma
    !enddo
    !print *,'init Q --- fronts and I_dist'
    !---------find pareto front and add to Q (tmp_set)----------------------------------
    CALL pareto_front(input_set=tmp_set,nondominated_set=new_set)
    CALL crowding_distance(new_set)
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
            sel_set(i) = new_set(i)
        enddo
        CALL new_front(tmp_set,sel_set,new_set)
        CALL crowding_distance(new_set)
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
   ! do i=1,cmaes_opt%lambda_mo
   !     print *,'fronts',cmaes_opt%a_k(i)%front
   ! enddo
    print *,'init_selection done'
    !-------there should be everythin in a_k and a_k_parents
    end subroutine init_selection
!===========================================================================
    subroutine selection(cmaes_opt)
    implicit none
    type(cmaes_opt_type),TARGET                               :: cmaes_opt
    TYPE(individual_type),DIMENSION(:),ALLOCATABLE,TARGET     :: tmp_set 
    TYPE(individual_type),DIMENSION(:),ALLOCATABLE,TARGET     :: sel_set
    REAL(kind=dp),DIMENSION(:),ALLOCATABLE                    :: newsort
    INTEGER,DIMENSION(:),ALLOCATABLE                          :: idxsel 
    integer                                                   :: idx
    integer                                                   :: i,j,k
    integer                                                   :: fsize
    integer                                                   :: psize
    integer                                                   :: front
    type(individual_type),POINTER                             :: a_k
    !--------everything could be done more efficient with pointers!!!--------------------
    k=1
    front = 1
    psize = 0
    do i = 1,cmaes_opt%lambda_mo
        !---------copy everythin in a_k_temp array-----------------
        call a_k_copy(cmaes_opt%a_k_temp(i),cmaes_opt%a_k(i))
        call a_k_copy(cmaes_opt%a_k_temp(i+cmaes_opt%lambda_mo),cmaes_opt%a_k_parents(i))
     !   print *,'f values for selection',cmaes_opt%a_k_temp(i)%f,cmaes_opt%a_k_temp(i+cmaes_opt%lambda_mo)%f
    enddo
    allocate(sel_set(cmaes_opt%lambda_mo))
    do i=1,cmaes_opt%lambda_mo
        a_k => sel_set(i)
      CALL a_k_init(a_k,cmaes_opt%n,cmaes_opt%lambda_mo)
      ALLOCATE(a_k%f(size(cmaes_opt%a_k(1)%f)))
    enddo
    do while (k .le. cmaes_opt%lambda_mo)
        !-------get front size-------------
        fsize = 0 
        do i=1,2*cmaes_opt%lambda_mo
          if (front .eq. cmaes_opt%a_k_temp(i)%front) then
          fsize = fsize + 1
          endif
        enddo
        !-------------front size lower of what needed?----------
        if (fsize+psize .le. cmaes_opt%lambda_mo) then
        ! add all individual from front to Q    
           do i=1,2*cmaes_opt%lambda_mo
             if (front .eq. cmaes_opt%a_k_temp(i)%front) then
             call a_k_copy(sel_set(k),cmaes_opt%a_k_temp(i))
             !sel_set(k) = cmaes_opt%a_k_temp(i)
             k = k + 1
             endif
           enddo
           psize = psize + fsize
        else
          ! check crowding distance and add n best
          allocate(tmp_set(fsize))
            do i=1,fsize
              a_k => tmp_set(i)
              CALL a_k_init(a_k,cmaes_opt%n,cmaes_opt%lambda_mo)
              !yannick this is hardcoded again
              ALLOCATE(a_k%f(size(cmaes_opt%a_k(1)%f)))
            enddo
             !----add to temporary set
             allocate(newsort(fsize))
             allocate(idxsel(fsize))
             j = 1
             do i=1,2*cmaes_opt%lambda_mo
               if (front .eq. cmaes_opt%a_k_temp(i)%front) then
               call a_k_copy(tmp_set(j),cmaes_opt%a_k_temp(i))
               !tmp_set(j) = cmaes_opt%a_k_temp(i)
               newsort(j) = cmaes_opt%a_k_temp(i)%I_dist
               j = j + 1
               endif
             enddo
           !----sort according to I_dist-----------
           CALL d_mrgrnk(newsort,idxsel)
           !----add to sel set Q------------
           !print *,'popsize',psize,'fsize',fsize,'needed',cmaes_opt%lambda_mo-psize
           do i = 1,cmaes_opt%lambda_mo-psize
               idx = size(idxsel)+1-i
               !print *,'selected',tmp_set(idxsel(idx))%I_dist
               call a_k_copy(sel_set(k),tmp_set(idxsel(idx)))
               !sel_set(k) = tmp_set(idxsel(i))
               k = k + 1
           enddo
           psize = psize + i
           deallocate(tmp_set,newsort,idxsel)
        endif 
        front = front + 1 
    enddo
    !print *,'should be the saaaame',cmaes_opt%lambda_mo,k-1,psize-1

    do i=1,cmaes_opt%lambda_mo
       call a_k_copy(cmaes_opt%a_k_temp(i),sel_set(i))
       !print *,'selected f',cmaes_opt%a_k_temp(i)%f,cmaes_opt%a_k_temp(i)%s_i
    enddo
    deallocate(sel_set)
    end subroutine selection
!===========================================================================
    subroutine dominates(a_k,a_k_parents,boolean)
    type(individual_type)                             :: a_k
    type(individual_type)                             :: a_k_parents
    LOGICAL                                           :: boolean
    IF (a_k%front .le. a_k_parents%front) then
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

    ! input:
    ! n, lambda, BD, xmean, sigma
    ! output:
    ! arx,arz
    n = cmaes_opt%n
    allocate(tmp(n))
    tmp=0._dp
    alpha = 1.e-6_dp
    !CALL reset_to_next_rng_substream(cmaes_opt%rng_normal_stream, error=error)
    do j=1,a_k%lambda
        do i=1,n; a_k%arz(i,j) = next_random_number(cmaes_opt%rng_normal_stream, error=error); enddo
    CALL DGEMV('N',N,N,1.0d0,a_k%BD,N,a_k%arz(:,j),1,0.0d0,a_k%multmp_vN,1)
    a_k%arx(:,j) = a_k%xmean + a_k%sigma * a_k%multmp_vN
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
    !        do i=1,n
    !                if (cmaes_opt%arx(i,k) .lt. 0._dp) cmaes_opt%arx(i,k) = 0._dp
    !                if (cmaes_opt%arx(i,k) .gt. 1._dp) cmaes_opt%arx(i,k) = 1._dp
    !        enddo
    enddo
    !---yannick this is stupid
    !print *,'new generating x...'
    !do k=1,a_k%lambda
    !    print *,a_k%arx(:,k)
    !    print *,a_k%arz(:,k)
    !enddo
    a_k%penalty = alpha*norm2(tmp)**2
    deallocate(tmp)
    end subroutine
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
    end subroutine
!===========================================================================
    subroutine update_path(a_k)
    implicit none
    type(individual_type),POINTER                             :: a_k
    TYPE(cp_error_type),POINTER                               :: error
    INTEGER                                                   :: n,k,i
    n = size(a_k%xmean)
    CALL dgemv('N',n,n,1.0d0,a_k%B,n,a_k%zmean,1,0.0d0,a_k%multmp_vN,1)
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
                        (a_k%xmean-a_k%xold) 
       else
       a_k%pc = (1._dp - a_k%cc)*a_k%pc
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
    n = size(a_k%xmean)
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
      CALL dgemm('N','T',a_k%mu,n,a_k%mu,1.0d0,a_k%weightsMat, &
                 a_k%mu,a_k%mutmp,n,0.0d0,a_k%multmp_mMU,a_k%mu)
      CALL dgemm('N','N',n,n,a_k%mu,1.0d0,a_k%mutmp,n,&
                 a_k%multmp_mMU,a_k%mu,0.0d0,a_k%multmp_mN2,n)
!      cmaes_opt%C = (1._dp-cmaes_opt%ccov+(1._dp-cmaes_opt%hsig)*cmaes_opt%ccov* &
!                     cmaes_opt%cc*(2._dp-cmaes_opt%cc)/cmaes_opt%mucov)*cmaes_opt%C & ! old matrix
!                   + cmaes_opt%ccov*(1._dp/cmaes_opt%mucov)*cmaes_opt%multmp_mN &  ! plus rank one update
!                   + cmaes_opt%ccov*(1._dp-1._dp/cmaes_opt%mucov) &  ! plus rank mu update
!                   *cmaes_opt%sigma**(-2) * cmaes_opt%multmp_mN2
      if (a_k%p_succ .lt. a_k%p_tresh) then
      a_k%C = (1._dp-a_k%ccov)*a_k%C & ! old matrix
                   + a_k%ccov*(a_k%multmp_mN) ! plus rank one update
      else
      a_k%C = (1._dp-a_k%ccov)*a_k%C & ! old matrix
                   + a_k%ccov*(a_k%multmp_mN &  ! plus rank one update
                   + a_k%ccov*(2._dp-a_k%ccov)*a_k%C)
      endif
  !    print *,'this is the new C'
  !    print *,'======================================================'
  !    print *,cmaes_opt%C     
    
    end subroutine
!===========================================================================
    SUBROUTINE pareto_front(input_set,nondominated_set)
         IMPLICIT NONE
         TYPE(individual_type_p),DIMENSION(:)  :: input_set
         TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE :: nondominated_Set
         TYPE(individual_type),POINTER                    :: a_k,a_k2
         REAL(kind=dp),DIMENSION(:,:),ALLOCATABLE    :: newsort
         REAL(kind=dp),DIMENSION(:),ALLOCATABLE      :: tmp_array
         INTEGER,DIMENSION(:),ALLOCATABLE            :: idxsel
         INTEGER                                     :: nondominated_size
         INTEGER                                     :: num_sel,popsize
         INTEGER                                     :: i,j,k 
         INTEGER                                     :: j_m,j_p 
         INTEGER                                     :: numrows,numcols
         numrows=size(input_set)
         !----watch out yannick this is hardcoded!!!!
         numcols=2 
         
         ! calculate fron with dominated count d_count and sub_set dominated s_i 
         do i =1,numrows 
            a_k => input_set(i)%p
            k = 1
            do j = 1,numrows
                a_k2 => input_set(j)%p
                !----yannick watch out...f now is hardcoded to only have 2 obj!!!!
                if ((a_k%f(1) .lt. a_k2%f(1)) .and. &
                    (a_k%f(2) .lt. a_k2%f(2)) ) then
                    a_k%s_i(k) = j
                    a_k%s_i_count = a_k%s_i_count + 1
                    k = k +1
                else if (a_k%f(1) .gt. a_k2%f(1) .and. &
                         a_k%f(2) .gt. a_k2%f(2) ) then

                   a_k%d_count = a_k%d_count + 1
                endif
            enddo 
         enddo
         nondominated_size = 0
         do i =1,numrows
            a_k => input_set(i)%p
            if (a_k%d_count .eq. 0) &
                nondominated_Size = nondominated_size + 1
        enddo
        allocate(nondominated_set(nondominated_size))
        allocate(idxsel(nondominated_size))
        allocate(tmp_array(nondominated_size))
        allocate(newsort(nondominated_size,numcols))
        k=1
        do i =1,numrows
            a_k => input_set(i)%p
            if (a_k%d_count .eq. 0) then 
            nondominated_set(k)%p => a_k             
            newsort(k,:) = a_k%f
            k = k+1
            endif
        enddo
        END SUBROUTINE pareto_front
!===========================================================================
     SUBROUTINE crowding_distance(input_set)
         IMPLICIT NONE
         TYPE(individual_type_p),DIMENSION(:)  :: input_Set
!         TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE :: output_Set
!         TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE :: tmp_Set
         REAL(kind=dp),DIMENSION(:,:),ALLOCATABLE    :: newsort
         REAL(kind=dp),DIMENSION(:),ALLOCATABLE      :: maxvals,minvals
         INTEGER,DIMENSION(:),ALLOCATABLE            :: idxsel
         INTEGER                                     :: nondominated_size
         INTEGER                                     :: numcols
         INTEGER                                     :: sel_target
         INTEGER                                     :: i,j,k 
         INTEGER                                     :: j_m,j_p 
         nondominated_size=size(input_set)
         ! this should not be hardcoded
         numcols=size(input_set(1)%p%f)
!         if (allocated(output_Set)) deallocate(output_set)
!         allocate(output_set(sel_target)) 
         allocate(newsort(nondominated_size,numcols))
         allocate(idxsel(nondominated_size))
         allocate(maxvals(numcols))
         maxvals(1) = 1._dp
         maxvals(2) = 300._dp
     ! crowding-distance-assignment
         do i = 1,nondominated_size
             newsort(i,:) = input_set(i)%p%f(:)
             input_set(i)%p%I_dist = 0._dp
         enddo
        
        do k = 1,numcols     ! num objectives 
           CALL d_mrgrnk(newsort(:,numcols),idxsel)
           input_set(idxsel(1))%p%I_dist = huge(1._dp) 
           input_set(idxsel(nondominated_size))%p%I_dist = huge(1._dp) 
           do i = 2,nondominated_size-1
               j = idxsel(i)
               j_p = idxsel(i+1)
               j_m = idxsel(i-1)
               input_set(j)%p%I_dist = input_set(j)%p%I_dist + & 
                                (input_set(j_p)%p%f(numcols)- &
                                 input_set(j_m)%p%f(numcols))/& 
                                 maxvals(numcols) !---yannick doesnt know what to put here
                                !(maxval(newsort(:,numcols))-minval(newsort(:,numcols)))
           enddo 
        enddo 
!        allocate(tmp_array(size(input_set))) 
!      do i = 1, nondominated_size
!            tmp_array(i) = input_set(i)%p%I_dist
!      enddo 
!      CALL d_mrgrnk(tmp_array,idxsel)
!      DO i=1,sel_target
!        output_set(i) = input_set(idxsel(i))
!      END DO
       deallocate(newsort,idxsel) 
       END SUBROUTINE crowding_distance
!===========================================================================
    SUBROUTINE new_front(global_set,input_set,output_set)
       IMPLICIT NONE
       TYPE(individual_type_p),DIMENSION(:)  :: input_Set
       TYPE(individual_type_p),DIMENSION(:)  :: global_Set
       TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE :: output_Set
       TYPE(individual_type_p),DIMENSION(:),ALLOCATABLE :: tmp_Set
       INTEGER                                          :: i,j,k,idx
       INTEGER                                          :: numrows
       TYPE(individual_type),POINTER                    :: a_k,a_k2

       allocate(tmp_set(size(global_set)))
       numrows = size(input_set)
       do i=1,size(global_set)
           tmp_set(i) = global_set(i)
       enddo
       if (allocated(output_set)) deallocate(output_set)
       
       k = 1
       do i = 1,numrows
           a_k => input_set(i)%p
 !          print *,a_k%f,a_k%s_i,a_k%s_i_count,a_k%d_count
           do j = 1,a_k%s_i_count
               idx = a_k%s_i(j)
               global_set(idx)%p%d_count = global_set(idx)%p%d_count - 1
               if (global_set(idx)%p%d_count .eq. 0) then
                  tmp_set(k)= global_set(idx)
                  k = k + 1
               endif
           enddo

       enddo
       allocate(output_set(k-1))
       do i=1,k-1
           output_set(i) = tmp_set(i)
       enddo 
!       print *,size(output_set)

    END SUBROUTINE
!========================================================================================================
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
      
      SUBROUTINE tool_eigendecomp(A,n,eVals,eVecs,info)
      IMPLICIT NONE
      
      !-------------------------------------------------------------------------
      !  Parameters
      !-------------------------------------------------------------------------
      INTEGER, INTENT(in)							:: n
      REAL(dp),DIMENSION(n,n),INTENT(in)			:: A
      REAL(dp),DIMENSION(n,n),INTENT(out)			:: eVals
      REAL(dp),DIMENSION(n,n),OPTIONAL,INTENT(out)	:: eVecs
      INTEGER                                       :: info
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
      INTEGER                                           :: allocStat
     
      
      
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
      IF(info .GT. 0) THEN
        WRITE(*,*) 'EVD failed to converge'
      END IF
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
! *****************************************************************************
subroutine read4file(n,x,ex)
    implicit none
    integer,intent(in)       :: n
    integer                  :: i,fileid,k
    real(kind=dp),intent(out) :: x(n)
    character(200)            :: str,str2,str3,filename
    logical,intent(out)      :: ex

    inquire(file="para.xml.restart",exist=ex)
    k=1
    if (ex) then
        print *,'read from restart file'
        filename="para.xml.restart"
    else
        filename="para.xml"
    endif


    fileid=9999
    open(fileid,file=filename)
    
    do i=1,59 
    select case (i)
    case default
     read(fileid,*) str
    case (17)
     read(fileid,*) str,x(k),str2
     k=k+1
    case (19:20)        
     read(fileid,*) str,x(k),str2
     k=k+1
     
    case (25)
     read(fileid,*) str,str3,x(k),str2
     k=k+1

    case (28:30)
     read(fileid,*) str,x(k),str2
     k=k+1

    case (36)
     read(fileid,*) str,x(k),str2
     k=k+1

    case (38:39)
     read(fileid,*) str,x(k),str2
     k=k+1

    case (44:45) 
     read(fileid,*) str,x(k),str2
     k=k+1

    case (50:51) 
     read(fileid,*) str,str3,x(k),str2
     k=k+1 
    
    case (54:56) 
     read(fileid,*) str,x(k),str2
     k=k+1 

    end select
    
    end do
    if (k-1 .ne. n) write(*,*) "something is very wrong"

    close(fileid)
    end subroutine

! *****************************************************************************
! *****************************************************************************
subroutine para2file(n,x,inputstr,outputstr)
    implicit none
    integer,intent(in)          :: n
    integer                     :: fileid2
    integer                     :: i,rank,ierr,fileid,k
    real(kind=dp),intent(in)    :: x(n)
    real(kind=dp)               :: dummy=0._dp
    character(len=*),intent(in) :: inputstr
    character(len=*),intent(in) :: outputstr
    character(200)              :: line
    character(24)              :: str,str2,str3
    
    fileid = 2222
    fileid=fileid2+100 
    open(fileid,file=adjustl(trim(inputstr)))
    open(fileid2,status='replace',file=adjustl(trim(outputstr)))
    k=1 
!   print *,'x to file',k,x
    do i=1,59
!    print *,'k',k
    select case (i)
    case default
     read(fileid,'(A)') line
!     str=adjustl(trim(str))
     write(fileid2,'(A)') line
    case (17)
     read(fileid,*) str,dummy,str2
     str=adjustl(trim(str))
!     str2=adjustl(trim(str2))
     str2="</Charge_width>"
     write(fileid2,'(8X,A,F13.7,X,A)') str,x(k),str2
     k=k+1
    case (19:20)        
     read(fileid,'(A,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))d
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,A)') str,x(k),str2
     k=k+1
    case (25)
     read(fileid,*) str,str3,dummy,str2
     str=adjustl(trim(str))
     str3=adjustl(trim(str3))
!     str2=adjustl(trim(str2))
     str2="</ETA>"
     write(fileid2,'(12X,A4,X,A6,F13.7,X,A)') str,str3,x(k),str2
        k=k+1
    case (28:30)
     read(fileid,'(A16,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,X,A)') str,x(k),str2
        k=k+1
    case (36)
     read(fileid,*) str,dummy,str2
     str=adjustl(trim(str))
!     str2=adjustl(trim(str2))
     str2="</Charge_width>"
     write(fileid2,'(8X,A,F13.7,X,A)') str,x(k),str2
        k=k+1
    case (38:39) 
     read(fileid,'(A,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,A)') str,x(k),str2
        k=k+1
    case (44:45) 
     read(fileid,'(A,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,A)') str,x(k),str2
        k=k+1
    case (50:51) 
     read(fileid,*) str,str3,dummy,str2
!     str=adjustl(trim(str))
     str3=adjustl(trim(str3))
!     str2=adjustl(trim(str2))
     str2="</ETA>"
     write(fileid2,'(12X,A4,X,A6,F13.7,X,A)') str,str3,x(k),str2
        k=k+1
    case (54:56) 
     read(fileid,'(A16,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,X,A)') str,x(k),str2
        k=k+1

     end select 
    enddo

    print *,'file written'
!    call RENAME(inputstr,"para.xml.restart",ierr)
!    if (ierr .ne. 0) print*,'shit...moving file failed'
!    call RENAME('temp',inputstr,ierr)
!    if (ierr .ne. 0) print*,'shit...moving file failed'
   
    close(fileid)
    close(fileid2)
end subroutine
! *****************************************************************************
! *****************************************************************************
      subroutine var_trans(x,y,lbnd,ubnd)
      real(kind=dp),dimension(:) :: x,lbnd,ubnd
      real(kind=dp),dimension(:),intent(out) :: y
      y = ((x - lbnd) / (ubnd - lbnd))
      end subroutine
! *****************************************************************************
! *****************************************************************************
      subroutine var_back_trans(x,y,lbnd,ubnd)
      real(kind=dp),dimension(:) :: x,y,lbnd,ubnd
      y = x*(ubnd-lbnd) + lbnd
      end subroutine
! *****************************************************************************
! *****************************************************************************
subroutine sample(this,n,x,l_bound,u_bound)
    !--------------------------------------------------------------
    TYPE(paramopt_master_type)               :: this
    integer,intent(in) :: n
    integer,parameter :: dp=8
    integer :: irand,t,j
    real(kind=dp) :: x(n)
    real(kind=dp) :: sampler,rnum,step,temp
    real(kind=dp) :: sampx(n),sr,tmpx(n),tmpx2(n),l_bound(n),u_bound(n)
    logical :: generate
    
    sampler=1._dp/(2._dp*sqrt(real(n)))
    generate=.TRUE.
!    step=1._dp/2._dp
    step = this%step_size
    tmpx = x
    !=================random uniform sampling======================
!   do i=1,n; tmpx(i) = next_random_number(this%rng_stream,error=this%error); enddo
!    call var_back_trans(tmpx,x,l_bound,u_bound)
    !==============================================================
!    t=1
!    irand=7
!            do while (generate)
!--------------------------random only one variable---------------------------
!                call random_number(rnum)
                 rnum = next_random_number(this%rng_stream,error=this%error)
                 irand=int(rnum*n)+1
                 print *,'change var num',irand
!
!                irand = 2 
!                call var_trans(x,tmpx,l_bound,u_bound)
                 rnum = next_random_number(this%rng_stream,error=this%error)
!                !print *,'after var trans',tmpx(irand)
                if (rnum .gt. 0.5_dp) then
                    step=step
                else
                    step=-step
                endif
                tmpx(irand) = tmpx(irand) + step
!               ! print *,'before check',tmpx(irand) 
                if (tmpx(irand) .gt. 1.0_dp) then 
                    tmpx(irand) = tmpx(irand) - 2*step
                endif
                
                if (tmpx(irand) .lt. 0._dp) then
                    tmpx(irand) = tmpx(irand) + 2*abs(step)
                endif
              x = tmpx  
              !  call var_back_trans(tmpx,x,l_bound,u_bound)
!             !   print *,'test',tmpx(irand)
!             !   print *,'old var',x(irand), &
!             !   'new var',(tmpx(irand)*((u_bound(irand)) - (l_bound(irand)))) + (l_bound(irand))
!--------------------------random ball walk-------------------------------------  
!                call random_number(sampx)   
!                call var_trans(x,tmpx,l_bound,u_bound)
!                sampx = sampx-tmpx
!                sampx(7) = 0._dp
!                sr = sqrt(sum(sampx**2))
!                stepr = 1._dp/(2._dp*sqrt(real(n)))
!                sampx = stepr*sampx/sr
!                tmpx = tmpx+sampx
!--------------------------random sample---------------------------------------
!                call random_number(sampx)
!                tmpx=sampx
!                tmpx(7)=1._dp
!              !  if (tmpx(4) .gt. tmpx(5)) then 
!              !      temp=tmpx(4)
!              !      tmpx(4)=tmpx(5)
!              !      tmpx(5)=temp 
!              !  endif
!              !  if (tmpx(21) .gt. tmpx(22)) then 
!              !      temp=tmpx(21)
!              !      tmpx(21)=tmpx(22)
!              !      tmpx(22)=temp 
!              !  endif
!--------------------------change one random variable---------------------------
!                print *,'generation loop',t
!                do while (irand .eq. 7)
!                 call random_number(rnum)
!                 irand=int(rnum*n)+1
!                enddo
!                call random_number(rnum)
!                call var_trans(x,tmpx,l_bound,u_bound)
!                tmpx(irand) = rnum
!                print *,'change var',irand
!                print *,'test',tmpx(irand)
!                print *,'old var',x(irand), &
!                'new var',(tmpx(irand)*((u_bound(irand)) - (l_bound(irand)))) + (l_bound(irand))
!-------------------------------------------------------------------------------
!                call var_back_trans(tmpx,x,l_bound,u_bound)

            !    do j=1,n
            !        if (tmpx2(j) .lt. l_bound(j)) then
            !            generate = .TRUE. 
            !            print *,j,tmpx2(j)
            !            exit
            !        else
            !            generate = .FALSE.
            !        endif
            !        if (tmpx2(j) .gt. u_bound(j)) then
            !            generate = .TRUE.
            !            print *,j,tmpx2(j)
            !            exit
            !        else
            !            generate = .FALSE.
            !        endif   
            !        !    call random_number(random)
            !        !    j = random*n
            !        !    if (j .eq. 0) j = 1
            !        !    call random_number(random)
            !        !    x(j) = random*abs(u_bound(j)-l_bound(j)) + l_bound(j)
            !    enddo
!                    generate = .FALSE.
!            enddo
!            x = tmpx2

!            if (this%para_env%mepos .eq. 0) print *,'change var',irand,'new value',x(irand)
end subroutine
! *****************************************************************************
! *****************************************************************************
 SUBROUTINE calc_obj_f(this, Epot, obj_f)
    TYPE(paramopt_master_type)               :: this
    REAL(KIND=dp),intent(in)                 :: Epot(:)
    REAL(KIND=dp),intent(out)                :: obj_f
    REAL(KIND=dp)                            :: energy
    real(kind=dp) :: h2o8s4_ref,h2o_eaip_ref,h2o_ref,h3op_ref,ohm_ref
    integer,parameter    :: h2o_id=1,h3o_id=16,ohm_id=22,h2o8s4_id=11 !,h2o_eaip_id=31
    integer                                  :: i 


        !---------------energy references-------------------------------------------
        h2o_ref      = Epot(h2o_id)
        !h2o_eaip_ref = Epot(h2o_eaip_id)
        h3op_ref     = Epot(h3o_id)
        ohm_ref      = Epot(ohm_id) 
        h2o8s4_ref   = Epot(h2o8s4_id)
        !---------------------------------------------------------------------------
                    write(*,*) '==============================asdfkjalsdkjfl==================='
                    DO i=1,this%num_ref
                          ! calculate energy error
                          energy = (this%ener_ref(i)%stoch1*Epot(i) + &
                          this%ener_ref(i)%stoch2*h3op_ref + &
                          this%ener_ref(i)%stoch3*ohm_ref + &
                          this%ener_ref(i)%stoch4*h2o_ref + & 
                          this%ener_ref(i)%stoch5*h2o8s4_ref)*this%kcalmol
                          this%res(i) = abs(energy-this%ener_ref(i)%ref)
                          ! print eneregy error
                          IF(this%iw>0) THEN
!                            write(*,'(A15,1X,I2,1X,I1,1X,I1,1X,I2,1X,I2,F10.2, &
!                                      1X,F10.2,1X,F10.2,F10.2,1X,F10.2,1X,L1,1X,F10.2)') &
                            write(*,'(A15,1X,I2,1X,I1,1X,I1,1X,I2,1X,I2,F10.2, &
                                      1X,F10.2,1X,F10.2,F10.2,1X,F10.2,1X)') &
                            trim(this%ener_ref(i)%mol), &
                            this%ener_ref(i)%stoch1, &
                            this%ener_ref(i)%stoch2, &
                            this%ener_ref(i)%stoch3, &
                            this%ener_ref(i)%stoch4, &
                            this%ener_ref(i)%stoch5, &
                            Epot(i), &
                            Epot(i)*this%kcalmol, &
                            energy, &
                            this%ener_ref(i)%ref, &
                            this%res(i) !, &
!                            dummy, &
!                            report_timings(i)
                          END IF
                    END DO
                    write(*,*) '==============================asdfkjalsdkjfl==================='
          obj_f = sum(this%res)
                    write(*,*) obj_f
 END SUBROUTINE
! *****************************************************************************
! *****************************************************************************
 !-----------------------Gaussian Adaption Algorithm-----------------------------
    subroutine GaussAdapt(n,x,gauss_opt)

    implicit none
    type(gauss_opt_type),target                               :: gauss_opt
    CHARACTER(len=default_string_length),pointer              :: task
    integer,intent(in)                                        :: n
    real(kind=dp),parameter                                   :: sto_eps=1e-2_dp
    real(kind=dp),pointer,intent(inout)                       :: x(:)
	real(kind=dp),pointer                                     :: f,c_t,r,eta(:),m(:)
	real(kind=dp),pointer                                     :: Q(:,:),C(:,:)
	real(kind=dp),pointer                                     :: l_bound(:),u_bound(:)
	real(kind=dp),pointer                                     :: fmin,xmin(:)
	real(kind=dp)                                             :: detQ,dQ(n,n),delta(n)
	real(kind=dp)                                             :: Id(n,n),eig_val(n),work(3*n-1),test,temp(n,n),x_old(n)
	real(kind=dp)                                             :: wm,wc,wt,beta,p,fe,fc
    integer                                                   :: i,j,info,algorithm
    integer,pointer                                           :: nf
    logical                                                   :: u_check(n),l_check(n),file_ex
    TYPE(cp_error_type),POINTER                               :: error
!----------------------------set pointers to type struct-----------------------
    f => gauss_opt%f
    c_t => gauss_opt%c_t
    r => gauss_opt%r
    eta => gauss_opt%eta
    m => gauss_opt%m
    Q => gauss_opt%Q
    C => gauss_opt%C
    fmin => gauss_opt%fmin
    xmin => gauss_opt%xmin
    task => gauss_opt%task
    nf => gauss_opt%nf
!----------------------------set GauAdapt prameter------------------------------
    print *,'in gauss adapt    ',trim(task),r,c_t,f
    algorithm=1
    x_old=x
    Id=0._dp
    do i=1,n; Id(i,i)=1._dp; enddo
    wm=exp(1._dp)*n
    wc=(n+1._dp)**2/(log(n+1._dp))
    wt=wc/2._dp
    beta=1._dp/wc
    p=1._dp/exp(1._dp)
    fe=1._dp + beta*(1._dp-p)
    fc=1._dp-beta*p
    !other than default values for things
   wm=1._dp
   wt=1._dp
!	wt=exp(1._dp)*n
!	wt=10._dp*n**2

!-------------------------------------------------------------------------------
    if (task(1:5).eq.'new_x') then
        if (f .lt. c_t) then
            delta = x - m
            task='new_x_accepted'
!----------------------------save current best----------------------------------
            if (f .lt. fmin) then
            fmin = f 
            xmin = x 
            endif
!------------------------------adapt parameters---------------------------------
            r=fe*r
            c_T = (1._dp-1._dp/wt)*c_T+f/wt
            m = (1._dp-1._dp/wm)*m + x/wm
!-----------------------------update convariance--------------------------------

!-----------------------------rank 1 update ------------------------------------
     select case (algorithm)
        case(1)
            dQ=(1._dp-1._dp/wc)*Id
            call DSYR('l',n,1._dp/wc,eta,int(1),dQ,n)
        case(2)
            dQ=(1._dp-1._dp/wc)*C
!            call DSYR('l',n,1._dp/wc,delta,int(1),dQ,n)
            call DGER(n,n,1._dp/wc,delta,int(1),delta,int(1),dQ,n)
            C=dQ
     end select
!-----------------------------eig.val decomposition-----------------------------
            call DSYEV( 'V', 'l', n, dQ, n, eig_val, WORK, 3*n-1, INFO )
            if (info.ne.0) task='stop_eigendecomp'
            if (any(isnan(eig_val))) task='stop'
            if (any(eig_val.lt.0)) then
                task='stop eig_val'
                return 
            endif
!-----------------------------normalization of Q and C--------------------------
            detQ=product(eig_val)
            detQ=detQ**(1.d0/n)
            C=C/detQ
            eig_val=eig_val/detQ
            eig_val=sqrt(eig_val)
!-----------------------------update Q such that Q=Q*D^1/2------------------------
      select case (algorithm)
         case(1)
            do i=1,n; dQ(:,i)=dQ(:,i)*sqrt(eig_val(i)); enddo
            call dgemm('n','t',n,n,n,1._dp,dQ,n,dQ,n,0._dp,temp,n)
            dQ=temp
            call dsymm('r','l',n,n,1._dp,dQ,n,Q,n,0._dp,temp,n)
            Q=temp
            call dgemm('n','t',n,n,n,1._dp,Q,n,Q,n,0._dp,temp,n)
            C=temp
          case(2)
            do i=1,n; dQ(:,i)=dQ(:,i)*(eig_val(i)); enddo
            Q=dQ
!            call dgemm('n','t',n,n,n,1._dp,Q,n,Q,n,0._dp,temp,n)
!            C=temp
!            print *,temp
!            print *,'---------------------'
!            print *,C
!            if(any(temp.ne.C)) print *,'something wroing'
      end select
!--------------------------------update ---------------------------------------
        else if (isnan(f)) then
            task='new_x_rejected'
            r=fc*r
        else
!---------------rejected lower step size dont adopt cov. + mean-----------------
            task='new_x_rejected'
            r=fc*r  
        endif
!------------------------------------sample-------------------------------------
        if (r.lt.1e-9) then
            task='stop_r'
            return
        endif
            do i=1,n; eta(i) = next_random_number(gauss_opt%rng_normal_stream, error=error); enddo
            !CALL reset_to_next_rng_substream(rng_stream)
            x=m
            call DGEMV('N',n,n,r,Q,n,eta,int(1),1._dp,x,int(1))
            
            if (any(isnan(x))) then 
            task='stop_sampling x'
            ! write nex x to se_para_file
            endif
            
    endif

    if (task(1:5).eq.'start') then
     print *,'This is the gauss start - yeah'
     inquire(file="gauss.restart",exist=file_ex)
     if (file_ex) then
        print *,'Gauss parameters from Restart file'
        ! restart file should be read in main!!!!!
        !call read_gauss_file(eta,m,C,Q,r,c_t,xmin,fmin,file_id)
     else    
        Q = Id 
        C = Q
        x = x_old
        m = x
        task='new_x'
        fmin=c_t
        xmin=x
        nf = 0
      endif
    endif
END SUBROUTINE
END MODULE paramopt_master

