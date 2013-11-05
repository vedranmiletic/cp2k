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
  USE cp_files,                        ONLY: open_file,&
                                             close_file
  USE cma_es,                          ONLY: cmaes,gen_pop
  USE cma_es_types,                    ONLY: individual_type, &
                                             individual_type_p, &
                                             cmaes_opt_type
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
  USE mathconstants,                   ONLY: pi
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
   REAL(KIND=dp),POINTER                                 :: init_x(:)
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
   LOGICAL                                               :: rand_inp
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
    TYPE(paramopt_master_type)                  :: this
    TYPE(cp_para_env_type), POINTER             :: para_env
    TYPE(section_vals_type), POINTER            :: swarm_section
    CHARACTER(len=*), PARAMETER :: routineN = 'paramopt_master_init', &
      routineP = moduleN//':'//routineN
    CHARACTER(LEN=*), INTENT(IN)                :: input_path
    CHARACTER(len=default_string_length)        :: output_path, str
    INTEGER, INTENT(IN)                         :: n_walkers, iw
    TYPE(cp_error_type), INTENT(inout),TARGET   :: error
    TYPE(section_vals_type), POINTER            :: paramopt_section => null()
    TYPE(section_vals_type), POINTER            :: cmaes_section => null()
    INTEGER                                     :: i,charge
    INTEGER                                     :: time(3)
    LOGICAL                                     :: restart,explicit_key
    REAL(kind=dp),DIMENSION(3,2)                :: seed

    type(individual_type),POINTER               :: a_k
    type(individual_type),POINTER               :: a_k_parents
    INTEGER                                     :: k
    INTEGER                                     :: unit_number
    REAL(kind=dp)                               :: ini_sigma

    this%error => error
    !---------------------
    !IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init called"
    !====================num para set======================== 
    this%num_para_set = 1
    !-------------------
    paramopt_section => section_vals_get_subs_vals(swarm_section, "PARAM_OPT", error=error)
    CALL section_vals_val_get(paramopt_section,"RUN_TYPE",i_val=this%run_type_id,error=error)
    !------------------
    CALL section_vals_val_get(paramopt_section,"NUM_VAR", i_val=this%num_var, error=error)
    CALL section_vals_val_get(paramopt_section,"SEED", i_val=this%input_seed, error=error)
    CALL section_vals_val_get(paramopt_section,"BENCHMARK", l_val=this%benchmark, error=error)

    if (this%run_type_id .eq. paramopt_cmaes_run) then
      cmaes_section => section_vals_get_subs_vals(paramopt_section, "CMAES_OPT", error=error)
      CALL section_vals_val_get(cmaes_section,"LAMBDA_MO",i_val=this%num_para_set,error=error)
    endif
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
    

    CALL section_vals_val_get(paramopt_section,"INITIAL_X" ,explicit=explicit_key,error=error)
      IF(explicit_key)THEN
        CALL section_vals_val_get(paramopt_section,"INITIAL_X", r_vals=this%init_x, error=error)
         if (size(this%init_x) .eq. size(this%x)) then
           this%x = this%init_x 
           this%rand_inp = .false.
         else 
           call cp_assert( .false. ,cp_failure_level,cp_assertion_failed,&
                       routineP, "init_x does not match num_var",this%error)
         endif
      ELSE
           this%rand_inp = .true.
      ENDIF

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
    if (this%rand_inp) then
        do i=1,this%num_var; this%xtrans(i) = next_random_number(this%rng_stream,error=this%error); enddo           
        call var_back_trans(this%xtrans,this%x,this%l_bound,this%u_bound)
        !end random x
    endif
    if (.not. this%benchmark) then
      !if (.not. this%rand_inp) CALL read4file(this%num_var,this%x,restart)
      !CALL para2file(this%num_var,this%x,"para.xml","para.xml.out")
    endif
    !--------------------------------------------
    print *,'num var',this%num_var
    print *,'run_type',this%run_type_id,'step size',this%step_size
    print *,'upper bounds',this%u_bound
    print *,'lower bounds',this%l_bound
    print *,'this is inital x'
    print *, this%x
    ! in case sigma is not given use sqrt(var(x))
    ini_sigma = sqrt(variance(this%x))
    IF (this%step_size .eq. 0_dp) this%step_size = ini_sigma
    print *,'this should be sigma',this%step_size
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
      allocate(this%cmaes_opt%insigma(this%num_var))
      !CALL section_vals_val_get(paramopt_section,"INSIGMA", r_vals=this%cmaes_opt%insigma, error=error)
      this%cmaes_opt%insigma = 0.5_dp*(this%u_bound-this%l_bound)
      this%step_size = maxval(this%cmaes_opt%insigma)/3_dp
      print *,'this is insigma',this%cmaes_opt%insigma
      print *,'this is step_size',this%step_size
      this%cmaes_opt%error => this%error
      this%cmaes_opt%lambda_mo = this%num_para_set
      allocate(this%cmaes_opt%population(2*this%cmaes_opt%lambda_mo))
      allocate(this%cmaes_opt%population_p(2*this%cmaes_opt%lambda_mo))
      allocate(this%cmaes_opt%a_k(this%cmaes_opt%lambda_mo))
      allocate(this%cmaes_opt%a_k_parents(this%cmaes_opt%lambda_mo))
      allocate(this%cmaes_opt%fmax(this%num_obj))
      allocate(this%cmaes_opt%fmin(this%num_obj))
      this%cmaes_opt%n = this%num_var
      this%cmaes_opt%l_bound => this%l_bound
      this%cmaes_opt%u_bound => this%u_bound
      allocate(this%cmaes_opt%rng_normal_stream)
      allocate(this%cmaes_opt%rng_uniform_stream)
      CALL create_rng_stream(this%cmaes_opt%rng_normal_stream,'gaussian',seed=seed, &
                             distribution_type=GAUSSIAN,error=this%error)
      CALL create_rng_stream(this%cmaes_opt%rng_uniform_stream,'normal',distribution_type=UNIFORM, &
                           seed=seed,error=this%error)
      this%cmaes_opt%generation = 1
      !=============NEEEEEEEDS TO BE CHANGED AND MOVED TO INPUT FILE!!!!!!!==========================
      this%cmaes_opt%fmax = (/100000_dp,10_dp/)
      this%cmaes_opt%fmin = (/0_dp,0_dp/)
     !--------------init population-----------------
     do k=1,this%cmaes_opt%lambda_mo
      this%cmaes_opt%population_p(k)%p => this%cmaes_opt%population(k)
      this%cmaes_opt%a_k(k)%p => this%cmaes_opt%population(k)
      a_k => this%cmaes_opt%a_k(k)%p
      CALL a_k_init(a_k,this%cmaes_opt%lambda_mo,this%num_obj, &
                    this%step_size,this%cmaes_opt%insigma,this)
      !--------------------------------------   
      a_k%f = 0._dp
      a_k%id = k
      CALL gen_pop(a_k,this%cmaes_opt)
      print *,'initial x from a_k',a_k%arx
      !----init parents too---------------
      this%cmaes_opt%population_p(k+this%cmaes_opt%lambda_mo)%p => this%cmaes_opt%population(k+this%cmaes_opt%lambda_mo)
      this%cmaes_opt%a_k_parents(k)%p => this%cmaes_opt%population(k+this%cmaes_opt%lambda_mo)
      a_k_parents => this%cmaes_opt%a_k_parents(k)%p
      CALL a_k_init(a_k_parents,this%cmaes_opt%lambda_mo,this%num_obj, &
                    this%step_size,this%cmaes_opt%insigma,this,a_k%arx(:,1))
      a_k_parents%f = huge(1._dp)
      a_k_parents%id = k+this%cmaes_opt%lambda_mo
      !print *,'initial x from a_k_parents',a_k_parents%arx
      !----init tmp too---------------
     enddo
     nullify(a_k,a_k_parents) 
    endif
!================================================================================
    if (.not. this%benchmark) then
    !---------------------
    call open_file(file_name='WATER27structures/mol_list.txt',unit_number=unit_number, &
                          file_action='read')
    DO i=1,this%num_ref !-5
          !-----watch out num_inp 30 or 35 !!
          READ(88,*) str,charge
          this%charges(i) = charge
          this%input_files(i)= TRIM(str) // ".inp"
          this%coord_files(i)= "WATER27structures/" // trim(str) // ".xyz"
    ENDDO
    call close_file(unit_number)
!    this%input_files(this%num_inp-4) = 'eaip_H2O.inp'
!    this%input_files(this%num_inp-3) = 'eaip_H2Op.inp'
!    this%input_files(this%num_inp-2) = 'eaip_H2On.inp'
!    this%input_files(this%num_inp-1) = 'dipole.inp'
    this%input_files(this%num_ref+1) = 'geo_opt.inp' 
    this%charges(this%num_ref+1) = 0
    this%coord_files(this%num_ref+1) = "WATER27structures/H2O.xyz" 
    !---------------------
    call open_file(file_name='WATER27.dat',unit_number=unit_number, &
                          file_action='read')
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
    call close_file(unit_number)
    endif
    !------------------------
!    allocate(this%function_matrix(this%num_para_set,this%num_inp))
    !------------------------
    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init done"
 END SUBROUTINE paramopt_master_init

 SUBROUTINE a_k_init(a_k,lambda_mo,num_obj,step_size,insigma,this,x)
    IMPLICIT NONE
    TYPE(paramopt_master_type)               :: this
    type(individual_type),POINTER            :: a_k
    REAL(kind=dp),DIMENSION(:)               :: insigma
    REAL(kind=dp),DIMENSION(:),OPTIONAL      :: x
    REAL(kind=dp)                            :: step_size
    INTEGER                                  :: i,n
    INTEGER                                  :: lambda_mo
    INTEGER                                  :: num_obj
    INTEGER                                  :: allocstat
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
                  a_k%D(i,i) = insigma(i)/maxval(insigma)
                  !a_k%D(i,i) = 1._dp 
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
             ALLOCATE(a_k%triuC(n,n),stat=allocStat)
             ALLOCATE(a_k%arx(n,a_k%lambda),stat=allocStat)
             ALLOCATE(a_k%arz(n,a_k%lambda),stat=allocStat)
             ALLOCATE(a_k%xmean(n))
             ALLOCATE(a_k%zmean(n))
             ALLOCATE(a_k%xold(n))
             ALLOCATE(a_k%f(num_obj))
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
             do i=1,lambda_mo; a_k%s_i(i)%p=>null(); enddo
             a_k%front = 0
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
             a_k%cc = 2._dp/(real(n) + 2._dp)
             a_k%ccov = 2._dp/(real(n)**2 + 6._dp)
             a_k%p_tresh = 0.44_dp
             a_k%p_succ = a_k%p_target
             !-----------------------------------------------------------
             a_k%arx(:,1) = this%x
             a_k%xmean = this%x
             a_k%sigma = step_size
             a_k%pc = 0._dp
             a_k%ps = 0._dp
         !    this%cmaes_opt%a_k%hsig = 1._dp
             a_k%ps = 0._dp
             a_k%chiN = sqrt(REAL(n))*((1.0_dp-1.0_dp/(4.0_dp*REAL(n))) &
                                   +1.0_dp/(21.0_dp*REAL(n*n)))
             a_k%d_count = 0
             a_k%s_i_count = 0
             a_k%I_dist = 0._dp
             a_k%kernel_dist = 0._dp
             a_k%hypervol = 0._dp
 END SUBROUTINE
! *****************************************************************************
 SUBROUTINE paramopt_master_steer(this, report, cmd, should_stop)
    TYPE(paramopt_master_type)               :: this
    TYPE(swarm_message_type), INTENT(IN)     :: report
    TYPE(swarm_message_type), INTENT(INOUT)  :: cmd
    LOGICAL, INTENT(INOUT)                   :: should_stop
    CHARACTER(len=*), PARAMETER :: routineN = 'paramopt_master_steer', &
      routineP = moduleN//':'//routineN
    INTEGER                                  :: i,j,k,i_inp,r_iter
    INTEGER                                  :: report_worker_id
    integer,parameter    :: h2o_id=1,h3o_id=16,ohm_id=22,h2o8s4_id=11,h2o_eaip_id=31 
    real(kind=dp) :: h2o8s4_ref,h2o_eaip_ref,h2o_ref,h3op_ref,ohm_ref
    REAL(KIND=dp)                            :: energy
    REAL(KIND=dp)                            :: report_rmsd
    REAL(KIND=dp)                            :: E_res_temp
    CHARACTER(len=default_string_length)     :: report_status
    type(individual_type),POINTER            :: a_k
    type(individual_type),POINTER            :: a_k_parents
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
             !-------------------for cmaes_opt------------------------------
             if (associated(this%cmaes_opt)) then
             a_k => this%cmaes_opt%a_k(j)%p
             !    print *,'penalty of a_k',j,'is',this%cmaes_opt%a_k(j)%penalty
             call calc_obj_f(this, this%function_matrix(:,j), this%E_res(j))
             a_k%f(1) = this%E_res(j) + a_k%penalty
             a_k%f(2) = this%function_matrix(this%num_inp,j) + a_k%penalty
             else
             !-----------------for the rest???--------------------------------
             call calc_obj_f(this, this%function_matrix(:,j), this%E_res(j))
             this%obj_f(j) = this%weights(1)*this%E_res(j) + &
                          this%weights(2)*this%function_matrix(this%num_inp,j) + &
                          this%penalty
             endif
          else
             if (associated(this%cmaes_opt)) then
             a_k => this%cmaes_opt%a_k(j)%p
             !    print *,'penalty of a_k',j,'is',this%cmaes_opt%a_k(j)%penalty
             this%obj_f(j) = this%function_matrix(1,j) !+ this%penalty(j)**2
             a_k%f(1) = this%obj_f(j) + a_k%penalty
             a_k%f(2) = this%function_matrix(2,j) + a_k%penalty
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
        !--------------------new x to file?--------------------
        if (.not. this%benchmark) then
        !CALL para2file(this%num_var,this%x,"para.xml","para.xml.old.out") 
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
         !    do i=1,this%cmaes_opt%lambda_mo
         ! !      if (this%cmaes_opt%a_k(i)%sigma .lt. 1e-9_dp) STOP
         !        if (this%cmaes_opt%a_k(i)%f(1) .lt. 1e-6) this%cmaes_opt%a_k(i)%f(1) = 0._dp
         !        if (this%cmaes_opt%a_k(i)%f(2) .lt. 1e-6) this%cmaes_opt%a_k(i)%f(2) = 0._dp
         !        print *,'f to decide',this%cmaes_opt%a_k(i)%p%f,this%cmaes_opt%a_k_parents(i)%p%f
         !    enddo
             this%cmaes_opt%generation = this%cmaes_opt%generation + 1
        !     if (this%cmaes_opt%generation .eq. 2000) STOP
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
          !CALL para2file(this%num_var,this%x,"para.xml","para.xml.out")
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
         a_k => this%cmaes_opt%a_k(j)%p
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
              this%x_temp = a_k%arx(:,1)
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
            !print *,a_k%arx
            this%x_temp = a_k%arx(:,1)
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


!========================================================================================================
     
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
  !  case (17)
  !   read(fileid,*) str,x(k),str2
  !   k=k+1
    case (19:20)        
     read(fileid,*) str,x(k),str2
     k=k+1
     
  !  case (25)
  !   read(fileid,*) str,str3,x(k),str2
  !   k=k+1

    case (28:30)
     read(fileid,*) str,x(k),str2
     k=k+1

  !  case (36)
  !   read(fileid,*) str,x(k),str2
  !   k=k+1

    case (38:39)
     read(fileid,*) str,x(k),str2
     k=k+1

    case (44:45) 
     read(fileid,*) str,x(k),str2
     k=k+1

  !  case (50:51) 
  !   read(fileid,*) str,str3,x(k),str2
  !   k=k+1 
    
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
  !  case (17)
  !   read(fileid,*) str,dummy,str2
  !   str=adjustl(trim(str))
! !    str2=adjustl(trim(str2))
  !   str2="</Charge_width>"
  !   write(fileid2,'(8X,A,F13.7,X,A)') str,x(k),str2
  !   k=k+1
   ! case (19:20)        
   !  read(fileid,'(A,F13.7,A)') str,dummy,str2
!  !   str=adjustl(trim(str))d
   !  str2=adjustl(trim(str2))
   !  write(fileid2,'(A,F13.7,A)') str,x(k),str2
   !  k=k+1
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
 !   case (36)
 !    read(fileid,*) str,dummy,str2
 !    str=adjustl(trim(str))
!!     str2=adjustl(trim(str2))
 !    str2="</Charge_width>"
 !    write(fileid2,'(8X,A,F13.7,X,A)') str,x(k),str2
 !       k=k+1
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
  !  case (50:51) 
  !   read(fileid,*) str,str3,dummy,str2
! !    str=adjustl(trim(str))
  !   str3=adjustl(trim(str3))
! !    str2=adjustl(trim(str2))
  !   str2="</ETA>"
  !   write(fileid2,'(12X,A4,X,A6,F13.7,X,A)') str,str3,x(k),str2
  !      k=k+1
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
FUNCTION Variance (Array)
IMPLICIT NONE
REAL(kind=dp)::Variance,X
REAL(kind=dp),INTENT(IN),DIMENSION(:)::Array
X = SUM(Array)/SIZE(Array)
Variance = SUM((Array-X)**2)/SIZE(Array)
END FUNCTION Variance

END MODULE paramopt_master

