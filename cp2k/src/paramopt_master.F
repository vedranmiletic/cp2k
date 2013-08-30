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
  USE swarm_message,                   ONLY: swarm_message_type

  USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                             section_vals_type,&
                                             section_vals_val_get
  USE input_constants,                 ONLY: paramopt_no_run,&
                                             paramopt_sampling_run,&
                                             paramopt_powell_run,&
                                             paramopt_gaussian_run
  USE kinds,                           ONLY: default_string_length,&
                                             dp
  USE message_passing,                 ONLY: mp_bcast
  USE parallel_rng_types,              ONLY: rng_stream_type, &
                                             create_rng_stream, &
                                             next_random_number, &
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

 TYPE gauss_opt_type
   INTEGER                                         :: n,nf
   REAL(KIND=dp),DIMENSION(:),ALLOCATABLE          :: eta,m,x,xmin
   REAL(KIND=dp),DIMENSION(:),ALLOCATABLE          :: l_bound,u_bound
   REAL(KIND=dp),DIMENSION(:,:),ALLOCATABLE        :: Q,C
   REAL(KIND=dp)                                   :: c_t,f,fmin,r
   REAL(KIND=dp)                                   :: wm,wc,wt,beta,p,fe,fc
   CHARACTER(len=default_string_length)            :: task
   TYPE(rng_stream_type),POINTER                   :: rng_normal_stream
 
 END TYPE gauss_opt_type

 TYPE paramopt_master_type
   PRIVATE
   INTEGER                                          :: iw
   INTEGER                                          :: num_inp
   INTEGER                                          :: num_calc
   INTEGER                                          :: num_ref
   INTEGER                                          :: num_var
   INTEGER                                          :: num_obj
   INTEGER                                          :: num_para_set
   INTEGER                                          :: n_walkers
   INTEGER                                          :: i_inp
   INTEGER                                          :: inp_iter
   INTEGER                                          :: run_iter
   INTEGER                                          :: run_type_id
   INTEGER,ALLOCATABLE                              :: i_recv(:)
   TYPE(cp_error_type),POINTER                      :: error
   TYPE(ener_ref_type),POINTER                      :: ener_ref(:)
   CHARACTER(len=default_string_length),POINTER     :: input_files(:)
   CHARACTER(len=default_string_length),POINTER     :: coord_files(:)
   INTEGER,POINTER                                  :: charges(:)
   REAL(KIND=dp),ALLOCATABLE                        :: Epot(:),timings(:)
   REAL(KIND=dp),POINTER                            :: res(:)
   REAL(KIND=dp)                                    :: E_res
   REAL(KIND=dp)                                    :: rmsd
   REAL(KIND=dp),POINTER                            :: function_matrix(:,:)
   REAL(KIND=dp),POINTER                            :: obj_f
   REAL(KIND=dp),ALLOCATABLE                        :: x(:)
   REAL(KIND=dp),POINTER                            :: xtrans(:)
   REAL(KIND=dp),POINTER                            :: xtransold(:)
   REAL(KIND=dp),POINTER                            :: xold(:)
   REAL(KIND=dp),POINTER                            :: u_bound(:)
   REAL(KIND=dp),POINTER                            :: l_bound(:)
   REAL(KIND=dp)                                    :: pi=2*asin(1._dp), &
                                                       kcalmol=6.27509468713739E+02_dp, &
                                                       ev=27.2116_dp,&
                                                       ev_kcal=23.06035_dp
   TYPE(cp_para_env_type), POINTER                  :: para_env
   TYPE(rng_stream_type),POINTER                    :: rng_stream
   LOGICAL,POINTER                                  :: sampling
   REAL(KIND=dp)                                    :: step_size
   REAL(KIND=dp),POINTER                            :: weights(:)
   TYPE(opt_state_type),POINTER                     :: powell_opt
   TYPE(gauss_opt_type),POINTER                     :: gauss_opt
 END TYPE paramopt_master_type

 TYPE :: ener_ref_type
   PRIVATE
   INTEGER                                          :: id
   CHARACTER(len=default_string_length)             :: mol
   CHARACTER(len=default_string_length)             :: mol_ref1
   CHARACTER(len=default_string_length)             :: mol_ref2
   CHARACTER(len=default_string_length)             :: mol_ref3
   CHARACTER(len=default_string_length)             :: mol_ref4
   INTEGER                                          :: stoch1
   INTEGER                                          :: stoch2
   INTEGER                                          :: stoch3
   INTEGER                                          :: stoch4
   INTEGER                                          :: stoch5
   REAL(KIND=dp)                                    :: ref  
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
    TYPE(section_vals_type), POINTER         :: paramopt_section
    INTEGER                                  :: i,charge
    INTEGER                                  :: time(3)
    LOGICAL                                  :: restart
    REAL(kind=dp),DIMENSION(3,2)             :: seed

    !---------------------
    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init called"
    !------------------
    this%num_var = 16
    !------------------
    this%num_obj = 2
    !-------------------
    this%num_inp = 31
    !-------------------
    this%num_calc = 31
    !--------------------
    this%num_ref = 30
    !--------------------
    this%iw = iw
    !------------------------
    this%para_env => para_env
    this%n_walkers = n_walkers
    !-------------------
    nullify(this%powell_opt,this%gauss_opt,this%sampling)
    !-------------------
    allocate(this%ener_ref(this%num_ref))
    allocate(this%Epot(this%num_ref))

    allocate(this%x(this%num_var))
    allocate(this%xtrans(this%num_var))
    allocate(this%xtransold(this%num_var))
    allocate(this%xold(this%num_var))
    allocate(this%l_bound(this%num_var))
    allocate(this%u_bound(this%num_var))
    ALLOCATE(this%input_files(this%num_inp))
    ALLOCATE(this%coord_files(this%num_inp))
    ALLOCATE(this%charges(this%num_inp))
    allocate(this%res(this%num_inp))
    allocate(this%rng_stream)

    allocate(this%timings(this%num_inp))
    allocate(this%weights(this%num_obj))
    allocate(this%i_recv(10))
    !-------------------
    paramopt_section => section_vals_get_subs_vals(swarm_section, "PARAM_OPT", error=error)
    CALL section_vals_val_get(paramopt_section,"STEP_SIZE", r_val=this%step_size, error=error)
    CALL section_vals_val_get(paramopt_section,"UPPER_BOUNDS", r_vals=this%u_bound, error=error)
    CALL section_vals_val_get(paramopt_section,"LOWER_BOUNDS", r_vals=this%l_bound, error=error)
    CALL section_vals_val_get(paramopt_section,"WEIGHTS", r_vals=this%weights, error=error)
    CALL section_vals_val_get(paramopt_section,"RUN_TYPE",i_val=this%run_type_id,error=error)
    print *,'run_type',this%run_type_id,'step size',this%step_size
    print *,'upper bounds',this%u_bound
    print *,'lower bounds',this%l_bound
    
    !-------------------
    this%num_para_set = 1
    this%i_inp = 1
    this%inp_iter = this%num_inp
    this%run_iter = 1
    this%i_recv = 0
    this%Epot = 0._dp
    call itime(time)
    seed(:,1) = REAL(time)
    seed(:,2) = (/ 1._dp, 2._dp, 3._dp /)
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
        CASE default
       ! CPAssert(.FALSE.,cp_failure_level,routineP,error,failure)
    END SELECT
    !-----read x --------
    CALL read4file(this%num_var,this%x,restart)
    CALL para2file(this%num_var,this%x,"para.xml","para.xml.out")
    CALL var_trans(this%x,this%xtrans,this%l_bound,this%u_bound)
    !----------------set pointers for sampling------------------------
    if (associated(this%sampling)) this%sampling = .true.
    !----------------set pointers gauss adapt------------------------
    if (associated(this%gauss_opt)) then
    CALL section_vals_val_get(paramopt_section,"C_T", r_val=this%gauss_opt%c_t, error=error)
    allocate(this%gauss_opt%eta(this%num_var),this%gauss_opt%m(this%num_var), &
             this%gauss_opt%xmin(this%num_var))
    allocate(this%gauss_opt%l_bound(this%num_var),this%gauss_opt%u_bound(this%num_var))
    allocate(this%gauss_opt%x(this%num_var),this%gauss_opt%Q(this%num_var,this%num_var), &
             this%gauss_opt%C(this%num_var,this%num_var))
    allocate(this%gauss_opt%rng_normal_stream)
    CALL create_rng_stream(this%gauss_opt%rng_normal_stream,'gaussian',distribution_type=GAUSSIAN,error=this%error)
    this%gauss_opt%task = 'startx'
    this%gauss_opt%l_bound = this%l_bound
    this%gauss_opt%u_bound = this%u_bound
    this%gauss_opt%r = this%step_size
    this%obj_f => this%gauss_opt%f
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
    this%obj_f => this%powell_opt%f
    CALL powell_optimize(this%num_var,this%xtrans,this%powell_opt)
    endif
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
    !------------------------
!    allocate(this%function_matrix(this%num_para_set,this%num_inp))
    !------------------------
    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init done"
 END SUBROUTINE paramopt_master_init

! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_steer(this, report, cmd, should_stop)
    TYPE(paramopt_master_type)               :: this
    TYPE(swarm_message_type), INTENT(IN)     :: report
    TYPE(swarm_message_type), INTENT(INOUT)  :: cmd
    LOGICAL, INTENT(INOUT)                   :: should_stop
    INTEGER                                  :: i,i_inp,r_iter
    INTEGER                                  :: report_worker_id
    integer,parameter    :: h2o_id=1,h3o_id=16,ohm_id=22,h2o8s4_id=11,h2o_eaip_id=31 
    real(kind=dp) :: h2o8s4_ref,h2o_eaip_ref,h2o_ref,h3op_ref,ohm_ref
    REAL(KIND=dp)                            :: energy
    REAL(KIND=dp)                            :: report_rmsd
    CHARACTER(len=default_string_length)     :: report_status

    r_iter = 1
    CALL swarm_message_get(report, "worker_id", report_worker_id)
    CALL swarm_message_get(report, "status",report_status)
!    print *,"reported status is",report_status,this%inp_iter
    IF (report_status == "ene_calc") THEN
    !---------we got a new message, deal with it :)---------------
       CALL swarm_message_get(report, "num_inp", i_inp)
       CALL swarm_message_get(report, "iter", r_iter)
       CALL swarm_message_get(report, "Epot", this%Epot(i_inp))
       !------------change status---------------------------------
       report_status = "wait_done"
       !------------deal with condition---------------------------
       this%i_recv(r_iter) = this%i_recv(r_iter) + 1
       !---------------print------------------------------------
!         IF(this%iw>0) THEN
!            WRITE (this%iw,'(A,20X,I8,I8,I8)')&
!                   " PRMOPT| Reporting worker ", report_worker_id, i_inp, r_iter
!            WRITE (this%iw,'(A,20X,E15.8)')&
!                   " PRMOPT| Reported potential Energy [Hartree] ", this%Epot(i_inp)
!         END IF

    ELSE IF (report_status == 'rmsd_calc') THEN
    !---------we got a new message, deal with it :)---------------
       CALL swarm_message_get(report, "num_inp", i_inp)
       CALL swarm_message_get(report, "iter", r_iter)
       CALL swarm_message_get(report, "rmsd", this%rmsd)
       !------------change status---------------------------------
       report_status = "wait_done"
       !------------deal with condition---------------------------
       this%i_recv(r_iter) = this%i_recv(r_iter) + 1
       !---------------print------------------------------------
!         IF(this%iw>0) THEN
!            WRITE (this%iw,'(A,20X,I8,I8,I8)')&
!                   " PRMOPT| Reporting worker ", report_worker_id, i_inp, r_iter
!            WRITE (this%iw,'(A,20X,E15.8)')&
!                   " PRMOPT| Reported rmsd ", this%rmsd
!         END IF
    ENDIF
    
    IF (this%i_recv(r_iter) .eq. this%num_calc) THEN
    !--------------we are done with the iteration/condition--------   
        call calc_obj_f(this, this%x, this%Epot, this%E_res)
        this%obj_f = this%weights(1)*this%E_res + this%weights(2)*this%rmsd
        IF(this%iw>0) THEN
          print *,'-------Rmsd-------'
          print *,this%rmsd
          print *,'-------Epot-------'
          print *,this%Epot
          print *,'-------obj_f-------'
          print *,this%obj_f
          open(999,file='Epot.out',position='append')
          write(999,*) this%Epot
          close(999)
          open(998,file='obj_f.out',position='append')
          write(998,*) this%xtrans,this%obj_f,this%E_res,this%rmsd
          close(998)
         
        ENDIF
        !--------------------new x to file?--------------------
        CALL para2file(this%num_var,this%x,"para.xml","para.xml.old.out") 
        !----new x-------------------
        !------------------------optimizer-----------------------------
          CALL var_trans(this%x,this%xtrans,this%l_bound,this%u_bound)
          if (associated(this%powell_opt)) CALL powell_optimize(this%num_var, &
              this%xtrans,this%powell_opt)
          if (associated(this%gauss_opt)) CALL GaussAdapt(this%num_var, &
              this%xtrans,this%gauss_opt)
          if (associated(this%sampling)) CALL sample(this,this%num_var, &
              this%x,this%l_bound,this%u_bound)
          !------------------check constrains-------------------------
          do i=1,this%num_var 
              if (this%xtrans(i) .lt. 0._dp) this%xtrans(i) = 0._dp
              if (this%xtrans(i) .gt. 1._dp) this%xtrans(i) = 1._dp
          enddo
          !condition on outer H_core???
          !if (this%xtrans(13) .lt. this%xtrans(11)) this%xtrans(13) = this%xtrans(11)
          CALL var_back_trans(this%xtrans,this%x,this%l_bound,this%u_bound)
        !---------reset inp_iter--------
         this%inp_iter = this%num_calc
         this%i_recv(r_iter) = 0
         !--------------------new x to file?--------------------
          CALL para2file(this%num_var,this%x,"para.xml","para.xml.out") 
    ENDIF
    
    !--------------master have a task to distr---------
    IF (this%inp_iter .gt. 0 .and. report_status == "wait_done" ) THEN
            !send new task to idle worker
         if (this%inp_iter .lt. this%num_calc) then
            CALL swarm_message_add(cmd, "command", "energy")
         else if (this%inp_iter .eq. this%num_calc) then
            CALL swarm_message_add(cmd, "command", "geo_opt")
         endif
         CALL swarm_message_add(cmd, "input", this%input_files(this%inp_iter))
         CALL swarm_message_add(cmd, "coord", this%coord_files(this%inp_iter))
         CALL swarm_message_add(cmd, "charge", this%charges(this%inp_iter))
         CALL swarm_message_add(cmd, "num_inp",this%inp_iter)
         CALL swarm_message_add(cmd, "iter", this%run_iter)
         CALL swarm_message_add(cmd, "x", this%x)
!       IF(this%iw>0) THEN
!          WRITE (this%iw,'(A,20X,I8,I8,I8)')&
!                 " PRMOPT| cmd send worker ", report_worker_id,  &
!                 this%inp_iter, this%run_iter
!       END IF
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


! *****************************************************************************
! *****************************************************************************
subroutine read4file(n,x,ex)
    implicit none
    integer,intent(in)       :: n
    integer                  :: i,fileid,k
    real(kind=dp),intent(out) :: x(n)
    character(200)            :: str,str2,filename
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
    case (17:18)
     read(fileid,*) str,x(k),str2
     k=k+1
    case (20:21)        
     read(fileid,*) str,x(k),str2
     k=k+1
     
    case (28:30)
     read(fileid,*) str,x(k),str2
     k=k+1

    case (36:37)
     read(fileid,*) str,x(k),str2
     k=k+1

    case (39:40)
     read(fileid,*) str,x(k),str2
     k=k+1

    case (45:46)
     read(fileid,*) str,x(k),str2
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
    character(24)              :: str,str2
    
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
    case (17:18)
     read(fileid,'(A24,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A24,F13.7,A1,A20)') str,x(k),' ',str2
     k=k+1
    case (20:21)        
     read(fileid,'(A24,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))d
     str2=adjustl(trim(str2))
     write(fileid2,'(A24,F13.7,A20)') str,x(k),str2
     k=k+1
    case (28:30)
     read(fileid,'(A16,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A16,F13.7,A,A16)') str,x(k),' ',str2
        k=k+1
    case (36:37)
     read(fileid,'(A24,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A24,F13.7,A,A20)') str,x(k),' ',str2
        k=k+1
    case (39:40)
     read(fileid,'(A24,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A24,F13.7,A24)') str,x(k),str2
        k=k+1
    case (45:46)
     read(fileid,'(A24,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A24,F13.7,A20)') str,x(k),str2
        k=k+1
    case (54:56) 
     read(fileid,'(A16,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A16,F13.7,A,A16)') str,x(k),' ',str2
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
!
!                irand = 4 
                call var_trans(x,tmpx,l_bound,u_bound)
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
                call var_back_trans(tmpx,x,l_bound,u_bound)
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
 SUBROUTINE calc_obj_f(this, x, Epot, obj_f)
    TYPE(paramopt_master_type)               :: this
    REAL(KIND=dp),intent(in)                 :: x(:)
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
!    wm=1._dp
!	wt=exp(1._dp)*n
!	wt=10._dp*n**2
!    wt=1._dp

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

