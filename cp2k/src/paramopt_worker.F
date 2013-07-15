!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2012  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief performs global geometry optimization
!> \par History
!> \author Ole
! *****************************************************************************
MODULE paramopt_worker
  USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                             get_atomic_kind,&
                                             get_atomic_kind_set 
  USE cp_control_types,                ONLY: dft_control_type, &
                                             qs_control_type, &
                                             scptb_control_type 
  USE cp_output_handling,              ONLY: cp_print_key_unit_nr
  USE cp_para_types,                   ONLY: cp_para_env_type
  USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                             cp_subsys_type,&
                                             pack_subsys_particles
  USE f77_interface,                   ONLY: create_force_env,&
                                             calc_energy, &
                                             calc_energy_force, &
                                             destroy_force_env,&
                                             f_env_add_defaults,&
                                             f_env_rm_defaults,&
                                             f_env_type, &
                                             get_energy, &
                                             get_natom
  USE force_env_types,                 ONLY: force_env_get,&
                                             force_env_type
  USE geo_opt,                         ONLY: cp_geo_opt
  USE global_types,                    ONLY: global_environment_type
  USE input_section_types,             ONLY: section_vals_type,&
                                             section_vals_val_get,&
                                             section_vals_val_set
  USE kinds,                           ONLY: default_path_length,&
                                             default_string_length,&
                                             dp
  USE message_passing,                 ONLY: mp_comm_split,&
                                             mp_sum
  USE md_run,                          ONLY: qs_mol_dyn
  USE mdctrl_types,                    ONLY: mdctrl_type
  USE parallel_rng_types,              ONLY: reset_to_next_rng_substream
  USE physcon,                         ONLY: angstrom,&
                                             femtoseconds,&
                                             kelvin
  USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                             section_vals_type,&
                                             section_vals_val_get
  USE qs_environment_types,            ONLY: get_qs_env,&
                                             qs_environment_type
  USE scptb_types,                     ONLY: scptb_parameter_type, &
                                             set_scptb_parameter,&
                                             get_scptb_parameter
  USE swarm_message,                   ONLY: swarm_message_type,&
                                             swarm_message_add,&
                                             swarm_message_get,&
                                             swarm_message_length,&
                                             swarm_message_haskey
#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'paramopt_worker'

 PUBLIC :: paramopt_worker_init, paramopt_worker_finalize
 PUBLIC :: paramopt_worker_execute
 PUBLIC :: paramopt_worker_type

  TYPE paramopt_worker_type
   PRIVATE
   INTEGER                                          :: id
   INTEGER                                          :: pid
   INTEGER                                          :: iw
   INTEGER                                          :: num_inp
   INTEGER                                          :: num_var
   TYPE(cp_error_type)                              :: error
   TYPE(cp_error_type)                              :: error_glob
   TYPE(cp_para_env_type), POINTER                  :: para_env
   TYPE(cp_para_env_type), POINTER                  :: para_env_glob
   CHARACTER(len=default_string_length),POINTER     :: input_files(:)
   INTEGER                                          :: subgroup
   REAL(kind=dp),POINTER                            :: x(:)
   REAL(kind=dp),POINTER                            :: Epot(:)
   REAL(kind=dp),POINTER                            :: timings(:)
   INTEGER                                          :: f_env_id
   TYPE(f_env_type), POINTER                        :: f_env
   TYPE(force_env_type), POINTER                    :: force_env
   TYPE(cp_subsys_type), POINTER                    :: subsys
   TYPE(section_vals_type), POINTER                 :: subsys_section
   TYPE(section_vals_type), POINTER                 :: print_section
   TYPE(global_environment_type), POINTER           :: globenv
   TYPE(section_vals_type), POINTER                 :: root_section
   TYPE(section_vals_type), POINTER                 :: root_section_glob
   TYPE(scptb_parameter_type), POINTER              :: scptb_para
   TYPE(section_vals_type), POINTER                 :: dft_section
   TYPE(dft_control_type), POINTER                  :: dft_control
   TYPE(qs_control_type), POINTER                   :: qs_control
   TYPE(section_vals_type), POINTER                 :: qs_section
   TYPE(qs_environment_type), POINTER               :: qs_env
   TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                       :: atomic_kind_set 
   TYPE(scptb_control_type), POINTER                :: scptb_control
   TYPE(scptb_parameter_type), POINTER              :: scptb_parameter
   INTEGER                                          :: n_atoms = -1
   REAL(KIND=dp),DIMENSION(:),ALLOCATABLE           :: pos
   INTEGER                                          :: nkind
   INTEGER, DIMENSION(0:3)                          :: norb
 END TYPE paramopt_worker_type


 CONTAINS




! *****************************************************************************
! *****************************************************************************
   SUBROUTINE paramopt_worker_init(this, para_env, root_section,&
                 input_path, worker_id, iw, error)
        TYPE(paramopt_worker_type), INTENT(INOUT):: this
        TYPE(cp_para_env_type), POINTER          :: para_env
        TYPE(section_vals_type), POINTER         :: root_section
        CHARACTER(LEN=*), INTENT(IN)             :: input_path
        INTEGER, INTENT(in)                      :: worker_id
        INTEGER, INTENT(in)                      :: iw
        TYPE(cp_error_type), INTENT(inout)       :: error
        INTEGER                                  :: ierr
        INTEGER                                  :: i,subgroup,worker_group
        INTEGER                                  :: n_groups_created
        CHARACTER(len=default_string_length)     :: str
        CHARACTER(len=default_string_length)     :: output_path 
        INTEGER, DIMENSION(:), POINTER           :: group_distribution_p 
        INTEGER, &
           DIMENSION(0:para_env%num_pe-1), &
           TARGET                                :: group_distribution 
        LOGICAL                                  :: failure    
        TYPE(cp_logger_type), POINTER            :: logger  
        INTEGER                                  :: natom_kind
        INTEGER                                  :: ikind
        TYPE(atomic_kind_type),POINTER           :: atomic_kind 
        CHARACTER(LEN=default_string_length)     :: key, atomname
        REAL(KIND=dp), DIMENSION(10, 0:3)        :: hcore
        

         
        failure = .FALSE.

        this%iw = iw
        this%para_env_glob => para_env
        this%root_section_glob => root_section
        this%error_glob = error
        this%error = error
        this%id = worker_id
        this%pid = this%para_env_glob%mepos+1
        !---------------
        this%num_inp = 35
        this%num_var = 16
        allocate(this%x(this%num_var))
        allocate(this%input_files(this%num_inp))
        allocate(this%Epot(this%num_inp))
        allocate(this%timings(this%num_inp))
        this%Epot = 0._dp
        this%timings = 0._dp
        IF (this%iw>0) WRITE(this%iw,*) "paramopt_worker_init called."

!---------------------input file(s)--------------------------------------
!     file i/o should be changed!!!!
      open(88,file='WATER27structures/mol_list.txt')
      do i=1,this%num_inp-5
          read(88,*) str
          this%input_files(i)= trim(str) // ".inp"
      enddo
      close(88)
      this%input_files(this%num_inp-4) = 'eaip_H2O.inp'
      this%input_files(this%num_inp-3) = 'eaip_H2Op.inp'
      this%input_files(this%num_inp-2) = 'eaip_H2On.inp'
      this%input_files(this%num_inp-1) = 'dipole.inp'
      this%input_files(this%num_inp-0) = 'geo_opt.inp'

      ! IF (para_env%mepos+1 .le. this%num_inp) WRITE(*,*) "input files",this%input_files(para_env%mepos+1),para_env%mepos

      ! ====== Create Comm for each Calculation ======
      group_distribution_p => group_distribution
      CALL mp_comm_split(comm=this%para_env_glob%group, sub_comm=this%subgroup, ngroups=n_groups_created, &
                        group_distribution=group_distribution_p, n_subgroups=this%num_inp)

      ! ======= Create f_env =======
      write(output_path,*) this%para_env_glob%mepos
      output_path=trim(adjustl(output_path))//"out"
      CALL create_force_env(this%f_env_id, &
                            input_path=trim(adjustl(this%input_files(this%pid))), &
                            output_path=output_path, &
                            mpi_comm=this%subgroup, &
                            ierr=ierr)
      CALL get_natom(this%f_env_id,this%n_atoms,ierr)
      allocate(this%pos(3*this%n_atoms))

      ! ======= More setup stuff =======
      CALL f_env_add_defaults(this%f_env_id, this%f_env, this%error, failure)
      IF(failure) STOP "paramopt_drive_worker: f_env_add_defaults failed"
      this%force_env => this%f_env%force_env 
      this%qs_env => this%force_env%qs_env 
      
      CALL force_env_get(this%force_env, globenv=this%globenv, subsys=this%subsys, &
                         root_section=this%root_section, &
                         para_env=this%para_env, error=this%error)

      ! getting an output unit for logging
      logger => cp_error_get_logger(this%error_glob)
      this%iw = cp_print_key_unit_nr(logger,this%root_section_glob,&
          "SWARM%PRINT%WORKER_RUN_INFO",extension=".workerLog", error=this%error)
 
      ! getting qs and scptb control and parameters 
      !CALL cp_subsys_get(this%subsys, natom=this%n_atoms, error=error)
      !allocate(this%pos(3*this%n_atoms))
      
                    !dft section
      this%dft_section =>  section_vals_get_subs_vals(this%qs_env%input, "DFT", error = error )
      CALL get_qs_env(this%qs_env, dft_control=this%dft_control, &
                      atomic_kind_set=this%atomic_kind_set,error=error)
                    ! here more things should be added 
      
                    !qs section 
      this%qs_section => section_vals_get_subs_vals(this%dft_section, "QS", error = error )
      this%qs_control => this%dft_control%qs_control
      
                    ! scptb section
      this%scptb_control => this%qs_control%scptb_control
      this%nkind = SIZE(this%atomic_kind_set)
   
      this%subsys_section => section_vals_get_subs_vals(this%force_env%force_env_section, &
                                                        "SUBSYS",error=error) 
      this%print_section => section_vals_get_subs_vals(this%subsys_section, &
                                                        "PRINT",error=this%error)
     ! IF (this%para_env_glob%ionode) THEN
!     CALL scptb_parameter_init(this%atomic_kind_set, &
!     this%scptb_control,this%print_section,this%para_env_glob,this%error)  
     !  DO ikind=1,this%nkind
       !  atomic_kind => this%atomic_kind_set(ikind)
       !  CALL get_atomic_kind(atomic_kind=atomic_kind,scptb_parameter=this%scptb_parameter)
       !  CALL scptb_parameter_from_file(this%scptb_parameter,this%scptb_control,error) 
       !  CALL scptb_parameter_broadcast(this%scptb_parameter,this%para_env_glob,this%error)
       !  CALL get_scptb_parameter(this%scptb_parameter,norb=this%norb,hcore=hcore)
       !  print *,hcore(1:2,0:1)
     !  END DO
     ! END IF
     !    CALL scptb_parameter_broadcast(this%scptb_parameter,this%para_env_glob,this%error)
                  ! calc initial energy
      CALL cp_get_pos(this%f_env_id, this%pos, 3*this%n_atoms, ierr)
      CALL cp_calc_energy(this%f_env_id, this%pos,3*this%n_atoms,this%Epot(this%pid),ierr) 
      print *,'initial energy',this%Epot(this%pid),this%pid,this%iw
      
      !print *,this%scptb_parameter%atomname
      !--------------destroy f env------------
      !CALL f_env_rm_defaults(this%f_env, this%error)
      !CALL destroy_force_env(this%f_env_id, ierr)

  END SUBROUTINE paramopt_worker_init



! *****************************************************************************
! *****************************************************************************
   SUBROUTINE paramopt_worker_finalize(this)
    TYPE(paramopt_worker_type), INTENT(INOUT)  :: this
    INTEGER                                    :: ierr

     IF (this%iw>0) WRITE(this%iw,*) "paramopt_worker_finalize called"

!     WRITE(*,*) "paramopt_worker_finalize called. worker_id=", this%id, &
!      " rank=",this%para_env%mepos 
     deallocate(this%pos)
     CALL f_env_rm_defaults(this%f_env, this%error)
     CALL destroy_force_env(this%f_env_id, ierr)
     IF(ierr /= 0) STOP "paramopt_worker_finalize: destroy_force_env failed"
   END SUBROUTINE paramopt_worker_finalize


! *****************************************************************************
! *****************************************************************************
   SUBROUTINE paramopt_worker_execute(this, cmd, report)
       TYPE(paramopt_worker_type), INTENT(INOUT)  :: this
       TYPE(swarm_message_type), INTENT(IN)       :: cmd
       TYPE(swarm_message_type), INTENT(INOUT)    :: report
       
       CHARACTER(len=default_string_length)     :: command

       
       TYPE(qs_control_type), POINTER           :: qs_control
       TYPE(section_vals_type), POINTER         :: qs_section

       INTEGER                                  :: i,ierr
       TYPE(cp_error_type)                      :: error
       LOGICAL                                  :: failure
       INTEGER                                  :: ikind
       TYPE(atomic_kind_type),POINTER           :: atomic_kind
       CHARACTER(LEN=default_string_length)     :: key, atomname, iname
       !------------------timing----------------------------------
       INTEGER                                  :: t1,t2,clock_rate,clock_max
       !-----------------parameter---------------------------------
       INTEGER                                :: z          !atomic number
       REAL(KIND=dp)                          :: zeff       !effective core charge
       INTEGER                                :: natorb     !total number of orbitals
       INTEGER                                :: lmaxorb    !max angular momentum orbitals
       INTEGER                                :: lmaxscp    !max angular momentum scp density
       INTEGER, DIMENSION(0:3)                :: norb       !number of orbitals per l QN
       INTEGER, DIMENSION(10,0:3)             :: nqm        !principal quantum numbers
       REAL(KIND=dp), DIMENSION(10,0:3)       :: zeta       !orbital exponents
       REAL(KIND=dp), DIMENSION(10,0:3)       :: hcore      !core potential energy
       REAL(KIND=dp), DIMENSION(10,0:3)       :: occupation !free atom occupation
       REAL(KIND=dp)                          :: energy     !free atom energy
       REAL(KIND=dp), DIMENSION(3)            :: crep       !core repulsion parameters
       REAL(KIND=dp), DIMENSION(1:3)          :: pol        !polarization for SCP
       REAL(KIND=dp)                          :: ag         !width of SCP charge
       REAL(KIND=dp)                          :: rcpair     !cutoff radius for core pair potentials
       !----------------------------------------------------------
       CHARACTER(len=default_string_length)     :: output_path 
       TYPE(cp_logger_type), POINTER            :: logger    
       failure = .FALSE.

       this%Epot = 0._dp
       this%timings = 0._dp

       CALL swarm_message_get(cmd, "command", command)
       IF (this%para_env_glob%ionode) WRITE(*,*) "paramopt_worker_execute called with command ",command
       If (command .eq. "nop") call swarm_message_get(cmd, "newx",this%x)  
        
       !IF (this%para_env_glob%ionode) WRITE(*,*) 'this is new x',this%x 
       !WRITE (*,*) output_path,this%para_env%mepos,this%para_env%group, &
       !           trim(adjustl(this%input_files(this%para_env%mepos+1))),f_env_id,this%Epot(id),i
    
    !==================================================================
       
       DO ikind=1,this%nkind
         atomic_kind => this%atomic_kind_set(ikind)
         CALL get_atomic_kind(atomic_kind=atomic_kind, &
                               name=iname, scptb_parameter=this%scptb_parameter)
         !----------update parameters-------------------------------------
         CALL get_scptb_parameter(this%scptb_parameter,norb=this%norb,hcore=hcore, &
                                  energy=energy, ag=ag, zeta=zeta, crep=crep)
         !IF (this%para_env_glob%ionode) print *,'first get',hcore(1:2,0:1)
         !-----------map x to parameter----------------------------------- 
         IF (iname .eq. 'H') THEN
           energy = this%x(1)
           ag = this%x(2)
           zeta(1,0) = this%x(3)
           hcore(1,0) = this%x(4)
           crep(1) = this%x(5)
           crep(2) = this%x(6)
           crep(3) = this%x(7)
         ELSE IF (iname .eq. 'O') THEN
           energy = this%x(8)
           ag = this%x(9)
           zeta(1,0) = this%x(10)
           hcore(1,0) = this%x(11)
           zeta(1,1) = this%x(12)
           hcore(1,1) = this%x(13)
           crep(1) = this%x(14)
           crep(2) = this%x(15)
           crep(3) = this%x(16)
         ELSE
            print *,'failure'
         ENDIF

         CALL set_scptb_parameter(this%scptb_parameter,hcore=hcore, &
                                  energy=energy, ag=ag, zeta=zeta, crep=crep)
         CALL get_scptb_parameter(this%scptb_parameter,norb=this%norb,hcore=hcore)
         !IF (this%para_env_glob%ionode) print *,'second get',hcore(1:2,0:1)
       END DO

    !===================================================================

       ! ======= Energy Calculations and timings =======
       call system_clock ( t1, clock_rate, clock_max )
       CALL calc_energy_force(this%f_env_id, .false. , ierr)
       CALL get_energy(this%f_env_id,this%Epot(this%pid),ierr)
       call system_clock ( t2, clock_rate, clock_max )  
       this%timings(this%pid) =  real ( t2 - t1 ) / real ( clock_rate )
    
        
       ! should be mp_reduce...mpi_reduce or something more efficient?
       CALL mp_sum(this%Epot,this%para_env_glob%group)
       CALL mp_sum(this%timings,this%para_env_glob%group)
              ! assemble report
       !CALL swarm_message_add(report, "iframe", worker%iframe) 
       CALL swarm_message_add(report, "Epot", this%Epot)
       CALL swarm_message_add(report, "timings", this%timings)
!       
   END SUBROUTINE paramopt_worker_execute


END MODULE paramopt_worker

