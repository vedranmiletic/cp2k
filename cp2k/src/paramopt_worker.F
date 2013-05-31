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
  USE cp_output_handling,              ONLY: cp_print_key_unit_nr
  USE cp_para_types,                   ONLY: cp_para_env_type
  USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                             cp_subsys_type,&
                                             pack_subsys_particles
  USE f77_interface,                   ONLY: create_force_env,&
                                             destroy_force_env,&
                                             f_env_add_defaults,&
                                             f_env_rm_defaults,&
                                             f_env_type
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
  USE md_run,                          ONLY: qs_mol_dyn
  USE mdctrl_types,                    ONLY: mdctrl_type
  USE parallel_rng_types,              ONLY: reset_to_next_rng_substream
  USE physcon,                         ONLY: angstrom,&
                                             femtoseconds,&
                                             kelvin
  USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                             section_vals_type,&
                                             section_vals_val_get
  USE swarm_message,                   ONLY: swarm_message_type,&
                                             swarm_message_add,&
                                             swarm_message_get,&
                                             swarm_message_length
#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'paramopt_worker'

 PUBLIC :: paramopt_worker_init, paramopt_worker_finalize
 PUBLIC :: paramopt_worker_execute
 PUBLIC :: paramopt_worker_type

  TYPE paramopt_worker_type
   PRIVATE
   INTEGER                                  :: id
   INTEGER                                  :: iw
   TYPE(cp_error_type)                      :: error
   INTEGER                                  :: f_env_id
   TYPE(f_env_type), POINTER                :: f_env
   TYPE(force_env_type), POINTER            :: force_env
   TYPE(cp_subsys_type), POINTER            :: subsys
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

        this%iw = iw

        IF (this%iw>0) WRITE(this%iw,*) "paramopt_worker_init called"

     !   CHARACTER(len=default_path_length)       :: output_path
     !   CHARACTER(len=default_string_length)     :: new_project_name, &
     !                                               project_name, worker_name
     !   INTEGER                                  :: i
     !   LOGICAL                                  :: failure
     !   TYPE(section_vals_type), POINTER         :: paramopt_section
     !   REAL(kind=dp)                            :: dist_in_angstrom 
     !   TYPE(cp_logger_type), POINTER            :: logger
     !   failure = .FALSE.
     !
     !
     !   this%error = error
     !   this%root_section => root_section
     !   this%iframe = 0
     !   this%id = worker_id
     !   this%iw = iw
     !
     !  !TODO: project name can also be obtained/manipulated in globenv
     !  ! change the PROJECT_NAME to avoid clashes during backup of RESTART-files
     !  !output_path = "__STD_OUT__"
     !  IF(worker_id > 999) STOP "worker_create_f_env: Did not expect so many workers."
     !  CALL section_vals_val_get(root_section, "GLOBAL%PROJECT_NAME", c_val=project_name,error=error)
     !  WRITE(worker_name,"('WORKER',I3.3)"), worker_id
     !  new_project_name = TRIM(project_name)//"-"//worker_name
     !  CALL section_vals_val_set(root_section, "GLOBAL%PROJECT_NAME", c_val=new_project_name,error=error)
     !  output_path = TRIM(new_project_name)//".out"
     !
     ! ! ======= Create f_env =======
     ! CALL create_force_env(this%f_env_id, &
     !                       input_path=input_path, &
     !                       input=root_section, &
     !                       output_path=output_path, &
     !                       mpi_comm=para_env%group)
     !
     !
     ! ! ======= More setup stuff =======
     ! CALL f_env_add_defaults(this%f_env_id, this%f_env, this%error, failure)
     ! IF(failure) STOP "paramopt_drive_worker: f_env_add_defaults failed"
     ! worker%force_env => worker%f_env%force_env
     ! CALL force_env_get(this%force_env, globenv=this%globenv, subsys=this%subsys, error=this%error)
     !
     !
     ! ! getting an output unit for logging
     ! logger => cp_error_get_logger(this%error)
     ! worker%iw = cp_print_key_unit_nr(logger,root_section,&
     !     "SWARM%PRINT%WORKER_RUN_INFO",extension=".workerLog", error=this%error)
     !
  END SUBROUTINE paramopt_worker_init



! *****************************************************************************
! *****************************************************************************
   SUBROUTINE paramopt_worker_finalize(this)
    TYPE(paramopt_worker_type), INTENT(INOUT)  :: this
    INTEGER                                    :: ierr

     IF (this%iw>0) WRITE(this%iw,*) "paramopt_worker_finalize called"
     !CALL f_env_rm_defaults(this%f_env, worker%error)
     !CALL destroy_force_env(this%f_env_id, ierr)
     !IF(ierr /= 0) STOP "paramopt_worker_finalize: destroy_force_env failed"
   END SUBROUTINE paramopt_worker_finalize


! *****************************************************************************
! *****************************************************************************
   SUBROUTINE paramopt_worker_execute(this, cmd, report)
       TYPE(paramopt_worker_type), INTENT(INOUT)  :: this
       TYPE(swarm_message_type), INTENT(IN)       :: cmd
       TYPE(swarm_message_type), INTENT(OUT)      :: report

       CHARACTER(len=default_string_length)     :: command

       CALL swarm_message_get(cmd, "command", command)
       IF (this%iw>0) WRITE(this%iw,*) "paramopt_worker_execute called with command ",command

       !INTEGER                                  :: n_atoms
       !REAL(KIND=dp)                            :: temperature, Epot
       !REAL(KIND=dp), DIMENSION(:), POINTER     :: positions
       !CHARACTER(len=default_string_length)     :: command
       !
       !CALL swarm_message_get(cmd, "command", command)
       !IF(TRIM(command) == "md_and_gopt") THEN
       !    CALL swarm_message_get(cmd, "temperature", temperature)
       !    CALL swarm_message_get(cmd, "iframe", worker%iframe)
       !    CALL run_mdgopt(worker, temperature)
       !ELSE
       !   STOP "Worker: received unkown command"
       !END IF
       !
       !! assemble report
       !CALL swarm_message_add(report, "worker_id", worker%id)
       !CALL swarm_message_add(report, "iframe", worker%iframe)
       !CALL force_env_get(worker%force_env, potential_energy=Epot, error=worker%error)
       !CALL swarm_message_add(report, "Epot", Epot)
       !IF (worker%iw>0) WRITE (worker%iw,'(A,25X,E20.10)')' PARAMOPT| Potential Energy [Hartree]',Epot
       !CALL cp_subsys_get(worker%subsys, natom=n_atoms, error=worker%error)
       !ALLOCATE(positions(3*n_atoms))
       !CALL pack_subsys_particles(worker%subsys, r=positions, error=worker%error)
       !CALL swarm_message_add(report, "positions", positions)
   END SUBROUTINE paramopt_worker_execute


END MODULE paramopt_worker

