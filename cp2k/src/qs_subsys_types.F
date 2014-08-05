!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2014  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
! *****************************************************************************
MODULE qs_subsys_types

  USE atomic_kind_list_types,          ONLY: atomic_kind_list_release,&
                                             atomic_kind_list_retain,&
                                             atomic_kind_list_type
  USE cell_methods,                    ONLY: cell_type,&
                                             real_to_scaled,&
                                             scaled_to_real
  USE colvar_types,                    ONLY: colvar_p_release,&
                                             colvar_p_type
  USE cp_para_env,                     ONLY: cp_para_env_release,&
                                             cp_para_env_retain
  USE cp_para_types,                   ONLY: cp_para_env_type
  USE distribution_1d_types,           ONLY: distribution_1d_release,&
                                             distribution_1d_retain,&
                                             distribution_1d_type
  USE f77_blas
  USE kinds,                           ONLY: dp
  USE mol_kind_new_list_types,         ONLY: mol_kind_new_list_release,&
                                             mol_kind_new_list_retain,&
                                             mol_kind_new_list_type
  USE mol_new_list_types,              ONLY: mol_new_list_release,&
                                             mol_new_list_retain,&
                                             mol_new_list_type
  USE molecule_types_new,              ONLY: deallocate_global_constraint,&
                                             global_constraint_type
  USE multipole_types,                 ONLY: multipole_type,&
                                             release_multipole_type,&
                                             retain_multipole_type
  USE particle_list_types,             ONLY: particle_list_release,&
                                             particle_list_retain,&
                                             particle_list_type
  USE cp_subsys_types,                 ONLY: cp_subsys_type,&
                                             cp_subsys_get,&
                                             cp_subsys_retain,&
                                             cp_subsys_release

#include "./common/cp_common_uses.f90"

  IMPLICIT NONE
  PRIVATE

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_subsys_types'

  PUBLIC :: qs_subsys_type,&
            qs_subsys_p_type

  PUBLIC :: qs_subsys_create,&
            qs_subsys_retain,&
            qs_subsys_release,&
            qs_subsys_get,&
            qs_subsys_set

! *****************************************************************************
! *****************************************************************************
  TYPE qs_subsys_type
     PRIVATE
     INTEGER                              :: ref_count = 1
     TYPE (cp_subsys_type), POINTER       :: cp_subsys => Null()
  END TYPE qs_subsys_type

! *****************************************************************************
! *****************************************************************************
  TYPE qs_subsys_p_type
     TYPE(qs_subsys_type), POINTER :: subsys
  END TYPE qs_subsys_p_type

CONTAINS

! *****************************************************************************
! *****************************************************************************
  SUBROUTINE qs_subsys_create(this, cp_subsys, error)
    TYPE(qs_subsys_type), POINTER            :: this
    TYPE(cp_subsys_type), POINTER            :: cp_subsys
    TYPE(cp_error_type), INTENT(inout)       :: error

    CHARACTER(len=*), PARAMETER :: routineN = 'qs_subsys_create', &
      routineP = moduleN//':'//routineN

    CPPreconditionNoFail(ASSOCIATED(cp_subsys),cp_failure_level,routineP,error)
    ALLOCATE(this)
    this%cp_subsys => cp_subsys
    CALL cp_subsys_retain(cp_subsys, error)

  END SUBROUTINE qs_subsys_create

! *****************************************************************************
!> \brief retains a subsys (see doc/ReferenceCounting.html)
!> \param subsys the subsys to retain
!> \param error variable to control error logging, stopping,...
!>        see module cp_error_handling
!> \par History
!>      07.2003 created [fawzi]
!> \author Fawzi Mohamed
! *****************************************************************************
  SUBROUTINE qs_subsys_retain(this, error)
    TYPE(qs_subsys_type), POINTER            :: this
    TYPE(cp_error_type), INTENT(inout)       :: error

    CHARACTER(len=*), PARAMETER :: routineN = 'qs_subsys_retain', &
      routineP = moduleN//':'//routineN

    CPPreconditionNoFail(ASSOCIATED(this),cp_failure_level,routineP,error)
    CPPreconditionNoFail(this%ref_count>0,cp_failure_level,routineP,error)

    this%ref_count = this%ref_count+1
  END SUBROUTINE qs_subsys_retain

! *****************************************************************************
!> \brief releases a subsys (see doc/ReferenceCounting.html)
!> \param subsys the subsys to release
!> \param error variable to control error logging, stopping,...
!>        see module cp_error_handling
!> \par History
!>      07.2003 created [fawzi]
!> \author Fawzi Mohamed
! *****************************************************************************
  SUBROUTINE qs_subsys_release(this, error)
    TYPE(qs_subsys_type), POINTER            :: this
    TYPE(cp_error_type), INTENT(inout)       :: error

    CHARACTER(len=*), PARAMETER :: routineN = 'qs_subsys_release', &
      routineP = moduleN//':'//routineN

    IF (ASSOCIATED(this)) THEN
       CPPreconditionNoFail(this%ref_count>0,cp_failure_level,routineP,error)
       this%ref_count = this%ref_count-1
       IF (this%ref_count==0) THEN
          CALL cp_subsys_release(this%cp_subsys, error)
          DEALLOCATE(this)
       END IF
    END IF
  END SUBROUTINE qs_subsys_release

! *****************************************************************************
! *****************************************************************************
  SUBROUTINE qs_subsys_set(this, error)
  !, atomic_kinds, particles, local_particles,&
  !     molecules_new, molecule_kinds_new, local_molecules_new, para_env,&
  !     colvar_p, shell_particles, core_particles, gci, multipoles, error)
    TYPE(qs_subsys_type), POINTER            :: this
    !TYPE(atomic_kind_list_type), OPTIONAL, &
    !  POINTER                                :: atomic_kinds
    !TYPE(particle_list_type), OPTIONAL, &
    !  POINTER                                :: particles
    !TYPE(distribution_1d_type), OPTIONAL, &
    !  POINTER                                :: local_particles
    !TYPE(mol_new_list_type), OPTIONAL, &
    !  POINTER                                :: molecules_new
    !TYPE(mol_kind_new_list_type), OPTIONAL, &
    !  POINTER                                :: molecule_kinds_new
    !TYPE(distribution_1d_type), OPTIONAL, &
    !  POINTER                                :: local_molecules_new
    !TYPE(cp_para_env_type), OPTIONAL, &
    !  POINTER                                :: para_env
    !TYPE(colvar_p_type), DIMENSION(:), &
    !  OPTIONAL, POINTER                      :: colvar_p
    !TYPE(particle_list_type), OPTIONAL, &
    !  POINTER                                :: shell_particles, &
    !                                            core_particles
    !TYPE(global_constraint_type), OPTIONAL, &
    !  POINTER                                :: gci
    !TYPE(multipole_type), OPTIONAL, POINTER  :: multipoles
    TYPE(cp_error_type), INTENT(inout)       :: error

    CHARACTER(len=*), PARAMETER :: routineN = 'qs_subsys_set', &
      routineP = moduleN//':'//routineN


    CPPreconditionNoFail(ASSOCIATED(this),cp_failure_level,routineP,error)
    CPPreconditionNoFail(this%ref_count>0,cp_failure_level,routineP,error)
!    IF (.NOT. failure) THEN
!       IF (PRESENT(multipoles)) THEN
!          CALL retain_multipole_type(multipoles, error)
!          CALL release_multipole_type(subsys%multipoles, error)
!          subsys%multipoles => multipoles
!       END IF
!       IF (PRESENT(atomic_kinds)) THEN
!          CALL atomic_kind_list_retain(atomic_kinds,error=error)
!          CALL atomic_kind_list_release(subsys%atomic_kinds, error=error)
!          subsys%atomic_kinds => atomic_kinds
!       END IF
!       IF (PRESENT(particles)) THEN
!          CALL particle_list_retain(particles, error=error)
!          CALL particle_list_release(subsys%particles, error=error)
!          subsys%particles => particles
!       END IF
!       IF (PRESENT(local_particles)) THEN
!          CALL distribution_1d_retain(local_particles,error=error)
!          CALL distribution_1d_release(subsys%local_particles,error=error)
!          subsys%local_particles => local_particles
!       END IF
!       IF (PRESENT(local_molecules_new)) THEN
!          CALL distribution_1d_retain(local_molecules_new,error=error)
!          CALL distribution_1d_release(subsys%local_molecules_new,error=error)
!          subsys%local_molecules_new => local_molecules_new
!       END IF
!       IF (PRESENT(molecule_kinds_new)) THEN
!          CALL mol_kind_new_list_retain(molecule_kinds_new, error=error)
!          CALL mol_kind_new_list_release(subsys%molecule_kinds_new, error=error)
!          subsys%molecule_kinds_new => molecule_kinds_new
!       END IF
!       IF (PRESENT(molecules_new)) THEN
!          CALL mol_new_list_retain(molecules_new, error=error)
!          CALL mol_new_list_release(subsys%molecules_new, error=error)
!          subsys%molecules_new => molecules_new
!       END IF
!       IF (PRESENT(para_env)) THEN
!          CALL cp_para_env_retain(para_env, error=error)
!          CALL cp_para_env_release(subsys%para_env, error=error)
!          subsys%para_env => para_env
!       END IF
!       IF (PRESENT(colvar_p)) THEN
!          CPPreconditionNoFail(.NOT.ASSOCIATED(subsys%colvar_p),cp_failure_level,routineP,error)
!          subsys%colvar_p=>colvar_p
!       ENDIF
!       IF (PRESENT(shell_particles)) THEN
!          IF(ASSOCIATED(shell_particles)) THEN
!             CALL particle_list_retain(shell_particles, error=error)
!             CALL particle_list_release(subsys%shell_particles, error=error)
!             subsys%shell_particles => shell_particles
!          END IF
!       END IF
!       IF (PRESENT(core_particles)) THEN
!          IF(ASSOCIATED(core_particles)) THEN
!             CALL particle_list_retain(core_particles, error=error)
!             CALL particle_list_release(subsys%core_particles, error=error)
!             subsys%core_particles => core_particles
!          END IF
!       END IF
!       IF (PRESENT(gci)) THEN
!          CPPreconditionNoFail(.NOT.ASSOCIATED(subsys%gci),cp_failure_level,routineP,error)
!          subsys%gci => gci
!       ENDIF
!    END IF
  END SUBROUTINE qs_subsys_set

! *****************************************************************************
! *****************************************************************************
  SUBROUTINE qs_subsys_get(this, cp_subsys, &
                           atomic_kinds, particles,&
                           local_particles, molecules_new, molecule_kinds_new,&
                           local_molecules_new, para_env, colvar_p,&
                           shell_particles, core_particles, gci, multipoles,&
                           natom, nparticle, ncore, nshell, error)
    TYPE(qs_subsys_type), POINTER            :: this
    TYPE(cp_subsys_type), POINTER, OPTIONAL   :: cp_subsys
    !INTEGER, INTENT(out), OPTIONAL           :: id_nr, ref_count
    TYPE(atomic_kind_list_type), OPTIONAL, &
      POINTER                                :: atomic_kinds
    TYPE(particle_list_type), OPTIONAL, &
      POINTER                                :: particles
    TYPE(distribution_1d_type), OPTIONAL, &
      POINTER                                :: local_particles
    TYPE(mol_new_list_type), OPTIONAL, &
      POINTER                                :: molecules_new
    TYPE(mol_kind_new_list_type), OPTIONAL, &
      POINTER                                :: molecule_kinds_new
    TYPE(distribution_1d_type), OPTIONAL, &
      POINTER                                :: local_molecules_new
    TYPE(cp_para_env_type), OPTIONAL, &
      POINTER                                :: para_env
    TYPE(colvar_p_type), DIMENSION(:), &
      OPTIONAL, POINTER                      :: colvar_p
    TYPE(particle_list_type), OPTIONAL, &
      POINTER                                :: shell_particles, &
                                                core_particles
    TYPE(global_constraint_type), OPTIONAL, &
      POINTER                                :: gci
    TYPE(multipole_type), OPTIONAL, POINTER  :: multipoles
    INTEGER, INTENT(out), OPTIONAL           :: natom, nparticle, ncore, &
                                                nshell
    TYPE(cp_error_type), INTENT(inout)       :: error
    !
    CHARACTER(LEN=*), PARAMETER :: routineN = 'qs_subsys_get', &
      routineP = moduleN//':'//routineN

      IF(PRESENT(cp_subsys)) cp_subsys => this%cp_subsys

      ! handle fields inherited from cp_subsys (would be much easier with Fortran 2003)
      !IF(PRESENT(id_nr)) CALL cp_subsys_get(this%cp_subsysthis%cp_subsys, id_nr=id_nr, error=error)
      !IF(PRESENT(ref_count)) CALL cp_subsys_get(this%cp_subsys, ref_count=ref_count, error=error)
      IF(PRESENT(atomic_kinds)) CALL cp_subsys_get(this%cp_subsys, atomic_kinds=atomic_kinds, error=error)
      IF(PRESENT(particles)) CALL cp_subsys_get(this%cp_subsys, particles=particles, error=error)
      IF(PRESENT(local_particles)) CALL cp_subsys_get(this%cp_subsys, local_particles=local_particles, error=error)
      IF(PRESENT(molecules_new)) CALL cp_subsys_get(this%cp_subsys, molecules_new=molecules_new, error=error)
      IF(PRESENT(molecule_kinds_new)) CALL cp_subsys_get(this%cp_subsys, molecule_kinds_new=molecule_kinds_new, error=error)
      IF(PRESENT(local_molecules_new)) CALL cp_subsys_get(this%cp_subsys, local_molecules_new=local_molecules_new, error=error)
      IF(PRESENT(para_env)) CALL cp_subsys_get(this%cp_subsys, para_env=para_env, error=error)
      IF(PRESENT(colvar_p)) CALL cp_subsys_get(this%cp_subsys, colvar_p=colvar_p, error=error)
      IF(PRESENT(shell_particles)) CALL cp_subsys_get(this%cp_subsys, shell_particles=shell_particles, error=error)
      IF(PRESENT(core_particles)) CALL cp_subsys_get(this%cp_subsys, core_particles=core_particles, error=error)
      IF(PRESENT(gci)) CALL cp_subsys_get(this%cp_subsys, gci=gci, error=error)
      IF(PRESENT(multipoles)) CALL cp_subsys_get(this%cp_subsys, multipoles=multipoles, error=error)
      IF(PRESENT(natom)) CALL cp_subsys_get(this%cp_subsys, natom=natom, error=error)
      IF(PRESENT(nparticle)) CALL cp_subsys_get(this%cp_subsys, nparticle=nparticle, error=error)
      IF(PRESENT(ncore)) CALL cp_subsys_get(this%cp_subsys, ncore=ncore, error=error)
      IF(PRESENT(nshell)) CALL cp_subsys_get(this%cp_subsys, nshell=nshell, error=error)



    !INTEGER                                  :: n_atom, n_core, n_shell
    !LOGICAL                                  :: failure
    !
    !failure = .FALSE.
    !n_atom = 0
    !n_core = 0
    !n_shell = 0

    CPPreconditionNoFail(ASSOCIATED(this),cp_failure_level,routineP,error)
    CPPreconditionNoFail(this%ref_count>0,cp_failure_level,routineP,error)

    !IF (.NOT.failure) THEN
    !   IF (PRESENT(id_nr)) id_nr = subsys%id_nr
    !   IF (PRESENT(ref_count)) ref_count = subsys%ref_count
    !   IF (PRESENT(atomic_kinds)) atomic_kinds => subsys%atomic_kinds
    !   IF (PRESENT(particles)) particles => subsys%particles
    !   IF (PRESENT(local_particles)) local_particles => subsys%local_particles
    !   IF (PRESENT(molecules_new)) molecules_new => subsys%molecules_new
    !   IF (PRESENT(molecule_kinds_new)) molecule_kinds_new => subsys%molecule_kinds_new
    !   IF (PRESENT(local_molecules_new)) local_molecules_new => subsys%local_molecules_new
    !   IF (PRESENT(para_env)) para_env => subsys%para_env
    !   IF (PRESENT(colvar_p)) colvar_p => subsys%colvar_p
    !   IF (PRESENT(shell_particles)) shell_particles => subsys%shell_particles
    !   IF (PRESENT(core_particles)) core_particles => subsys%core_particles
    !   IF (PRESENT(gci)) gci => subsys%gci
    !   IF (PRESENT(multipoles)) multipoles => subsys%multipoles
    !   IF (PRESENT(natom).OR.PRESENT(nparticle).OR.PRESENT(nshell)) THEN
    !      ! An atomic particle set should be present in each subsystem at the moment
    !      CPPreconditionNoFail(ASSOCIATED(subsys%particles),cp_failure_level,routineP,error)
    !      n_atom = subsys%particles%n_els
    !      ! Check if we have other kinds of particles in this subsystem
    !      IF (ASSOCIATED(subsys%shell_particles)) THEN
    !         n_shell = subsys%shell_particles%n_els
    !         CPPreconditionNoFail(ASSOCIATED(subsys%core_particles),cp_failure_level,routineP,error)
    !         n_core = subsys%core_particles%n_els
    !         ! The same number of shell and core particles is assumed
    !         CPPreconditionNoFail((n_core == n_shell),cp_failure_level,routineP,error)
    !      ELSE IF (ASSOCIATED(subsys%core_particles)) THEN
    !         ! This case should not occur at the moment
    !         CPPreconditionNoFail(ASSOCIATED(subsys%shell_particles),cp_failure_level,routineP,error)
    !      ELSE
    !         n_core = 0
    !         n_shell = 0
    !      END IF
    !      IF (PRESENT(natom)) natom = n_atom
    !      IF (PRESENT(nparticle)) nparticle = n_atom + n_shell
    !      IF (PRESENT(ncore)) ncore = n_core
    !      IF (PRESENT(nshell)) nshell = n_shell
    !   END IF
    !END IF

  END SUBROUTINE qs_subsys_get


END MODULE qs_subsys_types
