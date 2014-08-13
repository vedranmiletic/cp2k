!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2014  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief   Define the atomic kind types and their sub types
!> \author  Matthias Krack (MK)
!> \date    02.01.2002
!> \version 1.0
!>
!> <b>Modification history:</b>
!> - 01.2002 creation [MK]
!> - 04.2002 added pao [fawzi]
!> - 09.2002 adapted for POL/KG use [GT]
!> - 02.2004 flexible normalization of basis sets [jgh]
!> - 03.2004 attach/detach routines [jgh]
!> - 10.2004 removed pao [fawzi]
! *****************************************************************************
MODULE atomic_kind_types

  USE cp_control_types,                ONLY: dft_control_type
  USE cp_para_types,                   ONLY: cp_para_env_type
  USE damping_dipole_types,            ONLY: damping_p_type
  USE external_potential_types,        ONLY: all_potential_type,&
                                             fist_potential_type,&
                                             gth_potential_type,&
                                             local_potential_type
  USE f77_blas
  USE input_section_types,             ONLY: section_vals_type
  USE kinds,                           ONLY: default_string_length,&
                                             dp
  USE shell_potential_types,           ONLY: shell_kind_type
#include "./common/cp_common_uses.f90"

  IMPLICIT NONE

  PRIVATE

  ! Global parameters (only in this module)

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'atomic_kind_types'


!> Provides all information about an atomic kind
! *****************************************************************************
  TYPE atomic_kind_type
     TYPE(all_potential_type), POINTER      :: all_potential
     TYPE(local_potential_type), POINTER    :: tnadd_potential
     TYPE(fist_potential_type), POINTER     :: fist_potential
     TYPE(gth_potential_type), POINTER      :: gth_potential
     CHARACTER(LEN=default_string_length)   :: name
     CHARACTER(LEN=2)                       :: element_symbol
     REAL(KIND = dp)                        :: mass
     INTEGER                                :: kind_number,&
                                               natom
     INTEGER, DIMENSION(:), POINTER         :: atom_list
     LOGICAL                                :: shell_active
     TYPE(shell_kind_type), POINTER         :: shell
     TYPE(damping_p_type), POINTER          :: damping
  END TYPE atomic_kind_type

!> Provides a vector of pointers of type atomic_kind_type
! *****************************************************************************
  TYPE atomic_kind_p_type
     TYPE(atomic_kind_type),DIMENSION(:),&
          POINTER                             :: atomic_kind_set
  END TYPE atomic_kind_p_type

  ! Public subroutines

  PUBLIC :: allocate_atomic_kind_set,&
            check_atomic_kind_set,&
            deallocate_atomic_kind_set,&
            get_atomic_kind,&
            get_atomic_kind_set,&
            init_atomic_kind_set,&
            read_atomic_kind_set,&
            set_atomic_kind,&
            write_atomic_kind_set,&
            is_hydrogen

  ! Public data types
  PUBLIC :: atomic_kind_type

CONTAINS

! *****************************************************************************
!> \brief   Constructor routine for a set of atomic kinds
!> \param atomic_kind_set ...
!> \param nkind ...
!> \param error ...
!> \date    02.01.2002
!> \author  Matthias Krack (MK)
!> \version 2.0
! *****************************************************************************
  SUBROUTINE allocate_atomic_kind_set(atomic_kind_set,nkind,error)

    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    INTEGER, INTENT(IN)                      :: nkind
    TYPE(cp_error_type), INTENT(inout)       :: error

     STOP "atomic_kind_types"
  END SUBROUTINE allocate_atomic_kind_set

! *****************************************************************************
!> \brief   Destructor routine for a set of atomic kinds
!> \param atomic_kind_set ...
!> \param error ...
!> \date    02.01.2002
!> \author  Matthias Krack (MK)
!> \version 2.0
! *****************************************************************************
  SUBROUTINE deallocate_atomic_kind_set(atomic_kind_set,error)

    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "atomic_kind_types"
  END SUBROUTINE deallocate_atomic_kind_set

! *****************************************************************************
!> \brief Get attributes of an atomic kind.
! *****************************************************************************
  SUBROUTINE get_atomic_kind(atomic_kind, all_potential, tnadd_potential, fist_potential,&
       gth_potential,  &
       element_symbol, name, mass, kind_number, natom, atom_list, &
       alpha_core_charge, ccore_charge, core_charge, &
       core_charge_radius, rcov, rvdw, z, zeff, qeff, apol, cpol, mm_radius, &
       shell, shell_active, damping)

    TYPE(atomic_kind_type)                   :: atomic_kind
    TYPE(all_potential_type), OPTIONAL, &
      POINTER                                :: all_potential
    TYPE(local_potential_type), OPTIONAL, &
      POINTER                                :: tnadd_potential
    TYPE(fist_potential_type), OPTIONAL, &
      POINTER                                :: fist_potential
    TYPE(gth_potential_type), OPTIONAL, &
      POINTER                                :: gth_potential
    CHARACTER(LEN=2), INTENT(OUT), OPTIONAL  :: element_symbol
    CHARACTER(LEN=default_string_length), &
      INTENT(OUT), OPTIONAL                  :: name
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: mass
    INTEGER, INTENT(OUT), OPTIONAL           :: kind_number, natom
    INTEGER, DIMENSION(:), OPTIONAL, POINTER :: atom_list
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: alpha_core_charge, &
                                                ccore_charge, core_charge, &
                                                core_charge_radius
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: rcov, rvdw
    INTEGER, INTENT(OUT), OPTIONAL           :: z
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: zeff, qeff, apol, cpol, &
                                                mm_radius
    TYPE(shell_kind_type), OPTIONAL, POINTER :: shell
    LOGICAL, INTENT(OUT), OPTIONAL           :: shell_active
    TYPE(damping_p_type), OPTIONAL, POINTER  :: damping

    STOP "atomic_kind_types"
  END SUBROUTINE get_atomic_kind

! *****************************************************************************
!> \brief Get attributes of an atomic kind set.
! *****************************************************************************
  SUBROUTINE get_atomic_kind_set(atomic_kind_set,&
       atom_of_kind, kind_of, natom_of_kind,&
       maxatom,&
       maxlppl,maxlppnl,maxppnl, maxpol,&
       natom,nelectron,&
       nset,nshell,&
       all_potential_present,tnadd_potential_present,fist_potential_present,&
       gth_potential_present,&
       shell_present, shell_adiabatic,&
       shell_check_distance,&
       damping_present)

    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    INTEGER, INTENT(OUT), OPTIONAL :: maxatom, &
      maxlppl, maxlppnl, maxpol, maxppnl, &
      natom, nelectron, nset, nshell
    LOGICAL, INTENT(OUT), OPTIONAL :: all_potential_present, &
      tnadd_potential_present, fist_potential_present, gth_potential_present, &
      shell_present, shell_adiabatic, shell_check_distance, damping_present
    INTEGER, DIMENSION(:), INTENT(OUT), &
      OPTIONAL                               :: atom_of_kind, kind_of, natom_of_kind

    STOP "atomic_kind_types"
  END SUBROUTINE get_atomic_kind_set

! *****************************************************************************
!> \brief Initialise an atomic kind data set.
!> \param atomic_kind ...
!> \param para_env ...
!> \param force_env_section ...
!> \param error ...
!> \author Creation (11.01.2002,MK)
!>                20.09.2002 adapted for pol/kg use, gtb
! *****************************************************************************
  SUBROUTINE init_atomic_kind(atomic_kind,para_env,force_env_section,error)
    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: force_env_section
    TYPE(cp_error_type), INTENT(inout)       :: error

     STOP "atomic_kind_types"
  END SUBROUTINE init_atomic_kind

! *****************************************************************************
!> \brief Initialise an atomic kind set data set.
!> \param atomic_kind_set ...
!> \param para_env ...
!> \param force_env_section ...
!> \param error ...
!> \author - Creation (17.01.2002,MK)
!>      - 20.09.2002 para_env passed (gt)
! *****************************************************************************
  SUBROUTINE init_atomic_kind_set(atomic_kind_set,para_env,force_env_section,error)

    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: force_env_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "atomic_kind_types"
  END SUBROUTINE init_atomic_kind_set


! *****************************************************************************
!> \brief Read an atomic kind data set from the input file.
!> \param atomic_kind ...
!> \param kind_section ...
!> \param para_env ...
!> \param force_env_section ...
!> \param no_fail ...
!> \param method_id ...
!> \param error ...
!> \par History
!>      - Creation (09.02.2002,MK)
!>      - 20.09.2002,gt: adapted for POL/KG use (elp_potential)
!>      - 05.03.2010: split elp_potential into fist_potential and kg_potential
! *****************************************************************************
  SUBROUTINE read_atomic_kind(atomic_kind,kind_section,para_env,force_env_section,&
       no_fail, method_id, error)

    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    TYPE(section_vals_type), POINTER         :: kind_section
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: force_env_section
    LOGICAL, INTENT(IN)                      :: no_fail
    INTEGER, INTENT(IN)                      :: method_id
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "atomic_kind_types"

  END SUBROUTINE read_atomic_kind

! *****************************************************************************
!> \brief Read an atomic kind set data set from the input file.
!> \param atomic_kind_set ...
!> \param kind_section ...
!> \param para_env ...
!> \param force_env_section ...
!> \param error ...
!> \par History
!>      - Creation (09.02.2002,MK)
!>      - 20.09.2002,gt: adapted for POL/KG use (elp_potential)
! *****************************************************************************
  SUBROUTINE read_atomic_kind_set(atomic_kind_set,kind_section,para_env,force_env_section,error)

    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    TYPE(section_vals_type), POINTER         :: kind_section
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: force_env_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    CHARACTER(len=*), PARAMETER :: routineN = 'read_atomic_kind_set', &
      routineP = moduleN//':'//routineN

    STOP "atomic_kind_types"
  END SUBROUTINE read_atomic_kind_set

! *****************************************************************************
!> \brief This routines should perform only checks. no settings are allowed at
!>     this level anymore..
!> \param atomic_kind ...
!> \param dft_control ...
!> \param subsys_section ...
!> \param error ...
! *****************************************************************************
  SUBROUTINE check_atomic_kind(atomic_kind,dft_control,subsys_section,error)

    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    TYPE(dft_control_type), INTENT(IN)       :: dft_control
    TYPE(section_vals_type), POINTER         :: subsys_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "atomic_kind_types"
  END SUBROUTINE check_atomic_kind

! *****************************************************************************
!> \brief ...
!> \param atomic_kind_set ...
!> \param dft_control ...
!> \param para_env ...
!> \param subsys_section ...
!> \param error ...
! *****************************************************************************
  SUBROUTINE check_atomic_kind_set(atomic_kind_set,dft_control,para_env,subsys_section,error)

    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    TYPE(dft_control_type), INTENT(IN)       :: dft_control
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: subsys_section
    TYPE(cp_error_type), INTENT(inout)       :: error

     STOP "atomic_kind_types"
  END SUBROUTINE check_atomic_kind_set

! *****************************************************************************
!> \brief Set the components of an atomic kind data set.
! *****************************************************************************
  SUBROUTINE set_atomic_kind(atomic_kind,element_symbol,name,mass,kind_number,&
       natom,atom_list,&
       fist_potential,zeff,shell,&
       shell_active,damping,error)

    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    CHARACTER(LEN=*), INTENT(IN), OPTIONAL   :: element_symbol, name
    REAL(KIND=dp), INTENT(IN), OPTIONAL      :: mass
    INTEGER, INTENT(IN), OPTIONAL            :: kind_number, natom
    INTEGER, DIMENSION(:), INTENT(IN), &
      OPTIONAL                               :: atom_list
    TYPE(fist_potential_type), OPTIONAL, &
      POINTER                                :: fist_potential
    REAL(KIND=dp), INTENT(IN), OPTIONAL      :: zeff
    TYPE(shell_kind_type), OPTIONAL, POINTER :: shell
    LOGICAL, INTENT(IN), OPTIONAL            :: shell_active
    TYPE(damping_p_type), OPTIONAL, POINTER  :: damping
    TYPE(cp_error_type), INTENT(inout), &
      OPTIONAL                               :: error

    STOP "atomic_kind_types"
  END SUBROUTINE set_atomic_kind

! *****************************************************************************
!> \brief Write an atomic kind data set to the output unit.
!> \param atomic_kind ...
!> \param output_unit ...
!> \param error ...
!> \par History
!>      Creation (09.02.2002,MK)
! *****************************************************************************
  SUBROUTINE write_atomic_kind(atomic_kind,output_unit,error)

    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    INTEGER, INTENT(in)                      :: output_unit
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "atomic_kind_types"
  END SUBROUTINE write_atomic_kind

! *****************************************************************************
!> \brief Write an atomic kind set data set to the output unit.
!> \param atomic_kind_set ...
!> \param subsys_section ...
!> \param error ...
!> \par History
!>      Creation (09.02.2002,MK)
! *****************************************************************************
  SUBROUTINE write_atomic_kind_set(atomic_kind_set,subsys_section,error)
    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    TYPE(section_vals_type), POINTER         :: subsys_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "atomic_kind_types"
  END SUBROUTINE write_atomic_kind_set

! *****************************************************************************
!> \brief Determines if the atomic_kind is HYDROGEN
!> \param atomic_kind ...
!> \retval res ...
!> \author Teodoro Laino [tlaino] - University of Zurich 10.2008
! *****************************************************************************
  PURE FUNCTION is_hydrogen(atomic_kind) RESULT(res)
    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    LOGICAL                                  :: res

    res = TRIM(atomic_kind%element_symbol)=="H"
  END FUNCTION is_hydrogen

END MODULE atomic_kind_types
