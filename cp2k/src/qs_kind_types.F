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
MODULE qs_kind_types

  USE basis_set_types,                 ONLY: &
       allocate_geminal_basis_set, allocate_gto_basis_set, &
       deallocate_geminal_basis_set, deallocate_gto_basis_set, &
       geminal_basis_set_type, get_geminal_basis_set, get_gto_basis_set, &
       gto_basis_set_type, init_aux_basis_set, init_orb_basis_set, &
       read_geminal_basis_set, read_gto_basis_set, set_gto_basis_set, &
       write_gto_basis_set, write_orb_basis_set
  USE cp_control_types,                ONLY: dft_control_type,&
                                             qs_control_type
  USE cp_output_handling,              ONLY: cp_print_key_finished_output,&
                                             cp_print_key_unit_nr
  USE cp_para_types,                   ONLY: cp_para_env_type
  USE external_potential_types,        ONLY: &
       all_potential_type, allocate_potential, deallocate_potential, &
       fist_potential_type, get_potential, gth_potential_type, &
       init_potential, local_potential_type, read_potential, &
       set_default_all_potential, set_potential, write_potential
  USE f77_blas
  USE input_constants,                 ONLY: &
       do_fist, do_method_am1, do_method_dftb, do_method_mndo, &
       do_method_mndod, do_method_pdg, do_method_pm3, do_method_pm6, &
       do_method_pnnl, do_method_rm1, do_method_scptb, do_qs, &
       use_aux_basis_set, use_aux_fit_basis_set, use_lri_basis_set, &
       use_orb_basis_set, use_ri_aux_basis_set, use_scp_basis_set
  USE input_section_types,             ONLY: section_vals_get,&
                                             section_vals_get_subs_vals,&
                                             section_vals_type,&
                                             section_vals_val_get
  USE kinds,                           ONLY: default_string_length,&
                                             dp
  USE orbital_pointers,                ONLY: init_orbital_pointers,&
                                             nco,&
                                             ncoset
  USE paw_proj_set_types,              ONLY: allocate_paw_proj_set,&
                                             deallocate_paw_proj_set,&
                                             get_paw_proj_set,&
                                             paw_proj_set_type,&
                                             projectors
  USE periodic_table,                  ONLY: get_ptable_info
  USE physcon,                         ONLY: bohr,&
                                             evolt,&
                                             massunit
  USE qs_dftb_types,                   ONLY: qs_dftb_atom_type
  USE qs_dftb_utils,                   ONLY: deallocate_dftb_atom_param,&
                                             get_dftb_atom_param,&
                                             write_dftb_atom_param
  USE qs_dispersion_types,             ONLY: qs_atom_dispersion_type
  USE qs_grid_atom,                    ONLY: allocate_grid_atom,&
                                             deallocate_grid_atom,&
                                             grid_atom_type
  USE qs_harmonics_atom,               ONLY: allocate_harmonics_atom,&
                                             deallocate_harmonics_atom,&
                                             harmonics_atom_type
  USE scptb_types,                     ONLY: deallocate_scptb_parameter,&
                                             get_scptb_parameter,&
                                             scptb_parameter_type,&
                                             write_scptb_parameter
  USE semi_empirical_types,            ONLY: get_se_param,&
                                             semi_empirical_create,&
                                             semi_empirical_release,&
                                             semi_empirical_type,&
                                             write_se_param
  USE semi_empirical_utils,            ONLY: init_se_param,&
                                             se_param_set_default
  USE shell_potential_types,           ONLY: shell_kind_type,&
                                             shell_release,&
                                             shell_retain
  USE soft_basis_set,                  ONLY: create_soft_basis
  USE string_utilities,                ONLY: uppercase
  USE termination,                     ONLY: stop_program
  USE timings,                         ONLY: timeset,&
                                             timestop
#include "./common/cp_common_uses.f90"

  IMPLICIT NONE

  PRIVATE

  ! Global parameters (only in this module)

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_kind_types'

!> Input parameters for the DFT+U method
! *****************************************************************************
  TYPE dft_plus_u_type
     INTEGER                                :: l,max_scf
     REAL(KIND=dp)                          :: eps_u_ramping,&
                                               eps_scf,&
                                               u_minus_j_target,&
                                               u_minus_j,&
                                               u_ramping
     INTEGER, DIMENSION(:), POINTER         :: orbitals
     LOGICAL                                :: init_u_ramping_each_scf,&
                                               smear
  END TYPE dft_plus_u_type

!> Provides all information about an atomic kind
! *****************************************************************************
  TYPE qs_kind_type
     TYPE(semi_empirical_type), POINTER     :: se_parameter
     TYPE(semi_empirical_type), POINTER     :: scp_se_parameter
     TYPE(qs_dftb_atom_type), POINTER       :: dftb_parameter
     TYPE(scptb_parameter_type), POINTER    :: scptb_parameter
     TYPE(gto_basis_set_type), POINTER      :: aux_basis_set
     TYPE(gto_basis_set_type), POINTER      :: ri_aux_basis_set
     TYPE(gto_basis_set_type), POINTER      :: orb_basis_set
     TYPE(gto_basis_set_type), POINTER      :: scp_basis_set
     TYPE(gto_basis_set_type), POINTER      :: soft_basis_set
     TYPE(gto_basis_set_type), POINTER      :: hard_basis_set
     TYPE(gto_basis_set_type), POINTER      :: aux_fit_basis_set
     TYPE(gto_basis_set_type), POINTER      :: lri_basis_set
     TYPE(geminal_basis_set_type), POINTER  :: geminal_basis_set
     TYPE(paw_proj_set_type),  POINTER      :: paw_proj_set
     REAL(dp)                               :: hard_radius     ! for hard and soft exp
     REAL(dp)                               :: hard0_radius    ! for hard exp of rho0
     REAL(dp)                               :: max_rad_local   ! max GTO radius used in GAPW
     LOGICAL                                :: paw_atom        ! needs atomic rho1
     LOGICAL                                :: qm_atom         ! needs atomic rho1
     LOGICAL                                :: gpw_type_forced ! gpw atom even if with hard exponents
     LOGICAL                                :: ghost
     REAL(KIND = dp)                        :: alpha_scp, I_scp
     REAL(KIND = dp)                        :: dudq_dftb3
     INTEGER, DIMENSION(:,:), POINTER       :: addel, laddel, naddel
     TYPE(harmonics_atom_type), POINTER     :: harmonics
     TYPE(grid_atom_type), POINTER          :: grid_atom
     INTEGER                                :: ngrid_rad,ngrid_ang
     INTEGER                                :: lmax_rho0
     INTEGER, DIMENSION(:), POINTER         :: elec_conf ! used to set up the initial atomic guess
     LOGICAL                                :: bs_occupation
     TYPE(dft_plus_u_type), POINTER         :: dft_plus_u
     LOGICAL                                :: no_optimize
     !
     TYPE(qs_atom_dispersion_type), POINTER :: dispersion
     REAL(KIND=dp), DIMENSION(:,:), POINTER :: reltmat
  END TYPE qs_kind_type

!> Provides a vector of pointers of type qs_kind_type
! *****************************************************************************
  TYPE qs_kind_p_type
     TYPE(qs_kind_type),DIMENSION(:),&
          POINTER                             :: qs_kind_set
  END TYPE qs_kind_p_type

  ! Public subroutines

  PUBLIC :: allocate_qs_kind_set,&
            check_qs_kind_set,&
            deallocate_qs_kind_set,&
            get_qs_kind,&
            get_qs_kind_set,&
            init_qs_kind_set,&
            init_gapw_basis_set,&
            read_qs_kind_set,&
            set_qs_kind,&
            write_qs_kind_set,&
            write_gto_basis_sets

  ! Public data types
  PUBLIC :: qs_kind_type

CONTAINS

! *****************************************************************************
!> \brief   Constructor routine for a set of atomic kinds
!> \param qs_kind_set ...
!> \param nkind ...
!> \param error ...
!> \date    02.01.2002
!> \author  Matthias Krack (MK)
!> \version 2.0
! *****************************************************************************
  SUBROUTINE allocate_qs_kind_set(qs_kind_set,nkind,error)

    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    INTEGER, INTENT(IN)                      :: nkind
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"
  END SUBROUTINE allocate_qs_kind_set

! *****************************************************************************
!> \brief   Destructor routine for a set of atomic kinds
!> \param qs_kind_set ...
!> \param error ...
!> \date    02.01.2002
!> \author  Matthias Krack (MK)
!> \version 2.0
! *****************************************************************************
  SUBROUTINE deallocate_qs_kind_set(qs_kind_set,error)

    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"
  END SUBROUTINE deallocate_qs_kind_set

! *****************************************************************************
!> \brief Get attributes of an atomic kind.
! *****************************************************************************
  SUBROUTINE get_qs_kind(qs_kind, se_parameter, scp_se_parameter, dftb_parameter, scptb_parameter, &
       aux_basis_set, ri_aux_basis_set, aux_fit_basis_set, orb_basis_set, scp_basis_set, geminal_basis_set, &
       lri_basis_set, &
       alpha_scp, I_scp, &
       ncgf, nsgf, &
       soft_basis_set, elec_conf, hard_basis_set, paw_proj_set, softb, &
       paw_atom, hard_radius, hard0_radius, max_rad_local, &
       gpw_type_forced, harmonics, max_iso_not0, max_s_harm, grid_atom, &
       ngrid_ang, ngrid_rad, lmax_rho0, &
       dft_plus_u_atom, l_of_dft_plus_u, u_minus_j, dispersion, basis_set_id, &
       npgf, bs_occupation, no_optimize, addel, laddel, naddel, orbitals, &
       max_scf, eps_scf, smear, u_ramping, u_minus_j_target, eps_u_ramping, &
       init_u_ramping_each_scf, reltmat, ghost)

    TYPE(qs_kind_type)                       :: qs_kind
    TYPE(semi_empirical_type), OPTIONAL, &
      POINTER                                :: se_parameter, scp_se_parameter
    TYPE(qs_dftb_atom_type), OPTIONAL, &
      POINTER                                :: dftb_parameter
    TYPE(scptb_parameter_type), OPTIONAL, &
      POINTER                                :: scptb_parameter
    TYPE(gto_basis_set_type), OPTIONAL, &
      POINTER                                :: aux_basis_set, &
                                                ri_aux_basis_set, &
                                                aux_fit_basis_set, &
                                                orb_basis_set, scp_basis_set
    TYPE(geminal_basis_set_type), OPTIONAL, &
      POINTER                                :: geminal_basis_set
    TYPE(gto_basis_set_type), OPTIONAL, &
      POINTER                                :: lri_basis_set
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: alpha_scp, I_scp
    INTEGER, INTENT(OUT), OPTIONAL           :: ncgf, nsgf
    TYPE(gto_basis_set_type), OPTIONAL, &
      POINTER                                :: soft_basis_set
    INTEGER, DIMENSION(:), OPTIONAL, POINTER :: elec_conf
    TYPE(gto_basis_set_type), OPTIONAL, &
      POINTER                                :: hard_basis_set
    TYPE(paw_proj_set_type), OPTIONAL, &
      POINTER                                :: paw_proj_set
    LOGICAL, INTENT(IN), OPTIONAL            :: softb
    LOGICAL, INTENT(OUT), OPTIONAL           :: paw_atom
    REAL(dp), INTENT(OUT), OPTIONAL          :: hard_radius, hard0_radius, &
                                                max_rad_local
    LOGICAL, INTENT(OUT), OPTIONAL           :: gpw_type_forced
    TYPE(harmonics_atom_type), OPTIONAL, &
      POINTER                                :: harmonics
    INTEGER, INTENT(OUT), OPTIONAL           :: max_iso_not0, max_s_harm
    TYPE(grid_atom_type), OPTIONAL, POINTER  :: grid_atom
    INTEGER, INTENT(OUT), OPTIONAL           :: ngrid_ang, ngrid_rad, &
                                                lmax_rho0
    LOGICAL, INTENT(OUT), OPTIONAL           :: dft_plus_u_atom
    INTEGER, INTENT(OUT), OPTIONAL           :: l_of_dft_plus_u
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: u_minus_j
    TYPE(qs_atom_dispersion_type), &
      OPTIONAL, POINTER                      :: dispersion
    INTEGER, INTENT(IN), OPTIONAL            :: basis_set_id
    INTEGER, INTENT(OUT), OPTIONAL           :: npgf
    LOGICAL, INTENT(OUT), OPTIONAL           :: bs_occupation, no_optimize
    INTEGER, DIMENSION(:, :), OPTIONAL, &
      POINTER                                :: addel, laddel, naddel
    INTEGER, DIMENSION(:), OPTIONAL, POINTER :: orbitals
    INTEGER, OPTIONAL                        :: max_scf
    REAL(KIND=dp), OPTIONAL                  :: eps_scf
    LOGICAL, OPTIONAL                        :: smear
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: u_ramping, u_minus_j_target, &
                                                eps_u_ramping
    LOGICAL, OPTIONAL                        :: init_u_ramping_each_scf, ghost
    REAL(KIND=dp), DIMENSION(:, :), &
      OPTIONAL, POINTER                      :: reltmat

    STOP "qs_kind_types"
  END SUBROUTINE get_qs_kind

! *****************************************************************************
!> \brief Get attributes of an atomic kind set.
! *****************************************************************************
  SUBROUTINE get_qs_kind_set(qs_kind_set,maxatom,maxcgf,&
       maxco,maxco_proj,maxgtop,maxgtops,maxlgto,maxlprj,&
       maxnset,maxpgf,maxsgf,maxsgf_set,maxshell,&
       ncgf,ncgf_aux, ncgf_ri_aux,&
       npgf,nset,nsgf,nshell,&
       paw_atom_present,&
       qm_atom_present, dft_plus_u_atom_present,&
       maxder,&
       maxlgem,max_ngrid_rad,&
       max_sph_harm,maxg_iso_not0,lmax_rho0,&
       zetsoft_max,basis_set_id)

    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    INTEGER, INTENT(OUT), OPTIONAL :: maxatom, maxcgf, maxco, maxco_proj, &
      maxgtop, maxgtops, maxlgto, maxlprj, &
      maxnset, maxpgf, maxsgf, maxsgf_set, maxshell, &
      ncgf, ncgf_aux, ncgf_ri_aux, npgf, nset, nsgf, nshell
    LOGICAL, INTENT(OUT), OPTIONAL :: &
      paw_atom_present, qm_atom_present, dft_plus_u_atom_present
    INTEGER, INTENT(IN), OPTIONAL            :: maxder
    INTEGER, INTENT(OUT), OPTIONAL           :: maxlgem(2), max_ngrid_rad, &
                                                max_sph_harm, maxg_iso_not0, &
                                                lmax_rho0
    REAL(kind=dp), INTENT(out), OPTIONAL     :: zetsoft_max
    INTEGER, INTENT(IN), OPTIONAL            :: basis_set_id

    STOP "qs_kind_types"
  END SUBROUTINE get_qs_kind_set

! *****************************************************************************
!> \brief Initialise an atomic kind data set.
!> \param qs_kind ...
!> \param para_env ...
!> \param force_env_section ...
!> \param error ...
!> \author Creation (11.01.2002,MK)
!>                20.09.2002 adapted for pol/kg use, gtb
! *****************************************************************************
  SUBROUTINE init_qs_kind(qs_kind,para_env,force_env_section,error)
    TYPE(qs_kind_type), POINTER          :: qs_kind
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: force_env_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"
  END SUBROUTINE init_qs_kind

! *****************************************************************************
!> \brief Initialise an atomic kind set data set.
!> \param qs_kind_set ...
!> \param para_env ...
!> \param force_env_section ...
!> \param error ...
!> \author - Creation (17.01.2002,MK)
!>      - 20.09.2002 para_env passed (gt)
! *****************************************************************************
  SUBROUTINE init_qs_kind_set(qs_kind_set,para_env,force_env_section,error)

    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: force_env_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"
  END SUBROUTINE init_qs_kind_set

! *****************************************************************************
!> \brief ...
!> \param qs_kind_set ...
!> \param qs_control ...
!> \param force_env_section ...
!> \param error ...
! *****************************************************************************
  SUBROUTINE init_gapw_basis_set(qs_kind_set,qs_control,force_env_section,&
       error)

    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    TYPE(qs_control_type), POINTER           :: qs_control
    TYPE(section_vals_type), POINTER         :: force_env_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"

  END SUBROUTINE init_gapw_basis_set

! *****************************************************************************
!> \brief Read an atomic kind data set from the input file.
!> \param qs_kind ...
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
  SUBROUTINE read_qs_kind(qs_kind,kind_section,para_env,force_env_section,&
       no_fail, method_id, error)

    TYPE(qs_kind_type), POINTER          :: qs_kind
    TYPE(section_vals_type), POINTER         :: kind_section
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: force_env_section
    LOGICAL, INTENT(IN)                      :: no_fail
    INTEGER, INTENT(IN)                      :: method_id
    TYPE(cp_error_type), INTENT(inout)       :: error

   STOP "qs_kind_types"

  END SUBROUTINE read_qs_kind

! *****************************************************************************
!> \brief Read an atomic kind set data set from the input file.
!> \param qs_kind_set ...
!> \param kind_section ...
!> \param para_env ...
!> \param force_env_section ...
!> \param error ...
!> \par History
!>      - Creation (09.02.2002,MK)
!>      - 20.09.2002,gt: adapted for POL/KG use (elp_potential)
! *****************************************************************************
  SUBROUTINE read_qs_kind_set(qs_kind_set,kind_section,para_env,force_env_section,error)

    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    TYPE(section_vals_type), POINTER         :: kind_section
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: force_env_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"

  END SUBROUTINE read_qs_kind_set

! *****************************************************************************
!> \brief This routines should perform only checks. no settings are allowed at
!>     this level anymore..
!> \param qs_kind ...
!> \param dft_control ...
!> \param subsys_section ...
!> \param error ...
! *****************************************************************************
  SUBROUTINE check_qs_kind(qs_kind,dft_control,subsys_section,error)

    TYPE(qs_kind_type), POINTER          :: qs_kind
    TYPE(dft_control_type), INTENT(IN)       :: dft_control
    TYPE(section_vals_type), POINTER         :: subsys_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"

  END SUBROUTINE check_qs_kind

! *****************************************************************************
!> \brief ...
!> \param qs_kind_set ...
!> \param dft_control ...
!> \param para_env ...
!> \param subsys_section ...
!> \param error ...
! *****************************************************************************
  SUBROUTINE check_qs_kind_set(qs_kind_set,dft_control,para_env,subsys_section,error)

    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    TYPE(dft_control_type), INTENT(IN)       :: dft_control
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: subsys_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"
  END SUBROUTINE check_qs_kind_set

! *****************************************************************************
!> \brief Set the components of an atomic kind data set.
! *****************************************************************************
  SUBROUTINE set_qs_kind(qs_kind,elec_conf,paw_atom,hard_radius,hard0_radius,&
       soft_basis_set,hard_basis_set,lmax_rho0,zeff,&
       no_optimize,dispersion,u_minus_j,reltmat,dftb_parameter,&
       scptb_parameter,error)

    TYPE(qs_kind_type), INTENT(INOUT)        :: qs_kind
    INTEGER, DIMENSION(:), INTENT(IN), &
      OPTIONAL                               :: elec_conf
    LOGICAL, INTENT(IN), OPTIONAL            :: paw_atom
    REAL(dp), INTENT(IN), OPTIONAL           :: hard_radius, hard0_radius
    TYPE(gto_basis_set_type), OPTIONAL, &
      POINTER                                :: soft_basis_set, hard_basis_set
    INTEGER, INTENT(IN), OPTIONAL            :: lmax_rho0
    REAL(KIND=dp), INTENT(IN), OPTIONAL      :: zeff
    LOGICAL, INTENT(IN), OPTIONAL            :: no_optimize
    TYPE(qs_atom_dispersion_type), &
      OPTIONAL, POINTER                      :: dispersion
    REAL(KIND=dp), INTENT(IN), OPTIONAL      :: u_minus_j
    REAL(KIND=dp), DIMENSION(:, :), &
      OPTIONAL, POINTER                      :: reltmat
   TYPE(qs_dftb_atom_type), OPTIONAL, POINTER       :: dftb_parameter
   TYPE(scptb_parameter_type), OPTIONAL, POINTER    :: scptb_parameter
    TYPE(cp_error_type), INTENT(inout), &
      OPTIONAL                               :: error

      STOP "qs_kind_types"
  END SUBROUTINE set_qs_kind

! *****************************************************************************
!> \brief Write an atomic kind data set to the output unit.
!> \param qs_kind ...
!> \param output_unit ...
!> \param error ...
!> \par History
!>      Creation (09.02.2002,MK)
! *****************************************************************************
  SUBROUTINE write_qs_kind(qs_kind,output_unit,error)

    TYPE(qs_kind_type), POINTER          :: qs_kind
    INTEGER, INTENT(in)                      :: output_unit
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"
  END SUBROUTINE write_qs_kind

! *****************************************************************************
!> \brief Write an atomic kind set data set to the output unit.
!> \param qs_kind_set ...
!> \param subsys_section ...
!> \param error ...
!> \par History
!>      Creation (09.02.2002,MK)
! *****************************************************************************
  SUBROUTINE write_qs_kind_set(qs_kind_set,subsys_section,error)
    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    TYPE(section_vals_type), POINTER         :: subsys_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    STOP "qs_kind_types"

  END SUBROUTINE write_qs_kind_set

! *****************************************************************************
!> \brief Write all the GTO basis sets of an atomic kind set to the output
!>     unit (for the printing of the unnormalized basis sets as read from
!>           database).
!> \param qs_kind_set ...
!> \param subsys_section ...
!> \param error ...
!> \par History
!>      Creation (17.01.2002,MK)
! *****************************************************************************
  SUBROUTINE write_gto_basis_sets(qs_kind_set,subsys_section,error)

    TYPE(qs_kind_type), DIMENSION(:), &
      POINTER                                :: qs_kind_set
    TYPE(section_vals_type), POINTER         :: subsys_section
    TYPE(cp_error_type), INTENT(inout)       :: error

   STOP "qs_kind_types"! CALL timestop(handle)

  END SUBROUTINE write_gto_basis_sets


END MODULE qs_kind_types
