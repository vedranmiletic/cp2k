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
  USE damping_dipole_types,            ONLY: damping_p_release,&
                                             damping_p_type
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
  USE physcon,                         ONLY: bohr,&
                                             evolt,&
                                             massunit
  USE basis_set_types,                 ONLY: geminal_basis_set_type,&
                                             gto_basis_set_type
  USE cp_control_types,                ONLY: dft_control_type,&
                                             qs_control_type
  USE cp_para_types,                   ONLY: cp_para_env_type
  USE f77_blas
  USE input_section_types,             ONLY: section_vals_type
  USE kinds,                           ONLY: dp
  USE paw_proj_set_types,              ONLY: paw_proj_set_type
  USE qs_dftb_types,                   ONLY: qs_dftb_atom_type
  USE qs_dispersion_types,             ONLY: qs_atom_dispersion_type
  USE qs_grid_atom,                    ONLY: grid_atom_type
  USE qs_harmonics_atom,               ONLY: harmonics_atom_type
  USE scptb_types,                     ONLY: scptb_parameter_type
  USE semi_empirical_types,            ONLY: semi_empirical_type
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
     TYPE(semi_empirical_type), POINTER     :: se_parameter      => Null()
     TYPE(semi_empirical_type), POINTER     :: scp_se_parameter  => Null()
     TYPE(qs_dftb_atom_type), POINTER       :: dftb_parameter    => Null()
     TYPE(scptb_parameter_type), POINTER    :: scptb_parameter   => Null()
     TYPE(gto_basis_set_type), POINTER      :: aux_basis_set     => Null()
     TYPE(gto_basis_set_type), POINTER      :: ri_aux_basis_set  => Null()
     TYPE(gto_basis_set_type), POINTER      :: orb_basis_set     => Null()
     TYPE(gto_basis_set_type), POINTER      :: scp_basis_set     => Null()
     TYPE(gto_basis_set_type), POINTER      :: soft_basis_set    => Null()
     TYPE(gto_basis_set_type), POINTER      :: hard_basis_set    => Null()
     TYPE(gto_basis_set_type), POINTER      :: aux_fit_basis_set => Null()
     TYPE(gto_basis_set_type), POINTER      :: lri_basis_set     => Null()
     TYPE(geminal_basis_set_type), POINTER  :: geminal_basis_set => Null()
     TYPE(paw_proj_set_type),  POINTER      :: paw_proj_set      => Null()
     REAL(dp)                               :: hard_radius  = 0.8_dp*bohr    ! for hard and soft exp
     REAL(dp)                               :: hard0_radius = 0.8_dp*bohr    ! for hard exp of rho0
     REAL(dp)                               :: max_rad_local = 13.2_dp*bohr  ! max GTO radius used in GAPW
     LOGICAL                                :: paw_atom = .FALSE.        ! needs atomic rho1
     !LOGICAL                                :: qm_atom = .FALSE.        ! needs atomic rho1
     LOGICAL                                :: gpw_type_forced = .FALSE. ! gpw atom even if with hard exponents
     LOGICAL                                :: ghost = .FALSE.
     REAL(KIND = dp)                        :: alpha_scp = 0.0_dp
     REAL(KIND = dp)                        :: I_scp = 0.0_dp
     REAL(KIND = dp)                        :: alpha_scp = 0.0_dp
     REAL(KIND = dp)                        :: dudq_dftb3
     INTEGER, DIMENSION(:,:), POINTER       :: addel  => Null()
     INTEGER, DIMENSION(:,:), POINTER       :: laddel => Null()
     INTEGER, DIMENSION(:,:), POINTER       :: naddel => Null()
     TYPE(harmonics_atom_type), POINTER     :: harmonics => Null()
     TYPE(grid_atom_type), POINTER          :: grid_atom => Null()
     INTEGER                                :: ngrid_rad = 50
     INTEGER                                :: ngrid_ang = 50
     INTEGER                                :: lmax_rho0 = 0
     INTEGER, DIMENSION(:), POINTER         :: elec_conf => Null() ! used to set up the initial atomic guess
     LOGICAL                                :: bs_occupation = .FALSE.
     TYPE(dft_plus_u_type), POINTER         :: dft_plus_u => Null()
     LOGICAL                                :: no_optimize = .TRUE.
     !
     TYPE(qs_atom_dispersion_type), POINTER :: dispersion => Null()
     REAL(KIND=dp), DIMENSION(:,:), POINTER :: reltmat => Null()
  END TYPE qs_kind_type

!> Provides a vector of pointers of type qs_kind_type
! *****************************************************************************
  TYPE qs_kind_p_type
     TYPE(qs_kind_type),DIMENSION(:),&
          POINTER                             :: qs_kind_set
  END TYPE qs_kind_p_type

  ! Public subroutines

  PUBLIC :: check_qs_kind_set,&
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
!> \brief   Destructor routine for a set of qs kinds
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

    CHARACTER(len=*), PARAMETER :: routineN = 'deallocate_qs_kind_set', &
      routineP = moduleN//':'//routineN

    INTEGER                                  :: ikind, nkind, stat
    LOGICAL                                  :: failure

    failure = .FALSE.
    IF (ASSOCIATED(qs_kind_set)) THEN

       nkind = SIZE(qs_kind_set)

       DO ikind=1,nkind
          IF (ASSOCIATED(qs_kind_set(ikind)%se_parameter)) THEN
             CALL semi_empirical_release(qs_kind_set(ikind)%se_parameter,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%scp_se_parameter)) THEN
             CALL semi_empirical_release(qs_kind_set(ikind)%scp_se_parameter,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%dftb_parameter)) THEN
             CALL deallocate_dftb_atom_param(qs_kind_set(ikind)%dftb_parameter,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%scptb_parameter)) THEN
             CALL deallocate_scptb_parameter(qs_kind_set(ikind)%scptb_parameter,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%aux_basis_set)) THEN
             CALL deallocate_gto_basis_set(qs_kind_set(ikind)%aux_basis_set,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%ri_aux_basis_set)) THEN
             CALL deallocate_gto_basis_set(qs_kind_set(ikind)%ri_aux_basis_set,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%aux_fit_basis_set)) THEN
             CALL deallocate_gto_basis_set(qs_kind_set(ikind)%aux_fit_basis_set,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%lri_basis_set)) THEN
             CALL deallocate_gto_basis_set(qs_kind_set(ikind)%lri_basis_set,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%soft_basis_set).AND.&
               qs_kind_set(ikind)%paw_atom) THEN
             CALL deallocate_gto_basis_set(qs_kind_set(ikind)%soft_basis_set,error)
          ELSEIF(ASSOCIATED(qs_kind_set(ikind)%soft_basis_set).AND.&
               (.NOT.qs_kind_set(ikind)%paw_atom)) THEN
             NULLIFY (qs_kind_set(ikind)%soft_basis_set)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%orb_basis_set)) THEN
             CALL deallocate_gto_basis_set(qs_kind_set(ikind)%orb_basis_set,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%scp_basis_set)) THEN
             CALL deallocate_gto_basis_set(qs_kind_set(ikind)%scp_basis_set,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%geminal_basis_set)) THEN
             CALL deallocate_geminal_basis_set(qs_kind_set(ikind)%geminal_basis_set,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%paw_proj_set)) THEN
             CALL deallocate_paw_proj_set(qs_kind_set(ikind)%paw_proj_set,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%harmonics)) THEN
             CALL deallocate_harmonics_atom(qs_kind_set(ikind)%harmonics,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%grid_atom)) THEN
             CALL deallocate_grid_atom(qs_kind_set(ikind)%grid_atom,error)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%elec_conf)) THEN
             DEALLOCATE (qs_kind_set(ikind)%elec_conf,STAT=stat)
             CPPostcondition(stat==0, cp_failure_level, routineP, error, failure)
          END IF

          IF (ASSOCIATED(qs_kind_set(ikind)%dft_plus_u)) THEN
             IF (ASSOCIATED(qs_kind_set(ikind)%dft_plus_u%orbitals)) THEN
                DEALLOCATE (qs_kind_set(ikind)%dft_plus_u%orbitals,STAT=stat)
                CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
             END IF
             DEALLOCATE (qs_kind_set(ikind)%dft_plus_u,STAT=stat)
             CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
          END IF

          IF (ASSOCIATED(qs_kind_set(ikind)%dispersion)) THEN
             DEALLOCATE (qs_kind_set(ikind)%dispersion,STAT=stat)
             CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%addel)) THEN
             DEALLOCATE (qs_kind_set(ikind)%addel,STAT=stat)
             CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%naddel)) THEN
             DEALLOCATE (qs_kind_set(ikind)%naddel,STAT=stat)
             CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%laddel)) THEN
             DEALLOCATE (qs_kind_set(ikind)%laddel,STAT=stat)
             CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
          END IF
          IF (ASSOCIATED(qs_kind_set(ikind)%reltmat)) THEN
             DEALLOCATE (qs_kind_set(ikind)%reltmat,STAT=stat)
             CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
          END IF

       END DO
       DEALLOCATE (qs_kind_set,STAT=stat)
       CPPostcondition(stat==0, cp_failure_level, routineP, error, failure)
    ELSE
       CALL stop_program(routineN,moduleN,__LINE__,&
            "The pointer qs_kind_set is not associated and "//&
            "cannot be deallocated")
    END IF

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
    LOGICAL, OPTIONAL                        :: init_u_ramping_each_scf
    REAL(KIND=dp), DIMENSION(:, :), &
      OPTIONAL, POINTER                      :: reltmat
    LOGICAL, OPTIONAL                        :: ghost

    CHARACTER(len=*), PARAMETER :: routineN = 'get_qs_kind', &
      routineP = moduleN//':'//routineN

    INTEGER                                  :: i, j, l, my_basis_set_id, &
                                                n_set
    INTEGER, DIMENSION(:), POINTER           :: lmax, lmin, npgf_set
    LOGICAL                                  :: softb_local

    STOP "qs_kind_types"
    IF( PRESENT(basis_set_id) ) THEN
      my_basis_set_id = basis_set_id
    ELSE
      my_basis_set_id = use_orb_basis_set
    END IF

    softb_local = .FALSE.

    IF (PRESENT(softb)) softb_local = softb
!    IF (PRESENT(all_potential)) all_potential => qs_kind%all_potential
!    IF (PRESENT(tnadd_potential)) tnadd_potential => qs_kind%tnadd_potential
!    IF (PRESENT(fist_potential)) fist_potential => qs_kind%fist_potential
!    IF (PRESENT(gth_potential)) gth_potential => qs_kind%gth_potential
    IF (PRESENT(se_parameter))  se_parameter => qs_kind%se_parameter
    IF (PRESENT(scp_se_parameter))  scp_se_parameter => qs_kind%scp_se_parameter
    IF (PRESENT(dftb_parameter))  dftb_parameter => qs_kind%dftb_parameter
    IF (PRESENT(scptb_parameter))  scptb_parameter => qs_kind%scptb_parameter
    IF (PRESENT(aux_basis_set)) aux_basis_set => qs_kind%aux_basis_set
    IF (PRESENT(ri_aux_basis_set)) ri_aux_basis_set => qs_kind%ri_aux_basis_set
    IF (PRESENT(aux_fit_basis_set)) aux_fit_basis_set => qs_kind%aux_fit_basis_set
    IF (PRESENT(lri_basis_set)) lri_basis_set => qs_kind%lri_basis_set
    IF (PRESENT(orb_basis_set)) THEN
       IF(softb_local) THEN
          orb_basis_set => qs_kind%soft_basis_set
       ELSE
          orb_basis_set => qs_kind%orb_basis_set
       END IF
    END IF
    IF (PRESENT(geminal_basis_set)) geminal_basis_set => qs_kind%geminal_basis_set
    IF (PRESENT(scp_basis_set)) scp_basis_set => qs_kind%scp_basis_set
!    IF (PRESENT(element_symbol)) element_symbol = qs_kind%element_symbol
!    IF (PRESENT(name)) name = qs_kind%name
!    IF (PRESENT(mass)) mass = qs_kind%mass
    IF (PRESENT(alpha_scp)) alpha_scp = qs_kind%alpha_scp
    IF (PRESENT(I_scp)) I_scp = qs_kind%I_scp
!    IF (PRESENT(kind_number)) kind_number = qs_kind%kind_number
!    IF (PRESENT(natom)) natom = qs_kind%natom
!    IF (PRESENT(atom_list)) atom_list => qs_kind%atom_list
    IF (PRESENT(elec_conf)) elec_conf => qs_kind%elec_conf
!    IF (PRESENT(alpha_core_charge)) THEN
!       IF (ASSOCIATED(qs_kind%all_potential)) THEN
!          CALL get_potential(potential=qs_kind%all_potential,&
!               alpha_core_charge=alpha_core_charge)
!       ELSE IF (ASSOCIATED(qs_kind%gth_potential)) THEN
!          CALL get_potential(potential=qs_kind%gth_potential,&
!               alpha_core_charge=alpha_core_charge)
!       ELSE
!          alpha_core_charge = 1.0_dp
!       END IF
!    END IF
!    IF (PRESENT(ccore_charge)) THEN
!       IF (ASSOCIATED(qs_kind%all_potential)) THEN
!          CALL get_potential(potential=qs_kind%all_potential,&
!               ccore_charge=ccore_charge)
!       ELSE IF (ASSOCIATED(qs_kind%gth_potential)) THEN
!          CALL get_potential(potential=qs_kind%gth_potential,&
!               ccore_charge=ccore_charge)
!       ELSE
!          ccore_charge = 0.0_dp
!       END IF
!    END IF
!    IF (PRESENT(core_charge_radius)) THEN
!       IF (ASSOCIATED(qs_kind%all_potential)) THEN
!          CALL get_potential(potential=qs_kind%all_potential,&
!               core_charge_radius=core_charge_radius)
!       ELSE IF (ASSOCIATED(qs_kind%gth_potential)) THEN
!          CALL get_potential(potential=qs_kind%gth_potential,&
!               core_charge_radius=core_charge_radius)
!       ELSE
!          core_charge_radius = 0.0_dp
!       END IF
!    END IF
!    IF (PRESENT(core_charge)) THEN
!       IF (ASSOCIATED(qs_kind%all_potential)) THEN
!          CALL get_potential(potential=qs_kind%all_potential,&
!               zeff=core_charge)
!       ELSE IF (ASSOCIATED(qs_kind%gth_potential)) THEN
!          CALL get_potential(potential=qs_kind%gth_potential,&
!               zeff=core_charge)
!       ELSE
!          core_charge = 0.0_dp
!       END IF
!    END IF
    IF (PRESENT(ncgf)) THEN
       SELECT CASE ( my_basis_set_id)
       CASE (use_orb_basis_set)
         IF (ASSOCIATED(qs_kind%orb_basis_set)) THEN
           CALL get_gto_basis_set(gto_basis_set=qs_kind%orb_basis_set,&
                ncgf=ncgf)
         ELSE IF(ASSOCIATED(qs_kind%dftb_parameter)) THEN
           l = qs_kind%dftb_parameter%lmax
           ncgf = ((l+1)*(l+2)*(l+3))/6
         ELSE
           ncgf = 0
         END IF
       CASE (use_scp_basis_set)
         IF (ASSOCIATED(qs_kind%scp_basis_set)) THEN
           CALL get_gto_basis_set(gto_basis_set=qs_kind%scp_basis_set,&
                ncgf=ncgf)
         ELSE
           ncgf = 0
         END IF
       CASE (use_lri_basis_set)
         IF (ASSOCIATED(qs_kind%lri_basis_set)) THEN
           CALL get_gto_basis_set(gto_basis_set=qs_kind%lri_basis_set,&
                ncgf=ncgf)
         ELSE
           ncgf = 0
         END IF
       CASE (use_aux_fit_basis_set)
         IF (ASSOCIATED(qs_kind%aux_fit_basis_set)) THEN
           CALL get_gto_basis_set(gto_basis_set=qs_kind%aux_fit_basis_set,&
                ncgf=ncgf)
         ELSE IF(ASSOCIATED(qs_kind%dftb_parameter)) THEN
           l = qs_kind%dftb_parameter%lmax
           ncgf = ((l+1)*(l+2)*(l+3))/6
         ELSE
           ncgf = 0
         END IF
       END SELECT
    END IF
    IF (PRESENT(nsgf)) THEN
       SELECT CASE ( my_basis_set_id)
       CASE (use_orb_basis_set)
         IF (ASSOCIATED(qs_kind%orb_basis_set)) THEN
           CALL get_gto_basis_set(gto_basis_set=qs_kind%orb_basis_set,&
                nsgf=nsgf)
         ELSE IF(ASSOCIATED(qs_kind%dftb_parameter)) THEN
           nsgf = qs_kind%dftb_parameter%natorb
         ELSE
           nsgf = 0
         END IF
       CASE (use_scp_basis_set)
         IF (ASSOCIATED(qs_kind%scp_basis_set)) THEN
           CALL get_gto_basis_set(gto_basis_set=qs_kind%scp_basis_set,&
                nsgf=nsgf)
         ELSE
           nsgf = 0
         END IF
       CASE (use_lri_basis_set)
         IF (ASSOCIATED(qs_kind%lri_basis_set)) THEN
           CALL get_gto_basis_set(gto_basis_set=qs_kind%lri_basis_set,&
                nsgf=nsgf)
         ELSE
           nsgf = 0
         END IF
       CASE (use_aux_fit_basis_set)
         IF (ASSOCIATED(qs_kind%aux_fit_basis_set)) THEN
           CALL get_gto_basis_set(gto_basis_set=qs_kind%aux_fit_basis_set,&
                nsgf=nsgf)
         ELSE IF(ASSOCIATED(qs_kind%dftb_parameter)) THEN
           nsgf = qs_kind%dftb_parameter%natorb
         ELSE
           nsgf = 0
         END IF
       END SELECT
    END IF

    IF (PRESENT(npgf)) THEN
      IF (ASSOCIATED(qs_kind%orb_basis_set)) THEN
        CALL get_gto_basis_set(gto_basis_set=qs_kind%orb_basis_set,&
                               npgf=npgf_set, &
                               nset=n_set, &
                               lmax=lmax,&
                               lmin=lmin)
        npgf = 0
        DO i=1,n_set
          DO j = lmin(i),lmax(i)
            npgf = npgf+npgf_set(i)*nco(j)
          END DO
        END DO
      ELSE
        npgf = 0
      END IF
    END IF

!    IF (PRESENT(z)) THEN
!       IF (ASSOCIATED(qs_kind%all_potential)) THEN
!          CALL get_potential(potential=qs_kind%all_potential,z=z)
!       ELSE IF (ASSOCIATED(qs_kind%gth_potential)) THEN
!          CALL get_potential(potential=qs_kind%gth_potential,z=z)
!       ELSE
!          CALL get_ptable_info(qs_kind%element_symbol, number=z)
!       END IF
!    END IF
!    IF (PRESENT(zeff)) THEN
!       IF (ASSOCIATED(qs_kind%all_potential)) THEN
!          CALL get_potential(potential=qs_kind%all_potential,zeff=zeff)
!       ELSE IF (ASSOCIATED(qs_kind%gth_potential)) THEN
!          CALL get_potential(potential=qs_kind%gth_potential,zeff=zeff)
!       ELSE
!          zeff = 0.0_dp
!       END IF
!    END IF
!    IF (PRESENT(rcov)) THEN
!       CALL get_ptable_info(qs_kind%element_symbol, covalent_radius=rcov)
!    END IF
!    IF (PRESENT(rvdw)) THEN
!       CALL get_ptable_info(qs_kind%element_symbol, vdw_radius=rvdw)
!    END IF
!    IF (PRESENT(qeff)) THEN
!       IF (ASSOCIATED(qs_kind%fist_potential)) THEN
!          CALL get_potential(potential=qs_kind%fist_potential,qeff=qeff)
!       ELSE
!          qeff = -HUGE(0.0_dp)
!       END IF
!    END IF
!    IF (PRESENT(apol)) THEN
!       IF (ASSOCIATED(qs_kind%fist_potential)) THEN
!          CALL get_potential(potential=qs_kind%fist_potential,apol=apol)
!       ELSE
!          apol = -HUGE(0.0_dp)
!       END IF
!    END IF
!    IF (PRESENT(cpol)) THEN
!       IF (ASSOCIATED(qs_kind%fist_potential)) THEN
!          CALL get_potential(potential=qs_kind%fist_potential,cpol=cpol)
!       ELSE
!          cpol = -HUGE(0.0_dp)
!       END IF
!    END IF
!    IF (PRESENT(mm_radius)) THEN
!       IF (ASSOCIATED(qs_kind%fist_potential)) THEN
!          CALL get_potential(potential=qs_kind%fist_potential,mm_radius=mm_radius)
!       ELSE
!          mm_radius = -HUGE(0.0_dp)
!       END IF
!    END IF
    IF (PRESENT(soft_basis_set)) soft_basis_set => qs_kind%soft_basis_set
    IF (PRESENT(soft_basis_set)) soft_basis_set => qs_kind%soft_basis_set
    IF (PRESENT(hard_basis_set)) hard_basis_set => qs_kind%hard_basis_set
    IF (PRESENT(paw_proj_set))   paw_proj_set => qs_kind%paw_proj_set
    IF (PRESENT(paw_atom)) paw_atom = qs_kind%paw_atom
    !IF (PRESENT(qm_atom)) qm_atom = qs_kind%qm_atom
    IF (PRESENT(gpw_type_forced)) gpw_type_forced = qs_kind%gpw_type_forced
    IF (PRESENT(hard_radius)) hard_radius = qs_kind%hard_radius
    IF (PRESENT(hard0_radius)) hard0_radius = qs_kind%hard0_radius
    IF (PRESENT(max_rad_local)) max_rad_local = qs_kind%max_rad_local
    IF (PRESENT(harmonics))  harmonics => qs_kind%harmonics
    IF (PRESENT(max_s_harm)) THEN
       IF(ASSOCIATED(qs_kind%harmonics)) THEN
          max_s_harm = qs_kind%harmonics%max_s_harm
       ELSE
          max_s_harm = 0
       END IF
    END IF
    IF (PRESENT(max_iso_not0)) THEN
       IF(ASSOCIATED(qs_kind%harmonics)) THEN
          max_iso_not0 = qs_kind%harmonics%max_iso_not0
       ELSE
          max_iso_not0 = 0
       END IF
    END IF
    IF (PRESENT(grid_atom)) grid_atom => qs_kind%grid_atom
    IF (PRESENT(ngrid_ang)) ngrid_ang = qs_kind%ngrid_ang
    IF (PRESENT(ngrid_rad)) ngrid_rad = qs_kind%ngrid_rad
    IF (PRESENT(lmax_rho0)) lmax_rho0 = qs_kind%lmax_rho0
    IF (PRESENT(ghost)) ghost = qs_kind%ghost
!    IF (PRESENT(shell)) shell => qs_kind%shell
!    IF (PRESENT(shell_active)) shell_active = qs_kind%shell_active
    IF (PRESENT(dft_plus_u_atom)) dft_plus_u_atom = ASSOCIATED(qs_kind%dft_plus_u)
!    IF (PRESENT(damping)) damping => qs_kind%damping
    IF (PRESENT(l_of_dft_plus_u)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          l_of_dft_plus_u = qs_kind%dft_plus_u%l
       ELSE
          l_of_dft_plus_u = -1
       END IF
    END IF
    IF (PRESENT(u_minus_j)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          u_minus_j = qs_kind%dft_plus_u%u_minus_j
       ELSE
          u_minus_j = 0.0_dp
       END IF
    END IF
    IF (PRESENT(u_minus_j_target)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          u_minus_j_target = qs_kind%dft_plus_u%u_minus_j_target
       ELSE
          u_minus_j_target = 0.0_dp
       END IF
    END IF
    IF (PRESENT(init_u_ramping_each_scf)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          init_u_ramping_each_scf = qs_kind%dft_plus_u%init_u_ramping_each_scf
       ELSE
          init_u_ramping_each_scf = .FALSE.
       END IF
    END IF
    IF (PRESENT(u_ramping)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          u_ramping = qs_kind%dft_plus_u%u_ramping
       ELSE
          u_ramping = 0.0_dp
       END IF
    END IF
    IF (PRESENT(eps_u_ramping)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          eps_u_ramping = qs_kind%dft_plus_u%eps_u_ramping
       ELSE
          eps_u_ramping = 1.0E-5_dp
       END IF
    END IF
    IF (PRESENT(orbitals)) THEN
       NULLIFY (orbitals)
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          IF (ASSOCIATED(qs_kind%dft_plus_u%orbitals)) THEN
             orbitals => qs_kind%dft_plus_u%orbitals
          END IF
       END IF
    END IF
    IF (PRESENT(eps_scf)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          eps_scf = qs_kind%dft_plus_u%eps_scf
       ELSE
          eps_scf = 1.0E30_dp
       END IF
    END IF
    IF (PRESENT(max_scf)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          max_scf = qs_kind%dft_plus_u%max_scf
       ELSE
          max_scf = -1
       END IF
    END IF
    IF (PRESENT(smear)) THEN
       IF (ASSOCIATED(qs_kind%dft_plus_u)) THEN
          smear = qs_kind%dft_plus_u%smear
       ELSE
          smear = .FALSE.
       END IF
    END IF
    IF (PRESENT(dispersion)) dispersion => qs_kind%dispersion
    IF (PRESENT(bs_occupation)) bs_occupation=qs_kind%bs_occupation
    IF (PRESENT(addel))  addel  => qs_kind%addel
    IF (PRESENT(laddel)) laddel => qs_kind%laddel
    IF (PRESENT(naddel)) naddel => qs_kind%naddel

    IF (PRESENT(no_optimize)) no_optimize=qs_kind%no_optimize

    IF (PRESENT(reltmat)) reltmat => qs_kind%reltmat


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
      maxgtop, maxgtops, maxlgto, maxlprj, maxnset, maxpgf, maxsgf, &
      maxsgf_set, maxshell, ncgf, ncgf_aux, ncgf_ri_aux, npgf, nset, nsgf, &
      nshell
    LOGICAL, INTENT(OUT), OPTIONAL           :: paw_atom_present, &
                                                qm_atom_present, &
                                                dft_plus_u_atom_present
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
    TYPE(qs_kind_type), POINTER              :: qs_kind
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

    TYPE(qs_kind_type), POINTER              :: qs_kind
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

    TYPE(qs_kind_type), POINTER              :: qs_kind
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
    TYPE(qs_dftb_atom_type), OPTIONAL, &
      POINTER                                :: dftb_parameter
    TYPE(scptb_parameter_type), OPTIONAL, &
      POINTER                                :: scptb_parameter
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

    TYPE(qs_kind_type), POINTER              :: qs_kind
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
