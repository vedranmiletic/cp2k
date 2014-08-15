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
 USE input_section_types,             ONLY: section_vals_get,&
                                             section_vals_get_subs_vals,&
                                             section_vals_type,&
                                             section_vals_val_get
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
  use message_passing, ONLY: mp_abort
#include "./common/cp_common_uses.f90"

  IMPLICIT NONE

  PRIVATE

  ! Global parameters (only in this module)

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'atomic_kind_types'


!> Provides all information about an atomic kind
! *****************************************************************************
  TYPE atomic_kind_type
     TYPE(fist_potential_type), POINTER     :: fist_potential => Null()
     CHARACTER(LEN=default_string_length)   :: name = ""
     CHARACTER(LEN=2)                       :: element_symbol = ""
     REAL(KIND = dp)                        :: mass = 0.0_dp
     INTEGER                                :: kind_number = -1
     INTEGER                                :: natom = -1
     INTEGER, DIMENSION(:), POINTER         :: atom_list => Null()
     LOGICAL                                :: shell_active = .FALSE.
     TYPE(shell_kind_type), POINTER         :: shell => Null()
     TYPE(damping_p_type), POINTER          :: damping => Null()
  END TYPE atomic_kind_type

!> Provides a vector of pointers of type atomic_kind_type
! *****************************************************************************
  TYPE atomic_kind_p_type
     TYPE(atomic_kind_type),DIMENSION(:),&
          POINTER                             :: atomic_kind_set
  END TYPE atomic_kind_p_type

  ! Public subroutines

  PUBLIC :: deallocate_atomic_kind_set,&
            get_atomic_kind,&
            get_atomic_kind_set,&
            !read_atomic_kind_set,&
            set_atomic_kind,&
            is_hydrogen,&
            locate_kind_section_rep

  ! Public data types
  PUBLIC :: atomic_kind_type

CONTAINS

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

    CHARACTER(len=*), PARAMETER :: routineN = 'deallocate_atomic_kind_set', &
      routineP = moduleN//':'//routineN

    INTEGER                                  :: ikind, nkind, stat
    LOGICAL                                  :: failure

    failure = .FALSE.
    IF (ASSOCIATED(atomic_kind_set)) THEN

       nkind = SIZE(atomic_kind_set)

       DO ikind=1,nkind
          IF (ASSOCIATED(atomic_kind_set(ikind)%fist_potential)) THEN
             CALL deallocate_potential(atomic_kind_set(ikind)%fist_potential,error)
          END IF
          IF (ASSOCIATED(atomic_kind_set(ikind)%atom_list)) THEN
             DEALLOCATE (atomic_kind_set(ikind)%atom_list,STAT=stat)
             CPPostcondition(stat==0, cp_failure_level, routineP, error, failure)
          END IF
          CALL shell_release(atomic_kind_set(ikind)%shell,error)

          CALL damping_p_release(atomic_kind_set(ikind)%damping,error)
       END DO
       DEALLOCATE (atomic_kind_set,STAT=stat)
       CPPostcondition(stat==0, cp_failure_level, routineP, error, failure)
    ELSE
       CALL stop_program(routineN,moduleN,__LINE__,&
            "The pointer atomic_kind_set is not associated and "//&
            "cannot be deallocated")
    END IF

  END SUBROUTINE deallocate_atomic_kind_set

! *****************************************************************************
!> \brief Get attributes of an atomic kind.
! *****************************************************************************
  SUBROUTINE get_atomic_kind(atomic_kind, fist_potential,&
       element_symbol, name, mass, kind_number, natom, atom_list, &
       rcov, rvdw, z, qeff, apol, cpol, mm_radius, &
       shell, shell_active, damping)

    TYPE(atomic_kind_type)                   :: atomic_kind
    TYPE(fist_potential_type), OPTIONAL, &
      POINTER                                :: fist_potential
    CHARACTER(LEN=2), INTENT(OUT), OPTIONAL  :: element_symbol
    CHARACTER(LEN=default_string_length), &
      INTENT(OUT), OPTIONAL                  :: name
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: mass
    INTEGER, INTENT(OUT), OPTIONAL           :: kind_number, natom
    INTEGER, DIMENSION(:), OPTIONAL, POINTER :: atom_list
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: rcov, rvdw
    INTEGER, INTENT(OUT), OPTIONAL           :: z
    REAL(KIND=dp), INTENT(OUT), OPTIONAL     :: qeff, apol, cpol, &
                                                mm_radius
    TYPE(shell_kind_type), OPTIONAL, POINTER :: shell
    LOGICAL, INTENT(OUT), OPTIONAL           :: shell_active
    TYPE(damping_p_type), OPTIONAL, POINTER  :: damping

    CHARACTER(len=*), PARAMETER :: routineN = 'get_atomic_kind', &
      routineP = moduleN//':'//routineN

    INTEGER                                  :: i, j, l, my_basis_set_id, &
                                                n_set
    INTEGER, DIMENSION(:), POINTER           :: lmax, lmin, npgf_set
    LOGICAL                                  :: softb_local

    IF (PRESENT(fist_potential)) fist_potential => atomic_kind%fist_potential
    IF (PRESENT(element_symbol)) element_symbol = atomic_kind%element_symbol
    IF (PRESENT(name)) name = atomic_kind%name
    IF (PRESENT(mass)) mass = atomic_kind%mass
    IF (PRESENT(kind_number)) kind_number = atomic_kind%kind_number
    IF (PRESENT(natom)) natom = atomic_kind%natom
    IF (PRESENT(atom_list)) atom_list => atomic_kind%atom_list

    IF (PRESENT(z)) THEN
       CALL get_ptable_info(atomic_kind%element_symbol, number=z)
       WRITE (*,*) "get_atomic_kind z=", z
       !STOP "TODO: atomic_kind_types"
       !IF (ASSOCIATED(atomic_kind%all_potential)) THEN
       !   CALL get_potential(potential=atomic_kind%all_potential,z=z)
       !ELSE IF (ASSOCIATED(atomic_kind%gth_potential)) THEN
       !   CALL get_potential(potential=atomic_kind%gth_potential,z=z)
       !ELSE
       !   CALL get_ptable_info(atomic_kind%element_symbol, number=z)
       !END IF
    END IF
    IF (PRESENT(rcov)) THEN
       CALL get_ptable_info(atomic_kind%element_symbol, covalent_radius=rcov)
    END IF
    IF (PRESENT(rvdw)) THEN
       CALL get_ptable_info(atomic_kind%element_symbol, vdw_radius=rvdw)
    END IF

    !IF (PRESENT(ghost)) ghost = atomic_kind%ghost
    IF (PRESENT(shell)) shell => atomic_kind%shell
    IF (PRESENT(shell_active)) shell_active = atomic_kind%shell_active
    IF (PRESENT(damping)) damping => atomic_kind%damping

  END SUBROUTINE get_atomic_kind

! *****************************************************************************
!> \brief Get attributes of an atomic kind set.
! *****************************************************************************
  SUBROUTINE get_atomic_kind_set(atomic_kind_set,&
       atom_of_kind, kind_of, natom_of_kind,&
       maxatom,&
       natom,&
       nset,nshell,&
       fist_potential_present,&
       shell_present, shell_adiabatic,&
       shell_check_distance,&
       damping_present)

    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    INTEGER, DIMENSION(:), INTENT(OUT), &
      OPTIONAL                               :: atom_of_kind, kind_of, &
                                                natom_of_kind
    INTEGER, INTENT(OUT), OPTIONAL           :: maxatom, natom, nset, nshell
    LOGICAL, INTENT(OUT), OPTIONAL ::  fist_potential_present, &
      shell_present, shell_adiabatic, shell_check_distance, damping_present

  CHARACTER(len=*), PARAMETER :: routineN = 'get_atomic_kind_set', &
      routineP = moduleN//':'//routineN

    INTEGER :: atom_a, iatom, ikind, imax, ipgf, iset, lmax_rho0_kind, lmg, &
      ls, max_iso_not0, max_s_harm, my_basis_set_id, n, ngrid_rad, nkind
    INTEGER, DIMENSION(:), POINTER           :: lmgem, lsgem, npgf_kind
    LOGICAL                                  :: dft_plus_u_atom, paw_atom, &
                                                qm_atom
    REAL(KIND=dp)                            :: zeff, zeff_correction
    REAL(kind=dp), DIMENSION(:, :), POINTER  :: zet_kind
    TYPE(all_potential_type), POINTER        :: all_potential
    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    TYPE(damping_p_type), POINTER            :: damping
    TYPE(fist_potential_type), POINTER       :: fist_potential
    TYPE(geminal_basis_set_type), POINTER    :: geminal_basis_set
    TYPE(gth_potential_type), POINTER        :: gth_potential
    TYPE(gto_basis_set_type), POINTER :: aux_basis_set, aux_fit_basis_set, &
      lri_basis_set, orb_basis_set, ri_aux_basis_set, scp_basis_set, &
      soft_basis_set
    TYPE(local_potential_type), POINTER      :: tnadd_potential
    TYPE(paw_proj_set_type), POINTER         :: paw_proj_set
    TYPE(qs_dftb_atom_type), POINTER         :: dftb_parameter
    TYPE(shell_kind_type), POINTER           :: shell

    IF (ASSOCIATED(atomic_kind_set)) THEN

       IF (PRESENT(maxatom)) maxatom = 0
       IF (PRESENT(natom)) natom = 0
       IF (PRESENT(nshell)) nshell = 0
       IF (PRESENT(shell_present)) shell_present = .FALSE.
       IF (PRESENT(shell_adiabatic)) shell_adiabatic = .FALSE.
       IF (PRESENT(shell_check_distance)) shell_check_distance = .FALSE.
       IF (PRESENT(damping_present)) damping_present = .FALSE.
       IF (PRESENT(atom_of_kind)) atom_of_kind(:) = 0
       IF (PRESENT(kind_of)) kind_of(:) = 0
       IF (PRESENT(natom_of_kind)) natom_of_kind(:) = 0

       nkind = SIZE(atomic_kind_set)
       DO ikind=1,nkind
          atomic_kind => atomic_kind_set(ikind)
          CALL get_atomic_kind(atomic_kind=atomic_kind,&
               fist_potential=fist_potential,&
               shell=shell,&
               damping=damping)
          IF (PRESENT(maxatom)) THEN
             maxatom = MAX(maxatom,atomic_kind%natom)
          END IF
          IF (PRESENT(natom)) THEN
             natom = natom + atomic_kind_set(ikind)%natom
          END IF
          IF (PRESENT(fist_potential_present)) THEN
             IF (ASSOCIATED(fist_potential)) THEN
                fist_potential_present = .TRUE.
             END IF
          END IF
          IF (PRESENT(shell_present)) THEN
             IF (ASSOCIATED(shell)) THEN
                shell_present = .TRUE.
             END IF
          END IF
          IF (PRESENT(shell_adiabatic) .AND. ASSOCIATED(shell) )THEN
             IF(.NOT. shell_adiabatic) shell_adiabatic = (shell%massfrac /= 0.0_dp)
          END IF
          IF (PRESENT(shell_check_distance) .AND. ASSOCIATED(shell) )THEN
             IF(.NOT. shell_check_distance) shell_check_distance = (shell%max_dist > 0.0_dp)
          END IF
          IF (PRESENT(damping_present))THEN
             IF (ASSOCIATED(damping)) THEN
                damping_present=.TRUE.
             END IF
          END IF
          IF (PRESENT(atom_of_kind)) THEN
             DO iatom=1,atomic_kind%natom
                atom_a = atomic_kind%atom_list(iatom)
                atom_of_kind(atom_a) = iatom
             END DO
          END IF
          IF (PRESENT(kind_of)) THEN
             DO iatom=1,atomic_kind%natom
                atom_a = atomic_kind%atom_list(iatom)
                kind_of(atom_a) = ikind
             END DO
          END IF
          IF (PRESENT(natom_of_kind)) THEN
             natom_of_kind(ikind) = atomic_kind_set(ikind)%natom
          END IF
       END DO
    ELSE
       CALL stop_program(routineN,moduleN,__LINE__,&
            "The pointer atomic_kind_set is not associated")
    END IF

  END SUBROUTINE get_atomic_kind_set


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
  SUBROUTINE read_atomic_kind(atomic_kind,kind_section, no_fail, method_id, error)

    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    TYPE(section_vals_type), POINTER         :: kind_section
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
  SUBROUTINE read_atomic_kind_set(atomic_kind_set,kind_section,error)

    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    TYPE(section_vals_type), POINTER         :: kind_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    CHARACTER(len=*), PARAMETER :: routineN = 'read_atomic_kind_set', &
      routineP = moduleN//':'//routineN

    STOP "atomic_kind_types"
  END SUBROUTINE read_atomic_kind_set


! *****************************************************************************
!> \brief Set the components of an atomic kind data set.
! *****************************************************************************
  SUBROUTINE set_atomic_kind(atomic_kind,element_symbol,name,mass,kind_number,&
       natom,atom_list,&
       fist_potential,shell,&
       shell_active,damping,error)

    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    CHARACTER(LEN=*), INTENT(IN), OPTIONAL   :: element_symbol, name
    REAL(KIND=dp), INTENT(IN), OPTIONAL      :: mass
    INTEGER, INTENT(IN), OPTIONAL            :: kind_number, natom
    INTEGER, DIMENSION(:), INTENT(IN), &
      OPTIONAL                               :: atom_list
    TYPE(fist_potential_type), OPTIONAL, &
      POINTER                                :: fist_potential
    TYPE(shell_kind_type), OPTIONAL, POINTER :: shell
    LOGICAL, INTENT(IN), OPTIONAL            :: shell_active
    TYPE(damping_p_type), OPTIONAL, POINTER  :: damping
    TYPE(cp_error_type), INTENT(inout), &
      OPTIONAL                               :: error

    CHARACTER(len=*), PARAMETER :: routineN = 'set_atomic_kind', &
      routineP = moduleN//':'//routineN
    INTEGER                                  :: n, stat
    LOGICAL                                  :: failure

    failure = .FALSE.

    IF (ASSOCIATED(atomic_kind)) THEN

       IF (PRESENT(element_symbol)) atomic_kind%element_symbol = element_symbol
       IF (PRESENT(name)) atomic_kind%name = name
       IF (PRESENT(mass)) atomic_kind%mass = mass
       IF (PRESENT(kind_number)) atomic_kind%kind_number = kind_number
       IF (PRESENT(natom)) atomic_kind%natom = natom
       IF (PRESENT(atom_list)) THEN
          n = SIZE(atom_list)
          IF (n > 0) THEN
             IF (ASSOCIATED(atomic_kind%atom_list)) THEN
                DEALLOCATE (atomic_kind%atom_list,STAT=stat)
                CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
             END IF
             ALLOCATE (atomic_kind%atom_list(n),STAT=stat)
             CPPostcondition(stat==0,cp_failure_level,routineP,error,failure)
             atomic_kind%atom_list(:) = atom_list(:)
             atomic_kind%natom = n
          ELSE
             CALL stop_program(routineN,moduleN,__LINE__,&
                  "An invalid atom_list was supplied")
          END IF
       END IF
       !IF (PRESENT(elec_conf)) THEN
       !   IF (ASSOCIATED(atomic_kind%elec_conf)) THEN
       !      DEALLOCATE(atomic_kind%elec_conf)
       !   ENDIF
       !   ALLOCATE(atomic_kind%elec_conf(0:SIZE(elec_conf)-1))
       !   atomic_kind%elec_conf(:)=elec_conf(:)
       !ENDIF
       !IF (PRESENT(paw_atom)) atomic_kind%paw_atom = paw_atom
       !IF (PRESENT(qm_atom)) atomic_kind%qm_atom = qm_atom
       !IF (PRESENT(hard_radius)) atomic_kind%hard_radius = hard_radius
       !IF (PRESENT(hard0_radius)) atomic_kind%hard0_radius = hard0_radius
       IF (PRESENT(fist_potential)) atomic_kind%fist_potential => fist_potential
       !IF (PRESENT(soft_basis_set)) atomic_kind%soft_basis_set => soft_basis_set
       !IF (PRESENT(hard_basis_set)) atomic_kind%hard_basis_set => hard_basis_set
       !IF (PRESENT(lmax_rho0)) atomic_kind%lmax_rho0 = lmax_rho0
       !IF (PRESENT(zeff)) THEN
       !   IF (ASSOCIATED(atomic_kind%all_potential)) THEN
       !      CALL set_potential(potential=atomic_kind%all_potential,zeff=zeff)
       !   ELSE IF (ASSOCIATED(atomic_kind%gth_potential)) THEN
       !      CALL set_potential(potential=atomic_kind%gth_potential,zeff=zeff)
       !   END IF
       !END IF
       !IF (PRESENT(ghost)) atomic_kind%ghost = ghost
       IF (PRESENT(shell)) THEN
          atomic_kind%shell => shell
          CALL shell_retain(shell,error)
       END IF
       IF (PRESENT(shell_active)) atomic_kind%shell_active=shell_active

       !IF (PRESENT(no_optimize)) atomic_kind%no_optimize=no_optimize

       !IF (PRESENT(dispersion)) atomic_kind%dispersion => dispersion

       IF (PRESENT(damping)) atomic_kind%damping => damping

       !IF (PRESENT(u_minus_j)) THEN
       !   IF (ASSOCIATED(atomic_kind%dft_plus_u)) THEN
       !      atomic_kind%dft_plus_u%u_minus_j = u_minus_j
       !   END IF
       !END IF
       !
       !IF (PRESENT(reltmat)) atomic_kind%reltmat => reltmat

    ELSE
       CALL stop_program(routineN,moduleN,__LINE__,&
            "The pointer atomic_kind is not associated")
    END IF


  END SUBROUTINE set_atomic_kind

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

  
  
  
  
! *****************************************************************************
! *****************************************************************************
  SUBROUTINE locate_kind_section_rep(kind_section, kind_name,element_symbol, no_fail, i_rep, error)
    TYPE(section_vals_type), POINTER                     :: kind_section
    CHARACTER(LEN=default_string_length), INTENT(IN)     :: kind_name, element_symbol
    LOGICAL, INTENT(IN)                                  :: no_fail
    INTEGER, INTENT(out)                                 :: i_rep
    TYPE(cp_error_type), INTENT(inout)       :: error

    CHARACTER(LEN=*), PARAMETER :: routineN = 'locate_kind_section_rep', &
      routineP = moduleN//':'//routineN

    CHARACTER(LEN=default_string_length)                  :: kind_name_upper, keyword
       CHARACTER(LEN=2)                                   :: element_symbol_upper
    INTEGER :: handle, i, ipos, n_rep

    CALL timeset(routineN,handle)

    CALL section_vals_get(kind_section,n_repetition=n_rep,error=error)
    i_rep=-1
    kind_name_upper = kind_name
    CALL uppercase(kind_name_upper)
    ! First we use the atom_name to find out the proper KIND section
    DO i=1,n_rep
       CALL section_vals_val_get(kind_section,"_SECTION_PARAMETERS_",&
            c_val=keyword,i_rep_section=i,error=error)
       CALL uppercase(keyword)
       IF (keyword==kind_name_upper) THEN
          i_rep=i
          EXIT
       END IF
    END DO
    ! The search for the KIND section failed.. check for a QM/MM link atom
    IF (i_rep<1) THEN
       ipos = INDEX(kind_name,"_")
       IF (((ipos == 2).OR.(ipos == 3)).AND.(INDEX(kind_name,"_ghost")==0)) THEN
          ! If the atm_name could not match any KIND section it maybe be a QM/MM link atom.
          ! ghost atoms will be treated differently.
          kind_name_upper = kind_name(1:ipos-1)
          CALL uppercase(kind_name_upper)
          DO i=1,n_rep
             CALL section_vals_val_get(kind_section,"_SECTION_PARAMETERS_",&
                  c_val=keyword,i_rep_section=i,error=error)
             CALL uppercase(keyword)
             IF (keyword==kind_name_upper) THEN
                i_rep=i
                EXIT
             END IF
          END DO
       END IF
    END IF
    ! The search for the KIND section failed.. check element_symbol
    IF (i_rep<1) THEN
       ! If it's not a link atom let's check for the element and map
       ! the KIND section to the element.
       element_symbol_upper = element_symbol(1:2)
       CALL uppercase(element_symbol_upper)
       DO i=1,n_rep
          CALL section_vals_val_get(kind_section,"_SECTION_PARAMETERS_",&
               c_val=keyword,i_rep_section=i,error=error)
          CALL uppercase(keyword)
          IF (keyword==element_symbol_upper) THEN
             i_rep=i
             EXIT
          END IF
       END DO
    END IF
    ! In case it should not really match any possible KIND section
    ! let's look if a default one is defined..
    IF (i_rep<1) THEN
       DO i=1,n_rep
          CALL section_vals_val_get(kind_section,"_SECTION_PARAMETERS_",&
               c_val=keyword,i_rep_section=i,error=error)
          CALL uppercase(keyword)
          IF (keyword=="DEFAULT") THEN
             i_rep=i
             EXIT
          END IF
       END DO
    END IF
    IF (i_rep<0.AND.(.NOT.no_fail)) THEN
       CALL mp_abort("No &KIND section was possible to associate to the atomic kind <"//&
            TRIM(kind_name_upper)//">. The KIND section were also scanned for the"//&
            " corresponding element <"//TRIM(element_symbol)//"> "//&
            " and for the DEFAULT section but no match was found. Check your input file!")
    END IF

    CALL timestop(handle)

  END SUBROUTINE locate_kind_section_rep

END MODULE atomic_kind_types
