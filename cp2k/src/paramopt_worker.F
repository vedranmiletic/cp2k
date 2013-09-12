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
                                             get_atomic_kind
  USE cp_control_types,                ONLY: dft_control_type,&
                                             qs_control_type,&
                                             scptb_control_type
  USE cp_dbcsr_interface,              ONLY: cp_dbcsr_get_info
  USE cp_dbcsr_types,                  ONLY: cp_dbcsr_p_type,&
                                             cp_dbcsr_type
  USE cp_output_handling,              ONLY: cp_print_key_unit_nr
  USE cp_para_types,                   ONLY: cp_para_env_type
  USE cp_subsys_types,                 ONLY: cp_subsys_type
  USE f77_interface,                   ONLY: calc_energy_force,&
                                             create_force_env,&
                                             destroy_force_env,&
                                             f_env_add_defaults,&
                                             f_env_rm_defaults,&
                                             f_env_type,&
                                             get_energy,&
                                             get_natom, &
                                             get_pos, &
                                             calc_energy
  USE force_env_methods,               ONLY: ep_create_force_env,&
                                             force_env_calc_energy_force
  USE force_env_types,                 ONLY: force_env_get,&
                                             force_env_type

  USE geo_opt,                         ONLY: cp_geo_opt
  USE global_types,                    ONLY: global_environment_type
  USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                             section_vals_get,&
                                             section_vals_val_set,&
                                             section_vals_val_unset,&
                                             section_vals_val_get,&
                                             section_vals_type, &
                                             section_release, &
                                             section_get_subsection_index, &
                                             section_vals_remove_values
  USE kinds,                           ONLY: default_string_length,&
                                             default_path_length,&
                                             dp
  USE message_passing,                 ONLY: mp_comm_split,&
                                             mp_sum
  USE mulliken,                        ONLY: compute_multipoles
  USE particle_types,                  ONLY: particle_type
  USE qs_environment_types,            ONLY: get_qs_env,&
                                             qs_environment_type
  USE qs_rho_types,                    ONLY: qs_rho_type
  USE scp_environment_types,           ONLY: scp_environment_type
  USE scptb_types,                     ONLY: get_scptb_parameter,&
                                             scptb_parameter_type,&
                                             set_scptb_parameter
  USE swarm_message,                   ONLY: swarm_message_add,&
                                             swarm_message_get,&
                                             swarm_message_type
#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'paramopt_worker'

 PUBLIC :: paramopt_worker_init, paramopt_worker_finalize
 PUBLIC :: paramopt_worker_execute
 PUBLIC :: paramopt_worker_type

  TYPE paramopt_worker_type
   PRIVATE
   CHARACTER(len=default_path_length)               :: input_path
   INTEGER                                          :: id
   INTEGER                                          :: pid
   INTEGER                                          :: iw
   INTEGER                                          :: num_var
   INTEGER                                          :: group
   TYPE(cp_error_type)                              :: error
   TYPE(cp_para_env_type), POINTER                  :: para_env
   REAL(KIND=dp),ALLOCATABLE                            :: x(:)
   REAL(kind=dp)                                    :: Epot
   REAL(kind=dp)                                    :: timings
   REAL(kind=dp),DIMENSION(3,3)                     :: dipole
   REAL(kind=dp)                                    :: rmsd
   INTEGER                                          :: f_env_id
   TYPE(f_env_type), POINTER                        :: f_env
   TYPE(force_env_type), POINTER                    :: force_env
   TYPE(cp_subsys_type), POINTER                    :: subsys
   TYPE(section_vals_type), POINTER                 :: subsys_section
   TYPE(section_vals_type), POINTER                 :: topology_section
   TYPE(section_vals_type), POINTER                 :: print_section
   TYPE(global_environment_type), POINTER           :: globenv
   TYPE(section_vals_type), POINTER                 :: root_section
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
   REAL(KIND=dp),DIMENSION(:),ALLOCATABLE           :: pos,pos2
   REAL(dp), DIMENSION(3, 3)                        :: U
   REAL(dp), DIMENSION(3)                           :: center1, center2
   REAL(dp), ALLOCATABLE, DIMENSION(:, :)           :: g
   LOGICAL                                          :: calc_g = .FALSE.
   INTEGER                                          :: nkind
   INTEGER, DIMENSION(0:3)                          :: norb

 END TYPE paramopt_worker_type


 CONTAINS




! *****************************************************************************
! *****************************************************************************
   SUBROUTINE paramopt_worker_init(this, para_env, root_section,&
                 input_path, worker_id, iw, error)
    TYPE(paramopt_worker_type), &
      INTENT(INOUT)                          :: this
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: root_section
    CHARACTER(LEN=*), INTENT(IN)             :: input_path
    INTEGER, INTENT(in)                      :: worker_id, iw
    TYPE(cp_error_type), INTENT(inout)       :: error

    CHARACTER(len=default_string_length)     :: output_path, str
    INTEGER                                  :: i, ierr, n_groups_created
    TYPE(cp_logger_type), POINTER            :: logger

    LOGICAL                                  :: failure,restart
    INTEGER, DIMENSION(:), POINTER           :: group_distribution_p
    INTEGER, &
    DIMENSION(0:para_env%num_pe-1), TARGET :: group_distribution


        failure = .FALSE.
        this%iw = iw
        this%para_env => para_env
        this%group = this%para_env%group
        this%root_section => root_section
        this%error = error
        this%id = worker_id
        this%pid = this%para_env%mepos+1
        this%input_path = input_path
        !---------------
        this%num_var = 16
        allocate(this%x(this%num_var))
        this%x = 0._dp
        this%Epot = 0._dp
        this%timings = 0._dp
        this%rmsd = 0._dp
        IF (this%iw>0) WRITE(this%iw,*) "paramopt_worker_init called."
        IF (this%iw>0) WRITE(this%iw,*) "paramopt_worker_init done."
  END SUBROUTINE paramopt_worker_init



! *****************************************************************************
! *****************************************************************************
   SUBROUTINE paramopt_worker_finalize(this)
    TYPE(paramopt_worker_type), &
      INTENT(INOUT)                          :: this

    INTEGER                                  :: ierr

     IF (this%iw>0) WRITE(this%iw,*) "paramopt_worker_finalize called"

     WRITE(*,*) "paramopt_worker_finalize called. worker_id=", this%id, &
      " rank=",this%para_env%mepos 
!     DEALLOCATE(this%pos)
     CALL f_env_rm_defaults(this%f_env, this%error)
     CALL destroy_force_env(this%f_env_id, ierr)
     IF(ierr /= 0) STOP "paramopt_worker_finalize: destroy_force_env failed"
   END SUBROUTINE paramopt_worker_finalize


! *****************************************************************************
! *****************************************************************************
   SUBROUTINE paramopt_worker_execute(this, cmd, report)
    TYPE(paramopt_worker_type), &
      INTENT(INOUT)                          :: this
    TYPE(swarm_message_type), INTENT(IN)     :: cmd
    TYPE(swarm_message_type), INTENT(INOUT)  :: report

    CHARACTER(len=default_string_length)     :: command, iname
    CHARACTER(len=default_string_length)     :: output_path, input_file, coord_file
    INTEGER                                  :: clock_max, clock_rate, ierr, &
                                                ikind, t1, t2, i,j
    INTEGER                                  :: iter,i_inp,charge
    INTEGER, DIMENSION(0:3)                  :: norb
    LOGICAL                                  :: failure,explicit
    REAL(KIND=dp)                            :: ag, energy
    REAL(KIND=dp), DIMENSION(10, 0:3)        :: hcore, zeta
    REAL(KIND=dp), DIMENSION(3)              :: crep
    REAL(KIND=dp), DIMENSION(1:3)            :: pol
    REAL(KIND=dp), DIMENSION(:),allocatable  :: pos
    integer                                  :: natoms,test
    TYPE(atomic_kind_type), POINTER          :: atomic_kind
    TYPE(section_vals_type), POINTER         :: work_section

    CALL swarm_message_get(cmd, "command", command)

    if (command .eq. 'energy' .or. command .eq. 'geo_opt') then
    CALL swarm_message_get(cmd,"input",input_file)
    CALL swarm_message_get(cmd,"coord",coord_file)
    CALL swarm_message_get(cmd,"charge",charge)
    CALL swarm_message_get(cmd,"num_inp",i_inp)
    CALL swarm_message_get(cmd,"iter",iter)
    CALL swarm_message_get(cmd,"x",this%x)

    !debug
         j = mod(iter,17) + 1
         i = ceiling(real(iter)/real(17))


    write(output_path,*) this%id
    output_path=trim(adjustl(output_path))//".out"
    if (this%iw > 0 ) then
    !print *, this%id,i,j
    !print *, this%x
    endif
!    if (this%iw > 0 ) print *,'before set coordinate file name',trim(coord_file),charge
    CALL section_vals_val_set(this%root_section, &
                              "FORCE_EVAL%SUBSYS%TOPOLOGY%COORD_FILE_NAME", &
                              c_val=coord_file, error = this%error)


!    CALL section_vals_val_get(this%root_section, &
!                              "FORCE_EVAL%SUBSYS%COORD%UNIT", &
!                              c_val=test, error = this%error)
!    print *, "asdfasdf",test
    CALL section_vals_val_set(this%root_section, &
                              "FORCE_EVAL%DFT%CHARGE", &
                              i_val=charge, error = this%error)
!    if (this%iw > 0 )  print *,'before force env create'
!    CALL section_vals_val_get(this%root_section, &
!                              "FORCE_EVAL%SUBSYS%TOPOLOGY%COORD_FILE_NAME", &
!                              c_val=coord_file, error = this%error)
!    CALL section_vals_val_set(this%root_section, &
!                              "FORCE_EVAL%SUBSYS%TOPOLOGY%NUMBER_OF_ATOMS", &
!                              i_val=-1,error=this%error)
!    CALL section_vals_val_get(this%root_section, &
!                              "FORCE_EVAL%SUBSYS%TOPOLOGY%NUMBER_OF_ATOMS", &
!                              i_val=this%n_atoms,error=this%error)
!    if (this%iw > 0 ) print *,'after set',coord_file,this%n_atoms
    !-------------------create force env------------------------
!    CALL create_force_env(this%f_env_id, &
!                         input_path=TRIM(ADJUSTL(input_file)), &
!                         output_path=output_path, &
!                         mpi_comm=this%group, &
!                         ierr=ierr)
!    nullify(this%topology_section,atomic_kind,this%subsys_section)
!    nullify(this%atomic_kind_set)
!    nullify(this%force_env,this%qs_env,this%dft_section)
!    nullify(this%qs_section,this%qs_control)
!    nullify(this%scptb_control)
    CALL create_force_env(this%f_env_id, &
                            input_path=this%input_path, &
                            input=this%root_section, &
                            output_path=output_path, &
                            mpi_comm=this%group, &
                            ierr=ierr)
    CALL get_natom(this%f_env_id,this%n_atoms,ierr)
!    if (this%iw > 0 )  print *,'force env created',this%n_atoms,trim(output_path)
    IF (ierr .ne. 0) THEN
        print *, "Create force env failed",trim(input_file)
        STOP
    ENDIF
    !------------------------------------------------------------
    ! not sure if nullify is needed ? 
!    Nullify(this%f_env)
!    nullify(this%force_env)
!    nullify(this%qs_env)
!    nullify(this%scptb_parameter)
!    nullify(atomic_kind)
    CALL f_env_add_defaults(this%f_env_id, this%f_env, this%error, failure)
    IF(failure) STOP "paramopt_drive_worker: f_env_add_defaults failed"
      this%force_env => this%f_env%force_env 
      this%qs_env => this%force_env%qs_env

      CALL force_env_get(this%force_env, globenv=this%globenv, subsys=this%subsys, &
                         para_env=this%para_env, error=this%error)
                    !dft section
      this%dft_section =>  section_vals_get_subs_vals(this%qs_env%input, "DFT", error = this%error )
      CALL get_qs_env(this%qs_env, dft_control=this%dft_control, &
                      atomic_kind_set=this%atomic_kind_set,error=this%error)
                    ! here more things should be added 
      
                    !qs section 
      this%qs_section => section_vals_get_subs_vals(this%dft_section, "QS", error = this%error )
      this%qs_control => this%dft_control%qs_control
      
                    ! scptb section
      this%scptb_control => this%qs_control%scptb_control
      this%nkind = SIZE(this%atomic_kind_set)
      this%subsys_section => section_vals_get_subs_vals(this%force_env%force_env_section, &
                                                        "SUBSYS",error=this%error)
      this%topology_section => section_vals_get_subs_vals(this%subsys_section, &
                                                        "TOPOLOGY",error=this%error)  
      this%print_section => section_vals_get_subs_vals(this%subsys_section, &
                                                        "PRINT",error=this%error)
!    !----------------update parameters---------------------------
    !==================================================================
       DO ikind=1,this%nkind
         atomic_kind => this%atomic_kind_set(ikind)
         CALL get_atomic_kind(atomic_kind=atomic_kind, &
                               name=iname, scptb_parameter=this%scptb_parameter)
         !----------update parameters-------------------------------------
         CALL get_scptb_parameter(this%scptb_parameter,norb=this%norb,hcore=hcore, &
                                  energy=energy, ag=ag, zeta=zeta, crep=crep, pol=pol)
         !print *,pol
         !print *,'wid',this%id
         !IF (this%para_env%ionode) print *,'first get hcore',ikind,hcore(1:2,0:1)
         !IF (this%para_env%ionode) print *,'first get zeta',ikind,zeta(1:2,0:1)
         !IF (this%para_env%ionode) print *,'first get energy',ikind,energy,this%x(1),this%x(8)
         !-----------map x to parameter----------------------------------- 
         IF (iname .EQ. 'H') THEN
           ag = this%x(1)
           zeta(1,0) = this%x(2)
           hcore(1,0) = this%x(3)
           pol(1)  = this%x(4)
           crep(1) = this%x(5)
           crep(2) = this%x(6)
           crep(3) = this%x(7)
         ELSE IF (iname .EQ. 'O') THEN
           ag = this%x(8)
           zeta(1,0) = this%x(9)
           hcore(1,0) = this%x(10)
           zeta(1,1) = this%x(11)
           hcore(1,1) = this%x(12)
           pol(1) = this%x(13)
           pol(2) = this%x(14)
           crep(1) = this%x(15)
           crep(2) = this%x(16)
           crep(3) = this%x(17)
         ELSE
            PRINT *,'failure'
         ENDIF
         CALL set_scptb_parameter(this%scptb_parameter,hcore=hcore, &
                                  energy=energy, ag=ag, zeta=zeta, crep=crep, pol=pol)
         !CALL get_scptb_parameter(this%scptb_parameter,norb=this%norb,hcore=hcore,zeta=zeta)
         !IF (this%para_env%ionode) print *,'second get',ikind,hcore(1:2,0:1)
         !IF (this%para_env%ionode) print *,'second get zeta',ikind,zeta(1:2,0:1)
         !IF (this%para_env%ionode) print *,'second get energy',ikind,energy
       END DO
      !================================================================== 
    endif

    if (command .eq. 'energy') then
      CALL calc_energy_force(this%f_env_id, .FALSE. , ierr)
      CALL get_energy(this%f_env_id,energy,ierr)
      if (this%iw > 0) then
     ! print * ,'in worker',input_file
     ! print * ,this%x
     ! print * ,'energy',energy 
      endif
      !-------------report--------------------------------
      CALL swarm_message_add(report, "status", "ene_calc") 
      CALL swarm_message_add(report, "num_inp", i_inp) 
      CALL swarm_message_add(report, "iter", iter) 
      CALL swarm_message_add(report, "Epot", energy)

    else if (command .eq. 'geo_opt') then
        
      ALLOCATE(this%pos2(3*this%n_atoms))
      ALLOCATE(this%pos(3*this%n_atoms))
      ALLOCATE(this%g(2,3*this%n_atoms))
      CALL cp_get_pos(this%f_env_id, this%pos, 3*this%n_atoms, ierr)

      CALL cp_geo_opt(this%force_env, this%globenv, eval_opt_geo=.FALSE.,error=this%error)
      CALL cp_get_pos(this%f_env_id, this%pos2, 3*this%n_atoms, ierr)
      CALL rmsd(this%n_atoms,RESHAPE(this%pos,(/ 3, this%n_atoms /)), &
                             RESHAPE(this%pos2,(/ 3, this%n_atoms /) ), &
                             0, this%U,this%center2,this%center1, this%rmsd,this%calc_g,this%g)
      work_section => section_vals_get_subs_vals(this%subsys_section,"COORD",error=this%error)
      CALL section_vals_remove_values(work_section, this%error)
      !CALL section_vals_remove_values(this%subsys_section%coord,this%error)
      !CALL cp_set_pos(this%f_env_id, this%pos, 3*this%n_atoms, ierr)
      !CALL section_vals_get(this%subsys_section%coord,explicit=explicit,error=this%error)
      !print *,'adfadfad',explicit
!       CALL section_vals_val_unset(this%force_env%force_env_section,"SUBSYS%COORD",error=this%error)
!      CALL section_release(this%force_env%force_env_section%subsys_section, this%error)
!      CALL cp_subsys_release(this%subsys, this%error)
 
!      if (this%iw > 0 ) print *,'rmsd is doooooooone',this%rmsd
      deallocate(this%pos,this%pos2,this%g)
      
          !-------------report--------------------------------
      CALL swarm_message_add(report, "status", "rmsd_calc") 
      CALL swarm_message_add(report, "num_inp", i_inp) 
      CALL swarm_message_add(report, "iter", iter) 
      CALL swarm_message_add(report, "rmsd", this%rmsd)
    
     
    endif
    !!---------------cleanup 
    CALL f_env_rm_defaults(this%f_env, this%error)
    CALL destroy_force_env(this%f_env_id, ierr)
    IF(ierr /= 0) STOP "paramopt_worker: destroy_force_env failed"
   END SUBROUTINE paramopt_worker_execute



SUBROUTINE rmsd(n, coord1, coord2, option, U, x_center, y_center, & 
     error, calc_g, g)
!-----------------------------------------------------------------------
!  This subroutine calculates the least square rmsd of two coordinate
!  sets coord1(3,n) and coord2(3,n) using a method based on quaternion.
!  If option=1, then the rotation matrix U and the centers of coord are 
!  returned.
!-----------------------------------------------------------------------
! if calc_g == .true., derivative of RMSD with respect to coord1
! is returned
!-----------------------------------------------------------------------

    INTEGER, INTENT(in)                      :: n
    REAL(dp), DIMENSION(:, :), INTENT(in)    :: coord1, coord2
    INTEGER, INTENT(in)                      :: option
    REAL(dp), DIMENSION(:, :), INTENT(out)   :: U
    REAL(dp), DIMENSION(3), INTENT(out)      :: x_center, y_center
    REAL(dp), INTENT(out)                    :: error
    LOGICAL, INTENT(in)                      :: calc_g
    REAL(dp), INTENT(out)                    :: g(:,:)

    INTEGER                                  :: i, j
    REAL(dp)                                 :: lambda, tmp(3), x_norm, y_norm
    REAL(dp), DIMENSION(3, 3)                :: Rmatrix
    REAL(dp), DIMENSION(3, n)                :: x, y
    REAL(dp), DIMENSION(4)                   :: q
    REAL(dp), DIMENSION(4, 4)                :: S
    REAL(dp), DIMENSION(n)                   :: xi, yi

! make copies of the original coordinates

  x(:,1:n) = coord1(:,1:n)
  y(:,1:n) = coord2(:,1:n)

  ! calculate the barycenters, centroidal coordinates, and the norms
  x_norm = 0.0d0
  y_norm = 0.0d0
  DO i = 1, 3
     xi(:) = x(i,:)
     yi(:) = y(i,:)
     x_center(i) = SUM(xi(1:n))/DBLE(n)
     y_center(i) = SUM(yi(1:n))/DBLE(n)
     xi(:) = xi(:) - x_center(i)
     yi(:) = yi(:) - y_center(i)
     x(i,:) = xi(:)
     y(i,:) = yi(:)
     x_norm = x_norm + DOT_PRODUCT(xi, xi)
     y_norm = y_norm + DOT_PRODUCT(yi, yi)
  END DO

  ! calculate the R matrix
  DO i = 1, 3
     DO j = 1, 3
        Rmatrix(i,j) = DOT_PRODUCT(x(i,:),y(j,:))
     END DO
  END DO

  ! S matrix
  S(1, 1) = Rmatrix(1, 1) + Rmatrix(2, 2) + Rmatrix(3, 3)
  S(2, 1) = Rmatrix(2, 3) - Rmatrix(3, 2)
  S(3, 1) = Rmatrix(3, 1) - Rmatrix(1, 3)
  S(4, 1) = Rmatrix(1, 2) - Rmatrix(2, 1)

  S(1, 2) = S(2, 1)
  S(2, 2) = Rmatrix(1, 1) - Rmatrix(2, 2) - Rmatrix(3, 3)
  S(3, 2) = Rmatrix(1, 2) + Rmatrix(2, 1)
  S(4, 2) = Rmatrix(1, 3) + Rmatrix(3, 1)

  S(1, 3) = S(3, 1)
  S(2, 3) = S(3, 2)
  S(3, 3) =-Rmatrix(1, 1) + Rmatrix(2, 2) - Rmatrix(3, 3)
  S(4, 3) = Rmatrix(2, 3) + Rmatrix(3, 2)

  S(1, 4) = S(4, 1)
  S(2, 4) = S(4, 2)
  S(3, 4) = S(4, 3)
  S(4, 4) =-Rmatrix(1, 1) - Rmatrix(2, 2) + Rmatrix(3, 3) 

  ! Calculate eigenvalues and eigenvectors, and 
  ! take the maximum eigenvalue lambda and the corresponding eigenvector q.
  CALL dstmev(S, lambda, q)

  IF (option == 1) THEN
     ! convert quaternion q to rotation matrix U
     CALL rotation_matrix(q, U)
  END IF

  ! RMS Deviation
  error = SQRT(MAX(0.0d0,((x_norm+y_norm)-2.0d0*lambda))/DBLE(n))

  IF (calc_g) THEN
     DO i = 1, n
        DO j = 1, 3
           tmp(:) = MATMUL(TRANSPOSE(U(:,:)), y(:,i))
           g(j,i) = (x(j,i) - tmp(j))/(error*DBLE(n))
        END DO
     END DO
  END IF

END SUBROUTINE rmsd
!-----------------------------------------------------------------------
SUBROUTINE rotation_matrix(q, U)
!-----------------------------------------------------------------------
! This subroutine constructs rotation matrix U from quaternion q.
!-----------------------------------------------------------------------

    REAL(dp), DIMENSION(:), INTENT(in)       :: q
    REAL(dp), DIMENSION(:, :), INTENT(out)   :: U

    REAL(dp)                                 :: b0, b1, b2, b3, q0, q00, q01, &
                                                q02, q03, q1, q11, q12, q13, &
                                                q2, q22, q23, q3, q33

  q0 = q(1)
  q1 = q(2)
  q2 = q(3)
  q3 = q(4)

  b0 = 2.0d0*q0
  b1 = 2.0d0*q1
  b2 = 2.0d0*q2
  b3 = 2.0d0*q3

  q00 = b0*q0-1.0d0
  q01 = b0*q1
  q02 = b0*q2
  q03 = b0*q3

  q11 = b1*q1
  q12 = b1*q2
  q13 = b1*q3  

  q22 = b2*q2
  q23 = b2*q3

  q33 = b3*q3 

  U(1,1) = q00+q11
  U(1,2) = q12-q03
  U(1,3) = q13+q02

  U(2,1) = q12+q03
  U(2,2) = q00+q22
  U(2,3) = q23-q01

  U(3,1) = q13-q02
  U(3,2) = q23+q01
  U(3,3) = q00+q33

END SUBROUTINE rotation_matrix
!-----------------------------------------------------------------------
SUBROUTINE DSTMEV(A,lambda,evec)
!-----------------------------------------------------------------------
! a simple subroutine to compute the leading eigenvalue and eigenvector
! of a symmetric, traceless 4x4 matrix A by an inverse power iteration:
! (1) the matrix is converted to tridiagonal form by 3 Givens
! rotations;  V*A*V' = T
! (2) Gershgorin's theorem is used to estimate a lower
! bound for the leading negative eigenvalue:
! lambda_1 > g=min(T11-t12,-t21+T22-t23,-t32+T33-t34,-t43+T44)
!          =
! where tij=abs(Tij)
! (3) Form the positive definite matrix 
!     B = T-gI
! (4) Use svd (algorithm svdcmp from "Numerical Recipes")
!     to compute eigenvalues and eigenvectors for SPD matrix B
! (5) Shift spectrum back and keep leading singular vector
!     and largest eigenvalue.
! (6) Convert eigenvector to original matrix A, through 
!     multiplication by V'.  
!-----------------------------------------------------------------------
    REAL(dp), DIMENSION(4, 4)                :: A
    REAL(dp)                                 :: lambda
    REAL(dp), DIMENSION(4)                   :: evec

    INTEGER                                  :: i
    INTEGER, DIMENSION(1)                    :: max_loc
    REAL(dp), DIMENSION(4)                   :: SW
    REAL(dp), DIMENSION(4, 4)                :: SV, T, V
    REAL(dp), DIMENSION(8)                   :: rv1

! must be an array
!-----------------------------------------------------------------------
!(I).   Convert to tridiagonal form, keeping similarity transform
! (a product of 3 Givens rotations)

  CALL givens4(A,T,V)

  !-----------------------------------------------------------------------
  !(II)Estimate lower bound of smallest eigenvalue by Gershgorin's theorem
  lambda=MIN(T(1,1)-ABS(T(1,2)),-ABS(T(2,1))+T(2,2)-ABS(T(2,3)),&
       -ABS(T(3,2))+T(3,3)-ABS(T(3,4)),-ABS(T(4,3))+T(4,4))
  !-----------------------------------------------------------------------
  !(III). Form positive definite matrix     T <== lambda*I - T
  DO i = 1,4
     T(i,i) = T(i,i)-lambda
  ENDDO
  !-----------------------------------------------------------------------
  !(IV). Compute singular values/vectors of SPD matrix B
  CALL svdcmp(4,T,4,4,SW,SV,rv1)
  !-----------------------------------------------------------------------
  !(V). Shift spectrum back
  max_loc = MAXLOC(SW) 
  lambda = SW(max_loc(1)) + lambda
  !lambda = SW(1) + lambda
  !-----------------------------------------------------------------------
  !(VI). Convert eigenvector to original coordinates: (V is transposed!)
  evec = MATMUL(V,SV(:,max_loc(1)))
  !write(*,*)'-----------------------------------------------------------'
  !write(*,*) 'lambda = ', lambda,'  eigenvector:  '
  !write(*,99) evec
  !write(*,*)'-----------------------------------------------------------'
  !-----------------------------------------------------------------------
  !99 format(1x,4(d19.13,1x))

END SUBROUTINE dstmev
!-----------------------------------------------------------------------
SUBROUTINE givens4(S,T,V)
!-----------------------------------------------------------------------
    REAL(dp), DIMENSION(4, 4), INTENT(in)    :: S
    REAL(dp), DIMENSION(4, 4), INTENT(out)   :: T, V

    REAL(dp)                                 :: c1, c1c2, c2, c3, r1, r2, r3, &
                                                s1, s1c2, s2, s3

!double precision :: pythag
! external        pythag
!performs givens rotations to reduce symmetric 4x4 matrix to tridiagonal

  T=S; V = 0.d0
  !-----------------------------------------------------------------------
  !Zero out entries T(4,1) and T(1,4)
  ! compute cos and sin of rotation angle in the 3-4 plane
  r1 = pythag(T(3,1),T(4,1))
  IF(r1 .NE. 0.d0) THEN
     c1 = T(3,1)/r1; s1 = T(4,1)/r1
     V(3,3) = c1  ; V(3,4) = s1
     V(4,3) =-s1  ; V(4,4) = c1
     T(3,1) = r1  ; T(4,1) = 0.d0
     T(3:4,2:4) = MATMUL(V(3:4,3:4),T(3:4,2:4))
     T(1:2,3:4) = TRANSPOSE(T(3:4,1:2))
     T(3:4,3:4) = MATMUL(T(3:4,3:4),TRANSPOSE(V(3:4,3:4)))
  ELSE
     c1 = 1.d0; s1 = 0.d0
  ENDIF
  !-----------------------------------------------------------------------
  !Zero out entries T(3,1) and T(1,3)
  ! compute cos and sin of rotation angle in the 2-3 plane
  r2 = pythag(T(3,1), T(2,1))
  IF(r2 .NE. 0.d0) THEN
     c2 = T(2,1)/r2; s2 = T(3,1)/r2
     V(2,2) = c2  ; V(2,3) = s2
     V(3,2) =-s2  ; V(3,3) = c2
     T(2,1) = r2  ; T(3,1) = 0.d0
     T(2:3,2:4) = MATMUL(V(2:3,2:3),T(2:3,2:4))
     T(1,2:3)   = T(2:3,1);  T(4,2:3) = T(2:3,4)
     T(2:3,2:3) = MATMUL(T(2:3,2:3),TRANSPOSE(V(2:3,2:3)))
  ELSE
     c2 = 1.d0; s2 = 0.d0
  ENDIF
  !-----------------------------------------------------------------------
  !Zero out entries T(4,2) and T(2,4)
  ! compute cos and sin of rotation angle in the 3-4 plane
  r3 = pythag(T(4,2), T(3,2))
  IF(r3 .NE. 0.d0) THEN
     c3 = T(3,2)/r3; s3 = T(4,2)/r3
     V(3,3) = c3  ; V(3,4) = s3
     V(4,3) =-s3  ; V(4,4) = c3
     T(3,2) = r3  ; T(4,2) = 0.d0
     T(3:4,3:4) = MATMUL(V(3:4,3:4),T(3:4,3:4))
     T(1:2,3:4) = TRANSPOSE(T(3:4,1:2))
     T(3:4,3:4) = MATMUL(T(3:4,3:4),TRANSPOSE(V(3:4,3:4)))
  ELSE
     c3 = 1.d0; s3 = 0.d0
  ENDIF
  !-----------------------------------------------------------------------
  !Compute net rotation matrix (accumulate similarity for evec. computation)
  ! To save transposing later, This is the transpose!
  V(1,1)=1.d0; V(1,2:4) = 0.d0; V(2:4,1) = 0.d0
  V(2,2) = c2;  V(3,2) = c1*s2 ; V(4,2) = s1*s2; c1c2 = c1*c2; s1c2=s1*c2
  V(2,3) = -s2*c3 ; V(3,3) = c1c2*c3-s1*s3 ; V(4,3) =  s1c2*c3+c1*s3
  V(2,4) =  s2*s3 ; V(3,4) =-c1c2*s3-s1*c3 ; V(4,4) = -s1c2*s3+c1*c3
  !-----------------------------------------------------------------------
  !write(*,*) (V(1:4,i) - W(1:4,i),i=1,4)
END SUBROUTINE givens4
!-----------------------------------------------------------------------
SUBROUTINE svdcmp(mmax,a,m,n,w,v,rv1)
!-----------------------------------------------------------------------
    INTEGER                                  :: mmax
    REAL(dp)                                 :: a(mmax,*)
    INTEGER                                  :: m, n
    REAL(dp)                                 :: w(*), v(mmax,*), rv1(*)

    INTEGER                                  :: i, its, j, jj, k, l, nm
    REAL(dp)                                 :: anorm, c, f, g, h, s, scale, &
                                                x, y, z

!,pythag

  g = 0.0d0
  scale = 0.0d0
  anorm = 0.0d0
  DO i = 1, n
     l=i+1
     rv1(i)=scale*g
     g=0.0d0
     s=0.0d0
     scale=0.0d0
     IF(i.le.m)THEN
        DO k=i,m
           scale=scale+ABS(a(k,i))
        END DO
        IF(scale.ne.0.0d0)THEN
           DO k=i,m
              a(k,i)=a(k,i)/scale
              s=s+a(k,i)*a(k,i)
           END DO
           f=a(i,i)
           g=-SIGN(SQRT(s),f)
           h=f*g-s
           a(i,i)=f-g
           DO j=l,n 
              s=0.0d0
              DO k=i,m
                 s=s+a(k,i)*a(k,j)
              END DO
              f=s/h
              DO k=i,m
                 a(k,j)=a(k,j)+f*a(k,i)
              END DO
           END DO
           DO k=i,m
              a(k,i)=scale*a(k,i)
           END DO
        ENDIF
     ENDIF
     w(i)=scale *g
     g=0.0d0
     s=0.0d0
     scale=0.0d0
     IF((i.le.m).AND.(i.ne.n))THEN
        DO k=l,n
           scale=scale+ABS(a(i,k))
        END DO
        IF(scale.ne.0.0d0)THEN
           DO k=l,n
              a(i,k)=a(i,k)/scale
              s=s+a(i,k)*a(i,k)
           END DO
           f=a(i,l)
           g=-SIGN(SQRT(s),f)
           h=f*g-s
           a(i,l)=f-g
           DO k=l,n
              rv1(k)=a(i,k)/h
           END DO
           DO j=l,m
              s=0.0d0
              DO k=l,n
                 s=s+a(j,k)*a(i,k)
              END DO
              DO k=l,n
                 a(j,k)=a(j,k)+s*rv1(k)
              END DO
           END DO
           DO k=l,n
              a(i,k)=scale*a(i,k)
           END DO
        ENDIF
     ENDIF
     anorm=MAX(anorm,(ABS(w(i))+ABS(rv1(i))))
  END DO

  DO i = n, 1, -1
     IF(i .LT. n) THEN
        IF(g.ne.0.0d0)THEN
           DO j=l,n
              v(j,i)=(a(i,j)/a(i,l))/g
           END DO
           DO j=l,n
              s=0.0d0
              DO k=l,n
                 s=s+a(i,k)*v(k,j)
              END DO
              DO k=l,n
                 v(k,j)=v(k,j)+s*v(k,i)
              END DO
           END DO
        ENDIF
        DO j=l,n
           v(i,j)=0.0d0
           v(j,i)=0.0d0
        END DO
     ENDIF
     v(i,i)=1.0d0
     g=rv1(i)
     l=i
  END DO
!-----------------------------------------------------------------------
  DO i = MIN(m,n), 1, -1
     l=i+1
     g=w(i)
     DO j=l,n
        a(i,j)=0.0d0
     END DO
     IF(g.ne.0.0d0)THEN
        g=1.0d0/g
        DO j=l,n
           s=0.0d0
           DO k=l,m
              s=s+a(k,i)*a(k,j)
           END DO
           f=(s/a(i,i))*g
           DO k=i,m
              a(k,j)=a(k,j)+f*a(k,i)
           END DO
        END DO
        DO j=i,m
           a(j,i)=a(j,i)*g
        END DO
     ELSE
        DO j= i,m
           a(j,i)=0.0d0
        END DO
     ENDIF
     a(i,i)=a(i,i)+1.0d0
  END DO
!-----------------------------------------------------------------------
  DO k=n,1,-1
     DO its=1,30
        DO l=k,1,-1
           nm=l-1
           IF((ABS(rv1(l))+anorm).eq.anorm)  GOTO 2
           IF((ABS(w(nm))+anorm).eq.anorm)  GOTO 1
        END DO
1       c=0.0d0
        s=1.0d0
        DO i=l,k
           f=s*rv1(i)
           rv1(i)=c*rv1(i)
           IF((ABS(f)+anorm).eq.anorm) GOTO 2
           g=w(i)
           h=pythag(f,g)
           w(i)=h
           h=1.0d0/h
           c= (g*h)
           s=-(f*h)
           DO j=1,m     
              y=a(j,nm)
              z=a(j,i)
              a(j,nm)=(y*c)+(z*s)
              a(j,i)=-(y*s)+(z*c)
           END DO
        END DO
2       z=w(k)
        IF(l .EQ. k)THEN
           IF(z.lt.0.0d0)THEN
              w(k)=-z
              DO j=1,n
                 v(j,k)=-v(j,k)
              END DO
           ENDIF
           GOTO 3
        ENDIF
        IF(its.eq.30) THEN
           WRITE(*,*) 'no convergence in svdcmp'
           STOP
        ENDIF
        x=w(l)
        nm=k-1
        y=w(nm)
        g=rv1(nm)
        h=rv1(k)
        f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0d0*h*y)
        g=pythag(f,1.0d0)
        f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x
        c=1.0d0
        s=1.0d0
        DO j=l,nm
           i=j+1
           g=rv1(i)
           y=w(i)
           h=s*g
           g=c*g    
           z=pythag(f,h)
           rv1(j)=z
           c=f/z
           s=h/z
           f= (x*c)+(g*s)
           g=-(x*s)+(g*c)
           h=y*s
           y=y*c
           DO jj=1,n
              x=v(jj,j)
              z=v(jj,i)
              v(jj,j)= (x*c)+(z*s)
              v(jj,i)=-(x*s)+(z*c)
           END DO
           z=pythag(f,h)
           w(j)=z
           IF(z.ne.0.0d0)THEN
              z=1.0d0/z
              c=f*z
              s=h*z
           ENDIF
           f= (c*g)+(s*y)
           x=-(s*g)+(c*y)
           DO jj=1,m
              y=a(jj,j)
              z=a(jj,i)
              a(jj,j)= (y*c)+(z*s)
              a(jj,i)=-(y*s)+(z*c)
           END DO
        END DO
        rv1(l)=0.0d0
        rv1(k)=f       
        w(k)=x
     END DO
3    CONTINUE
  END DO
!-----------------------------------------------------------------------

END SUBROUTINE svdcmp
!-----------------------------------------------------------------------
FUNCTION pythag(a,b)
!-----------------------------------------------------------------------

    REAL(dp)                                 :: a, b, pythag

    REAL(dp)                                 :: absa, absb

  absa=ABS(a)
  absb=ABS(b)
  IF(absa.gt.absb)THEN
     pythag=absa*dsqrt(1.0d0+(absb/absa)**2)
  ELSE
     IF(absb.eq.0.0d0)THEN
        pythag=0.0d0
     ELSE
        pythag=absb*dsqrt(1.0d0+(absa/absb)**2)
     ENDIF
  ENDIF

END FUNCTION pythag

! *****************************************************************************
!> \brief does a geometry optimization on given force environments specified in input
!> \param default 
!> \note
!>     be sure force environments are set correctly and input file geo_opt section exists 
!> \author misteliy
! *****************************************************************************
  SUBROUTINE calc_multipole(f_env_id,dipole,ierr)
    
    INTEGER                                  :: f_env_id
    REAL(KIND=dp), DIMENSION(:, :), TARGET   :: dipole
    INTEGER                                  :: ierr

    INTEGER                                  :: handle, nblock, nspin
    LOGICAL                                  :: failure
    REAL(KIND=dp), DIMENSION(:), POINTER     :: charge
    REAL(KIND=dp), DIMENSION(:, :), POINTER  :: dipole_ptr
    REAL(KIND=dp), DIMENSION(:, :, :), &
      POINTER                                :: quadrupole
    TYPE(atomic_kind_type), DIMENSION(:), &
      POINTER                                :: atomic_kind_set
    TYPE(cp_dbcsr_p_type), DIMENSION(:), &
      POINTER                                :: matrix_p, matrix_s
    TYPE(cp_dbcsr_type), POINTER             :: pscp
    TYPE(cp_error_type)                      :: error
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(f_env_type), POINTER                :: f_env
    TYPE(force_env_type), POINTER            :: force_env
    TYPE(global_environment_type), POINTER   :: globenv
    TYPE(particle_type), DIMENSION(:), &
      POINTER                                :: particle_set
    TYPE(qs_environment_type), POINTER       :: qs_env
    TYPE(qs_rho_type), POINTER               :: rho
    TYPE(scp_environment_type), POINTER      :: scp_env

!-------------------error-----------------------------------
!-------------------------------------------------------------

    PRINT *,'in here'
    failure=.FALSE. 
    NULLIFY(f_env, force_env, globenv, matrix_p, rho, qs_env, dipole_ptr)
    CALL f_env_add_defaults(f_env_id,f_env,error, failure, handle)
    PRINT *,'after add defaults'
    IF (.not.failure) THEN
    force_env  => f_env%force_env
    globenv    => force_env%globenv
    qs_env     => force_env%qs_env
    PRINT *,'after force_env'
    PRINT *,ASSOCIATED(qs_env),'qs_env?',qs_env%id_nr

    CALL get_qs_env(qs_env=qs_env,atomic_kind_set=atomic_kind_set,particle_set=particle_set,&
                    rho=rho,matrix_s=matrix_s, &
                    scp_env=scp_env, &
                    para_env=para_env, &
                    error=error)
    PRINT *,ASSOCIATED(scp_env),'scp_env?'
    pscp => scp_env%pscp
    PRINT *,ASSOCIATED(qs_env%scp_env),'scp_env?'
    CALL calc_energy_force(f_env_id,.TRUE.,ierr)
    CALL get_qs_env(qs_env=qs_env,atomic_kind_set=atomic_kind_set,particle_set=particle_set,&
                    rho=rho,matrix_s=matrix_s,&
                    para_env=para_env,error=error)
    PRINT *,ASSOCIATED(rho),'rho'
    PRINT *,ASSOCIATED(atomic_kind_set),'atomic_kind_set'
    PRINT *,ASSOCIATED(particle_set),'particle set'
    PRINT *,ASSOCIATED(matrix_s),'matrix s'
    PRINT *,ASSOCIATED(para_env),'para env'
    CALL cp_error_check(error,failure)
    matrix_p   => rho%rho_ao
    ! here we get the numbers for charges
    nspin=SIZE(matrix_p)
    CALL cp_dbcsr_get_info(matrix_s(1)%matrix,nblkrows_total=nblock)
    ALLOCATE(charge(nblock))
!    ALLOCATE(dipole_ptr(3,nblock))
    ALLOCATE(quadrupole(3,3,nblock))
    dipole_ptr => dipole
    CALL compute_multipoles(atomic_kind_set,particle_set,&
                                matrix_p,pscp,matrix_s(1)%matrix,charge,dipole_ptr,quadrupole,para_env)
    ELSE
       PRINT *,'dipole failure'
      ! write out something?
      ierr=cp_failure_level
    ENDIF
    CALL f_env_rm_defaults(f_env,error,ierr)
  END SUBROUTINE calc_multipole
END MODULE paramopt_worker

