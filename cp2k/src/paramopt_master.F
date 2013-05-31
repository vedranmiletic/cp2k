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
  USE kinds,                           ONLY: dp
  USE physcon,                         ONLY: angstrom,&
                                             kelvin
  USE swarm_message,                   ONLY: swarm_message_type,&
                                             swarm_message_add,&
                                             swarm_message_get,&
                                             swarm_message_length
#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'paramopt'


 PUBLIC :: paramopt_master_type
 PUBLIC :: paramopt_master_init, paramopt_master_finalize
 PUBLIC :: paramopt_master_steer


 TYPE paramopt_master_type
   PRIVATE
   INTEGER                    :: iw
 END TYPE paramopt_master_type

 CONTAINS


! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_init(this, swarm_section, n_walkers, iw, error)
    TYPE(paramopt_master_type)               :: this
    TYPE(section_vals_type), POINTER         :: swarm_section
    INTEGER, INTENT(IN)                      :: n_walkers, iw
    TYPE(cp_error_type), INTENT(inout)       :: error
    TYPE(section_vals_type), POINTER         :: paramopt_section

    this%iw = iw
    paramopt_section => section_vals_get_subs_vals(swarm_section, "PARAM_OPT", error=error)

    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init called"

 END SUBROUTINE paramopt_master_init

! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_steer(this, report, cmd, should_stop)
    TYPE(paramopt_master_type)               :: this
    TYPE(swarm_message_type)                 :: report
    TYPE(swarm_message_type)                 :: cmd
    LOGICAL, INTENT(INOUT)                   :: should_stop

    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_steer called"

    CALL swarm_message_add(cmd, "command", "nop")

    !REAL(KIND=dp) :: report_Epot
    !INTEGER :: report_worker_id
    !
    !CALL swarm_message_get(report, "worker_id", report_worker_id)
    !CALL swarm_message_get(report, "Epot", report_Epot)
    !
    !paramopt%Elowest = MIN(paramopt%Elowest, report_Epot)
    !IF(paramopt%iw>0) THEN
    !   WRITE (paramopt%iw,'(A,46X,I8)')&
    !          " PARAMOPT| Reporting worker ", report_worker_id
    !   WRITE (paramopt%iw,'(A,20X,E15.8)')&
    !          " PARAMOPT| Reported potential Energy [Hartree] ", report_Epot
    !   WRITE (paramopt%iw,'(A,13X,E15.8)')&
    !    " PARAMOPT| Lowest reported potential Energy [Hartree] ",paramopt%Elowest
    !END IF
    !
    !!IF(paramopt%Elowest<paramopt%Emin .AND. .NOT. master%should_stop) THEN
    !IF(paramopt%Elowest<paramopt%Emin) THEN
    !   IF(paramopt%iw>0)  WRITE (paramopt%iw, *)  "PARAMOPT| Reached Emin > Epot. Quitting."
    !   should_stop = .TRUE.
    !ELSE
    !   CALL minhop_steer(paramopt%minhop, report, cmd)
    !ENDIF
 END SUBROUTINE paramopt_master_steer


! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_finalize(this)
    TYPE(paramopt_master_type)                 :: this

    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_finalize called"
 END SUBROUTINE paramopt_master_finalize


END MODULE paramopt_master

