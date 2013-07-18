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
  USE message_passing,                 ONLY: mp_bcast
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


 TYPE paramopt_master_type
   PRIVATE
   INTEGER                           :: iw
   INTEGER                           :: num_inp
   INTEGER                           :: num_var
   TYPE(ener_ref_type),POINTER       :: ener_ref(:)
   REAL(KIND=dp),POINTER             :: res(:)
   REAL(KIND=dp),POINTER             :: obj_f
   REAL(KIND=dp),POINTER             :: x(:)
   REAL(KIND=dp),POINTER             :: xtrans(:)
   REAL(KIND=dp),POINTER             :: xold(:)
   REAL(KIND=dp),POINTER             :: u_bound(:)
   REAL(KIND=dp),POINTER             :: l_bound(:)
   REAL(KIND=dp)                     :: pi=2*asin(1._dp), &
                                        kcalmol=6.27509468713739E+02_dp, &
                                        ev=27.2116_dp,&
                                        ev_kcal=23.06035_dp
   TYPE(opt_state_type),POINTER      :: powell_opt
   TYPE(cp_para_env_type), POINTER   :: para_env
 END TYPE paramopt_master_type

 TYPE :: ener_ref_type
   PRIVATE
   INTEGER :: id
   CHARACTER(80) :: mol
   character(80) :: mol_ref1
   character(80) :: mol_ref2
   character(80) :: mol_ref3
   character(80) :: mol_ref4
   integer   :: stoch1
   integer   :: stoch2
   integer   :: stoch3
   integer   :: stoch4
   integer   :: stoch5
   real(kind=dp) :: ref  
 END TYPE ener_ref_type   

 CONTAINS


! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_init(this, para_env, swarm_section, input_path, n_walkers, iw, error)
    TYPE(paramopt_master_type)               :: this
    TYPE(cp_para_env_type), POINTER          :: para_env
    TYPE(section_vals_type), POINTER         :: swarm_section
    CHARACTER(LEN=*), INTENT(IN)             :: input_path
    INTEGER, INTENT(IN)                      :: n_walkers, iw
    TYPE(cp_error_type), INTENT(inout)       :: error
    TYPE(section_vals_type), POINTER         :: paramopt_section
    INTEGER                                  :: i
    LOGICAL                                  :: restart
    !---------------------
    this%iw = iw
    paramopt_section => section_vals_get_subs_vals(swarm_section, "PARAM_OPT", error=error)
    this%para_env => para_env
    !---------------------
    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init called"
    !------------------
    this%num_var = 16
    !-------------------
    this%num_inp = 35
    allocate(this%ener_ref(this%num_inp))
    allocate(this%res(this%num_inp))
    allocate(this%x(this%num_var))
    allocate(this%xtrans(this%num_var))
    allocate(this%xold(this%num_var))
    allocate(this%l_bound(this%num_var))
    allocate(this%u_bound(this%num_var))
    this%res = 0._dp
    !this%l_bound=(/-2.0_dp,0.05_dp,1._dp,-1.5_dp,0._dp,0._dp,-100._dp,&
    !    -40._dp,0.05_dp,0.05_dp,-3._dp,0.05_dp,-5._dp,0._dp,0._dp,-100._dp/)
    !this%u_bound=(/-0.5_dp,1.0_dp,1.5_dp,-0.5_dp,10._dp,400._dp,300._dp,&
    !    -20._dp,0.4_dp,3._dp,-2._dp,5._dp,-2._dp,10._dp,400._dp,300._dp/)
                !  energy  charge  Zeta   H_core Zeta   H_core     A      C       D
    this%l_bound=(/-40._dp,0.05_dp,-5._dp,-5._dp ,                 0._dp ,-100._dp  ,-100._dp,&
                   -40._dp,0.05_dp,-5._dp,-5._dp ,-5._dp ,-5._dp , 0._dp ,-100._dp  ,-100._dp/)
    this%u_bound=(/5._dp  ,5._dp  ,5._dp ,0._dp  ,                 10._dp,400._dp,400._dp,&
                   5._dp  ,5._dp  ,5._dp ,0._dp  ,5._dp  ,0._dp  , 10._dp,400._dp,400._dp/)
    !-----read x --------
    CALL read4file(this%num_var,this%x,restart)
    IF(this%iw>0) CALL para2file(this%num_var,this%x,"para.xml",2222)
    IF(this%iw>0) WRITE (this%iw,*) "initial x: ",this%x
    !this%l_bound = this%x-this%x*0.05_dp
    !this%u_bound = this%x+this%x*0.05_dp
    CALL var_trans(this%x,this%xtrans,this%l_bound,this%u_bound)
    !----gauss------------
    allocate(this%powell_opt)
    this%powell_opt%state = 0
    this%powell_opt%maxfun = 500 !----change that to imput files value
    this%powell_opt%rhobeg = 0.5_dp
    this%powell_opt%rhoend = 0.000001_dp
    this%powell_opt%iprint = 1
    this%powell_opt%nvar = this%num_var
    this%obj_f => this%powell_opt%f
    CALL powell_optimize(this%num_var,this%xtrans,this%powell_opt)
    !---------------------
    open(99,file='WATER27.dat')
      ! only the first 30 mol have ref data...so 35 - 5 ;/
      do i=1,this%num_inp-5
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
 END SUBROUTINE paramopt_master_init

! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_steer(this, report, cmd, should_stop)
    TYPE(paramopt_master_type)               :: this
    TYPE(swarm_message_type), INTENT(IN)     :: report
    TYPE(swarm_message_type), INTENT(INOUT)  :: cmd
    LOGICAL, INTENT(INOUT)                   :: should_stop
    INTEGER                                  :: i
    REAL(KIND=dp),POINTER                    :: report_Epot(:),report_timings(:)
    INTEGER                                  :: report_worker_id
    integer,parameter    :: h2o_id=1,h3o_id=16,ohm_id=22,h2o8s4_id=11,h2o_eaip_id=31 
    real(kind=dp) :: h2o8s4_ref,h2o_eaip_ref,h2o_ref,h3op_ref,ohm_ref
    REAL(KIND=dp)                            :: energy
    REAL(KIND=dp)                            :: report_rmsd
    LOGICAL                                  :: dummy


    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_steer called"


    CALL swarm_message_add(cmd, "command", "nop")
    CALL swarm_message_add(cmd, "newx", this%x)
    CALL swarm_message_get(report, "worker_id", report_worker_id)

    IF(.NOT. swarm_message_haskey(report, "initial_hello")) THEN
        CALL swarm_message_get(report, "Epot", report_Epot)
        CALL swarm_message_get(report, "rmsd", report_rmsd)
        CALL swarm_message_get(report, "timings", report_timings)
        !print *,report_Epot
        !---------------energy references-------------------------------------------
        h2o_ref      = report_Epot(h2o_id)
        h2o_eaip_ref = report_Epot(h2o_eaip_id)
        h3op_ref     = report_Epot(h3o_id)
        ohm_ref      = report_Epot(ohm_id) 
        h2o8s4_ref   = report_Epot(h2o8s4_id)
        !---------------calcualte error difference----------------------------------
        !   WRITE (this%iw,'(A,46X,I8)')&
        !          " PARAMOPT| Reporting worker ", report_worker_id
        IF(this%iw>0) THEN
        WRITE (this%iw,*) '           !!! energy in kcalmol'
        WRITE (this%iw,*) '           mol ','s1 ','s2 ','s3 ','s4 ', &
                               's5 ','ener       ','ener     ', &
                               'ener diff ', &
                               'ref     ','error   ','converged',' time'
        END IF
                    DO i=1,30
                          ! calculate energy error
                          energy = (this%ener_ref(i)%stoch1*report_Epot(i) + &
                          this%ener_ref(i)%stoch2*h3op_ref + &
                          this%ener_ref(i)%stoch3*ohm_ref + &
                          this%ener_ref(i)%stoch4*h2o_ref + & 
                          this%ener_ref(i)%stoch5*h2o8s4_ref)*this%kcalmol
                          this%res(i) = abs(energy-this%ener_ref(i)%ref)
                          ! print eneregy error
                          IF(this%iw>0) THEN
                            write(*,'(A15,1X,I2,1X,I1,1X,I1,1X,I2,1X,I2,F10.2, &
                                      1X,F10.2,1X,F10.2,F10.2,1X,F10.2,1X,L1,1X,F10.2)') &
                            trim(this%ener_ref(i)%mol), &
                            this%ener_ref(i)%stoch1, &
                            this%ener_ref(i)%stoch2, &
                            this%ener_ref(i)%stoch3, &
                            this%ener_ref(i)%stoch4, &
                            this%ener_ref(i)%stoch5, &
                            report_Epot(i), &
                            report_Epot(i)*this%kcalmol, &
                            energy, &
                            this%ener_ref(i)%ref, &
                            this%res(i), &
                            dummy, &
                            report_timings(i)
                          END IF
                    END DO

        this%obj_f = sum(this%res) + 100000._dp*report_rmsd
        !this%obj_f = 10000._dp*report_rmsd
        this%xold  = this%x


        IF(this%iw>0) THEN
          WRITE (this%iw,'(A,5X,F15.8,F15.8)') &
             " PARAMOPT| Total time for Energy Calculations", sum(report_timings),maxval(report_timings)
          WRITE (this%iw,'(A,5X,F15.8)')&
             " PARAMOPT| Total Energy Error", sum(this%res)
          WRITE (this%iw,'(A,5X,F15.8)')&
             " PARAMOPT| Total RMSD        ", report_rmsd
          WRITE (this%iw,'(A,5X,F15.8)')&
             " PARAMOPT| Total             ", this%obj_f
        END IF
       
        !IF (this%para_env%mepos .eq. 0) THEN 
        !CALL powell_optimize(this%num_var,this%xtrans,this%powell_opt)
        !CALL var_back_trans(this%xtrans,this%x,this%l_bound,this%u_bound)
        !do i=1,this%num_var 
        !    if (this%x(i) .lt. this%l_bound(i)) this%x(i) = this%l_bound(i)
        !    if (this%x(i) .gt. this%u_bound(i)) this%x(i) = this%u_bound(i)
        !enddo
        !ENDIF
        !CALL mp_bcast(this%x,0,this%para_env%group)
        !CALL mp_bcast(this%xtrans,0,this%para_env%group)
        !------------------------------------------------------------------
        !=========have to make sure all process do the same sampling or bcast result???
        CALL sample(this,this%num_var,this%x,this%l_bound,this%u_bound)
        CALL var_trans(this%x,this%xtrans,this%l_bound,this%u_bound)
        !------------------------------------------------------------------
        IF(this%iw>0) CALL para2file(this%num_var,this%x,"para.xml",2222)
        IF(this%iw>0) THEN
        !  WRITE (this%iw,*) 'this should be xtrans',this%xtrans
        !  WRITE (this%iw,*) '-------------------------------------------------'
        !  WRITE (this%iw,*) 'this should be f',this%obj_f
        !  WRITE (this%iw,*) '-------------------------------------------------'
        !  WRITE (this%iw,*) 'this should be x diff',this%xold - this%x
        !  WRITE (this%iw,*) '-------------------------------------------------'
        !  WRITE (this%iw,*) 'this should be old x',this%xold
        !  WRITE (this%iw,*) '-------------------------------------------------'
        !  WRITE (this%iw,*) 'this should be new x',this%x
          open(999,ACCESS="APPEND")
          write(999,*) this%xold,sum(this%res),report_rmsd
          close(999)
        !  open(998,ACCESS="APPEND")
        !  write(998,*) this%xold-this%x
        !  close(998)
        END IF
    END IF

 END SUBROUTINE paramopt_master_steer


! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_finalize(this)
    TYPE(paramopt_master_type)                 :: this

    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_finalize called"
 END SUBROUTINE paramopt_master_finalize

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
        !print *,'read from restart file'
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

subroutine para2file(n,x,inputstr,fileid2)
    implicit none
    integer,intent(in)          :: n,fileid2
    integer                     :: i,rank,ierr,fileid,k
    real(kind=dp),intent(in)    :: x(n)
    real(kind=dp)               :: dummy=0._dp
    character(len=*),intent(in) :: inputstr
    character(200)              :: line
    character(24)              :: str,str2
  
    fileid=fileid2+100 
    open(fileid,file=adjustl(trim(inputstr)))
    open(fileid2,file='temp')
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
    call RENAME(inputstr,"para.xml.restart",ierr)
    if (ierr .ne. 0) print*,'shit...moving file failed'
    call RENAME('temp',inputstr,ierr)
    if (ierr .ne. 0) print*,'shit...moving file failed'
   
    close(fileid)
    close(fileid2)
end subroutine

      subroutine var_trans(x,y,lbnd,ubnd)
      real(kind=dp),dimension(:) :: x,lbnd,ubnd
      real(kind=dp),dimension(:),intent(out) :: y
      y = ((x - lbnd) / (ubnd - lbnd))
      end subroutine

      subroutine var_back_trans(x,y,lbnd,ubnd)
      real(kind=dp),dimension(:) :: x,y,lbnd,ubnd
      y = x*(ubnd-lbnd) + lbnd
      end subroutine
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
    
!    CALL init_random_seed() 
    sampler=1._dp/(2._dp*sqrt(real(n)))
    generate=.TRUE.
    step=1._dp/2._dp
    step=1._dp/4._dp
!    t=1
!    irand=7
            do while (generate)
!--------------------------random only one variable---------------------------
                if (this%para_env%mepos .eq. 0) call random_number(rnum)
                call mp_bcast(rnum,0,this%para_env%group)
                irand=int(rnum*n)+1

                !irand = 4 
                call var_trans(x,tmpx,l_bound,u_bound)
                !print *,'after var trans',tmpx(irand)
                if (this%para_env%mepos .eq. 0) call random_number(rnum)
                call mp_bcast(rnum,0,this%para_env%group)
                if (rnum .gt. 0.5_dp) then
                    step=step
                else
                    step=-step
                endif
                tmpx(irand) = tmpx(irand) + step
               ! print *,'before check',tmpx(irand) 
                if (tmpx(irand) .gt. 1.0_dp) then 
                    tmpx(irand) = tmpx(irand) - 2*step
                endif
                
                if (tmpx(irand) .lt. 0._dp) then
                    tmpx(irand) = tmpx(irand) + 2*abs(step)
                endif
             !   print *,'test',tmpx(irand)
             !   print *,'old var',x(irand), &
             !   'new var',(tmpx(irand)*((u_bound(irand)) - (l_bound(irand)))) + (l_bound(irand))
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
                call var_back_trans(tmpx,tmpx2,l_bound,u_bound)

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
                    generate = .FALSE.
            enddo
            x = tmpx2

            if (this%para_env%mepos .eq. 0) print *,'change var',irand,'new value',x(irand)
end subroutine
END MODULE paramopt_master

