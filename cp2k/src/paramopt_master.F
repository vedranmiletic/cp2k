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
  USE kinds,                           ONLY: default_string_length,&
                                             dp
  USE message_passing,                 ONLY: mp_bcast
  USE parallel_rng_types,              ONLY: rng_stream_type, &
                                             create_rng_stream, &
                                             next_random_number
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
   INTEGER                           :: num_para_set
   TYPE(cp_error_type),POINTER       :: error
   TYPE(ener_ref_type),POINTER       :: ener_ref(:)
   CHARACTER(len=default_string_length),POINTER     :: input_files(:)
   REAL(KIND=dp),ALLOCATABLE         :: Epot(:),timings(:)
   REAL(KIND=dp),POINTER             :: res(:)
   REAL(KIND=dp),POINTER             :: function_matrix(:,:)
   REAL(KIND=dp),POINTER             :: obj_f
   REAL(KIND=dp),ALLOCATABLE             :: x(:)
   REAL(KIND=dp),POINTER             :: xtrans(:)
   REAL(KIND=dp),POINTER             :: xtransold(:)
   REAL(KIND=dp),POINTER             :: xold(:)
   REAL(KIND=dp),POINTER             :: u_bound(:)
   REAL(KIND=dp),POINTER             :: l_bound(:)
   REAL(KIND=dp)                     :: pi=2*asin(1._dp), &
                                        kcalmol=6.27509468713739E+02_dp, &
                                        ev=27.2116_dp,&
                                        ev_kcal=23.06035_dp
   TYPE(opt_state_type),POINTER      :: powell_opt
   TYPE(cp_para_env_type), POINTER   :: para_env
   TYPE(rng_stream_type),POINTER     :: rng_stream
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
    CHARACTER(len=default_string_length)     :: output_path, str
    INTEGER, INTENT(IN)                      :: n_walkers, iw
    TYPE(cp_error_type), INTENT(inout)       :: error
    TYPE(section_vals_type), POINTER         :: paramopt_section
    INTEGER                                  :: i
    INTEGER                                  :: time(3)
    LOGICAL                                  :: restart
    REAL(kind=dp),DIMENSION(3,2)             :: seed

    this%iw = iw
    paramopt_section => section_vals_get_subs_vals(swarm_section, "PARAM_OPT", error=error)
    this%para_env => para_env
    !---------------------
    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init called"
    !------------------
    this%num_var = 16
    !-------------------
    this%num_inp = 35
    !-------------------
    this%num_para_set = 1
    !-------------------
    allocate(this%ener_ref(this%num_inp))
    allocate(this%res(this%num_inp))
    allocate(this%x(this%num_var))
    allocate(this%xtrans(this%num_var))
    allocate(this%xtransold(this%num_var))
    allocate(this%xold(this%num_var))
    allocate(this%l_bound(this%num_var))
    allocate(this%u_bound(this%num_var))
    ALLOCATE(this%input_files(this%num_inp))
    allocate(this%rng_stream)
    allocate(this%Epot(this%num_var))
    allocate(this%timings(this%num_var))
    
    call itime(time)
    seed(:,1) = REAL(time)
    seed(:,2) = (/ 1._dp, 2._dp, 3._dp /)
    CALL create_rng_stream(this%rng_stream,'uniform',seed=seed,error=this%error)
    !---------------------

    this%res = 0._dp
                !  energy  width  Zeta   H_core Zeta   H_core       A      C        D
    this%l_bound=(/-50._dp,0.05_dp,-5._dp,-50._dp ,                 0._dp ,-100._dp ,-100._dp,&
                   -50._dp,0.05_dp,-5._dp,-50._dp ,-5._dp ,-50._dp ,0._dp ,-100._dp ,-100._dp/)
    this%u_bound=(/5._dp  ,5._dp  ,5._dp ,0._dp  ,                  10._dp, 300._dp ,300._dp,&
                   5._dp  ,5._dp  ,5._dp ,0._dp  ,5._dp  ,0._dp  ,  10._dp, 300._dp ,300._dp/)
    !-----read x --------
    CALL read4file(this%num_var,this%x,restart)
    CALL para2file(this%num_var,this%x,"para.xml",2222)
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
    OPEN(88,file='WATER27structures/mol_list.txt')
    DO i=1,this%num_inp-5
          READ(88,*) str
          this%input_files(i)= TRIM(str) // ".inp"
    ENDDO
    CLOSE(88)
    this%input_files(this%num_inp-4) = 'eaip_H2O.inp'
    this%input_files(this%num_inp-3) = 'eaip_H2Op.inp'
    this%input_files(this%num_inp-2) = 'eaip_H2On.inp'
    this%input_files(this%num_inp-1) = 'dipole.inp'
    this%input_files(this%num_inp-0) = 'geo_opt.inp' 
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
    !------------------------
    allocate(this%function_matrix(this%num_para_set,this%num_inp))
    !------------------------
    IF(this%iw>0) WRITE (this%iw,*) "paramopt_master_init done"
 END SUBROUTINE paramopt_master_init

! *****************************************************************************
! *****************************************************************************
 SUBROUTINE paramopt_master_steer(this, report, cmd, should_stop)
    TYPE(paramopt_master_type)               :: this
    TYPE(swarm_message_type), INTENT(IN)     :: report
    TYPE(swarm_message_type), INTENT(INOUT)  :: cmd
    LOGICAL, INTENT(INOUT)                   :: should_stop
    INTEGER                                  :: i
    INTEGER                                  :: report_worker_id
    integer,parameter    :: h2o_id=1,h3o_id=16,ohm_id=22,h2o8s4_id=11,h2o_eaip_id=31 
    real(kind=dp) :: h2o8s4_ref,h2o_eaip_ref,h2o_ref,h3op_ref,ohm_ref
    REAL(KIND=dp)                            :: energy
    REAL(KIND=dp)                            :: report_rmsd

    CALL swarm_message_get(report, "worker_id", report_worker_id)

    IF(.NOT. swarm_message_haskey(report, "initial_hello")) THEN
       CALL swarm_message_get(report, "Epot", this%Epot(report_worker_id))
       IF(this%iw>0) THEN
          WRITE (this%iw,'(A,46X,I8)')&
                 " GLBOPT| Reporting worker ", report_worker_id
          WRITE (this%iw,'(A,20X,E15.8)')&
                 " GLBOPT| Reported potential Energy [Hartree] ", this%Epot(report_worker_id)
       END IF
    END IF   
    
    IF(swarm_message_haskey(report, "initial_hello")) THEN
       CALL swarm_message_add(cmd, "command", "nop")
       CALL swarm_message_add(cmd, "input", this%input_files(report_worker_id))
       CALL swarm_message_add(cmd, "x", this%x)
       IF(this%iw > 0) WRITE(this%iw,'(A,10X,A,I10)') &
                     " GLBOPT| Sending initial command to worker", &
                     trim(this%input_files(report_worker_id)),report_worker_id
       RETURN
    ENDIF
    CALL swarm_message_get(report, "Epot", this%Epot(report_worker_id))

     
    CALL swarm_message_add(cmd, "command", "nop")
    CALL swarm_message_add(cmd, "input", this%input_files(report_worker_id))
    CALL swarm_message_add(cmd, "x", this%x)
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
        print *,'read from restart file'
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
    
    sampler=1._dp/(2._dp*sqrt(real(n)))
    generate=.TRUE.
    step=1._dp/2._dp
    step=1._dp/4._dp
    !=================random uniform sampling======================
!    tmpx = next_random_number(this%rng_stream,error=this%error)
!    call var_back_trans(tmpx,x,l_bound,u_bound)
    !==============================================================
!    t=1
!    irand=7
!            do while (generate)
!--------------------------random only one variable---------------------------
!                call random_number(rnum)
                 rnum = next_random_number(this%rng_stream,error=this%error)
                 irand=int(rnum*n)+1
!
!                !irand = 4 
                call var_trans(x,tmpx,l_bound,u_bound)
                 rnum = next_random_number(this%rng_stream,error=this%error)
!                !print *,'after var trans',tmpx(irand)
!                call random_number(rnum)
                if (rnum .gt. 0.5_dp) then
                    step=step
                else
                    step=-step
                endif
                tmpx(irand) = tmpx(irand) + step
!               ! print *,'before check',tmpx(irand) 
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
                call var_back_trans(tmpx,x,l_bound,u_bound)

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
!                    generate = .FALSE.
!            enddo
!            x = tmpx2

            if (this%para_env%mepos .eq. 0) print *,'change var',irand,'new value',x(irand)
end subroutine
END MODULE paramopt_master

