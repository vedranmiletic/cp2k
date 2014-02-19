!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2012  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief performs global geometry optimization
!> \par History
!> \author Ole
! *****************************************************************************
MODULE paramopt_utils
  USE paramopt_types,                  ONLY: paramopt_master_type
  USE kinds,                           ONLY: default_string_length,&
                                             dp
  USE parallel_rng_types,              ONLY: rng_stream_type, &
                                             create_rng_stream, &
                                             next_random_number, &
                                             reset_to_next_rng_substream, &
                                             GAUSSIAN, &
                                             UNIFORM 

#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'paramopt_utils'


 PUBLIC :: Variance
 PUBLIC :: var_trans,var_back_trans
 PUBLIC :: sample

! TYPE minima_type
!   INTEGER                                             :: id = -1
!   REAL(KIND=dp), DIMENSION(:), ALLOCATABLE            :: pos
!   REAL(KIND=dp), DIMENSION(:), ALLOCATABLE            :: escape_hist
!   REAL,          DIMENSION(:), ALLOCATABLE            :: tempdist
!   REAL(KIND=dp)                                       :: Epot = -1.0
!   TYPE(history_fingerprint_type)                      :: fp
!   LOGICAL                                             :: disabled = .FALSE.
!   INTEGER                                             :: n_active = 0
!   INTEGER                                             :: n_sampled = 0
! END TYPE minima_type


 CONTAINS


 FUNCTION Variance (Array)
  IMPLICIT NONE
  REAL(kind=dp)::Variance,X
  REAL(kind=dp),INTENT(IN),DIMENSION(:)::Array
  X = SUM(Array)/SIZE(Array)
  Variance = SUM((Array-X)**2)/SIZE(Array)
 END FUNCTION Variance
 ! *****************************************************************************
! *****************************************************************************
 subroutine var_trans(x,y,lbnd,ubnd)
      real(kind=dp),dimension(:) :: x,lbnd,ubnd
      real(kind=dp),dimension(:),intent(out) :: y
      y = ((x - lbnd) / (ubnd - lbnd))
 end subroutine
! *****************************************************************************
! *****************************************************************************
 subroutine var_back_trans(x,y,lbnd,ubnd)
      real(kind=dp),dimension(:) :: x,y,lbnd,ubnd
      y = x*(ubnd-lbnd) + lbnd
 end subroutine
! *****************************************************************************
! *****************************************************************************
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
!    step=1._dp/2._dp
    step = this%step_size
    tmpx = x
    !=================random uniform sampling======================
!   do i=1,n; tmpx(i) = next_random_number(this%rng_stream,error=this%error); enddo
!    call var_back_trans(tmpx,x,l_bound,u_bound)
    !==============================================================
!    t=1
!    irand=7
!            do while (generate)
!--------------------------random only one variable---------------------------
!                call random_number(rnum)
                 rnum = next_random_number(this%rng_stream,error=this%error)
                 irand=int(rnum*n)+1
                 print *,'change var num',irand
!
!                irand = 2 
!                call var_trans(x,tmpx,l_bound,u_bound)
                 rnum = next_random_number(this%rng_stream,error=this%error)
!                !print *,'after var trans',tmpx(irand)
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
              x = tmpx  
              !  call var_back_trans(tmpx,x,l_bound,u_bound)
!             !   print *,'test',tmpx(irand)
!             !   print *,'old var',x(irand), &
!             !   'new var',(tmpx(irand)*((u_bound(irand)) - (l_bound(irand)))) + (l_bound(irand))
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
!                call var_back_trans(tmpx,x,l_bound,u_bound)

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

!            if (this%para_env%mepos .eq. 0) print *,'change var',irand,'new value',x(irand)
 end subroutine
! *****************************************************************************

! *****************************************************************************
 subroutine read4file(n,x,ex)
    implicit none
    integer,intent(in)       :: n
    integer                  :: i,fileid,k
    real(kind=dp),intent(out) :: x(n)
    character(200)            :: str,str2,str3,filename
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
  !  case (17)
  !   read(fileid,*) str,x(k),str2
  !   k=k+1
    case (19:20)        
     read(fileid,*) str,x(k),str2
     k=k+1
     
  !  case (25)
  !   read(fileid,*) str,str3,x(k),str2
  !   k=k+1

    case (28:30)
     read(fileid,*) str,x(k),str2
     k=k+1

  !  case (36)
  !   read(fileid,*) str,x(k),str2
  !   k=k+1

    case (38:39)
     read(fileid,*) str,x(k),str2
     k=k+1

    case (44:45) 
     read(fileid,*) str,x(k),str2
     k=k+1

  !  case (50:51) 
  !   read(fileid,*) str,str3,x(k),str2
  !   k=k+1 
    
    case (54:56) 
     read(fileid,*) str,x(k),str2
     k=k+1 

    end select
    
    end do
    if (k-1 .ne. n) write(*,*) "something is very wrong"

    close(fileid)
 end subroutine

! *****************************************************************************
! *****************************************************************************
 subroutine para2file(n,x,inputstr,outputstr)
    implicit none
    integer,intent(in)          :: n
    integer                     :: fileid2
    integer                     :: i,rank,ierr,fileid,k
    real(kind=dp),intent(in)    :: x(n)
    real(kind=dp)               :: dummy=0._dp
    character(len=*),intent(in) :: inputstr
    character(len=*),intent(in) :: outputstr
    character(200)              :: line
    character(24)              :: str,str2,str3
    
    fileid = 2222
    fileid=fileid2+100 
    open(fileid,file=adjustl(trim(inputstr)))
    open(fileid2,status='replace',file=adjustl(trim(outputstr)))
    k=1 
!   print *,'x to file',k,x
    do i=1,59
!    print *,'k',k
    select case (i)
    case default
     read(fileid,'(A)') line
!     str=adjustl(trim(str))
     write(fileid2,'(A)') line
  !  case (17)
  !   read(fileid,*) str,dummy,str2
  !   str=adjustl(trim(str))
! !    str2=adjustl(trim(str2))
  !   str2="</Charge_width>"
  !   write(fileid2,'(8X,A,F13.7,X,A)') str,x(k),str2
  !   k=k+1
   ! case (19:20)        
   !  read(fileid,'(A,F13.7,A)') str,dummy,str2
!  !   str=adjustl(trim(str))d
   !  str2=adjustl(trim(str2))
   !  write(fileid2,'(A,F13.7,A)') str,x(k),str2
   !  k=k+1
    case (25)
     read(fileid,*) str,str3,dummy,str2
     str=adjustl(trim(str))
     str3=adjustl(trim(str3))
!     str2=adjustl(trim(str2))
     str2="</ETA>"
     write(fileid2,'(12X,A4,X,A6,F13.7,X,A)') str,str3,x(k),str2
        k=k+1
    case (28:30)
     read(fileid,'(A16,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,X,A)') str,x(k),str2
        k=k+1
 !   case (36)
 !    read(fileid,*) str,dummy,str2
 !    str=adjustl(trim(str))
!!     str2=adjustl(trim(str2))
 !    str2="</Charge_width>"
 !    write(fileid2,'(8X,A,F13.7,X,A)') str,x(k),str2
 !       k=k+1
    case (38:39) 
     read(fileid,'(A,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,A)') str,x(k),str2
        k=k+1
    case (44:45) 
     read(fileid,'(A,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,A)') str,x(k),str2
        k=k+1
  !  case (50:51) 
  !   read(fileid,*) str,str3,dummy,str2
! !    str=adjustl(trim(str))
  !   str3=adjustl(trim(str3))
! !    str2=adjustl(trim(str2))
  !   str2="</ETA>"
  !   write(fileid2,'(12X,A4,X,A6,F13.7,X,A)') str,str3,x(k),str2
  !      k=k+1
    case (54:56) 
     read(fileid,'(A16,F13.7,A)') str,dummy,str2
!     str=adjustl(trim(str))
     str2=adjustl(trim(str2))
     write(fileid2,'(A,F13.7,X,A)') str,x(k),str2
        k=k+1

     end select 
    enddo

    print *,'file written'
!    call RENAME(inputstr,"para.xml.restart",ierr)
!    if (ierr .ne. 0) print*,'shit...moving file failed'
!    call RENAME('temp',inputstr,ierr)
!    if (ierr .ne. 0) print*,'shit...moving file failed'
   
    close(fileid)
    close(fileid2)
 end subroutine

END MODULE paramopt_utils

