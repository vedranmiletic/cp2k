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
 PUBLIC :: Deviation
 PUBLIC :: plugin
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
  Variance = abs(Variance)/X
 END FUNCTION Variance
 ! *****************************************************************************
! *****************************************************************************
 subroutine Deviation (Array)
  IMPLICIT NONE
  REAL(kind=dp)::X
  REAL(kind=dp),INTENT(INOUT),DIMENSION(:)::Array
  X = SUM(Array)/SIZE(Array)
  array = (Array-X)
 end subroutine deviation
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

!       revised Nov. 20, 1996 by petra

!       This subroutine calculates the density at points X2
!       using points X.  It is returned in dens1.  The number
!       of variables in X is given by ncol1. The bandwidth is
!       passed in hd1.
!       The program expects to receive R, the interquartile range
!       of the data which it uses in calculating the silverman
!       optimal bandwidth formula for a biweight kernel

      SUBROUTINE PLUGIN(X,N,Z,M,F,H)                      
!-----------------------------------------------------------------------
!       Version: 1995
!       fortran77 
!                                                                       
!       Purpose:                                                        
!                                                                       
!       Simple  Subroutine for kernel density estimation 
!       with iterative plug-in bandwidth selection
!       
!       This version only uses the gauss kernel and estimates only
!       the density itself and not its derivatives.
!                                                                       
!  INPUT    X(N)   DOUBLE PRECISION   sorted data
!  INPUT    N      INTEGER            length of  X
!  INPUT    Z(M)   DOUBLE PRECISION   output grid (sorted)
!  INPUT    M      INTEGER            length of Z
!  OUTPUT   F(M)   DOUBLE PRECISION   estimated density
!  OUTPUT   H      DOUBLE PRECISION   estimated iterative plugin bandwidth
!                                                                       
!                                                                       
!-----------------------------------------------------------------------
      INTEGER N,M
      INTEGER i,j
      INTEGER it,iter
      INTEGER jbegin,jend
      REAL(kind=8) h
      REAL(kind=8) h2,h3,s2,s3,d2,d3,e2,e3,rhat2,rhat3,co2
      REAl(kind=8) const,a,pi,t,r2,co1,rt2pi,s,xiqr,xn
      REAL(kind=8) X(N),Z(M),F(M)
      XN=DBLE(N)
      XIQR=X(NINT(.75*XN))-X(NINT(.25*XN)+1)	
      PI=3.141592645D0
      RT2PI = DSQRT(2.D0*PI)
      ITER=5
!-------  Estimate inflation constant C
          H2=(.920*XIQR)/(XN**(1.D0/7.D0))
          H3=(.912*XIQR)/(XN**(1.D0/9.D0))
          S2 = 0.D0
          S3 = 0.D0
          DO 20 I = 1, N-1
           DO 30 J = I+1, N
                     D2 = ((X(I) - X(J))/H2)**2
                     D3 = ((X(I) - X(J))/H3)**2
                     IF(D2.GT.50.AND.D3.GT.60) GOTO 20
                     E2 = DEXP(-0.5D0*D2)
                     E3 = DEXP(-0.5D0*D3)
                     S2 = S2 + (D2**2 - 6.D0*D2 + 3.D0)*E2
               S3 = S3 + (D3**3 - 15.D0*D3**2 + 45.D0*D3 - 15.D0)*E3
30          CONTINUE
20         CONTINUE
          RHAT2 = (2.D0*S2)/((XN**2)*(H2**5)*RT2PI)
          RHAT2 = RHAT2 + 3.D0/(RT2PI*XN*(H2**5))
          RHAT3 = (-2.D0*S3)/((XN**2)*(H3**7)*RT2PI)
          RHAT3 = RHAT3 + 15.D0/(RT2PI*XN*(H3**7))
          CO1 = 1.357D0*(RHAT2/RHAT3)**(1.D0/7.D0)
!-
!-------  Compute constant of asymptotic formula
!-
       CONST=1.D0/(2.D0*DSQRT(PI))
       A=1.132795764/RHAT3**(1.D0/7.D0)*XN**(-1.D0/2.D0)
!-
!------  Loop over iterations
!-
       DO 100 IT=1,ITER
!-                                                                     
!-------  Estimate functional
!
        S=0.D0                                                    
        DO 40 I = 1, N-1
           DO 50 J = I+1, N
                     D2 = ((X(I) - X(J))/A)**2
                     IF(D2.GT.50) GOTO 40
                     E2 = DEXP(-0.5D0*D2)
                     S = S + (D2**2 - 6.D0*D2 + 3.D0)*E2
50          CONTINUE
40       CONTINUE
        R2 = (2.D0*S)/((XN**2)*(A**5)*RT2PI)
        R2 = R2 + 3.D0/(RT2PI*XN*(A**5))
!-                                                                     
!-------  Estimate bandwidth by asymptotic formula
!-
         H=(CONST/(R2*XN))**(.2D0)                                    
         A=CO1*H**(5./7.)
100     CONTINUE
!-
!------- Estimate density with plugin bandwidth
!-            
      if (n .lt. 4) H = 1.D0 
      if (isnan(h)) h = 1.D0
      if (h .eq. 0.) h = 1.D0 
      JBEGIN=1
      JEND=1                                             
      DO 200 I=1,M                                                      
         S=0.D0    
         DO 210 J=JBEGIN,JEND
                T=(Z(I)-X(J))/H
                IF(T.GT.5.0.AND.JBEGIN.LT.N) THEN
                   JBEGIN=JBEGIN+1
                   GOTO 210
                END IF
                S=S+DEXP(-T*T/2.)
210             CONTINUE         
         DO 220 JEND=J,N
                T=(Z(I)-X(JEND))/H
                IF(T.LT.-5.0) GOTO 230
                S=S+DEXP(-T*T/2.)
220             CONTINUE
230         F(I)=S/(XN*H*RT2PI)
            JEND=JEND-1
!-                         
200      CONTINUE                                             
      RETURN                                                            
      END SUBROUTINE          

END MODULE paramopt_utils

