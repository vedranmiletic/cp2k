!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2014  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief Implemenation of machine interface based on Fortran 2003 and POSIX
!> \author Ole Schuett
! *****************************************************************************
  USE kinds,                           ONLY: dp, int_8, default_path_length,&
                                             default_string_length
  USE ISO_C_BINDING

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: m_cputime, m_flush, m_memory, &
            m_hostnm, m_getcwd, m_getlog, m_getuid, m_getpid, m_getarg, &
            m_iargc, m_abort, m_chdir, m_mov, &
            m_memory_details, m_procrun

CONTAINS

! *****************************************************************************
! can be used to get a nice core
! *****************************************************************************
  SUBROUTINE m_abort()
    INTERFACE
      SUBROUTINE abort() BIND(C,name="abort")
        USE ISO_C_BINDING
      END SUBROUTINE
    END INTERFACE

    CALL abort()
  END SUBROUTINE m_abort


! *****************************************************************************
! the number of arguments of the fortran program
! *****************************************************************************
  FUNCTION m_iargc() RESULT (ic)
    INTEGER                                  :: ic

    ic = COMMAND_ARGUMENT_COUNT ()
  END FUNCTION m_iargc


! *****************************************************************************
!  cpu time in seconds
! *****************************************************************************
  FUNCTION m_cputime() RESULT (ct)
    REAL(KIND=dp)                            :: ct

    CALL CPU_TIME(ct)
  END FUNCTION m_cputime


! *****************************************************************************
! flush a given unit
! *****************************************************************************
  SUBROUTINE m_flush(lunit)
    INTEGER, INTENT(IN)                      :: lunit

    FLUSH(lunit)
  END SUBROUTINE m_flush


! *****************************************************************************
! returns if a process is running on the local machine
! 1 if yes and 0 if not
! *****************************************************************************
  FUNCTION m_procrun(pid) RESULT (run_on)
    INTEGER, INTENT(IN)       ::   pid
    INTEGER                   ::   run_on
    INTEGER                   ::   istat

    INTERFACE
      FUNCTION kill(pid, sig) BIND(C,name="kill") RESULT(errno)
        USE ISO_C_BINDING
        INTEGER(KIND=C_INT),VALUE                :: pid, sig
        INTEGER(KIND=C_INT)                      :: errno
      END FUNCTION
    END INTERFACE

    ! If sig is 0, then no signal is sent, but error checking is still
    ! performed; this can be used to check for the existence of a process
    ! ID or process group ID.

    istat = kill(pid=pid, sig=0)
    IF(istat == 0) THEN
       run_on = 1 ! no error, process exists
    ELSE
       run_on = 0 ! error, process probably does not exist
    ENDIF

  END FUNCTION m_procrun


! *****************************************************************************
! returns the total amount of memory [bytes] in use, if known, zero otherwise
! *****************************************************************************
  FUNCTION m_memory()

      INTEGER(KIND=int_8)                      :: m_memory
      INTEGER(KIND=int_8)                      :: m1,m2,m3

      !
      ! __NO_STATM_ACCESS can be used to disable the stuff, if getpagesize
      ! lead to linking errors or /proc/self/statm can not be opened
      !
#if defined(__NO_STATM_ACCESS)
      m_memory=0
#else
      CHARACTER(LEN=80) :: DATA
      INTEGER :: iostat,i

      ! the size of a page, might not be available everywhere
      INTERFACE
       FUNCTION getpagesize() BIND(C,name="getpagesize") RESULT(RES)
         USE ISO_C_BINDING
         INTEGER(C_INT) :: RES
       END FUNCTION
      END INTERFACE

      !
      ! reading from statm
      !
      m_memory=-1
      DATA=""
      OPEN(121245,FILE="/proc/self/statm",ACTION="READ",STATUS="OLD",ACCESS="STREAM")
      DO I=1,80
         READ(121245,END=999) DATA(I:I)
      ENDDO
999   CLOSE(121245)
      DATA(I:80)=""
      ! m1 = total
      ! m2 = resident
      ! m3 = shared
      READ(DATA,*,IOSTAT=iostat) m1,m2,m3
      IF (iostat.NE.0) THEN
         m_memory=0
      ELSE
         m_memory=m2
#if defined(__STATM_TOTAL)
         m_memory=m1
#endif
#if defined(__STATM_RESIDENT)
         m_memory=m2
#endif
         m_memory=m_memory*getpagesize()
      ENDIF
#endif

  END FUNCTION m_memory

! *****************************************************************************
! *** get more detailed memory info, all units are bytes.
! *** the only 'useful' option is MemLikelyFree which is an estimate of remaining memory
! *** assumed to give info like /proc/meminfo while MeMLikelyFree is the amount of
! *** memory we're likely to be able to allocate, but not necessarily in one chunk
! *** zero means not available
! *****************************************************************************
  SUBROUTINE m_memory_details(MemTotal,MemFree,Buffers,Cached,Slab,SReclaimable,MemLikelyFree)

     INTEGER(kind=int_8), OPTIONAL :: MemTotal,MemFree,Buffers,Cached,Slab,SReclaimable,MemLikelyFree

     INTEGER, PARAMETER :: Nbuffer=10000
     CHARACTER(LEN=Nbuffer) :: meminfo


     INTEGER :: i

     MemTotal=0
     MemFree=0
     Buffers=0
     Cached=0
     Slab=0
     SReclaimable=0
     MemLikelyFree=0
     meminfo=""

     OPEN(UNIT=8123,file="/proc/meminfo",ACCESS="STREAM",ERR=901)
     i=0
     DO
       i=i+1
       IF (i>Nbuffer) EXIT
       READ(8123,END=900,ERR=900) meminfo(i:i)
     ENDDO
 900 CONTINUE
     meminfo(i:Nbuffer)=""
 901 CONTINUE
     CLOSE(8123,ERR=902)
 902 CONTINUE
     MemTotal=get_field_value_in_bytes('MemTotal:')
     MemFree=get_field_value_in_bytes('MemFree:')
     Buffers=get_field_value_in_bytes('Buffers:')
     Cached=get_field_value_in_bytes('Cached:')
     Slab=get_field_value_in_bytes('Slab:')
     SReclaimable=get_field_value_in_bytes('SReclaimable:')
     ! opinions here vary but this might work
     MemLikelyFree=MemFree+Buffers+Cached+SReclaimable


  CONTAINS
        INTEGER(int_8) FUNCTION get_field_value_in_bytes(field)
           CHARACTER(LEN=*) :: field
           INTEGER :: start
           INTEGER(KIND=int_8) :: value
           get_field_value_in_bytes=0
           start=INDEX(meminfo,field)
           IF (start.NE.0) THEN
              start=start+LEN_TRIM(field)
              IF (start.LT.Nbuffer) THEN
                 READ(meminfo(start:),*,ERR=999,END=999) value
                 ! XXXXXXX convert from Kb to bytes XXXXXXXX
                 get_field_value_in_bytes=value*1024
 999             CONTINUE
              ENDIF
           ENDIF
        END FUNCTION
  END SUBROUTINE m_memory_details


! *****************************************************************************
! *****************************************************************************
  SUBROUTINE m_mov(source,TARGET)

    CHARACTER(LEN=*), INTENT(IN)             :: source, TARGET

    INTEGER                                  :: istat

    INTERFACE
      FUNCTION unlink(path) BIND(C,name="unlink") RESULT(errno)
        USE ISO_C_BINDING
        CHARACTER(KIND=C_CHAR), DIMENSION(*)     :: path
        INTEGER(KIND=C_INT)                      :: errno
      END FUNCTION
    END INTERFACE

    INTERFACE
      FUNCTION rename(src, dest) BIND(C,name="rename") RESULT(errno)
        USE ISO_C_BINDING
        CHARACTER(KIND=C_CHAR), DIMENSION(*)     :: src, dest
        INTEGER(KIND=C_INT)                      :: errno
      END FUNCTION
    END INTERFACE

    IF (TARGET==source) THEN
       WRITE(*,*) "Warning: m_mov ",TRIM(TARGET)," equals ", TRIM(source)
       RETURN
    ENDIF

    ! first remove target (needed on windows / mingw)
    istat = unlink(TRIM(TARGET)//c_null_char)
    ! ignore istat of unlink

    ! now move
    istat = rename(TRIM(source)//c_null_char, TRIM(TARGET)//c_null_char)
    IF (istat .NE. 0) THEN
      WRITE(*,*) "Trying to move "//TRIM(source)//" to "//TRIM(TARGET)//"."
      WRITE(*,*) "rename returned status: ",istat
      STOP "Problem moving file"
    ENDIF
  END SUBROUTINE m_mov


! *****************************************************************************
! *****************************************************************************
  SUBROUTINE m_hostnm(hname)
    CHARACTER(len=*), INTENT(OUT)            :: hname

    INTEGER                                  :: istat, i
    CHARACTER(len=default_path_length)       :: buf

    INTERFACE
      FUNCTION  gethostname(buf, buflen) BIND(C,name="gethostname") RESULT(errno)
        USE ISO_C_BINDING
        CHARACTER(KIND=C_CHAR), DIMENSION(*)     :: buf
        INTEGER(KIND=C_INT), VALUE               :: buflen
        INTEGER(KIND=C_INT)                      :: errno
      END FUNCTION
    END INTERFACE

    istat = gethostname(buf, LEN(buf))
    IF(istat /= 0) STOP "m_hostnm failed"
    i = INDEX(buf, c_null_char) -1
    hname = buf(1:i)
  END SUBROUTINE m_hostnm


! *****************************************************************************
! *****************************************************************************
  SUBROUTINE m_getcwd(curdir)
    CHARACTER(len=*), INTENT(OUT)            :: curdir
    TYPE(C_PTR)                              :: stat
    INTEGER                                  :: i
    CHARACTER(len=default_path_length), TARGET  :: tmp

    INTERFACE
      FUNCTION  getcwd(buf, buflen) BIND(C,name="getcwd") RESULT(stat)
        USE ISO_C_BINDING
        CHARACTER(KIND=C_CHAR), DIMENSION(*)     :: buf
        INTEGER(KIND=C_INT), VALUE               :: buflen
        TYPE(C_PTR)                              :: stat
      END FUNCTION
    END INTERFACE

    stat = getcwd(tmp, LEN(tmp))
    IF(.NOT. C_ASSOCIATED(stat)) STOP "m_getcwd failed"
    i = INDEX(tmp, c_null_char) -1
    curdir = tmp(1:i)
  END SUBROUTINE m_getcwd


! *****************************************************************************
! *****************************************************************************
  SUBROUTINE m_chdir(dir,ierror)
    CHARACTER(len=*), INTENT(IN)             :: dir
    INTEGER, INTENT(OUT)                     :: ierror

    INTERFACE
      FUNCTION chdir(path) BIND(C,name="chdir") RESULT(errno)
        USE ISO_C_BINDING
        CHARACTER(KIND=C_CHAR), DIMENSION(*)     :: path
        INTEGER(KIND=C_INT)                      :: errno
      END FUNCTION
    END INTERFACE

    ierror = chdir(TRIM(dir)//c_null_char)
  END SUBROUTINE m_chdir


! *****************************************************************************
! *****************************************************************************
  SUBROUTINE m_getlog(user)
    CHARACTER(len=*), INTENT(OUT)            :: user

    ! this is needed to load a statically linked binary on some architectures.
#if defined(__HAS_NO_SHARED_GLIBC)
    user="root ;-)"
#else
    TYPE, BIND(C) :: passwd_struct
      TYPE(C_PTR)             :: name
      !... more fields, which we don't need and where Linux deviates from POSIX
    END TYPE passwd_struct

    INTEGER                                  :: uid, i
    TYPE(passwd_struct), POINTER             :: pwd
    TYPE(C_PTR)                              :: pwd_cptr
    CHARACTER, DIMENSION(:), POINTER         :: pwd_name_p
    CHARACTER(len=default_string_length)     :: name_long

    INTERFACE
     FUNCTION getpwuid(uid) BIND(C,name="getpwuid") RESULT(result)
       USE ISO_C_BINDING
       INTEGER(KIND=C_INT), VALUE               :: uid
       TYPE(C_PTR)                              :: result
     END FUNCTION
    END INTERFACE

    pwd_cptr = C_NULL_PTR
    NULLIFY(pwd, pwd_name_p)
    name_long = ""

    CALL m_getuid(uid)
    WRITE(user,'(I16)') uid
    pwd_cptr = getpwuid(uid)

    IF(.NOT.C_ASSOCIATED(pwd_cptr)) RETURN
    CALL C_F_POINTER(pwd_cptr, pwd)
    IF(.NOT.C_ASSOCIATED(pwd%name)) RETURN
    CALL C_F_POINTER(pwd%name, pwd_name_p, (/LEN(name_long)/) )

    DO i=1, LEN(name_long)
      IF(pwd_name_p(i) .EQ. C_NULL_CHAR) EXIT
      name_long(i:i) = pwd_name_p(i)
    END DO
    IF(i > LEN(name_long)) RETURN

    user = TRIM(name_long)
#endif
  END SUBROUTINE m_getlog


! *****************************************************************************
! *****************************************************************************
  SUBROUTINE m_getuid(uid)
   INTEGER, INTENT(OUT)                     :: uid

   INTERFACE
     FUNCTION getuid() BIND(C,name="getuid") RESULT(uid)
       USE ISO_C_BINDING
       INTEGER(KIND=C_INT)              :: uid
     END FUNCTION
   END INTERFACE

   uid = getuid()
  END SUBROUTINE m_getuid


! *****************************************************************************
! *****************************************************************************
  SUBROUTINE m_getpid(pid)
   INTEGER, INTENT(OUT)                     :: pid

   INTERFACE
     FUNCTION getpid() BIND(C,name="getpid") RESULT(pid)
       USE ISO_C_BINDING
       INTEGER(KIND=C_INT)              :: pid
     END FUNCTION
   END INTERFACE

   pid = getpid()
  END SUBROUTINE m_getpid


! *****************************************************************************
! *****************************************************************************
  SUBROUTINE m_getarg(i,arg)
    INTEGER, INTENT(IN)                      :: i
    CHARACTER(len=*), INTENT(OUT)            :: arg
    CHARACTER(len=1024)                      :: tmp
    INTEGER                                  :: istat

    CALL GET_COMMAND_ARGUMENT(i, tmp, status=istat)
    IF(istat /= 0) STOP "m_getarg failed"
    arg = TRIM(tmp)
  END SUBROUTINE m_getarg

