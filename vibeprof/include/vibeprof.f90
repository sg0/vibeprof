! vibeprof: phase markers for Fortran.
!
!   use vibeprof
!   call vibeprof_phase_begin('solver')
!   ...
!   call vibeprof_phase_end('solver')
!
! Compile this file with your Fortran compiler and link with -lvibeprof (or -lvibeprof_stub).
module vibeprof
  use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
  implicit none
  private
  public :: vibeprof_phase_begin, vibeprof_phase_end, vibeprof_mark, vibeprof_enabled

  interface
    subroutine c_phase_begin(name) bind(C, name="vibeprof_phase_begin")
      import :: c_char
      character(kind=c_char), dimension(*), intent(in) :: name
    end subroutine
    subroutine c_phase_end(name) bind(C, name="vibeprof_phase_end")
      import :: c_char
      character(kind=c_char), dimension(*), intent(in) :: name
    end subroutine
    subroutine c_mark(name) bind(C, name="vibeprof_mark")
      import :: c_char
      character(kind=c_char), dimension(*), intent(in) :: name
    end subroutine
    function c_enabled() bind(C, name="vibeprof_enabled") result(on)
      import :: c_int
      integer(c_int) :: on
    end function
  end interface

contains

  subroutine vibeprof_phase_begin(name)
    character(len=*), intent(in) :: name
    call c_phase_begin(trim(name) // c_null_char)
  end subroutine

  subroutine vibeprof_phase_end(name)
    character(len=*), intent(in) :: name
    call c_phase_end(trim(name) // c_null_char)
  end subroutine

  subroutine vibeprof_mark(name)
    character(len=*), intent(in) :: name
    call c_mark(trim(name) // c_null_char)
  end subroutine

  logical function vibeprof_enabled()
    vibeprof_enabled = c_enabled() /= 0
  end function

end module vibeprof
