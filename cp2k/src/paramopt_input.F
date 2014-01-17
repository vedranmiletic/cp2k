!-----------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations         !
!   Copyright (C) 2000 - 2012  CP2K developers group                          !
!-----------------------------------------------------------------------------!

! *****************************************************************************
!> \brief performs global geometry optimization
!> \par History
!> \author Ole
! *****************************************************************************
MODULE paramopt_input
  USE cp_output_handling,              ONLY: cp_print_key_section_create
  USE input_constants,                 ONLY: add_last_numeric,&
                                             low_print_level,&
                                             swarm_do_glbopt,&
                                             swarm_do_paramopt,&
                                             paramopt_no_run,&
                                             paramopt_sampling_run,&
                                             paramopt_powell_run,&
                                             paramopt_gaussian_run, &
                                             paramopt_lbfgs_run, &
                                             paramopt_cmaes_run
  USE input_keyword_types,             ONLY: keyword_create,&
                                             keyword_release,&
                                             keyword_type
  USE input_section_types,             ONLY: section_add_keyword,&
                                             section_add_subsection,&
                                             section_create,&
                                             section_release,&
                                             section_type
  USE input_val_types,                 ONLY: integer_t,&
                                             real_t
  USE kinds,                           ONLY: dp
  USE string_utilities,                ONLY: s2a

#include "cp_common_uses.h"

 IMPLICIT NONE
 PRIVATE

  CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'paramopt_input'

 PUBLIC :: paramopt_declare_input

  CONTAINS



!! *****************************************************************************
!! *****************************************************************************
  SUBROUTINE paramopt_declare_input(swarm_section,error)
    TYPE(section_type), POINTER              :: swarm_section
    TYPE(cp_error_type), INTENT(inout)       :: error

    TYPE(keyword_type), POINTER              :: keyword
    TYPE(section_type), POINTER              :: paramopt_section
    TYPE(section_type), POINTER              :: cmaes_section

    NULLIFY(paramopt_section, cmaes_section, keyword)

    CALL section_create(paramopt_section,name="PARAM_OPT",&
         description="blah TODO.",&
         repeats=.FALSE., supported_feature=.TRUE.,error=error)

    CALL keyword_create(keyword=keyword,&
         name="BENCHMARK",&
         description="blabla", &
         usage="BENCHMARK {LOGICAL}",&
         default_l_val=.FALSE., lone_keyword_l_val=.TRUE., error=error)
    CALL section_add_keyword(paramopt_section,keyword,error=error)
    CALL keyword_release(keyword,error=error)  
   
    CALL keyword_create(keyword, name="NUM_VAR",&
         description="balba TODO.",&
         default_i_val= 1,error=error)
    CALL section_add_keyword(paramopt_section, keyword, error=error)
    CALL keyword_release(keyword,error=error) 

    CALL keyword_create(keyword, name="SEED",&
         description="balba TODO.",&
         default_i_val= -1,error=error)
    CALL section_add_keyword(paramopt_section, keyword, error=error)
    CALL keyword_release(keyword,error=error) 
     
    CALL keyword_create(keyword, name="RUN_TYPE",&
         description="Type of run that you want to perform ",&
         usage="RUN_TYPE SAMPLING",&
         default_i_val=paramopt_powell_run,&
         enum_c_vals=s2a("NONE","SAMPLING","POWELL","GAUSSIAN","LBFGS","CMAES"),&
         enum_i_vals=(/paramopt_no_run, paramopt_sampling_run, &
         paramopt_powell_run, paramopt_gaussian_run, paramopt_lbfgs_run, paramopt_cmaes_run/),&
         supported_feature=.FALSE.,error=error)
    CALL section_add_keyword(paramopt_section,keyword,error=error)
    CALL keyword_release(keyword,error=error)

    CALL keyword_create(keyword, name="STEP_SIZE",&
         description="balba TODO.",&
         default_r_val= 0.0_dp,error=error)
    CALL section_add_keyword(paramopt_section, keyword, error=error)
    CALL keyword_release(keyword,error=error)

     CALL keyword_create(keyword, name="C_T",&
         description="balba TODO.",&
         default_r_val= 1._dp,error=error)
    CALL section_add_keyword(paramopt_section, keyword, error=error)
    CALL keyword_release(keyword,error=error)   

    CALL keyword_create(keyword, name="INSIGMA",&
         description="blabla TODO",&
         n_var=-1,type_of_var=real_t,default_r_val= 0.0_dp,error=error)
    CALL section_add_keyword(paramopt_section,keyword,error=error)
    CALL keyword_release(keyword,error=error)

    CALL keyword_create(keyword, name="INITIAL_X",&
         description="blabla TODO",&
         n_var=-1,type_of_var=real_t,required=.FALSE.,default_r_vals=(/0._dp/),error=error)
    CALL section_add_keyword(paramopt_section,keyword,error=error)
    CALL keyword_release(keyword,error=error)

    CALL keyword_create(keyword, name="UPPER_BOUNDS",&
         description="blabla TODO",&
         n_var=-1,type_of_var=real_t,error=error)
    CALL section_add_keyword(paramopt_section,keyword,error=error)
    CALL keyword_release(keyword,error=error)
    
    CALL keyword_create(keyword, name="LOWER_BOUNDS",&
         description="blabla TODO",&
         n_var=-1,type_of_var=real_t,error=error)
    CALL section_add_keyword(paramopt_section,keyword,error=error)
    CALL keyword_release(keyword,error=error)   
     
    CALL keyword_create(keyword, name="WEIGHTS",&
         description="blabla TODO",&
         n_var=-1,type_of_var=real_t,error=error)
    CALL section_add_keyword(paramopt_section,keyword,error=error)
    CALL keyword_release(keyword,error=error)   
     
    CALL section_create(cmaes_section,name="CMAES_OPT",&
         description="blah TODO.",&
         repeats=.FALSE., supported_feature=.TRUE.,error=error)

    CALL keyword_create(keyword, name="LAMBDA_MO",&
         description="balba TODO.",&
         default_i_val=1,error=error)
    CALL section_add_keyword(cmaes_section, keyword, error=error)
    CALL keyword_release(keyword,error=error)

    CALL keyword_create(keyword, name="ADDITIONAL_CONS",&
         description="balba TODO.",&
         default_l_val=.FALSE.,&
         lone_keyword_l_val = .TRUE.,&
         error=error)
    CALL section_add_keyword(cmaes_section, keyword, error=error)
    CALL keyword_release(keyword,error=error)

    CALL keyword_create(keyword, name="PENALTY",&
         description="balba TODO.",&
         default_r_val= 1._dp,error=error)
    CALL section_add_keyword(cmaes_section, keyword, error=error)
    CALL keyword_release(keyword,error=error)

    CALL section_add_subsection(paramopt_section, cmaes_section, error=error)
    CALL section_add_subsection(swarm_section, paramopt_section, error=error)
    CALL section_release(cmaes_section,error=error)
    CALL section_release(paramopt_section,error=error)

  END SUBROUTINE paramopt_declare_input
END MODULE paramopt_input

