!***********************************************************************
!
!                 AUTOMATICALLY GENERATED FILE 
!            ALL CHANGES TO THIS FILE WILL BE LOST
!
! File created by mkdummyinc from:
!   Source file:  eos_idealgas.f90
! and
!   Dummy file:   noeos.f90
!
!***********************************************************************
    subroutine init_eos(f)
!
      real, dimension (mx,my,mz,mfarray), intent(inout) :: f
!
      call keep_compiler_quiet(f)
!
    endsubroutine init_eos
!*********************************************************************** 
    subroutine ioncalc(f)
!
    real, dimension (mx,my,mz,mfarray) :: f
!
    call keep_compiler_quiet(f)
!
    endsubroutine ioncalc
!*********************************************************************** 
    subroutine gettemperature(f,TT_tmp)
!
     real, dimension (mx,my,mz,mfarray), optional :: f
     real, dimension (mx,my,mz), intent(out) :: TT_tmp
!
     call not_implemented('gettemperature','for this EOS')
!
     call keep_compiler_quiet(present(f))
     call keep_compiler_quiet(TT_tmp)
!
    endsubroutine gettemperature
!*********************************************************************** 
    subroutine getpressure(pp_tmp,TT_tmp,rho_tmp,mu1_tmp)
!
     real, dimension (nx), intent(out) :: pp_tmp
     real, dimension (nx), intent(in)  :: TT_tmp,rho_tmp,mu1_tmp
!
     call not_implemented('getpresssure','for this EOS')
!
     call keep_compiler_quiet(pp_tmp)
     call keep_compiler_quiet(TT_tmp)
     call keep_compiler_quiet(rho_tmp)
     call keep_compiler_quiet(mu1_tmp)
!
    endsubroutine getpressure
!*********************************************************************** 
    subroutine isothermal_lnrho_ss(lnrho,T0,rho0,ss)
!
      real, dimension (mx,my,mz), intent(out) :: lnrho,ss
      real, intent(in) :: T0,rho0
!
      call not_implemented('isothermal_lnrho_ss','for this EOS')
!
      call keep_compiler_quiet(lnrho,ss)
      call keep_compiler_quiet(T0,rho0)
!
    endsubroutine isothermal_lnrho_ss
!*********************************************************************** 
    subroutine bc_ss_flux(f,topbot,lone_sided)
!
!  constant flux boundary condition for entropy (called when bcz='c1')
!  This implementation is intended to work for all supported EOS.
!
!  23-jan-2002/wolf: coded
!  11-jun-2002/axel: moved into the entropy module
!   8-jul-2002/axel: split old bc_ss into two
!  26-aug-2003/tony: distributed across ionization modules
!  13-mar-2011/pete: c1 condition for z-boundaries with Kramers' opacity
!   4-jun-2015/MR: factor cp added in front of tmp_xy
!  30-sep-2016/MR: changes for use of one-sided BC formulation (chosen by setting new optional switch lone_sided)
!  21-jun-2024/Kishore: account for bounds on Kramers conductivity
!  07-dec-2024/Kishore: copied from eos_idealgas to eos_idealgas_vapor and modified to be EOS-agnostic. Also reduced duplicated code.
!  09-dec-2024/Kishore: moved to noeos.f90 to replace an earlier implementation that looked correct only for eos_idealgas.
!  TODO: after verifying that this works correctly for all EOS, move it to entropy.f90
!  NOTE: non-dummy implementations (excluding this) are in eos_ionization and eos_temperature_ionization
!
      use DensityMethods, only: getdlnrho_z, getderlnrho_z, getrho
      use Deriv, only: bval_from_neumann, set_ghosts_for_onesided_ders
      use General, only: loptest
      use SharedVariables, only: get_shared_variable
!
      real, pointer :: Flux,FbyK,chi
      real, pointer :: hcond0_kramers, nkramers, chimax_kramers, chimin_kramers
      logical, pointer :: lmultilayer, lheatc_chiconst, lheatc_kramers, lheatc_Kprof, lheatc_Kconst
!
      integer, intent(IN) :: topbot
      real, dimension (mx,my,mz,mfarray) :: f
      logical, optional :: lone_sided
!
      real, dimension (mx,my) :: FbyKT_xy, TT_xy, rho_xy, Krho1kr_xy, cp, cv, pp_xy
      integer :: i, il, im, ivars, n, ig1, ig2, dir
!
      if (ldebug) print*,'bc_ss_flux: ENTER - cs20,cs0=',cs20,cs0
!
      call get_shared_variable('lheatc_chiconst',lheatc_chiconst)
      call get_shared_variable('lheatc_kramers',lheatc_kramers)
      call get_shared_variable('lheatc_Kprof',lheatc_Kprof)
      call get_shared_variable('lheatc_Kconst',lheatc_Kconst)
!
!     TODO: is there a neat way to guard against simultaneous use of heat conductivities (which is currently not supported here)?
!
      if (lheatc_kramers) then
        call get_shared_variable('hcond0_kramers',hcond0_kramers)
        call get_shared_variable('nkramers',nkramers)
        call get_shared_variable('chimax_kramers',chimax_kramers)
        call get_shared_variable('chimin_kramers',chimin_kramers)
      endif
!
      if (lheatc_chiconst) then
        call get_shared_variable('chi',chi)
      endif
!
      select case (topbot)
!
      case(BOT)
!
        if (lheatc_chiconst.or.lheatc_kramers) then
          call get_shared_variable('Fbot',Flux)
        else
          call get_shared_variable('FbotKbot',FbyK)
        endif
        n=n1
        ig1=-1
        ig2=-nghost
        dir=-1
!
      case(TOP)
!
        if (lheatc_chiconst.or.lheatc_kramers) then
          call get_shared_variable('Ftop',Flux)
        else
          call get_shared_variable('FtopKtop',FbyK)
        endif
        n=n2
        ig1=1
        ig2=nghost
        dir=1
!
      case default
        call fatal_error('bc_ss_flux','invalid argument')
      endselect
!
!AXEL: not needed for noeos. Can remove if auto-test ok.
!KG: restored since this subroutine is used other EOS modules (even
!KG: eos_idealgas) through the eos_dummies.inc mechanism. Commenting this out
!KG: breaks most of the convection auto-tests (e.g. conv-slab)
      do il=1,mx
        do im =1,my
          call get_gamma_etc(cp=cp(il,im), cv=cv(il,im), f=f(il,im,n,:))
        enddo
      enddo
!
      if (pretend_lnTT) then
!
        if (.not.(lheatc_Kprof.or.lheatc_Kconst)) call not_implemented('bc_ss_flux', &
          'for this heat conductivity when lpretend_lnTT=T')
!
        FbyKT_xy=-FbyK/exp(f(:,:,n,iss))
        do i=ig1,ig2,dir
          f(:,:,n+i,iss) = f(:,:,n-i,iss) - dir*dz2_bound(i)*FbyKT_xy
        enddo
      else
!
        call getrho(f(:,:,n,ilnrho),rho_xy)
!
        if (ldensity_nolog) then
          ivars=irho_ss
        else
          ivars=ilnrho_ss
        endif
        do il=1,mx
          do im=1,my
            call eoscalc(ivars,f(il,im,n,:),lnTT=TT_xy(il,im), pp=pp_xy(il,im))
          enddo
        enddo
        TT_xy = exp(TT_xy)
!
!  Check whether we have chi=constant at the boundary, in which case
!  we have the nonconstant rho_xy*chi in FbyKT_xy.
!  Check also whether Kramers opacity is used, then hcond itself depends
!  on density and temperature.
!
        if (lheatc_chiconst) then
          !FbyKT_xy=Flux/(rho_xy*chi*TT_xy)
          !Kishore: I think the above misses a factor of cp.
          FbyKT_xy=Flux/(rho_xy*chi*cp*TT_xy)
        else if (lheatc_kramers) then
          Krho1kr_xy = hcond0_kramers*rho_xy**(-2*nkramers-1)*TT_xy**(6.5*nkramers)
!
          if (chimin_kramers>0) Krho1kr_xy = max(Krho1kr_xy, chimin_kramers*cp)
          if (chimax_kramers>0) Krho1kr_xy = min(Krho1kr_xy, chimax_kramers*cp)
!
          FbyKT_xy=Flux/(rho_xy*Krho1kr_xy*TT_xy)
        else
          if (headtt.and..not.(lheatc_Kprof.or.lheatc_Kconst)) then
            call warning('bc_ss_flux', 'FtopKtop and FbotKbot may not be correctly set.')
          endif
          FbyKT_xy=FbyK/TT_xy
        endif
!
!  enforce ds/dz + (p/(rho*TT))*dlnrho/dz = - cv*F/(K*TT)
!
        if (loptest(lone_sided)) then
          call not_implemented('bc_ss_flux', 'one-sided BC')
!           call getderlnrho_z(f,n,rho_xy)                           ! rho_xy=d_z ln(rho)
!           call bval_from_neumann(f,topbot,iss,3,rho_xy)
!           call set_ghosts_for_onesided_ders(f,topbot,iss,3,.true.)
        else
          pp_xy = pp_xy/(rho_xy*TT_xy) !pp_xy is now P/(rho*T)
          FbyKT_xy = dir*cv*FbyKT_xy !now (+-)cv*F/(K*T)
          do i=ig1,ig2,dir
            call getdlnrho_z(f(:,:,:,ilnrho),n,i,rho_xy) !rho_xy is now dlnrho
            f(:,:,n+i,iss) =   f(:,:,n-i,iss) - pp_xy*rho_xy &
                             - dz2_bound(i)*FbyKT_xy
          enddo
        endif
      endif
!
    endsubroutine bc_ss_flux
!*********************************************************************** 
    subroutine write_thermodyn
!
      call not_implemented('write_thermodyn','for this EOS')
!
    endsubroutine write_thermodyn
!*********************************************************************** 
    subroutine read_thermodyn(input_file)
!
      character (len=*), intent(in) :: input_file
!
      call not_implemented('read_thermodyn','for this EOS')
!
      call keep_compiler_quiet(input_file)
!
    endsubroutine read_thermodyn
!*********************************************************************** 
    subroutine read_species(input_file)
!
      character (len=*) :: input_file
!
      call not_implemented('read_species','for this EOS')
!
      call keep_compiler_quiet(input_file)
!
    endsubroutine read_species
!*********************************************************************** 
    subroutine find_species_index(species_name,ind_glob,ind_chem,found_specie)
!
      integer, intent(out) :: ind_glob
      integer, intent(inout) :: ind_chem
      character (len=*), intent(in) :: species_name
      logical, intent(out) :: found_specie
!
      call not_implemented('find_species_index','for this EOS')
!
      call keep_compiler_quiet(ind_glob)
      call keep_compiler_quiet(ind_chem)
      call keep_compiler_quiet(species_name)
      call keep_compiler_quiet(found_specie)
!
    endsubroutine find_species_index
!*********************************************************************** 
    subroutine find_mass(element_name,MolMass)
!
      character (len=*), intent(in) :: element_name
      real, intent(out) :: MolMass
!
      call not_implemented('find_mass','for this EOS')
!
      call keep_compiler_quiet(element_name)
      call keep_compiler_quiet(MolMass)
!
    endsubroutine find_mass
!*********************************************************************** 
    subroutine eos_before_boundary(f)
!
!     05-dec-2024/Kishore: added dummy routine
!
      real, dimension (mx,my,mz,mfarray), intent(inout) :: f
!
      call keep_compiler_quiet(f)
!
    endsubroutine eos_before_boundary
!*********************************************************************** 
