
subroutine stop_on_err(error_msg)
  use iso_fortran_env, only : error_unit
  character(len=*), intent(in) :: error_msg

  if(error_msg /= "") then
    write (error_unit,*) trim(error_msg)
    write (error_unit,*) "rrtmgp_garand_atmos stopping"
    stop
  end if

end subroutine stop_on_err
!-----------------------------
!
! This example program computes clear-sky fluxes for the Garand atmospheres
!   (doi:10.1029/2000JD000184) used to train the RRTMGP gas optics.
! Users supply a file containing the atmospheres and relevant boundary conditions
!   (separately for LW and SW); the program invokes RRTMGP gas optics and
!   computes fluxes up, down, and net (down minus up) with RTE, as well as
!   heating rates. Fluxes are reported for the broadband as well as within each
!   spectral band defined by RRTMGP.
! The program expects files in a certain, arbitrary netCDF file. Results are added
!    to the file containing the inputs.
! The code does either LW or SW calculations; the boundary conditions in the file
!   describing the atmosphere need to be consistent with the absorption coefficient data
! The code divides the work into blocks of columns (user-configurable, default of 4)
!   which makes the code longer and more complciated than it might be but
!   useful for testing e.g. threading implementations.
!
program rrtmgp_garand_atmos
  !
  ! Modules for working with rte and rrtmgp
  !
  ! Working precision for real variables
  !
  use mo_rte_kind,           only: wp
  !
  ! Optical properties of the atmosphere as array of values
  !   In the longwave we include only absorption optical depth (_1scl)
  !   Shortwave calculations would use optical depth, single-scattering albedo, asymmetry parameter (_2str)
  !
  use mo_optical_props,      only: ty_optical_props, &
                                   ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str
  !
  ! Gas optics: maps physical state of the atmosphere to optical properties
  !
  use mo_gas_optics_rrtmgp,         only: ty_gas_optics_rrtmgp
  !
  ! Gas optics uses a derived type to represent gas concentrations compactly...
  !
  use mo_gas_concentrations, only: ty_gas_concs
  !
  ! ... and another type to encapsulate the longwave source functions.
  !
  use mo_source_functions,   only: ty_source_func_lw
  !
  ! RTE  drivers
  !
  use mo_rte_lw,             only: rte_lw
  use mo_rte_sw,             only: rte_sw
  !
  ! Output fluxes by spectral band in addition to broadband
  !   in extensions/
  !
  use mo_fluxes,             only: ty_fluxes_broadband
  !
  ! Simple estimation of heating rates (in extensions/)
  !
  use mo_heating_rates,      only: compute_heating_rate
  !
  ! Serial netCDF I/O, provided in examples/
  !
  use mo_load_coefficients,  only: load_and_init
  use mo_multi_garand_io,    only: read_atmos, is_lw, is_sw, &
                                   read_lw_bc, read_sw_bc, read_lw_rt,  &
                                   write_fluxes_nCase, write_dir_fluxes_nCase, write_heating_rates_nCase, &
                                   write_spectral_disc, read_atmos_nCase
  implicit none
  ! ----------------------------------------------------------------------------------
  ! Variables
  ! ----------------------------------------------------------------------------------
  ! Arrays: dimensions (col, lay)
  real(wp), dimension(:,:,:),   allocatable :: p_lay, t_lay, p_lev
  real(wp), dimension(:,:,:),   allocatable :: col_dry
  !
  ! Longwave only
  !
  real(wp), dimension(:,:,:), allocatable :: t_lev
  real(wp), dimension(:),     allocatable :: t_sfc
  real(wp), dimension(:,:),   allocatable :: emis_sfc ! First dimension is band
  !
  ! Shortwave only
  !
  real(wp), dimension(:),     allocatable :: sza, tsi, mu0
  real(wp), dimension(:,:),   allocatable :: sfc_alb_dir, sfc_alb_dif ! First dimension is band
  real(wp)                                :: tsi_scaling = -999._wp
  !
  ! Source functions
  !
  !   Longwave
  type(ty_source_func_lw)               :: lw_sources
  !   Shortwave
  real(wp), dimension(:,:), allocatable :: toa_flux

  !
  ! Output variables
  !
  real(wp), dimension(:,:  ,:), target, &
                               allocatable ::     flux_up,      flux_dn, &
                                                  flux_net,     flux_dir
  real(wp), dimension(:,:,:),    allocatable :: heating_rate
  !
  ! Derived types from the RTE and RRTMGP libraries
  !
  type(ty_gas_optics_rrtmgp)    :: k_dist
  type(ty_gas_concs)     :: gas_concs_subset
  type(ty_gas_concs),allocatable,dimension(:)  :: gas_concs
  class(ty_optical_props_arry), &
             allocatable :: optical_props
  type(ty_fluxes_broadband) :: fluxes

  !
  ! Inputs to RRTMGP
  !
  logical :: top_at_1

  integer :: ncol, nlay, nbnd, ngpt, nUserArgs=0, nCase, icase
  integer :: b, nBlocks, colS, colE, nSubcols, nang
  integer :: blockSize = 0
  character(len=6) :: block_size_char
  !
  ! k-distribution file and input-output files must be paired: LW or SW
  !
  character(len=256) :: k_dist_file = 'coefficients.nc'
  character(len=256) :: input_file  = "rrtmgp-flux-inputs-outputs.nc"
  real(wp), parameter :: pi = acos(-1._wp)
  ! ----------------------------------------------------------------------------------
  ! Code
  ! ----------------------------------------------------------------------------------
  !
  ! Parse command line for any file names, block size
  !
  nUserArgs = command_argument_count()
  if (nUserArgs < 2) then
     print *, "usage: rrtmgp_garand_atmos <input_file> <k_dist_file> [blockSize]"
     print *, "arguments:                                                       "
     print *, "    input_file   file containing atmos profile information       "
     print *, "    k_dist_file  file containing spectral discretization         "
     print *, "    blockSize    number of columns to process at once (optional) "
     stop
  else
     call get_command_argument(1, input_file)
     call get_command_argument(2, k_dist_file)
  end if
  if (nUserArgs >= 3) then
    call get_command_argument(3, block_size_char)
    read(block_size_char, '(i6)') blockSize
  end if
  if (nUserArgs >  4) print *, "Ignoring command line arguments beyond the first three..."
  !
  ! Read temperature, pressure, gas concentrations, then variables specific
  !  to LW or SW problems. Arrays are allocated as they are read
  !
  call read_atmos_nCase(input_file,  p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry)
  if(is_lw(input_file)) then
    call read_lw_bc(input_file, t_sfc, emis_sfc)
    ! Number of quadrature angles
    call read_lw_rt(input_file, nang)
  else
    call read_sw_bc(input_file, sza, tsi, tsi_scaling, sfc_alb_dir, sfc_alb_dif)
    allocate(mu0(size(sza)))
    mu0 = cos(sza * pi/180.)
  end if

  !
  ! Load the gas optics class with data
  !
  call load_and_init(k_dist, k_dist_file, gas_concs(1))
  if(k_dist%source_is_internal() .neqv. is_lw(input_file)) &
    call stop_on_err("rrtmgp_garand_atmos: gas optics and input-output file disagree about SW/LW")

  !
  ! Problem sizes; allocate output arrays for full problem
  !
  ncol  = size(p_lay,1)
  nlay  = size(p_lay,2)
  nCase = size(p_lay,3)
  nbnd = k_dist%get_nband()
  ngpt = k_dist%get_ngpt()
  top_at_1 = p_lay(1, 1, 1) < p_lay(1, nlay, 1)

  allocate(flux_up(ncol,nlay+1,nCase), flux_dn(ncol,nlay+1,nCase), flux_net(ncol,nlay+1,nCase))
  allocate(heating_rate(ncol,nlay,nCase))
  if(is_sw(input_file)) &
    allocate(flux_dir(ncol,nlay+1,nCase))

  !
  ! LW calculations neglect scattering; SW calculations use the 2-stream approximation
  !   Here we choose the right variant of optical_props.
  !
  if(is_sw(input_file)) then
    allocate(ty_optical_props_2str::optical_props)
  else
    allocate(ty_optical_props_1scl::optical_props)
  end if
  call stop_on_err(optical_props%init(k_dist))

  !
  ! How many columns to do at once? Default is all but user may have specified something
  !
  if (blockSize == 0) blockSize = ncol
  nBlocks = ncol/blockSize ! Integer division

  !
  ! Allocate arrays for the optical properties themselves.
  !
  select type(optical_props)
    class is (ty_optical_props_1scl)
      call stop_on_err(optical_props%alloc_1scl(blockSize, nlay))
    class is (ty_optical_props_2str)
      call stop_on_err(optical_props%alloc_2str(blockSize, nlay))
    class default
      call stop_on_err("rrtmgp_garand_atmos: Don't recognize the kind of optical properties ")
  end select
  !
  ! Source function
  !
  if(is_sw(input_file)) then
    allocate(toa_flux(blockSize, ngpt))
  else
    call stop_on_err(lw_sources%alloc(blockSize, nlay, k_dist))
  end if
  do iCase=1,nCase

    !
    ! Loop over subsets of the problem
    !
    do b = 1, nBlocks
      colS = (b-1) * blockSize + 1
      colE =  b    * blockSize
      nSubcols = colE-colS+1
      call compute_fluxes(colS, colE, iCase)
    end do

    !
    ! Do any leftover columns
    !
    if(mod(ncol, blockSize) /= 0) then
      colS = ncol/blockSize * blockSize + 1  ! Integer arithmetic
      colE = ncol
      nSubcols = colE-colS+1
      !
      ! Reallocate optical properties and source function arrays
      !
      select type(optical_props)
        class is (ty_optical_props_1scl)
          call stop_on_err(optical_props%alloc_1scl(nSubcols, nlay))
        class is (ty_optical_props_2str)
          call stop_on_err(optical_props%alloc_2str(nSubcols, nlay))
        class default
          call stop_on_err("rrtmgp_garand_atmos: Don't recognize the kind of optical properties ")
      end select
      if(is_sw(input_file)) then
        if(allocated(toa_flux)) deallocate(toa_flux)
        allocate(toa_flux(nSubcols, ngpt))
      else
        call stop_on_err(lw_sources%alloc(nSubcols, nlay))
      end if

      call compute_fluxes(colS, colE, iCase)
    end if

    !
    ! Heating rates
    !
    call stop_on_err(compute_heating_rate(flux_up(:,:,iCase), flux_dn(:,:,iCase), &
                                          p_lev(:,:,iCase), heating_rate(:,:,iCase)))
  enddo
  !
  ! ... and write everything out
  !
  call write_spectral_disc(input_file, optical_props)
  call write_fluxes_nCase (input_file, flux_up, flux_dn, flux_net)
  call write_heating_rates_nCase(input_file, heating_rate)
  if(k_dist%source_is_external()) &
    call write_dir_fluxes_nCase(input_file, flux_dir)

contains

  subroutine compute_fluxes(colS, colE, iCase)
    integer, intent(in) :: colS, colE, iCase
    !
    ! This routine compute fluxes: LW or SW, without or without col_dry provided to gas_optics()
    !   Most variables come from the host program; this just keeps us from repeating a bunch of code
    !   in two places
    !
    fluxes%flux_up      => flux_up(colS:colE,:,iCase)
    fluxes%flux_dn      => flux_dn(colS:colE,:,iCase)
    fluxes%flux_net     => flux_net(colS:colE,:,iCase)
    if(is_sw(input_file)) then
      fluxes%flux_dn_dir     => flux_dir(colS:colE,:,iCase)
    end if
    call stop_on_err(gas_concs(iCase)%get_subset(colS, nSubcols, gas_concs_subset))
    if(is_sw(input_file)) then
      !
      ! Gas optics, including source functions
      !   There are two entries because the test files used during developement contain
      !   the field col_dry, the number of molecules per sq cm. Users will normally let
      !   RRTMGP compute this internally but it can be provided as an optional argument.
      !   We provide the value during validation to minimize one source of difference
      !   with reference calculations.
      !
      if(allocated(col_dry)) then
        call stop_on_err(k_dist%gas_optics(p_lay(colS:colE,:,iCase), &
                                           p_lev(colS:colE,:,iCase), &
                                           t_lay(colS:colE,:,iCase), &
                                           gas_concs_subset,   &
                                           optical_props,      &
                                           toa_flux,           &
                                           col_dry = col_dry(colS:colE,:,iCase)))
      else
        call stop_on_err(k_dist%gas_optics(p_lay(colS:colE,:,iCase), &
                                           p_lev(colS:colE,:,iCase), &
                                           t_lay(colS:colE,:,iCase), &
                                           gas_concs_subset,   &
                                           optical_props,      &
                                           toa_flux))
      end if
      if(tsi_scaling > 0.0_wp) toa_flux(:,:) =  toa_flux(:,:) * tsi_scaling
      !
      ! Radiative transfer
      !
      call stop_on_err(rte_sw(optical_props,               &
                                 top_at_1,                 &
                                 mu0(colS:colE),           &
                                 toa_flux,                 &
                                 sfc_alb_dir(:,colS:colE), &
                                 sfc_alb_dif(:,colS:colE), &
                                 fluxes))
    else
      !
      ! Gas optics, including source functions
      !
      if(allocated(col_dry)) then
        call stop_on_err(k_dist%gas_optics(p_lay(colS:colE,:,iCase), &
                                           p_lev(colS:colE,:,iCase), &
                                           t_lay(colS:colE,:,iCase), &
                                           t_sfc(colS:colE  ), &
                                           gas_concs_subset,   &
                                           optical_props,      &
                                           lw_sources,         &
                                           tlev    = t_lev  (colS:colE,:,iCase), &
                                           col_dry = col_dry(colS:colE,:,iCase)))
      else
        call stop_on_err(k_dist%gas_optics(p_lay(colS:colE,:,iCase), &
                                           p_lev(colS:colE,:,iCase), &
                                           t_lay(colS:colE,:,iCase), &
                                           t_sfc(colS:colE  ), &
                                           gas_concs_subset,   &
                                           optical_props,      &
                                           lw_sources,         &
                                           tlev    = t_lev  (colS:colE,:,iCase)))
      end if
      !
      ! Radiative transfer
      !
      call stop_on_err(rte_lw(optical_props,            &
                                 top_at_1,              &
                                 lw_sources,            &
                                 emis_sfc(:,colS:colE), &
                                 fluxes, n_gauss_angles = nang))
    end if
  end subroutine compute_fluxes
end program rrtmgp_garand_atmos
