#
# Need to set variables RRTMGP_ROOT, FC, FCGLAGS
# e.g. export FC=ifort; export FCFLAGS='-m64 -O3 -traceback -assume realloc_lhs -extend-source 132 -qopt-report-phase=vec'
#
#
# Location of RTE+RRTMGP libraries, module files.
#
RRTMGP_BUILD = $(RRTMGP_ROOT)/build
#
# RRTMGP library, module files
#
LDFLAGS   += -L$(RRTMGP_BUILD)
LIBS      += -lrrtmgp -lrte
FCINCLUDE += -I$(RRTMGP_BUILD)
#
# netcdf library, module files
# Environment variables NCHOME and NFHOME point to root of C and Fortran interfaces respectively -
#
FCINCLUDE += -I$(NFHOME)/include
LDFLAGS   += -L$(NFHOME)/lib -L$(NCHOME)/lib
LIBS      += -lnetcdff -lnetcdf

VPATH  = .:$(RRTMGP_ROOT)/examples:$(RRTMGP_ROOT)/extensions

# Compilation rules
%.o: %.F90
	$(FC) $(FCFLAGS) $(FCINCLUDE) -c $<
%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)
#
# Extra sources -- extensions to RRTMGP classes, shared infrastructure, local sources
#
ADDITIONS  = mo_load_coefficients.o mo_simple_netcdf.o
ADDITIONS += mo_test_files_io.o mo_heating_rates.o mo_fluxes_byband.o mo_fluxes_byband_kernels.o mo_rrtmgp_clr_all_sky.o

#
# Targets
#
all: rrtmgp_garand_atmos

rrtmgp_garand_atmos: $(ADDITIONS) rrtmgp_garand_atmos.o
rrtmgp_garand_atmos.o:  rrtmgp_garand_atmos.F90 $(ADDITIONS)
mo_fluxes_byband.o:     mo_fluxes_byband.F90 mo_fluxes_byband_kernels.o
mo_load_coefficients.o: mo_load_coefficients.F90 mo_simple_netcdf.o
mo_test_files_io.o:     mo_test_files_io.F90     mo_simple_netcdf.o

clean:
	-rm ../rrtmgp_garand_atmos *.o *.mod *.optrpt
