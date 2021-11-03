#!/usr/bin/env python

import os, sys
import xarray as xa
import numpy as np

ncFiles = [
    '/global/u1/p/pernak18/RRTMGP/sandbox/iter_optimizations/band01_coefficients_LW_g05-06_iter011.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/iter_optimizations/band02_coefficients_LW_g01-02_iter017.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band03.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band04.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band05.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band06.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band07.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band08.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band09.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band10.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band11.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band12.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band13.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band14.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band15.nc', 
    '/global/u1/p/pernak18/RRTMGP/sandbox/band_k_dist/coefficients_LW_band16.nc'
]

# initialize outputs
fullDict = {}

# what netCDF variables have a g-point dimension and will thus
# need to be modified in the combination iterations?
kMajor = ['kmajor', 'plank_fraction']

# kminor variables that are nontrivial to combine
# for minor contributors
regions = ['lower', 'upper']
kMinor = ['kminor_{}'.format(reg) for reg in regions]

strVars = ['identifier_minor', 'gas_minor', 'gas_names']
minorVars = []
for reg in regions:
    minorVars.append('minor_gases_{}'.format(reg))
    minorVars.append('scaling_gas_{}'.format(reg))
# end region loop

scalars = ['absorption_coefficient_ref_P', 'absorption_coefficient_ref_T', 
           'press_ref_trop']

nGpt = []

# number of minor contributors per band
nPerBandUp = np.zeros(16).astype(int)
nPerBandLo = np.zeros(16).astype(int)
for iNC, ncFile in enumerate(ncFiles):
    with xa.open_dataset(ncFile) as kDS:
        sizeDS = kDS.sizes
        nGpt.append(sizeDS['gpt'])
        ncVars = list(kDS.keys())
        for ncVar in ncVars:
            # don't know what to do with this! for the full band file, 
            # we probably don't need it anymore
            if ncVar == 'gpt_weights': continue

            varDA = kDS[ncVar]
            varArr = varDA.values
            varDims = varDA.dims
            varSizes = varDA.sizes

            # is the variable empty (e.g., no minor contributors)?
            if not varSizes:
                if ncVar in scalars: fullDict[ncVar] = varDA
                continue
            # endif empty var

            # minor contributors coming back to haunt me
            # no contributions in a given band
            minCond1 = ('minor_absorber_intervals_upper' in varDims or \
                'contributors_upper' in varDims) and \
                iNC in [3, 11, 13, 14, 15]
            minCond2 = ('minor_absorber_intervals_lower' in varDims or \
                'contributors_lower' in varDims) and iNC == 13
            if minCond1 or minCond2: continue

            vDim = 'minor_absorber_intervals_upper'
            if vDim in varDims:
                nPerBandUp[iNC] = varSizes[vDim]

            vDim = 'minor_absorber_intervals_lower'
            if vDim in varDims:
                nPerBandLo[iNC] = varSizes[vDim]

            concatAx = -1 if ncVar in kMajor + kMinor else 0
            if ncVar not in fullDict.keys():
                if ncVar in minorVars:
                    fullDict[ncVar] = varArr.tolist()
                else:
                    fullDict[ncVar] = varArr
                # endif ncVar
            elif '_ref' in ncVar or ncVar in strVars:
                # these are the same for every band and can be overwritten
                fullDict[ncVar] = varArr
            else:
                if ncVar in minorVars:
                    fullDict[ncVar] += varArr.tolist()
                elif ncVar in strVars:
                    fullDict[ncVar].append(varArr)
                else:
                    fullDict[ncVar] = np.concatenate(
                        (fullDict[ncVar], varArr), axis=concatAx)
                # endif ncVar
            # endif ncVar

            """
            varDims = varDA.dims
            if (any(modDim in varDims for modDim in modDims)):
                for modDim in modDims:
                    if modDim in varDims: break
                fullDict[ncVar] = xa.concat(
                    [fullDict[ncVar], varDA], dim=modDim)
            else:
                # don't do anything to this variable; identical for 
                # all bands and most recent cant be preserved
                fullDict[ncVar] = xa.DataArray(varDA)
            # endif modDim
            """
        # end ncVar loop
    # endwith
# end band loop

# transform string lists
for sv in strVars: fullDict[sv] = np.array(fullDict[sv]).T
for mv in minorVars: fullDict[mv] = np.array(fullDict[mv]).T

# recalculate g-point related indices
fullDict['bnd_limits_gpt'] = \
    (fullDict['bnd_limits_gpt']-[0,1]).flatten(
    ).cumsum().reshape(-1, 2)

i1, iStartLo, iStartUp = 1, 1, 1
lims_gpt_up, lims_gpt_lo = [], []
startUp, startLo = [], []
for nG, nUp, nLo in zip(nGpt, nPerBandUp, nPerBandLo):
    i2 = i1 + nG - 1
    for i in range(nUp):
        lims_gpt_up.append([i1, i2])
        startUp.append(iStartUp)
        iStartUp += nG
    # end i loop

    for i in range(nLo): 
        lims_gpt_lo.append([i1, i2])
        startLo.append(iStartLo)
        iStartLo += nG
    # end i loop

    i1 += nG
# end minor contributor loop

fullDict['minor_limits_gpt_lower'] = np.array(lims_gpt_lo)
fullDict['minor_limits_gpt_upper'] = np.array(lims_gpt_up)
fullDict['kminor_start_lower'] = np.array(startLo)
fullDict['kminor_start_upper'] = np.array(startUp)

#for key in fullDict.keys(): print(key, fullDict[key].shape)
outDims = {
    'bnd': len(ncFiles), 
    'atmos_layer': 2, 
    'pair': 2, 
    'temperature': 14, 
    'pressure_interp': 60, 
    'mixing_fraction': 9, 
    'gpt': sum(nGpt), 
    'temperature_Planck': 196, 
    'minor_absorber': 21, 
    'absorber': 19, 
    'minor_absorber_intervals_lower': len(lims_gpt_lo), 
    'minor_absorber_intervals_upper': len(lims_gpt_up), 
    'pressure': 59, 
    'absorber_ext': 20, 
    'fit_coeffs': 2, 
    'contributors_lower': fullDict['kminor_lower'].shape[-1], 
    'contributors_upper': fullDict['kminor_upper'].shape[-1]
}

# extract dimensions from reference LBL netCDF for consistency
refNC = '/global/project/projectdirs/e3sm/pernak18/reference_netCDF/' + \
    'g-point-reduce/rrtmgp-data-lw-g256-2018-12-04.nc'
outDict = {}
with xa.open_dataset(refNC) as refDS:
    for key in fullDict.keys():
        data = fullDict[key]
        dims = refDS[key].dims
        outDict[key] = {"dims": dims, "data": data}
# endwith

outCoord = {}
for key in outDims.keys(): 
    outCoord[key] = {'dims': (key), "data": range(outDims[key])}

# make an acceptable dictionary for xarray.Dataset.from_dict()
dsDict = {
    "coords": outCoord,
    "dims": outDims, 
    "data_vars": outDict
}

outDS = xa.Dataset.from_dict(dsDict)
outDS.to_netcdf('temp.nc')
print('Wrote temp.nc')

sys.exit()
import by_band_lib as BYBAND
BYBAND.fluxCompute('temp.nc', 
                   '/project/projectdirs/e3sm/pernak18/reference_netCDF/g-point-reduce/multi_garand_template_broadband.nc', 
                   'k-distribution-opt/rrtmgp_garand_atmos', '.', 'optimized_fluxes.nc')
print('Saved new fluxes to ./optimized_fluxes.nc')