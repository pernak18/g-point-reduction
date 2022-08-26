# standard libraries
import os, sys, shutil

# "standard" install
import numpy as np

# directory in which libraries installed with conda are saved
PIPPATH = '{}/.local/'.format(os.path.expanduser('~')) + \
    'cori/3.7-anaconda-2019.10/lib/python3.7/site-packages'
PATHS = ['common', PIPPATH]
for path in PATHS: sys.path.append(path)

# user must do `pip install xarray` on cori (or other NERSC machines)
import xarray as xa

# GLOBAL VARIABLES (paths)
PROJECT = '/global/project/projectdirs/e3sm/pernak18/'
EXE = '{}/g-point-reduction/garand_atmos/rrtmgp_garand_atmos'.format(
    PROJECT)
GARAND = '{}/reference_netCDF/g-point-reduce/'.format(PROJECT) + \
  'multi_garand_template_single_band.nc'
CWD = os.getcwd()

# these paths are needed for all-band flux calculations
EXEFULL = PROJECT + \
  '/g-point-reduction/k-distribution-opt/rrtmgp_garand_atmos'
NCFULLPROF = PROJECT + \
    '/reference_netCDF/g-point-reduce/multi_garand_template_broadband.nc'

# default cost function components, level indices (surface, , weights
CFCOMPS = ['flux_net', 'band_flux_net']
BOUNDS = [0, 26, 42]
CFLEVS = {'flux_net': BOUNDS, 'band_flux_net': BOUNDS}
CFWGT = [0.5, 0.5]

def pathCheck(path, mkdir=False):
    """
    Determine if file exists. If not, throw an Assertion Exception
    """

    if mkdir:
        # mkdir -p -- create dir tree
        if not os.path.exists(path): os.makedirs(path)
    else:
        assert os.path.exists(path), 'Could not find {}'.format(path)
    # endif mkdir
# end pathCheck

def fluxCompute(inK, profiles, exe, fluxDir, outFile):
    """
    Compute fluxes for a given k-distribution and set of atmospheric
    conditions

    Inputs
        inK -- string, k-distribution file to use in flux calcs
        profiles -- string, netCDF with profile specifications
        exe -- string, RRTMGP flux calculation executable path
        fluxDir -- string, directory in which the executable is run
        outFile -- string, file to which RRTMGP output is renamed
    """

    # standard library
    import subprocess as sub

    pathCheck(fluxDir, mkdir=True)
    cwd = os.getcwd()
    os.chdir(fluxDir)

    # file staging for RRTMGP run
    # trying to keep this simple/compact so we don't end up with a
    # bunch of files in a directory
    aPaths = [exe, inK]
    rPaths = ['./run_rrtmgp', 'coefficients.nc']
    for aPath, rPath in zip(aPaths, rPaths):
        if os.path.islink(rPath): os.unlink(rPath)
        os.symlink(aPath, rPath)
    # end rPath loop

    # so we don't overwrite the LBL results
    inRRTMGP = 'rrtmgp-inputs-outputs.nc'
    shutil.copyfile(profiles, inRRTMGP)

    # assuming the RRTMGP call sequence is `exe inputs k-dist`
    rPaths.insert(1, inRRTMGP)

    # run the model with inputs
    sub.call(rPaths)

    # save outputs (inRRTMGP gets overwritten every run)
    os.rename(inRRTMGP, outFile)
    #print('Wrote {}'.format(outFile))

    os.chdir(cwd)

    return outFile
# end fluxCompute()

def costCalc(lblNC, testNC, doLW, compNameCF, pLevCF, costComp0, scale, init):
    """
    Calculate cost of test dataset with respect to reference dataset 
    at a given number of pressure levels and for a given set of 
    components (upwelling flux, net flux, heating rate, etc.). Also keep 
    other diagnostics normalized to cost of initial 256 g-point k-distribution

    Call
        costDict = costCalc(
            lblDS, testDS, doLW, compNameCF, pLevCF, costComp0, scale, init)

    Inputs
        lblDS -- xarray dataset, LBLRTM reference
        testDS -- xarray dataset, RRTMGP trial dataset (i.e., g-points combined)
        doLW -- boolean, LW or SW parameters used in cost
        compNameCF -- string list of cost function component names (e.g., "flux_net")
        pLevCF -- dictionary; keys for each compNameCF, float iterable (list) values 
            of pressure levels at which each CF component is evaluated
        costComp0 -- dictionary; keys for each compNameCF, float scalar values of 
            initial cost associated with RRTMGP full k-distribution
        scale -- dictionary; keys for each compNameCF, float scalar values
        init -- boolean, calculate initial cost i.e., with the full
            256 g-point k-distribution)

    Outputs
        dictionary with following keys:
            allComps -- float array, weighted cost for each 
                cost function component, normalized WRT cost of full 
                k-distribution (= 100)
            totalCost -- float, total of allComps array
            dCost -- float, change in cost WRT cost of full k-distribution
            costComps -- dictionary; keys are cost components names, 
                values are weighted cost at given component and pressure 
                level, summed over all profiles, normalized with initial cost
            dCostComps -- dictionary; keys are cost components names, 
                values are weighted changes in cost at given component and 
                pressure level, summed over all profiles, normalized with 
                initial cost

    Keywords
        None
    """

    lblDS = xa.open_dataset(lblNC)
    testDS = xa.open_dataset(testNC)
    allComps = []

    # add diffuse to SW dataset
    # should this be done outside of code?
    # TO DO: BAND DIRECT NO WORKING YET
    if not doLW:
        lblDS['flux_dif_net'] = lblDS['flux_dif_dn'] - \
            lblDS['flux_up']

        if init:
            testDS['flux_dif_dn'] = testDS['flux_dn'] - \
                testDS['flux_dir_dn']
            testDS['flux_dif_net'] = testDS['flux_dif_dn'] - \
                testDS['flux_up']
        # endif init
    # endif LW

    # for diagnostics, we keep the cost and delta-cost for 
    # each component; in the total cost, we average over profiles 
    # AND pLevCF, but for diagnostics we break down by pLevCF
    costComps = {}
    dCostComps = {}

    # first calculate weighted cost for each component
    for comp in compNameCF:
        # pressure dimension will depend on parameter
        # layer for HR, level for everything else
        pStr = 'lay' if 'heating_rate' in comp else 'lev'

        if 'forcing' in comp:
            # assuming comp is following '*_forcing_N' where 
            # * is the parameter (flux_net, heating_rate, etc.), 
            # N is the forcing record index
            iForce = int(comp.split('_')[-1])-1
           
            # extract baseline and forcing scenarios
            # baseline is record 0 (Present Day) or 
            # 1 (Preindustrial) -- see 
            # https://github.com/pernak18/g-point-reduction/wiki/LW-Forcing-Number-Convention
            if not doLW:
                # keeping minor-19 for Eli (see LW-Forcing-Number-Convention link)
                # but code iForce needs to be recalibrated
                if iForce == 18: iForce -= 11
            # endif doLW

            iBase = 1 if iForce < 7 else 0

            selDict = {'record': iBase, pStr: pLevCF[comp]}
            bTest = testDS.isel(selDict)
            bLBL = lblDS.isel(selDict)

            # calculate forcing
            selDict['record'] = int(iForce)
            fTest = testDS.isel(selDict)
            fLBL = lblDS.isel(selDict)
            testDSf = fTest - bTest
            lblDSf = fLBL - bLBL
            subsetErr = testDSf - lblDSf

            # what parameter are we extracting from dataset?
            if doLW:
                compDS = comp.replace('_forcing_{}'.format(iForce+1), '')
            else:
                if iForce == 7:
                    compDS = comp.replace('_forcing_19', '')
                else:
                    compDS = comp.replace('_forcing_{}'.format(iForce+1), '')
                # endif iForce
            # end doLW
        else:
            # Compute differences in all variables in datasets at 
            # levels closest to user-provided pressure levels
            # particularly important for heating rate since its
            # vertical dimension is layers and not levels
            # baseline is record 0 (Garand Present Day)
            try:
                # allow for different atmospheric specs 
                # (PI, PI 2xCH4) to be requested using the 
                # "param_N" convention with "N" being the forcing
                # scenario index
                iForce = int(comp.split('_')[-1])-1
                compDS = comp.replace('_{}'.format(iForce+1), '')
            except:
                # default to present day Garand atm specs
                iForce = 0
                compDS = str(comp)
            # stop trying

            selDict = {'record': iForce, pStr: pLevCF[comp]}
            subsetErr = (testDS-lblDS).isel(selDict)
        # endif forcing

        # get array for variable, then compute its test-ref RMS
        # over all columns at given pressure levels for a given
        # forcing scenario
        cfDA = getattr(np.fabs(subsetErr), compDS)

        # determine which dimensions over which to average
        dims = subsetErr[compDS].dims
        calcDims = ['col', pStr]
        if 'band' in dims: calcDims.append('band')

        # components will be scaled by their own initial cost
        costComps[comp] = cfDA.sum(dim=['col'])

        # total cost (sum of compCosts) will be scaled to 100
        # WRT initial cost to keep HR and Flux in same range
        compCost = cfDA.sum(dim=calcDims).values * scale[comp]
        allComps.append(np.sum(compCost))
        if not init:
            dCostComps[comp] = (costComps[comp] - costComp0[comp])
    # end CF component loop

    # now calculate total cost with all components and its 
    # relative difference from 256 g-point reference cost (scaled to 100)
    allComps = np.array(allComps)
    totalCost = allComps.sum()
    dCost = totalCost - 100

    lblDS.close()
    testDS.close()

    return {'allComps': allComps, 'totalCost': totalCost, 'dCost': dCost, 
            'costComps': costComps, 'dCostComps': dCostComps}
# end costCalc()

def combineBands(iBand, fullNC, trialNC, lw, outNC='trial_band_combined.nc', 
                 finalDS=False):
    """
    Combine a given trial fluxes dataset in a given band with the full-band 
    fluxes from the rest of the bands

    Call
        outDS = combineBands(iBand, fullDS, trialDS)
    
    Input
        iBand -- int, zero-offset band number that was modified (i.e., 
            band for which g-points were combined)
        fullDS -- list of xarray Datasets, full-band fluxes for each 
            of the bands that were not modified
        trialDS -- xarray Dataset, fluxes for band where g-points were 
            combined
        lw -- boolean, longwave instead of shortwave flux 
            parameters saved to output netCDF
    
    Keywords
        finalDS -- boolean, merge all bands together after full 
            optimization is complete
        
    Output
        outDS -- xarray Dataset, fluxes for all bands
    """

    fullDS = [xa.open_dataset(fNC) for fNC in fullNC]
    trialDS = xa.open_dataset(trialNC)
    
    nForce = fullDS[0].sizes['record']
    bandVars = ['flux_up', 'flux_dn', 'flux_net', 'heating_rate', 
                'emis_sfc', 'band_lims_wvn']
    fluxVars = bandVars[:4]
    if not lw:
        bandVars.append('flux_dir_dn')
        fluxVars.append('flux_dir_dn')
    # end shortWave

    inBand = int(iBand)
    outDS = xa.Dataset()

    # replace original fluxes for trial band with modified one
    fluxesMod = list(fullDS)
    if not finalDS: fluxesMod[iBand] = trialDS
    nBands = len(fullDS)

    # TO DO: consider xarray.merge()
    ncVars = list(trialDS.keys())
    for ncVar in ncVars:
        if ncVar in bandVars:
            # concat variables on the band dimension
            modDS = [bandDS[ncVar] for bandDS in fluxesMod]
            outDat = xa.concat(modDS, 'band')

            # add record/forcing dimension
            if ncVar == 'emis_sfc':
                newDims = ('record', 'col', 'band')
            elif ncVar == 'band_lims_wvn':
                newDims = ('record', 'band', 'pair')
                outDat = outDat.expand_dims(
                    dim={'record': nForce}, axis=0)
            else:
                if ncVar == 'heating_rate':
                    pDim = 'lay'
                else:
                    pDim = 'lev'
                # endif HR

                newDims = ('record', pDim, 'col', 'band')
            # endif newDims

            outDat = outDat.transpose(*newDims)
        elif ncVar == 'band_lims_gpt':
            gptLims = []
            for iBand, bandDS in enumerate(fluxesMod):
                bandLims = bandDS['band_lims_gpt'].squeeze()
                if iBand == 0:
                    gptLims.append(bandLims)
                else:
                    offset = gptLims[-1][1]
                    gptLims.append(bandLims+offset)
                # endif iBand
            # end band loop

            # add record/forcing dimension
            modDims = {'record': np.arange(nForce), 
                       'band': np.arange(nBands), 
                       'pair': np.arange(2)}
            outDat = xa.DataArray(
                [gptLims] * nForce, dims=modDims)
        else:
            # retain any variables with no band dimension
            outDat = trialDS[ncVar]
        # endif ncVar

        outDS[ncVar] = outDat
    # end ncVar loop

    if not lw:
        outDS['flux_dif_dn'] = outDS['flux_dn'] - outDS['flux_dir_dn']
        outDS['flux_dif_net'] = outDS['flux_dif_dn'] - outDS['flux_up']
        fluxVars.append('flux_dif_dn')
        fluxVars.append('flux_dif_net')
    # endif LW

    # calculate broadband fluxes
    for fluxVar in fluxVars:
        pDim = 'lay' if 'heating_rate' in fluxVar else 'lev'
        dimsBB = ('record', pDim, 'col')
        outDS = outDS.rename({fluxVar: 'band_{}'.format(fluxVar)})
        broadband = outDS['band_{}'.format(
            fluxVar)].sum(dim='band')
        outDS[fluxVar] = xa.DataArray(broadband, dims=dimsBB)
    # end fluxVar loop

    for fDS in fullDS: fDS.close()
    trialDS.close()

    outDS.heating_rate.attrs['units'] = 'K/s'
    outDS.band_heating_rate.attrs['units'] = 'K/s'

    outDS.to_netcdf(outNC)

    return outNC
# end combineBands()

def combineBandsSgl(iBand,fullNC,trialNC,lw,outNC='trial_band_combined.nc', finalDS=False):
    """
    Combine a given trial fluxes dataset in a given band with the full-band 
    fluxes from the rest of the bands

    Call
        outDS = combineBands(iBand, trialNC,fullBandFluxes)
    
    Input
        iBand -- int, zero-offset band number that was modified (i.e., 
            band for which g-points were combined)
        trialNC -- string, netCDF file with fluxes for band where 
            g-points were combined
        fullBandFLuxes - list of nectdf files with full-band fluxes 
            for each band that was not modified
        lw -- boolean, longwave instead of shortwave flux 
            parameters saved to output netCDF
    
    Keywords
        finalDS -- boolean, merge all bands together after full 
            optimization is complete
        
    Output
        outDS -- xarray Dataset, fluxes for all bands
    """

    bandVars = ['flux_up', 'flux_dn', 'flux_net', 'heating_rate', 
                'emis_sfc', 'band_lims_wvn']
    fluxVars = bandVars[:4]
    if not lw:
        bandVars.append('flux_dir_dn')
        fluxVars.append('flux_dir_dn')
    # end shortWave

    inBand = int(iBand)
    outDS = xa.Dataset()
     
    #print ("in CombineBandsSgl")
    #print (trialNC)

    # If trial data and flux data are  coming in as NC, store in xarray
    with xa.open_dataset(trialNC) as trialDS:
            trialDS.load()

    fullDS = []
    for bandNC in fullNC:
        with xa.open_dataset(bandNC) as bandDS:
            bandDS.load()
            fullDS.append(bandDS)
        # end with
    # end bandNC loop

    nForce = fullDS[0].sizes['record']

    # replace original fluxes for trial band with modified one
    fluxesMod = list(fullDS)
    if not finalDS: fluxesMod[iBand] = trialDS
    nBands = len(fullDS)

    # TO DO: consider xarray.merge()
    ncVars = list(trialDS.keys())
    for ncVar in ncVars:
        if ncVar in bandVars:
            # concat variables on the band dimension
            modDS = [bandDS[ncVar] for bandDS in fluxesMod]
            outDat = xa.concat(modDS, 'band')

            # add record/forcing dimension
            if ncVar == 'emis_sfc':
                newDims = ('record', 'col', 'band')
            elif ncVar == 'band_lims_wvn':
                newDims = ('record', 'band', 'pair')
                outDat = outDat.expand_dims(
                    dim={'record': nForce}, axis=0)
            else:
                if ncVar == 'heating_rate':
                    pDim = 'lay'
                else:
                    pDim = 'lev'
                # endif HR

                newDims = ('record', pDim, 'col', 'band')
            # endif newDims

            outDat = outDat.transpose(*newDims)
        elif ncVar == 'band_lims_gpt':
            gptLims = []
            for iBand, bandDS in enumerate(fluxesMod):
                bandLims = bandDS['band_lims_gpt'].squeeze()
                if iBand == 0:
                    gptLims.append(bandLims)
                else:
                    offset = gptLims[-1][1]
                    gptLims.append(bandLims+offset)
                # endif iBand
            # end band loop

            # add record/forcing dimension
            modDims = {'record': np.arange(nForce), 
                       'band': np.arange(nBands), 
                       'pair': np.arange(2)}
            outDat = xa.DataArray(
                [gptLims] * nForce, dims=modDims)
        else:
            # retain any variables with no band dimension
            outDat = trialDS[ncVar]
        # endif ncVar

        outDS[ncVar] = outDat
    # end ncVar loop

    if not lw:
        outDS['flux_dif_dn'] = outDS['flux_dn'] - outDS['flux_dir_dn']
        outDS['flux_dif_net'] = outDS['flux_dif_dn'] - outDS['flux_up']
        fluxVars.append('flux_dif_dn')
        fluxVars.append('flux_dif_net')
    # endif LW

    # calculate broadband fluxes
    for fluxVar in fluxVars:
        pDim = 'lay' if 'heating_rate' in fluxVar else 'lev'
        dimsBB = ('record', pDim, 'col')
        outDS = outDS.rename({fluxVar: 'band_{}'.format(fluxVar)})
        broadband = outDS['band_{}'.format(
            fluxVar)].sum(dim='band')
        outDS[fluxVar] = xa.DataArray(broadband, dims=dimsBB)
    # end fluxVar loop

    for fDS in fullDS: fDS.close()
    trialDS.close()

    outDS.heating_rate.attrs['units'] = 'K/s'
    outDS.band_heating_rate.attrs['units'] = 'K/s'

    outDS.to_netcdf(outNC)

    return outNC
# end combineBandsSgl()

