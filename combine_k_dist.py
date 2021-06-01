#!/usr/bin/env python

import os, sys, glob

def kCombine(ncFiles, outNC='rrtmgp-combined_k_dist.nc'):
  """
  Combine k distributions from each band into a single file

  outNC = kCombine(ncFiles)

  Inputs
    ncFiles -- string list of netCDF files that contain absorption
      coefficient distibutions for each band (1 per band)

  Outputs
    outDS -- xarray dataset, RRTMGP-type netCDF for a k-distribution;
      combined distribution for all bands

  Keywords
    outNC -- string, netCDF file name to which outDS is written
  """

  import numpy as np
  import xarray as xa

  # read in xarray datasets for each band
  inDS = [xa.open_dataset(nc) for nc in ncFiles]
  nGpt = [ds.sizes['gpt'] for ds in inDS]

  # search for LW variables in datasets
  lw = ['plank_fraction' in list(ds.keys()) for ds in inDS]

  # needs to be all or nothing; nothing in between
  # TO DO: test this for LW and mixed LW/SW
  if all(lw):
    doLW = True
  elif not all(lw):
    doLW = False
  else:
    print('Inconsistent LW and SW fields')
    sys.exit()
  # endif LW

  regions = ['upper', 'lower']

  # more than just the g-point dimension changes if the number
  # of g-points is altered; namely, contributors_* and
  # minor_absorber_intervals_*
  # these are why we cannot simply use an xa.concat() or xa.merge()

  # what bands have minor contributors (upper and lower atmosphere)?
  nBands = 16 if doLW else 14
  noUp = [3, 11, 13, 14, 15] if doLW else [1, 3, 4, 12, 13]
  noLo = [13] if doLW else [12, 13]
  buffer = np.arange(nBands)
  noContrib = {'upper': noUp, 'lower': noLo}
  minorBands = {}
  for reg in regions:
    minorBands[reg] = np.delete(buffer, noContrib[reg])

  # how many minor contributors intervals per band?
  tempStr = 'minor_absorber_intervals'
  nInterval = {}
  for reg in regions: nInterval['{}_{}'.format(tempStr, reg)] = \
    [ds.sizes['{}_{}'.format(tempStr, reg)] for ds in inDS]

  # recalculate start/end indices/gpt limits -- we are given single-
  # band limits (e.g., 1-16) and need to convert to bandmerged limits
  # (e.g., 1-256)
  kMinStart, gLims = {}, {}
  for reg in regions:
    kMinStart[reg] = []
    gLims[reg] = []

    nPerBand = nInterval['{}_{}'.format(tempStr, reg)]
    i1, iStart = 1, 1

    for nG, nInt, iBand in zip(nGpt, nPerBand, range(nBands)):
      i2 = i1 + nG - 1

      # store indices for each interval of band, only if there are
      # minor contributors
      for i in range(nInt):
        if iBand in minorBands[reg]:
          gLims[reg].append([i1, i2])
          kMinStart[reg].append(iStart)
        # endif iBand

        iStart += nG
      # end i loop

      i1 += nG
    # end interval loop
  # end reg loop

  # more than just the g-point dimension changes if the number
  # of g-points is altered --> "g-point dimensions"
  minIntDims = list(nInterval.keys())
  contDims = ['contributors_lower', 'contributors_upper']
  gDims = ['gpt'] + contDims

  # now combine into a new dataset
  outDS = xa.Dataset()
  outVars = list(inDS[0].keys())
  for outVar in outVars:
    # grab dataArray corresponding to outVar for each band
    inDA = [ds[outVar] for ds in inDS]

    # all bands should have the same dimensions -- grab 1 set
    dims = inDA[0].dims

    # are there any g-point dimensions?
    gMatch = list(set(gDims).intersection(dims))
    minAbsMatch = list(set(minIntDims).intersection(dims))
    minStartMatch = list(set(contDims).intersection(dims))
    if gMatch:
      # determine which dimension over which to concatenate
      # should only be 1
      concatVar = gMatch[0]
      outDS[outVar] = xa.concat(inDA, concatVar)
    elif minAbsMatch:
      # if outVar is a minor absorption interval limit array,
      # we need to assign the previously calculated list
      # should only be 1 dimension
      minDim = minAbsMatch[0]
      reg = 'upper' if 'upper' in minDim else 'lower'
      if 'limits' in outVar:
        modDims = \
          {minDim: range(len(gLims[reg])), 'pair': np.arange(2)}
        outDS[outVar] = xa.DataArray(gLims[reg], dims=modDims)
      elif 'kminor_start' in outVar:
        # if outVar is a minor absorption start indice array,
        # we need to assign the previously calculated list
        # should only be 1 dimension
        modDims = {minDim: range(len(kMinStart[reg]))}
        outDS[outVar] = xa.DataArray(kMinStart[reg], dims=modDims)
      else:
        outDS[outVar] = xa.concat(
          np.array(inDA, dtype=object)[minorBands[reg]], minDim)
      # endif gpt_limits
    else:
      # should be the same for the each band -- keep just 1
      outDS[outVar] = inDA[0]
    # endif gMatch
  # end outVar loop

  # now ensure data encoding matches original (non-reduced) k-distributions
  origNC = 'rrtmgp-data-lw-g256-2018-12-04.nc' if doLW else \
    'rrtmgp-data-sw-g224-2018-12-04.nc'
  assert os.path.exists(origNC), 'Could not find {}'.format(origNC)
  encoding = {}

  with xa.open_dataset(origNC) as origDS:
    origVars = list(origDS.keys())
    for origVar in origVars: encoding[origVar] = origDS[origVar].encoding
  # endwith

  strVars = ['gas_minor', 'gas_names', 'identifier_minor', 
             'minor_gases_lower', 'minor_gases_upper', 
             'scaling_gas_lower', 'scaling_gas_upper']
  encode = {}
  for sv in strVars: 
    encode[sv] = {'zlib': True, 'complevel': 5, 'char_dim_name': 'string_len'}

  ncVars = list(outDS.keys())
  for ncVar in ncVars:
    if ncVar == 'gpt_weights': continue
    outDat = outDS[ncVar]
    if ncVar in strVars:
      outDat = outDat.astype(str)
      strings = [''.join(string) for string in outDat.values]
      outDS[ncVar] = xa.DataArray(
        np.array(strings, dtype=np.dtype(('S', 32))), 
        dims=[outDat.dims[0]], attrs=outDat.attrs)
    else:
      outDS[ncVar] = xa.DataArray(outDat)
      outDS[ncVar].encoding['dtype'] = origDS[ncVar].encoding['dtype']
    # endif ncVar
  # end ncVar loop
  
  outDS.to_netcdf(outNC)
  print('Wrote {}'.format(outNC))
# end kCombine()

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(\
    description='Combine k-distributions for separate bands into ' + \
    'a single netCDF file.')
  parser.add_argument('indir', type=str, \
    help='Directory with band??.nc files that contain ' + \
    'by-band k-distributions.')
  parser.add_argument('--out_nc', '-o', type=str,
    default='rrtmgp-combined_k_dist.nc',
    help='Path to output netCDF that will be written.')
  args = parser.parse_args()

  inDir = args.indir
  assert os.path.exists(inDir), 'Could not find {}'.format(inDir)
  ncFiles = sorted(glob.glob('{}/band??.nc'.format(inDir)))

  kCombine(ncFiles, outNC=args.out_nc)
# end main()
