#!/usr/bin/env python

import os, sys, glob

def kCombine(ncFiles, outNC='combined_k_dist.nc', doLW=False):
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
    doLW -- boolean, LW is being processed, otherwise SW
  """

  import numpy as np
  import xarray as xa

  inDS = [xa.open_dataset(nc) for nc in ncFiles]

  # zero-offset bands with no minor contributors
  nBands = 16 if doLW else 14
  noUp = [3, 11, 13, 14, 15] if doLW else [1, 3, 4, 12, 13]
  noLo = [13] if doLW else [12, 13]
  buffer = np.arange(nBands)
  upMinorBands = np.delete(buffer, noUp)
  loMinorBands = np.delete(buffer, noLo)

  # indices that represent minor contributors in various arrays
  regions = ['upper', 'lower']
  kMinorIdx = ['kminor_start_{}'.format(reg) for reg in regions]
  minorLims = ['minor_limits_gpt_{}'.format(reg) for reg in regions]
  minorVars = kMinorIdx + minorLims
  minAbs = ['minor_absorber_intervals_{}'.format(r) for r in regions]

  # more than just the g-point dimension changes if the number
  # of g-points is altered; namely, contributors_* and
  # minor_absorber_intervals_*
  # these are why we cannot simply use an xa.concat() or xa.merge()
  # "g-point variables"
  gVars = ['gpt', 'contributors_lower', 'contributors_upper',
    'minor_absorber_intervals_lower', 'minor_absorber_intervals_upper']

  # now combine into a new dataset
  outDS = xa.Dataset()
  outVars = list(inDS[0].keys())
  for outVar in outVars:
    # grab dataArray corresponding to outVar for each band
    inDA = [ds[outVar] for ds in inDS]

    # all bands should have the same dimensions -- grab 1 set
    dims = inDA[0].dims

    # pop out data arrays where there are no minor contributors
    minAbsDims = list(set(minAbs).intersection(dims))
    if minAbsDims:
      # should only be 1 dimension
      minDim = minAbsDims[0]
      minBands = upMinorBands if 'upper' in minDim else loMinorBands
      inDA = np.array(inDA)[minBands].tolist()
    # endif minAbsDims

    # are there any g-point variables?
    gDims = list(set(gVars).intersection(dims))
    if gDims:
      # determine which dimension over which to concatenate
      # should only be 1
      concatVar = gDims[0]
      outDS[outVar] = xa.concat(inDA, concatVar)
    else:
      # should be the same for the each band -- keep just 1
      outDS[outVar] = inDA[0]
    # endif gDims

    if outVar in minorVars:
      # print(outDS[outVar])
      print(outVar)
    # endif outVar
  # end outVar loop

  sys.exit()
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
  args = parser.parse_args()

  inDir = args.indir
  assert os.path.exists(inDir), 'Could not find {}'.format(inDir)
  ncFiles = sorted(glob.glob('{}/band??.nc'.format(inDir)))

  kCombine(ncFiles)
# end main()
