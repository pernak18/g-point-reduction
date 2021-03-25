#!/usr/bin/env python

import os, sys
import glob
import xarray as xa
import numpy as np

NCDIR = '/Users/rpernak/Work/RC/RRTMGP/by-band-g-reduce/sensitivities'

if __name__ == '__main__':

  import argparse

  parser = argparse.ArgumentParser(\
    description='Combine similar netCDFs on a new `record` ' + \
    'dimension that represents sensitivity studies used in ' + \
    'RRTMGP band generation (e.g., with forcing).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--indir', '-i', type=str, default=NCDIR, \
    help='Directory with RRTMGP netCDF flux files for each ' + \
    'sensitivity.')
  parser.add_argument('--outfile', '-o', type=str, \
    default='lblrtm-lw-flux-inputs-outputs-garandANDpreind.nc', \
    help='Name of new, consolidated netCDF with `record` dim.')
  args = parser.parse_args()

  outNC = args.outfile
  inDir = args.indir

  assert os.path.isdir(inDir), 'Could not find {}'.format(inDir)

  ncFiles = sorted(glob.glob('{}/*.nc'.format(args.indir)))

  # ad-hoc sorting to match Jen's numbering convention
  iSort = [0] + list(range(13, 19)) + [12] + list(range(1, 12))
  ncFiles = np.array(ncFiles)[iSort]

  # add record coordinate dimension to each dataset; write temp netcdf
  print('Adding record dimension (creating temporary files)')
  for iNC, nc in enumerate(ncFiles):
    base = os.path.basename(nc)
    outDS = xa.Dataset()
    with xa.open_dataset(nc) as ds:
      ncVars = ds.keys()
      for ncVar in ncVars:
        outDS[ncVar] = ds[ncVar].expand_dims(dim={'record': [iNC]})
      # end ncVar loop
      outDS.to_netcdf(base)
  # end nc loop

  if os.path.exists(outNC): os.remove(outNC)

  print('Concatenating datasets')
  ncFiles = [os.path.basename(nc) for nc in ncFiles]
  newDS = xa.open_mfdataset(ncFiles, concat_dim='record')

  # older files have the dimensions switched for these guys
  newDims = ('record', 'band', 'pair')
  newDS['band_lims_wvn'] = newDS['band_lims_wvn'].transpose(*newDims)

  newDS.to_netcdf(outNC)
  print('Wrote {}'.format(outNC))

  print('Removing temporary files')
  for nc in ncFiles: os.remove(nc)

# end main()
