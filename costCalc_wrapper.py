#!/usr/bin/env python

import os, sys, glob

PIPPATH = '{}/.local/'.format(os.path.expanduser('~')) + \
    'cori/3.7-anaconda-2019.10/lib/python3.7/site-packages'
paths = [PIPPATH, 'common']
for path in paths: sys.path.append(path)

# in common
import utils

# local library
import by_band_lib as REDUX

import xarray as xa
import numpy as np

# GLOBAL VARIABLE CONFIGURATION
# TO DO: better way to do this?
# TO DO: add provisions for COMPS and LEVELS consistency
# TO DO: provision for sum(weights) = 1
# COST FUNCTION DEFINITION (components, levels, weights)

# Eli LW
COMPS = ['flux_net', 'band_flux_net', 'heating_rate',
  'heating_rate_7', 'flux_net_forcing_5', 'flux_net_forcing_6',
  'flux_net_forcing_7', 'flux_net_forcing_9', 'flux_net_forcing_10',
  'flux_net_forcing_11', 'flux_net_forcing_12', 'flux_net_forcing_13',
  'flux_net_forcing_14', 'flux_net_forcing_15', 'flux_net_forcing_16',
  'flux_net_forcing_17', 'flux_net_forcing_18']

# Eli SW
COMPS = ['flux_dif_net', 'flux_dir_dn', 'heating_rate',
  'heating_rate_7', 'flux_net_forcing_5', 'flux_net_forcing_6',
  'flux_net_forcing_7', 'flux_net_forcing_19']

# 1 level key must exist for each COMPS string
# indices of pressure levels to use in cost calculation
# 0: Surface, 26: Tropopause, 42: TOA
# Eli LW
LEVELS = {}
LEVELS['flux_net'] = [0, 26, 42]
LEVELS['band_flux_net'] = [42]
LEVELS['heating_rate'] = range(42)
LEVELS['heating_rate_7'] = range(42)
LEVELS['flux_net_forcing_5'] = [0, 26, 42]
LEVELS['flux_net_forcing_6'] = [0, 26, 42]
LEVELS['flux_net_forcing_7'] = [0, 26, 42]
LEVELS['flux_net_forcing_9'] = [0, 26, 42]
LEVELS['flux_net_forcing_10'] = [0, 26, 42]
LEVELS['flux_net_forcing_11'] = [0, 26, 42]
LEVELS['flux_net_forcing_12'] = [0, 26, 42]
LEVELS['flux_net_forcing_13'] = [0, 26, 42]
LEVELS['flux_net_forcing_14'] = [0, 26, 42]
LEVELS['flux_net_forcing_15'] = [0, 26, 42]
LEVELS['flux_net_forcing_16'] = [0, 26, 42]
LEVELS['flux_net_forcing_17'] = [0, 26, 42]
LEVELS['flux_net_forcing_18'] = [0, 26, 42]

# Eli SW
LEVELS = {}
LEVELS['flux_dif_net'] = [0, 26, 42]
LEVELS['flux_dir_dn'] = range(42)
LEVELS['heating_rate'] = range(42)
LEVELS['heating_rate_7'] = range(42)
LEVELS['flux_net_forcing_5'] = [0,42]
LEVELS['flux_net_forcing_6'] = [0,42]
LEVELS['flux_net_forcing_7'] = [0,42]
LEVELS['flux_net_forcing_19'] = [0,42]

# 1 weight per COMPS string
# Eli LW
WEIGHTS =  [0.6, 0.04, 0.12, 0.12, 0.01, 0.02, 0.04, 0.005,
            0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.005]

# Eli SW
WEIGHTS = [0.1, 0.6, 0.05, 0.05, 0.02, 0.05, 0.11, 0.02]

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(\
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='')
  parser.add_argument('--ref_path', '-r', type=str,
    default='lblrtm-lw-flux-inputs-outputs-garandANDpreind.nc',
    help='Reference file (LBLRTM or CHARTS/LBL) path to ' + \
      'RRTMGP-type netCDF flux file.')
  parser.add_argument('--test_path', '-t', type=str, \
    default='rrtmgp-lw-flux-inputs-outputs-garandANDpreind.nc',
    help='Path to RRTMGP netCDF flux file for full k-distribution.')
  parser.add_argument('--others', '-o', type=str, nargs='+',
    default=['optimized_fluxes.nc', 'optimized_fluxes_iter001.nc'],
    help='List of paths to additional test files that are ' + \
      'similar to --test_path, but for additional configurations ' + \
      '(optimal angle, g-point reduction, etc.). The total cost ' + \
      'for these configurations are compared to the cost of ' + \
      '--test_file')
  parser.add_argument('--sw', '-sw', action='store_true', \
    help='Do shortwave cost instead of default longwave.')
  args = parser.parse_args()

  refNC = args.ref_path
  testNC = args.test_path
  others = args.others
  paths = [refNC, testNC] + others
  for path in paths: utils.file_check(path)

  doLW = not args.sw

  scale, cost0, totalCost = {}, {}, {}

  # normalize to get HR an fluxes on same scale
  # so each cost component has its own scale to 100
  # for the full-k RRTMGP, this is just 1
  for comp in COMPS: scale[comp] = 1

  with xa.open_dataset(refNC) as rDS, xa.open_dataset(testNC) as tDS:
    print('Calculating full k-distribution cost')
    # first calculate the cost of full k-distribution RRTMGP
    isInit = True
    costDict = REDUX.costCalc(
      rDS, tDS, doLW, COMPS, LEVELS, cost0, scale, isInit)

    # store initial cost for each component (integrated over all
    # pressure levels specified by user)
    for iComp, comp in enumerate(COMPS):
      cost0[comp] = costDict['allComps'][iComp]

    # save total cost for full k configuration
    totalCost['Full_k'] = costDict['totalCost']
  # endwith

  # now use initial cost in normalization
  scale = {}
  for comp, weight in zip(COMPS, WEIGHTS):
    scale[comp] = weight * 100 / cost0[comp]

  for i, other in enumerate(others):
    print('Calculating cost for {}'.format(other))
    with xa.open_dataset(refNC) as rDS, xa.open_dataset(other) as oDS:
      isInit = False
      costDict = REDUX.costCalc(
        rDS, oDS, doLW, COMPS, LEVELS, cost0, scale, isInit)
    # endwith

    # TO DO: will wanna label better
    totalCost[os.path.basename(other)] = costDict['totalCost']
  # end path loop

  # TO DO: save to a file? CSV?
  # print out cost for each configuration
  for key in totalCost.keys():
    norm = '(Non-normalized)' if 'Full_k' in key else '(Normalized)'
    print('{:100s}{:10.3f} {:s}'.format(key, totalCost[key], norm))
# endif main()
