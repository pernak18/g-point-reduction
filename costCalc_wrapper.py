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
# COST FUNCTION DEFINITION (components, levels, weights)
COMPS = ['flux_net', 'heating_rate']

# 1 level key must exist for each COMPS string
# indices of pressure levels to use in cost calculation
# 0: Surface, 26: Tropopause, 42: TOA
LEVELS = {}
LEVELS['flux_net'] = [0, 26, 42]
LEVELS['heating_rate'] = range(41)

# 1 weight per COMPS string
WEIGHTS = [0.5, 0.5]

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

  scale, costComp0, totalCost = {}, {}, {}

  # normalize to get HR an fluxes on same scale
  # so each cost component has its own scale to 100
  # for the full-k RRTMGP, this is just 1
  for comp in COMPS: scale[comp] = 1

  with xa.open_dataset(refNC) as rDS, xa.open_dataset(testNC) as tDS:
    print('Calculating full k-distribution cost')
    # first calculate the cost of full k-distribution RRTMGP
    isInit = True
    costDict = REDUX.costCalc(
      rDS, tDS, doLW, COMPS, LEVELS, costComp0, scale, isInit)

    # store initial cost components
    for comp in COMPS: costComp0[comp] = costDict['costComps'][comp]

    # save total cost for full k configuration
    totalCost['Full_k'] = costDict['totalCost']
  # endwith

  # now use initial cost in normalization
  scale = {}
  for comp, weight in zip(COMPS, WEIGHTS):
    scale[comp] = weight * 100 / costComp0[comp]

  for i, other in enumerate(others):
    print('Calculating cost for {}'.format(other))
    with xa.open_dataset(refNC) as rDS, xa.open_dataset(other) as oDS:
      isInit = False
      costDict = REDUX.costCalc(
        rDS, oDS, doLW, COMPS, LEVELS, costComp0, scale, isInit)
    # endwith
    totalCost[os.path.basename(other)] = costDict['totalCost']
  # end path loop

  # print out cost for each configuration
  for key in totalCost.keys():
    print('{:50s}{:10.3f}'.format(key, totalCost[key]))
# endif main()
