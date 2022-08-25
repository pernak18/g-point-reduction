#!/usr/bin/env python

import os, sys, glob

PIPPATH = '{}/.local/'.format(os.path.expanduser('~')) + \
    'cori/3.7-anaconda-2019.10/lib/python3.7/site-packages'
paths = [PIPPATH, 'common']
for path in paths: sys.path.append(path)

# in common
import utils

# local library
import by_band_lib as redux

# pip/conda installs
import xarray as xa
import numpy as np

# GLOBAL VARIABLE CONFIGURATION

class costCompare:
  def __init__(self, inYAML):
    """
    Compare user-defined cost functions of multiple models and/or 
    formulations of said models. Models can be (but are not required 
    to be) run with this object, given an executable path. Cost 
    comparisons are then performed by loading the output netCDF 
    files that are presumed to contain all of the cost components 
    that the user defines in the input YAML configuration file
    """

    self.config = str(inYAML)
    utils.file_check(self.config)

    # populated by readYAML()
    self.refNC, self.testNC, self.others, self.doLW = \
      None, None, None, None

    # populated by costCalc()
    self.scale, self.cost0, self.totalCost, self.costComps = \
      {}, {}, {}, {}
    self.compNames, self.compLevs, self.compWeights = [], {}, {}
    self.outComps = []

    # populated by aliases*()
    self.aliases = {}
    self.aliases['levels'] = {}
    self.aliases['names'] = {}

  # end constructor

  def aliasesGarandLW(self):
    """
    Map user-friendly names to what is used in code
    See https://github.com/pernak18/g-point-reduction/wiki/LW-Forcing-Number-Convention#g-point-reduction-convention-
    and "Reduction and Optimization" section in 
    https://github.com/pernak18/g-point-reduction/blob/master/band-g-point-reduction.ipynb

    Level aliases are handled in readYAML()
    """

    # combine broadband and by-band flux/HR fields
    fieldsNC = ['flux_dn', 'flux_up', 'flux_net', 'heating_rate']
    bandFieldsNC = ['band_{}'.format(field) for field in fieldsNC]
    fieldsNC += bandFieldsNC

    # `record` dimension in flux files span 19 scenarios (wee wiki page)
    # map the indices to the user-friendly names
    iRecords = range(1, 20)
    nameRecords = ['garand', 'preind', '2xch4', '2xco2', 'pi_pd-no2', 
                   '4xch4', '4xco2', 'xs-all', 'ccl4', 'cfc11', 
                   'cfc12', 'cfc22', 'hfc143a', 'hfc125', 'hfc23', 
                   'hfc32', 'hfc134a', 'cf4', 'no2xs']

    # start with present-day non-forcing parameters
    self.aliases['names'] = {
      'code': list(fieldsNC), 'user': list(fieldsNC)}

    for iRec, name in zip(iRecords, nameRecords):
      # non-forcing parameters for each record
      self.aliases['names']['code'] += [
        '{}_{:d}'.format(field, iRec) for field in fieldsNC]
      self.aliases['names']['user'] += [
        '{}_{}'.format(field, name) for field in fieldsNC]

      # concatenate forcing parameters
      self.aliases['names']['code'] += [
        '{}_forcing_{:d}'.format(field, iRec) for field in fieldsNC]
      self.aliases['names']['user'] += [
        '{}_forcing_{}'.format(field, name) for field in fieldsNC]
    # end record loop
  # end aliasesGarandLW()

  def readYAML(self):
    """
    Read in configuration file with cost function definition, paths, 
    etc.
    """

    import yaml

    # try to read YAML specs into memory
    with open(self.config, 'r') as inFP:
      try:
        config = yaml.safe_load(inFP)
      except:
        print('Could not load {}'.format(self.config))
        sys.exit(1)
      # end trying
    # endwith YAML

    # level alias mapping (names-names)
    iLevC = config['level_indices']
    self.aliases['levels']['toa'] = int(iLevC['top'])
    self.aliases['levels']['sfc'] = int(iLevC['surface'])
    self.aliases['levels']['tpause'] = int(iLevC['tropopause'])

    for iComp, comp in config['components'].items():
      name = comp[0]
      self.outComps.append(name)

      # what field name are we using in the code?
      # this should never fail, unless i messed up the aliases
      iMatch = self.aliases['names']['user'].index(name)
      codeName = self.aliases['names']['code'][iMatch]
      self.compNames.append(codeName)

      if len(comp) != 3:
        print('{} definition is not of length 3, returning'.format(name))
        sys.exit(1)
      # endif len(comp)

      lowComps = [levStr.lower() for levStr in comp[1]]

      self.compWeights[codeName] = float(comp[2])

      # level alias mapping (names-indices)
      if 'all' in lowComps:
        # replace "all" string with indices for all levels or layers
        # overwrites any other levels that might superfluously exist
        nLev = config['level_indices']['top']
        # for `range` we need n+1 to catch the last lev/lay
        # since level_indices are levels, heating rates already have 
        # the +1
        if 'heating_rate' not in name: nLev += 1
        iLevs = range(nLev)
      else:
        iLevs = []
        # we have aliases for toa (42), tpause (26), and surface (0), 
        # but none of the other indieces
        for iLev in lowComps:
          try:
            # grab alias
            iLevs.append(self.aliases['levels'][iLev])
          except:
            # grab explicit level index
            try:
              iLevs.append(int(iLev))
            except:
              print('{} index not understood'.format(iLev))
              sys.exit(1)
            # end trying
          # end trying
        # end iLev loop
      # endif all
      self.compLevs[codeName] = list(iLevs)
    # end componnent loop

    if sum(self.compWeights.values()) != 1.0:
      print('Weights do not integrate to unity, returning')
      sys.exit(1)
    # endif weights

    self.refNC = config['ref_path']
    self.testNC = config['test_path']
    self.others = config['others']
    paths = [self.refNC, self.testNC] + self.others
    for path in paths: utils.file_check(path)

    self.doLW = config['do_lw']

    # normalize to get HR an fluxes on same scale
    # so each cost component has its own scale to 100
    # for the full-k RRTMGP, this is just 1
    for comp in self.compNames: self.scale[comp] = 1
  # end readYAML()

  def runRRTMGP(self):
    """
    """
  # end runRRTMGP

  def costCalc(self):
    """
    """

    with xa.open_dataset(self.refNC) as rDS, \
      xa.open_dataset(self.testNC) as tDS:

      print('Calculating full k-distribution cost')
      # first calculate the cost of full k-distribution RRTMGP
      isInit = True
      costDict = redux.costCalc(
        rDS, tDS, self.doLW, self.compNames, self.compLevs, 
        self.cost0, self.scale, isInit)

      # store initial cost for each component (integrated over all
      # pressure levels specified by user)
      for iComp, comp in enumerate(self.compNames):
        self.cost0[comp] = costDict['allComps'][iComp]

      # save total cost for full k configuration
      costKey = 'Full_k'
      self.totalCost[costKey] = costDict['totalCost']
      self.costComps[costKey] = {}
      for comp in self.compNames: self.costComps[costKey][comp] = \
        costDict['costComps'][comp].sum().values
    # endwith

    # now use initial cost in normalization
    self.scale = {}
    for comp in self.compNames:
      weight = self.compWeights[comp]
      self.scale[comp] = weight * 100 / self.cost0[comp]

    for i, other in enumerate(self.others):
      print('Calculating cost for {}'.format(other))
      with xa.open_dataset(self.refNC) as rDS, \
        xa.open_dataset(other) as oDS:

        isInit = False
        costDict = redux.costCalc(
          rDS, oDS, self.doLW, self.compNames, self.compLevs, 
          self.cost0, self.scale, isInit)
      # endwith

      print(other)

      costKey = os.path.basename(other)
      self.totalCost[costKey] = costDict['totalCost']
      self.costComps[costKey] = {}
      for comp in self.compNames: self.costComps[costKey][comp] = \
        costDict['costComps'][comp].sum().values
    # end path loop

    # TO DO: save to a file? CSV?
    # print out cost for each configuration
    for key in self.totalCost.keys():
      norm = '(Non-normalized)' if 'Full_k' in key else '(Normalized)'
      print('{:100s}{:10.3f} {:s}'.format(
        key, self.totalCost[key], norm))

      for iComp, comp in enumerate(self.compNames):
        outComp = self.outComps[iComp]
        if 'Full_k' in key:
          print('{:>100s}{:10.3e}'.format(
            outComp, self.costComps[key][comp]))
        else:
          print('{:>100s}{:10.3f}'.format(outComp, 
            self.costComps[key][comp] * 100 / self.cost0[comp]))
        # endif key
      # end comp loop
    # end key loop
  # end costCalc()
# end costCompare

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(\
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='See `costCompare` constructor doc string')
  parser.add_argument('--in_yaml', '-i', type=str, 
    default='costCalc_wrapper_LW.yaml',
    help='YAML file with user-defined cost function ' + \
     '(fields, levels, weights) and other parameters ' + \
     '(paths, SW switch) for costCompare objects.')
  args = parser.parse_args()

  cObj = costCompare(args.in_yaml)
  cObj.aliasesGarandLW()
  cObj.readYAML()
  cObj.costCalc()
# endif main()
