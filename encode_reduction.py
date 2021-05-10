#!/usr/bin/env python

import os, sys

PIPPATH = '{}/.local/'.format(os.path.expanduser('~')) + \
    'cori/3.7-anaconda-2019.10/lib/python3.7/site-packages'
paths = [PIPPATH, 'common']
for path in paths: sys.path.append(path)
    
# in common
import utils

import xarray as xa
import numpy as np

# SW scalars
tsi_default = np.double(1360.85767381726)
mg_default = np.single(0.1567652)
sb_default = np.single(902.7126)

# http://xarray.pydata.org/en/latest/user-guide/io.html#writing-encoded-data
# https://stackoverflow.com/questions/48895227/output-int32-time-dimension-in-netcdf-using-xarray
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(\
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        description='Given a reduced k-distribution produced with ' + \
        '`by_band_lib.py`, convert the array types to their correct ' + \
        'encoding.')
    parser.add_argument('--infile', '-i', type=str, 
                        default='rrtmgp-data-LW-g-red.nc', 
                       help='Reduced k-distribution netCDF file.')
    parser.add_argument('--baseline', '-b', type=str, 
                        default='rrtmgp-data-lw-g256-2018-12-04.nc', 
                       help='netCDF with correct encoding.')
    parser.add_argument('--shortwave', '-sw', action='store_true', \
                       help='Add some scalars from full SW k-distribution')
    parser.add_argument('--outfile', '-o', 
                        default='encoded_rrtmgp-data-LW-g-red.nc', 
                       help='Name of modified k-dist netCDF')
    args = parser.parse_args()

    inFile = args.infile; utils.file_check(inFile)
    baseFile = args.baseline; utils.file_check(baseFile)
    outFile = args.outfile

    strVars = ['gas_minor', 'gas_names', 'identifier_minor', 
               'minor_gases_lower', 'minor_gases_upper', 
               'scaling_gas_lower', 'scaling_gas_upper']
    encode = {}
    for sv in strVars: 
        encode[sv] = {'zlib':True, 'complevel':5, 'char_dim_name': 'string_len'}

    outDS = xa.Dataset()
    with xa.open_dataset(inFile) as iDS, xa.open_dataset(baseFile) as bDS:
        ncVars = list(iDS.keys())
        for ncVar in ncVars:
            inDat = iDS[ncVar]
            if ncVar in strVars:
                inDat = inDat.astype(str)
                strings = [''.join(string) for string in inDat.values]
                outDS[ncVar] = xa.DataArray(
                    np.array(strings, dtype=np.dtype(('S', 32))), 
                    dims=[inDat.dims[0]], attrs=inDat.attrs)
            else:
                outDS[ncVar] = xa.DataArray(inDat)
                outDS[ncVar].encoding['dtype'] = bDS[ncVar].encoding['dtype']
            # endif ncVar
        # end ncVar loop
        outDS.to_netcdf(outFile, encoding=encode, format='NETCDF3_CLASSIC')
    # endwith

    print('Wrote {}'.format(outFile))
# endif main()
