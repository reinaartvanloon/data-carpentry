# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def convert_pr_units(darray):
    """Convert kg m-2 s-1 to mm day-1.
   
    Args:
      darray (xarray.DataArray): Precipitation data
   
    """
   
    darray.data = darray.data * 86400
    darray.attrs['units'] = 'mm/day'
   
    return darray
