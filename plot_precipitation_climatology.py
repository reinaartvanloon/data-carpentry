import argparse

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cmocean


def convert_pr_units(darray):
    """Convert kg m-2 s-1 to mm day-1.
    
    Args:
      darray (xarray.DataArray): Precipitation data
    
    """
    
    darray.data = darray.data * 86400
    darray.attrs['units'] = 'mm/day'
    
    return darray


def create_plot(clim, model, season, gridlines=False,levels=None):
    """Plot the precipitation climatology.
    
    Args:
      clim (xarray.DataArray): Precipitation climatology data
      model (str): Name of the climate model
      season (str): Season
      
    Kwargs:
      gridlines (bool): Select whether to plot gridlines    
    
    """
    if not levels:
        levels = np.arange(0, 13.5, 1.5)
    print(levels)
        
    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    clim.sel(season=season).plot.contourf(ax=ax,
                                          extend='max',
                                          transform=ccrs.PlateCarree(),
                                          cbar_kwargs={'label': clim.units},
                                          cmap=cmocean.cm.haline_r,
                                          levels=levels)
    ax.coastlines()
    if gridlines:
        plt.gca().gridlines()
    
    title = f'{model} precipitation climatology ({season})'
    plt.title(title)
    
def apply_mask(darray, sftlf_file, realm):
    dset_mask = xr.open_dataset(sftlf_file)
    if realm=='land':
        masked_array = darray.where(dset_mask['sftlf'] >= 0.5)
    elif realm=='ocean':
        masked_array = darray.where(dset_mask['sftlf'] < 0.5)
    return masked_array


def main(inargs):
    """Run the program."""

    dset = xr.open_dataset(inargs.pr_file)
    
    clim = dset['pr'].groupby('time.season').mean('time', keep_attrs=True)
    clim = convert_pr_units(clim)
    
    if inargs.mask:
        sftlf_file, realm = inargs.mask
        clim = apply_mask(clim,sftlf_file, realm)

    create_plot(clim, dset.attrs['source_id'], inargs.season, gridlines=inargs.gridlines, levels=inargs.cbar_levels)
    plt.savefig(inargs.output_file, dpi=200)


if __name__ == '__main__':
    description='Plot the precipitation climatology for a given season.'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument("pr_file", type=str, help="Precipitation data file")
    parser.add_argument("season", type=str, help="Season to plot",choices=['DJF','MAM','JJA','SON'])
    parser.add_argument("output_file", type=str, help="Output file name")
    parser.add_argument("--gridlines",action='store_true',default=False, help='Include gridlines on the plot')
    parser.add_argument("--cbar_levels",nargs='*',type=float, default=None,help='Define ticks on colorbar')
    parser.add_argument("--mask", type=str, nargs=2,
                        metavar=('SFTLF_FILE', 'REALM'), default=None,
                        help="""Provide sftlf file and realm to mask ('land' or 'ocean')""")


    args = parser.parse_args()
    
    main(args)
    
