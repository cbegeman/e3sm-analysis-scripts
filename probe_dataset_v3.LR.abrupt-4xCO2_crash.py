import xarray as xr
import numpy as np
import numpy as np
import os
from math import pi, nan
import gsw
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import datetime

def decimal_days_from_xtime(xtime, start_day=1):
    """
    Determine the time index closest to the target time

    Parameters
    ----------
    xtime : numpy.ndarray of numpy.char
        Times in the dataset

    dt_target : float
        Time in seconds since the first time in the list of ``xtime`` values

    start_xtime : str, optional
        The start time, the first entry in ``xtime`` by default

    Returns
    -------
    time_index : int
        Index in xtime that is closest to dt_target
    """
    sec_per_day = 3600. * 24.
    t0 = datetime.datetime.strptime(xtime.values[0].decode(),
                                    '%Y-%m-%d_%H:%M:%S')
    decimal_days = np.zeros((len(xtime.values)))
    for time_index in range(len(xtime.values)):
        tf = datetime.datetime.strptime(xtime.values[time_index].decode(),
                                        '%Y-%m-%d_%H:%M:%S')
        t = (tf - t0).total_seconds()
        decimal_days[time_index] = start_day + t / sec_per_day
    return decimal_days

def print_cpl_stats(filename, mask, mask_level, varlist=[], ntime_start=0, ntime_slices=1, create_plot=False, print_minmax=False):
    ds_all = xr.open_dataset(filename)
    keys = ds_all.keys()
    ntime = ds_all.sizes['time']
    times = np.arange(ntime-ntime_slices-ntime_start, ntime-ntime_start, 1)
    ds_all = ds_all.isel(domo_ny=0)
    ds_all = ds_all.isel(domr_ny=0)
    ds_all = ds_all.isel(doma_ny=0)
    ds_all = ds_all.isel(domi_ny=0)
    if create_plot:
        if len(varlist) < 1:
            varlist = ds_all.keys()
        for var in varlist:
            fig = plt.subplots(nrows=len(varlist), ncols=1, sharex=True, figsize=(8, 4 * len(varlist)))
            plt.subplot(nrows, 1, 1)
            plt.plot(times, ds.x2oacc_Foxx_sen.values[mask])
            ax = plt.gca()
            ax.set_ylabel('')
            
            
    if print_minmax:
        for time in times:
            print(ds_all['time'].values[time])
            ds = ds_all.isel(time=time)
            print(f'SST_cpl max,min = {np.nanmin(ds.o2x_So_t.values[mask])},{np.nanmax(ds.o2x_So_t.values[mask])}')
            print(f'SSS_cpl max,min = {np.nanmin(ds.o2x_So_s.values[mask])},{np.nanmax(ds.o2x_So_s.values[mask])}')
            print(f'SSu_cpl max,min = {np.nanmin(ds.o2x_So_u.values[mask])},{np.nanmax(ds.o2x_So_u.values[mask])}')
            print(f'SSv_cpl max,min = {np.nanmin(ds.o2x_So_v.values[mask])},{np.nanmax(ds.o2x_So_v.values[mask])}')
            print(f'SHF_cpl max,min = {np.nanmin(ds.x2oacc_Foxx_sen.values[mask])},{np.nanmax(ds.x2oacc_Foxx_sen.values[mask])}')
            print(f'LHF_cpl max,min = {np.nanmin(ds.x2oacc_Foxx_lat.values[mask])},{np.nanmax(ds.x2oacc_Foxx_lat.values[mask])}')
            if len(np.shape(ds.x2i_Fioo_frazil.values))==2:
                print(f'frazil_cpl max,min = {np.nanmin(ds.x2i_Fioo_frazil.values[0, mask])},{np.nanmax(ds.x2i_Fioo_frazil.values[0, mask])}')
            else:
                print(f'frazil_cpl max,min = {np.nanmin(ds.x2i_Fioo_frazil.values[mask])},{np.nanmax(ds.x2i_Fioo_frazil.values[mask])}')
            if len(np.shape(ds.i2x_Fioi_meltw.values))==2:
                print(f'icemelt_cpl max,min = {np.nanmin(ds.i2x_Fioi_meltw.values[0, mask])},{np.nanmax(ds.i2x_Fioi_meltw.values[0, mask])}')
            else:
                print(f'icemelt_cpl max,min = {np.nanmin(ds.i2x_Fioi_meltw.values[mask])},{np.nanmax(ds.i2x_Fioi_meltw.values[mask])}')
            #print(f'riverrunoff_cpl max,min = {np.nanmin(ds.r2xo_Forr_rofl.values[mask])},{np.nanmax(ds.r2xo_Forr_rofl.values[mask])}')
            #print(f'icerunoff_cpl max,min = {np.nanmin(ds.x2i_Fixx_rofi.values[0, mask])},{np.nanmax(ds.x2i_Fixx_rofi.values[0, mask])}')

def print_var_stats(filename, ds_mesh, mask, mask_level, mask_250m,
                    varlist=[], ntime_start=None, ntime_slices=None, show_all=False, show_minmax=False,
                    savepath='', last_time = 'time', start_day=0,
                    create_plot=True, create_hovmoller=False, xlabel=''):
    ds_all = xr.open_dataset(filename)
    keys = ds_all.keys()
    ntime = ds_all.sizes['Time']
    if ntime_slices == None:
        times = range(ntime)
    else:
        times = np.arange(ntime-ntime_slices-ntime_start, ntime-ntime_start, 1)
    if last_time != 'time':
        _, start_day, _ = last_time.split('-')
    decimal_days = decimal_days_from_xtime(ds_all.xtime, start_day=float(start_day))
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ls = [(1, ()), (0, (1, 1)), (0, (5, 5)),  (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
    nCells = np.sum(mask)
    if show_minmax:
        if create_hovmoller:
            if np.sum(mask) > 1:
                print('create_hovmoller not supported')
                create_hovmoller = False
            maxLevelCell = ds_mesh.maxLevelCell.values[mask]
            var_hovmoller = np.zeros((len(times), maxLevelCell[0]))
        for itime, time in enumerate(times):
            print(ds_all['xtime'].values[time])
            ds = ds_all.isel(Time=time)
            print(ds['daysSinceStartOfSim'].values)
            if create_plot:
                fig = plt.figure()
            if len(varlist) < 1:
                varlist = keys
            for iVar, key in enumerate(varlist):
                if key not in keys:
                    continue
                var = ds[key].values
                #mask_nan = np.isnan(var)
                #if np.sum(mask_nan) > 0:
                #    print(f'\t{key} contains nans at {np.sum(mask_nan)} points:')
                if ds[key].dtype == 'float64' or ds[key].dtype == 'float32':
                    if 'nCells' not in ds[key].dims:
                       continue
                    if show_all:
                        print(f'\t{key} all: {np.nanmin(var)}, {np.nanmax(var)}')
                        #lat_nan = ds_mesh.latCell.values[mask_nan]
                        #lon_nan = ds_mesh.lonCell.values[mask_nan]
                        #for i in range(5):
                        #    print(f'\t\t{lat_nan[i]}, {lon_nan[i]}')
                            #if 'nVertLevels' in ds[key].dims or 'nVertLevelsP1' not in ds[key].dims:
                            #    print(f'\t\t\t{ds[key].values}')
                    #mask_bad = np.logical_and(np.greater_equal(ds_mesh.bottomDepth.values, 250.),
                    #                          np.less_equal(var, -1.e33))
                    #if np.sum(mask_bad) > 0:
                    #    print(f'\t{key} contains bad values at {np.sum(mask_bad)} points:')
                    #    lat_nan = lat[mask_bad]
                    #    lon_nan = lon[mask_bad]
                    #    print(f'\tlat bad: {np.min(lat_nan)}, '
                    #          f'{np.max(lat_nan)}')
                    #    print(f'\tlon bad: {np.min(lon_nan)}, '
                    #          f'{np.max(lon_nan)}')
                    if np.sum(mask) > 0:
                        if 'nVertLevels' in ds[key].dims:
                            # print(f'--------- {key}: masking out levels')
                            var = var[mask, :]
                            maxLevelCell = ds_mesh.maxLevelCell.values[mask]
                            zMid = ds_mesh.zMid.values[mask, :]
                            masked_var = var[mask_level[mask]]
                        elif 'nVertLevelsP1' in ds[key].dims:
                            var = var[mask, :-1]
                            maxLevelCell = ds_mesh.maxLevelCell.values[mask] - 1
                            zMid = ds_mesh.zMid.values[mask, :]
                            masked_var = var
                        elif 'At250m' in key:
                            # print(f'--------- {key}: masking out >250m levels')
                            masked_var = var[mask_250m * mask]
                        else:
                            masked_var = var[mask]
                        print(f'\t{key} masked: {np.nanmin(masked_var)}, '
                              f'{np.nanmax(masked_var)}')
                        if key == 'ssh':
                            bottomDepth_masked = ds_mesh.bottomDepth.values[mask]
                            wct = bottomDepth_masked + masked_var
                            print(f'\tbottomDepth masked: {np.min(bottomDepth_masked)}, '
                                  f'{np.max(bottomDepth_masked)}')
                            print(f'\twct masked: {np.min(wct)}, '
                                  f'{np.max(wct)}')
                    if create_hovmoller:
                        iCell = 0
                        zidx = maxLevelCell[0]
                        var_hovmoller[itime, :zidx] = var[iCell, :zidx]
                    if create_plot:
                        for iCell in range(nCells):
                            plt.plot(var[iCell, :int(maxLevelCell[iCell])],
                                     zMid[iCell, :maxLevelCell[iCell]],
                                     color=color_list[iVar], linestyle=ls[iCell], label=key)
            else:
                print(f'\ttype of {key} not yet supported')
        if create_plot:
            ax = plt.gca()
            if nCells == 1:
                plt.legend()
            plt.ylim([-1500., 0.])
            plt.title(ds_all["xtime"].values[time])
            plt.xlabel(xlabel)
            plt.ylabel('Depth (m)')
            fig = plt.savefig(f'{xlabel}_{last_time}_{time:02g}.png', bbox_inches='tight')
            fig.close()
    if create_hovmoller:
        if np.sum(mask) > 1:
            print('create_hovmoller not supported')
            create_hovmoller = False
        maxLevelCell = ds_mesh.maxLevelCell.values[mask]
        for iVar, key in enumerate(varlist):
            print(key)
            if key not in keys:
                print(f'{key} not in dataset')
                continue
            var = ds_all[key].values
            if 'nVertLevels' in ds_all[key].dims:
                var = var[:, mask, :]
                maxLevelCell = ds_mesh.maxLevelCell.values[mask]
                zMid = ds_mesh.zMid.values[mask, :]
            elif 'nVertLevelsP1' in ds_all[key].dims:
                var = var[:, mask, :-1]
                maxLevelCell = ds_mesh.maxLevelCell.values[mask] - 1
                zMid = ds_mesh.zMid.values[mask, :]
            iCell = 0
            zidx = maxLevelCell[0]
            var_hovmoller = np.zeros((len(times), maxLevelCell[0]))
            var_hovmoller[:, :] = var[:, 0, :zidx]
            fig = plt.figure()
            if key == 'temperature':
                levels = [0.1, 1., 5., 10., 15., 20., 25., 30., 35., 40., 45.]
                vmin = 0.1
                vmax = 50.
                cNorm = LogNorm(vmin=vmin, vmax=vmax)
                #cNorm  = SymLogNorm(1, vmin=-40, vmax=40)
                var_hovmoller[var_hovmoller <= 0] = 1e-5
                cmap_name = 'viridis'
            elif key == 'vertVelocityTop':
                vmax = 0.05
                vmin = -vmax
                levels = np.arange(vmin, vmax, 0.005)
                cNorm  = Normalize(vmin=vmin, vmax=vmax, clip=False)
                cmap_name = 'coolwarm'
            elif key == 'salinity':
                vmax = 40
                vmin = 33
                levels = np.arange(vmin, vmax, 0.5)
                cNorm  = Normalize(vmin=vmin, vmax=vmax)
                cmap_name = 'viridis'
            elif key == 'velocityZonal' or key == 'velocityMeridional':
                vmax = 0.3
                vmin = -vmax
                levels = np.arange(vmin, vmax, 0.01)
                cNorm  = Normalize(vmin=vmin, vmax=vmax)
                cmap_name = 'coolwarm'
            cm = plt.get_cmap(cmap_name) 
            if cmap_name == 'viridis':
                cm.set_under('black')
                cm.set_over('yellow')
            if cmap_name == 'coolwarm':
                cm.set_under('blue')
                cm.set_over('red')
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            plt.contourf(decimal_days, zMid[0, :zidx], np.transpose(var_hovmoller[:, :zidx]),
                         cmap=cm, norm=cNorm, levels=levels)
            plt.ylim([-1500., 0.])
            plt.title(key)
            ax = plt.gca()
            #ax.invert_yaxis()
            cbar = fig.colorbar(scalarMap, ax=ax)
            plt.xlabel('Time (days)')
            plt.ylabel('Depth (m)')
            print('{savepath}/{key}_hovmoller_{last_time_rst}.png')
            fig = plt.savefig(f'{savepath}/{key}_hovmoller_{last_time}.png', bbox_inches='tight')
    return

# paths and files
#runname = 'v3.LR.abrupt-4xCO2_0101_cpl4'
#runname = 'v3.LR.abrupt-4xCO2_0101_cpl3'
#runname = 'v3.LR.abrupt-4xCO2_0101_cpl2'
#filepath = f'/lcrc/group/e3sm/ac.cbegeman/E3SMv3/{runname}/tests/custom-10_5x1_ndays_highFrequencyOceanOutput/run'
#filepath = f'/lcrc/group/e3sm/ac.cbegeman/E3SMv3/{runname}/tests/custom-10_5x1_ndays_highFreqOceanOutput/run'
#filepath = f'/lcrc/group/e3sm/ac.cbegeman/E3SMv3/{runname}/tests/custom-10_10x1_ndays_highFreqOceanOutput_day22/run'
meshfile = '/lcrc/group/e3sm/ac.xylar/compass_1.2/chrysalis/e3smv3-meshes/icoswisc30e3r5/ocean/global_ocean/IcoswISC/WOA23/files_for_e3sm/ocean_mesh/IcoswISC30E3r5.20231120.nc'

#last_time_rst = '0032-04-27'
##last_time_rst = '0032-04-28'
##last_time_hf = '0032-04-01'
#last_time_hf = '0032-04-27'
#
#rstfile = f'{runname}.mpaso.rst.{last_time_rst}_00000.nc'
#rstfile_ice = f'{runname}.mpassi.rst.{last_time_rst}.nc'
#
#hffile = f'{runname}.mpaso.hist.am.highFrequencyOutput.{last_time_hf}_00.00.00.nc'
#hffile_ice = f'{runname}.mpassi.hist.am.highFrequencyOutput.{last_time_hf}.nc'

# Get datasets just for last available time
ds_mesh = xr.open_dataset(meshfile)
ds_mesh = ds_mesh.isel(Time=0)
lat = ds_mesh.latCell.values
lon = ds_mesh.lonCell.values
nVertLevels = ds_mesh.sizes['nVertLevels']
maxLevelCell = ds_mesh.maxLevelCell.values - 1
bottomDepth = ds_mesh.bottomDepth.values
refBottomDepth = ds_mesh.refBottomDepth.values
pref = 9.81 * 1000. * ds_mesh.refZMid.values

if 'landIceDraft' in ds_mesh.keys():
    landIceDraft = ds_mesh.landIceDraft.values
    landIceMask = ds_mesh.landIceMask.values
    layerThickness = ds_mesh.layerThickness.values
    print(f'min landIceDraft {np.nanmin(landIceDraft)}')
    mask = np.logical_and(np.greater_equal(landIceDraft, 0.), np.greater_equal(landIceMask, 1.))
    print(f'nCells with land ice mask: {np.sum(landIceMask)}')
    print(f'nCells with land ice draft: {np.sum(np.less(landIceDraft, 0.))}')
    print(f'nCells with land ice mask and not land ice draft: {np.sum(mask)}')

        #np.logical_and(np.greater_equal(ds_rst.accumulatedLandIceFrazilMass.values, 800.),
        #mask_tracer = np.less_equal(ds['temperatureAt250m'].values, -1.e33)
        #mask_bad = np.logical_and(np.greater_equal(ds.ssh.values, -250.),
        #              np.logical_and(valid_250m, mask_tracer))
        #print(f'nCells with bad values: {np.sum(mask_bad)}')
        #print(f'nCells with bad values and landIceMask: {np.sum(np.logical_and(landIceMask, mask_bad))}')
        #print(f'nCells with bad values and landIce* wonky: {np.sum(np.logical_and(mask, mask_bad))}')
        #mask_frazil = np.greater_equal(ds_rst.accumulatedLandIceFrazilMass.values, 0.)
        #print(f'nCells with bad values and landIce* wonky and landIceFrazil: {np.sum(np.logical_and(np.logical_and(mask, mask_bad), mask_frazil))}')
#ds_rst = xr.open_dataset(f'{filepath}/{rstfile}')
#ds_rst = ds_rst.isel(Time=0)

#ds = xr.open_dataset(f'{filepath}/{hffile_ice}')
#for key in ds.keys():
#    if ds[key].dtype == 'float64' or ds[key].dtype == 'float32':
#        if 'nCells' in ds[key].dims:
#            masked_var = ds[key].values[mask]
#            print(f'{key}: {np.nanmin(masked_var)}, '
#                  f'{np.nanmax(ds[key].values)}')
#        else:
#            print(f'dimensions of {key} not yet supported')
#    else:
#        print(f'type of {key} not yet supported')
#ds = xr.open_dataset(f'{filepath}/{rstfile_ice}')
#ds_ice = ds.isel(Time=0)
#for key in ds.keys():
#    if 'Time' in ds[key].dims:
#        if ds[key].dtype == 'float64' or ds[key].dtype == 'float32':
#            V_ice = ds_ice[key].values # units of m
#            if 'nCategories' in ds_ice[key].dims:
#                for cat in range(5):
#                    masked_var = V_ice[mask, cat]
#                    print(f'\t{key} masked: {np.min(masked_var)}, '
#                      f'{np.max(masked_var)}')
#            else:
#                masked_var = V_ice[mask]
#                print(f'\t{key} masked: {np.min(masked_var)}, '
#                  f'{np.max(masked_var)}')
#    

lat1 = 25. * pi/180.
lat2 = 28. * pi/180.
lon1 = 142. * pi/180.
lon2 = 146. * pi/180.
mask_region = np.logical_and(
    np.logical_and(lat >= lat1,
                   lat <= lat2),
    np.logical_and(lon >= lon1,
                   lon <= lon2))
print(f'nCells in region: {np.sum(mask_region)}')

zlev_1d = np.arange(0, nVertLevels, 1)
zlev_2d, maxLevelCell_2d = np.meshgrid(zlev_1d, maxLevelCell)
mask_level = np.less_equal(zlev_2d, maxLevelCell_2d)
invalid_level = mask_level == 0

iLevel0250 = 0
for iLevel in range(nVertLevels):
    if (refBottomDepth[iLevel] > 250.):
       iLevel0250 = iLevel-1
       break
mask_250m = np.greater_equal(maxLevelCell, iLevel0250)

print(f'bottomDepth valid: {np.min(bottomDepth[mask_250m])}, {np.max(bottomDepth[mask_250m])}')
_, maxLevelCell_2d = np.meshgrid(zlev_1d, maxLevelCell)
zMask = np.less_equal(zlev_2d, maxLevelCell_2d)

#ds = xr.open_dataset(f'{filepath}/{rstfile}')
#mask_frazil = ds['accumulatedLandIceFrazilMass'].values > 1.e4
#mask_seaice = ds['seaIcePressure'].values[:] > 5e-4

#hffile_old = f'/lcrc/group/e3sm/ac.cbegeman/E3SMv3/v3.LR.abrupt-4xCO2_0101_cpl2/tests/custom-10_5x1_ndays_highFrequencyOceanOutput/run/v3.LR.abrupt-4xCO2_0101_cpl2.mpaso.hist.am.highFrequencyOutput.0032-04-01_00.00.00.nc'
#ds = xr.open_dataset(f'{hffile_old}')
#ds = ds.isel(Time=2)
#print(f'Getting temperature from {ds["xtime"].values}')
#var = ds['temperature'].values
#var[invalid_level] = nan
#columnwise_min_pt = np.nanmin(var, axis=1)
#mask_pt = columnwise_min_pt < 0. 

#print(f'pt mask: {var[mask, :]}')
#print(f'sa mask: {ds.salinity.values[mask, :]}')

#last_time_cpl = '0032-04-27'
#last_time_cpl = '0032-04-28'
#cpl_time = 45000.
##cplfile = f'{runname}.cpl.hi.{last_time_cpl}-{cpl_time:05g}.nc'
#cplfile = f'/lcrc/group/e3sm/ac.cbegeman/E3SMv3/v3.LR.abrupt-4xCO2_0101_cpl2/tests/custom-10_5x1_ndays_highFrequencyOceanOutput/run/v3.LR.abrupt-4xCO2_0101_cpl2.cpl.hi.{last_time_cpl}-{cpl_time:05g}.nc'
#ds_cpl = xr.open_dataset(f'{cplfile}')
#ds_cpl = ds_cpl.isel(time=-1)
#ds_cpl = ds_cpl.isel(domo_ny=0)
#velocity_mag = np.sqrt(np.square(ds_cpl.o2x_So_u.values) + np.square(ds_cpl.o2x_So_v.values))
#mask_velocity = velocity_mag > 10.
#mask_frazil = ds_cpl.x2i_Fioo_frazil.values != 0.
#print(f'max velocity: {np.nanmax(velocity_mag[mask_region])}')

#ds = xr.open_dataset(f'{filepath}/{hffile}')
#ds = ds.isel(Time=-1)
cell_idx = np.arange(0, ds_mesh.sizes['nCells'], 1)
#mask_salinity = ds['salinity'].values[] < 0.

#mask = np.logical_and(mask_frazil[0, :], mask_region)
#print(f'lat min,max: {np.min(lat[mask])*180./pi}, {np.max(lat[mask])*180./pi}')
#print(f'lon min,max: {np.min(lon[mask])*180./pi}, {np.max(lon[mask])*180./pi}')
#mask = np.logical_and(mask, columnwise_min_pt< 0.)
#print(f'nCells in mask: {np.sum(mask)}')
#print(f'nCells in mask with high velocity: {np.sum(np.logical_and(mask, mask_velocity))}')
#print(f'min pt in mask: {columnwise_min_pt[mask]}')
#print(f'lat min,max: {np.min(lat[mask])*180./pi}, {np.max(lat[mask])*180./pi}')
#print(f'lon min,max: {np.min(lon[mask])*180./pi}, {np.max(lon[mask])*180./pi}')
#for filename in [f'{filepath}/{hffile}']: #,f'{filepath}/{rstfile}',f'{filepath}/{rstfile_ice}']:
mask = cell_idx == 454295

last_time_hf = '0032-01-10'
#filepath='/lcrc/group/e3sm/ac.cbegeman/E3SMv3/v3.LR.abrupt-4xCO2_0101/tests/custom-10_5x1_ndays_drag_1e-2'
filepath='/lcrc/group/e3sm/ac.cbegeman/E3SMv3/v3.LR.abrupt-4xCO2_0101/tests/S_1x10_ndays_highFreqOutput'
#filepath='/lcrc/group/e3sm/ac.cbegeman/E3SMv3/v3.LR.abrupt-4xCO2_0101/tests/custom-10_5x1_ndays_split_implicit'
for filename in [f'{filepath}/run/v3.LR.abrupt-4xCO2_0101.mpaso.hist.am.highFrequencyOutput.{last_time_hf}_00.00.00.nc']:
#for filename in ['/lcrc/group/e3sm/ac.cbegeman/E3SMv3/v3.LR.abrupt-4xCO2_0101_cpl4/tests/custom-10_10x1_ndays_highFrequencyOceanOutput_day22/run/v3.LR.abrupt-4xCO2_0101_cpl4.mpaso.hist.am.highFrequencyOutput.0032-04-22.0032-04-28.nc']:
    #print_var_stats(filename, ds_mesh, mask, mask_level, mask_250m, show_minmax=True, create_plot=False, ntime_slices=4, 
    #    ntime_start=0)
    hf_variables = ['temperature', 'salinity', 'vertVelocityTop', 'velocityZonal', 'velocityMeridional']
    print_var_stats(filename, ds_mesh, mask, mask_level, mask_250m, last_time=last_time_hf,
                    varlist=hf_variables, create_plot=False, create_hovmoller=True,
                    savepath=f'{filepath}/plots')
    #hf_variables = ['temperatureHorizontalAdvectionTendency',
    #                'temperatureVerticalAdvectionTendency',
    #                'temperatureHorMixTendency',
    #                'temperatureVertMixTendency',
    #                'temperatureSurfaceFluxTendency',
    #                'temperatureNonLocalTendency',
    #                'temperatureShortWaveTendency',
    #                ]
    #print_var_stats(filename, ds_mesh, mask, mask_level, mask_250m,
    #                varlist=hf_variables, create_plot=True, xlabel='Temperature_tendency')

#days = np.arange(28., 29., 1.)
#for i, cpl_day in enumerate(days):
#    last_time_cpl = f'0032-04-{cpl_day:02g}'
#    times = np.arange(1800., 45000., 1800.)
#    for i, cpl_time in enumerate(times):
#        cplfile = f'{runname}.cpl.hi.{last_time_cpl}-{cpl_time:05g}.nc'
#        for filename in [f'{filepath}/{cplfile}']:
#            if not os.path.isfile(filename):
#                print(f'{filename} does not exist')
#                continue
#            print_cpl_stats(filename, mask, mask_level, varlist=[], ntime_slices=1)
