import xarray as xr
import numpy as np

#mesh_file = '/lcrc/group/e3sm/ac.xylar/compass_1.4/chrysalis/e3smv3-meshes/sowisc12to30e3r3/ocean/global_ocean/SOwISC12to30/WOA23/files_for_e3sm/ocean_initial_condition/mpaso.SOwISC12to30E3r3.20240705.nc'
#init_file = '/lcrc/group/e3sm/data/inputdata/ocn/mpas-o/SOwISC12to30E3r3/mpaso.SOwISC12to30E3r3.20240705.nc'
#init_file = '/lcrc/group/e3sm/ac.xylar/compass_1.4/chrysalis/e3smv3-meshes/sowisc12to30e3r3/ocean/global_ocean/SOwISC12to30/WOA23/init/initial_state/initial_state.nc'
init_file = '/lcrc/group/e3sm/data/inputdata/ocn/mpas-o/SOwISC12to30E3r3/mpaso.SOwISC12to30E3r3.20240711.nc'

ds = xr.open_dataset(init_file)
ds = ds.isel(Time=0)
maxLevelCell = ds.maxLevelCell - 1
if 'minLevelCell' in ds.keys():
    minLevelCell = ds.minLevelCell - 1
else:
    minLevelCell = xr.zeros_like(maxLevelCell)
layerThickness = ds.layerThickness
restingThickness = ds.restingThickness
bottomDepth = ds.bottomDepth
if 'ssh' in ds.keys():
    ssh = ds.ssh
landIceMask = ds.landIceMask

nVertLevels = layerThickness.shape[1]
verticalIndices = np.arange(nVertLevels)
vertical_mask = np.ones_like(layerThickness.values, dtype=bool)
for verticalIndex in verticalIndices:
    vertical_mask[:, verticalIndex] = np.logical_and(maxLevelCell.values >= minLevelCell.values,
        np.logical_and(verticalIndex <= maxLevelCell.values,
                       verticalIndex >= minLevelCell.values))
layerThicknessMasked = np.where(vertical_mask, layerThickness, np.nan)

land_ice_mask = landIceMask.values == 0

#columnThickness = np.nansum(layerThicknessMasked, axis=1)
columnThickness = np.nansum(layerThickness, axis=1)
restingColumnThickness = np.nansum(restingThickness, axis=1)
column_thickness_mask_isc = columnThickness < 5.
column_thickness_mask = columnThickness < 10.

print(f'Number of cells with column thickness less than 10. \t{np.nansum(column_thickness_mask)}')
mask = np.logical_and(land_ice_mask, column_thickness_mask)
print(f'Number of cells with column thickness less than 10. outside ice shelf cavities \t{np.nansum(mask)}')

mask = np.logical_and(land_ice_mask, column_thickness_mask_isc)
print(f'Number of cells with column thickness less than 5. outside ice shelf cavities \t{np.nansum(mask)}')

mask = np.logical_and(land_ice_mask, column_thickness_mask)
bottomDepthMasked = bottomDepth.values[mask]
if 'ssh' in ds.keys():
    sshMasked = ssh.values[mask]
layerThicknessThinMask = layerThickness.values[mask, :]
landIcePressureMasked = ds.landIcePressure.values[mask]
landIceDraftMasked = ds.landIceDraft.values[mask]
nCells = np.shape(layerThicknessThinMask)[0]
for i in range(nCells):
    print(i)
    kmin = minLevelCell.values[mask][i]
    kmax = maxLevelCell.values[mask][i]
    print(f'\tnlayers = {kmax + 1 - kmin}')
    #print(f'\trestingColumnThickness = {restingColumnThickness[mask][i]}')
    #print(f'\tlayerThickness = {layerThicknessThinMask[i, :]}')
    if 'ssh' in ds.keys():
        print(f'\tssh + bottomDepth = {sshMasked[i] + bottomDepthMasked[i]}')
    else:
        print(f'\tcolumnThickness = {columnThickness[mask][i]}')
    #print(f'\tbottomDepth = {bottomDepthMasked[i]}')
    #print(f'\tlandIcePressure = {landIcePressureMasked[i]}')
    print(f'\tlandIceDraft = {landIceDraftMasked[i]}')
    print(f'\tmin(layerThickness) = {np.nanmin(layerThicknessThinMask[i, kmin:kmax+1])}')

mask = np.logical_and(~land_ice_mask, column_thickness_mask_isc)
print(f'Number of cells with column thickness less than 5. inside ice shelf cavities \t{np.nansum(mask)}')

mask = np.logical_and(~land_ice_mask, column_thickness_mask)
if 'ssh' in ds.keys():
    sshMasked = ssh.values[mask]
bottomDepthMasked = bottomDepth.values[mask]
layerThicknessThinMask = layerThickness.values[mask, :]
landIcePressureMasked = ds.landIcePressure.values[mask]
landIceDraftMasked = ds.landIceDraft.values[mask]
nCells = np.shape(layerThicknessThinMask)[0]
for i in range(nCells):
    print(i)
    kmin = minLevelCell.values[mask][i]
    kmax = maxLevelCell.values[mask][i]
    #print(f'\tcolumnThickness = {columnThickness[mask][i]}')
    #print(f'\trestingColumnThickness = {restingColumnThickness[mask][i]}')
    #print(f'\tlayerThickness = {layerThicknessThinMask[i, :]}')
    print(f'\tnlayers = {kmax + 1 - kmin}')
    if 'ssh' in ds.keys():
        print(f'\tssh + bottomDepth = {sshMasked[i] + bottomDepthMasked[i]}')
    else:
        print(f'\tcolumnThickness = {columnThickness[mask][i]}')
    #print(f'\tbottomDepth = {bottomDepthMasked[i]}')
    #print(f'\tlandIcePressure = {landIcePressureMasked[i]}')
    print(f'\tlandIceDraft = {landIceDraftMasked[i]}')
    print(f'\tmin(layerThickness) = {np.nanmin(layerThicknessThinMask[i, kmin:kmax+1])}')
