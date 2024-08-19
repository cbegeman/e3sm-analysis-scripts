import xarray as xr
import numpy as np

filepath = '/lcrc/group/acme/ac.dcomeau/scratch/chrys/20240813.GMPAS-JRA1p5-DIB-PISMF.TL319_SOwISC12to30E3r3.mesh-review-test.chrysalis/run'

fileinit = '/lcrc/group/e3sm/data/inputdata/ocn/mpas-o/SOwISC12to30E3r3/mpaso.SOwISC12to30E3r3.20240801.nc'
ds_init = xr.open_dataset(fileinit)

fileabort = f'{filepath}/abort_seaice_0001-01-01_00.00.00.nc'
ds_abort = xr.open_dataset(fileabort)

fileseaice = f'{filepath}/20240813.GMPAS-JRA1p5-DIB-PISMF.TL319_SOwISC12to30E3r3.mesh-review-test.chrysalis.mpassi.hist.am.timeSeriesStatsDaily.0001-04-01.nc'
prefix = 'timeDaily_avg_'
ds_seaice = xr.open_dataset(fileseaice)

filehfocean = f'{filepath}/20240813.GMPAS-JRA1p5-DIB-PISMF.TL319_SOwISC12to30E3r3.mesh-review-test.chrysalis.mpaso.hist.am.highFrequencyOutput.0001-04-01_00.00.00.nc'
ds_hfocean = xr.open_dataset(filehfocean)

fileocean = f'{filepath}/20240813.GMPAS-JRA1p5-DIB-PISMF.TL319_SOwISC12to30E3r3.mesh-review-test.chrysalis.mpaso.hist.am.timeSeriesStatsMonthly.0001-03-01.nc'
ds_ocean = xr.open_dataset(fileocean).isel(Time=-1)

# get cell with max sea ice thickness
ds_seaice_lasttime = ds_seaice.isel(Time=-1)
iceVolumeCell = ds_seaice_lasttime[f'{prefix}iceVolumeCell']
iceAreaCell = ds_seaice_lasttime[f'{prefix}iceAreaCell']
iceThicknessCell = iceVolumeCell / iceAreaCell
idx_maxseaice = np.nanargmax(iceThicknessCell.values)
#idx_maxseaice = 676070 - 1
#idx_maxseaice = 122951 - 1
print(f'iCell = {idx_maxseaice + 1}')
ds_seaice_maxseaice = ds_seaice.isel(nCells=idx_maxseaice)
ds_ocean_maxseaice = ds_ocean.isel(nCells=idx_maxseaice)

ds_abort_maxseaice = ds_abort.isel(nCells=idx_maxseaice, Time=-1)
print(f'lat, lon = {ds_abort_maxseaice["latCell"].values*180./np.pi},'
      f'{ds_abort_maxseaice["lonCell"].values*180./np.pi}')
print(f'P sea ice = {ds_abort_maxseaice["icePressure"].values} N/m')
print(f'div = {ds_abort_maxseaice["divergence"].values} %/day')
#print(f'div = {ds_seaice_maxseaice["divergence"].values[-10:]} %/day')
print(f'frazil sea ice = {ds_abort_maxseaice["frazilFormation"].values} m/s')
print(f'tend sea ice transport = {ds_abort_maxseaice["iceVolumeTendencyTransport"].values} m/s')
print(f'tend sea ice thermo = {ds_abort_maxseaice["iceVolumeTendencyThermodynamics"].values} m/s')
print(f'tend sea ice basal melt = {ds_abort_maxseaice["basalIceMelt"].values} m/s')
print(f'tend sea ice lateral melt = {ds_abort_maxseaice["lateralIceMelt"].values} m/s')
#print(f'tau atm u = {ds_abort_maxseaice[""].values} m/s')
#print(f'tau atm v = {ds_abort_maxseaice[""].values} m/s')

ds_init_maxseaice = ds_init.isel(Time=0, nCells=idx_maxseaice)
cellsOnCell = ds_init_maxseaice['cellsOnCell'].values
bottomDepth = ds_init_maxseaice["bottomDepth"]
print(f'bottomDepth = {bottomDepth.values}')
print(f'maxLevelCell = {ds_init_maxseaice["maxLevelCell"].values}')
print(f"landIceFraction = {ds_init_maxseaice['landIceFraction'].values}")
print(f"landIceDraft = {ds_init_maxseaice['landIceDraft'].values}")
print(f"landIceFloatingMask = {ds_init_maxseaice['landIceFloatingMask'].values}")
print(f"nEdges = {ds_init_maxseaice['nEdgesOnCell'].values}")
print(f"latNeighbor = {ds_init['latCell'].values[cellsOnCell] * 180/np.pi}")
print(f"lonNeighbor = {ds_init['lonCell'].values[cellsOnCell] * 180/np.pi}")
print(f"landIceDraftNeighbor = {ds_init['landIceDraft'].values[0, cellsOnCell]}")
print(f"bottomDepthNeighbor = {ds_init['bottomDepth'].values[cellsOnCell]}")
print(f"landIceFloatingMaskNeighbor = {ds_init['landIceFloatingMask'].values[0, cellsOnCell]}")
print(f"maxLevelCellNeighbor = {ds_init['maxLevelCell'].values[cellsOnCell]}")

print(f"max H sea ice = {iceThicknessCell.isel(nCells=idx_maxseaice).values}")

ds_hfocean_maxseaice = ds_hfocean.isel(nCells=idx_maxseaice, Time=-1)
landIceFreshwaterFlux = ds_hfocean_maxseaice['landIceFreshwaterFluxTotal']
Tsurf = ds_hfocean_maxseaice['temperatureAtSurface']
Tbot = ds_hfocean_maxseaice['temperatureAtBottom']
Ssurf = ds_hfocean_maxseaice['salinityAtSurface']
Sbot = ds_hfocean_maxseaice['salinityAtBottom']
ssh = ds_hfocean_maxseaice['ssh']
pressureAdjSsh = ds_hfocean_maxseaice['pressureAdjustedSSH']
columnIntSpeed = ds_hfocean_maxseaice['columnIntegratedSpeed']
H = ssh + bottomDepth 

print(f"landIceFreshwaterFlux = {landIceFreshwaterFlux.values}")
print(f"SSH = {ssh.values}")
print(f"|U| = {columnIntSpeed.values/H.values}")
print(f"pAdj SSH = {pressureAdjSsh.values}")
print(f"Tsurf, Tbot = {Tsurf.values}, {Tbot.values}")
print(f"Ssurf, Sbot = {Ssurf.values}, {Sbot.values}")

prefix = 'timeMonthly_avg_'
icebergFreshWaterFlux = ds_ocean_maxseaice[f'{prefix}icebergFreshWaterFlux'].values
frazilIceFreshwaterFlux = ds_ocean_maxseaice[f'{prefix}frazilIceFreshwaterFlux'].values
landIceFrictionVelocity = ds_ocean_maxseaice[f'{prefix}landIceFrictionVelocity'].values
landIceFreshwaterFlux = ds_ocean_maxseaice[f'{prefix}landIceFreshwaterFlux'].values
seaIceFreshwaterFlux = ds_ocean_maxseaice[f'{prefix}seaIceFreshWaterFlux'].values
iceRunoffFlux = ds_ocean_maxseaice[f'{prefix}iceRunoffFlux'].values
riverRunoffFlux = ds_ocean_maxseaice[f'{prefix}riverRunoffFlux'].values
windStressZonal = ds_ocean_maxseaice[f'{prefix}windStressZonal'].values
windStressMeridional = ds_ocean_maxseaice[f'{prefix}windStressMeridional'].values
#print(f"land ice u* = {landIceFrictionVelocity}")
print(f"iceberg fw flux = {icebergFreshWaterFlux}")
print(f"sea ice fw flux = {seaIceFreshwaterFlux}")
print(f"land ice fw flux = {landIceFreshwaterFlux}")
print(f"sea ice frazil fw flux = {frazilIceFreshwaterFlux}")
print(f"ice runoff flux = {iceRunoffFlux}")
print(f"river runoff flux = {riverRunoffFlux}")
print(f"wind stress u, v = {windStressZonal}, {windStressMeridional}")
