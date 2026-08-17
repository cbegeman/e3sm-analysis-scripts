import glob
import os
import re

import numpy as np
import xarray as xr

#FATMLNDFRC = '/lcrc/group/e3sm/data/inputdata/share/domains/domain.lnd.r05_IcoswISC30E3r5.231121.nc'
#FINIDAT = '/lcrc/group/e3sm/data/inputdata/lnd/clm2/initdata/v3.LR.historical_0091.elm.r.2010-01-01-00000.nc'
#total column area:        1.496679e+14 m^2
#surface-water area:       3.815827e+11 m^2
#surface-water volume:     1.956889e+12 m^3
#mean depth where wet:     5.128349e+00 m
#number of wet columns:    5748
#max wet-column depth:     1.019988e+05 m
#wet fraction of that col: 1.000000e-02
#wet area of that column:  2.285344e+06 m^2
#FINIDAT = '/lcrc/group/e3sm2/ac.jwolfe/E3SMv3/20260720.baseline.WCYCL2010.bluepulse.chrysalis/archive/rest/0026-01-01-00000/20260720.baseline.WCYCL2010.bluepulse.chrysalis.elm.r.0026-01-01-00000.nc'
#total column area:        1.496679e+14 m^2
#surface-water area:       5.001216e+11 m^2
#surface-water volume:     3.099795e+11 m^3
#mean depth where wet:     6.198084e-01 m
#number of wet columns:    7920
#max wet-column depth:     8.821367e+03 m
#wet fraction of that col: 1.000000e-02
#wet area of that column:  3.403303e+05 m^2
#FINIDAT = '/lcrc/group/e3sm2/ac.jwolfe/E3SMv3/20260720.baseline.WCYCL2010.bluepulse.chrysalis/archive/rest/0026-01-01-00000/20260720.baseline.WCYCL2010.bluepulse.chrysalis.elm.r.0026-01-01-00000.nc.orig'
#total column area:        1.496679e+14 m^2
#surface-water area:       5.001216e+11 m^2
#surface-water volume:     2.143894e+12 m^3
#mean depth where wet:     4.286746e+00 m
#number of wet columns:    7920
#max wet-column depth:     1.031644e+05 m
#wet fraction of that col: 1.000000e-02
#wet area of that column:  2.285344e+06 m^2
FATMLNDFRC = (
    '/lcrc/group/e3sm/data/inputdata/share/domains/'
    'domain.lnd.r05_SOwISC12to30E3r3.250515.nc'
)
FINIDAT = (
    '/lcrc/group/e3sm/data/inputdata/lnd/clm2/initdata/'
    'elmi.v3-SORRM.ne30pg2_r05_SOwISC12to30E3r3.2010-01-01-00000.'
    'c20260304.nc'
)
#total column area:        1.496450e+14 m^2
#surface-water area:       3.807108e+11 m^2
#surface-water volume:     1.820293e+12 m^3
#mean depth where wet:     4.781301e+00 m
#number of wet columns:    5616
#columns deeper than 100m: 256
#10th pct wet fraction:    9.151715e-05
#50th pct wet fraction:    1.000000e-02
#90th pct wet fraction:    1.466888e-01
#max wet-column volume:    2.323064e+11 m^3
#depth of that column:     1.019988e+05 m
#wet fraction of that col: 1.000000e-02
#wet area of that column:  2.277541e+06 m^2
#FREST = '/lcrc/group/e3sm2/ac.cbegeman/E3SMv3_dev/20260723.v3.SORRME3r3.CRYO2010.bluepulse-control.nucleate-ice-subgrid-1.4.chrysalis/run/20260723.v3.SORRME3r3.CRYO2010.bluepulse-control.nucleate-ice-subgrid-1.4.chrysalis.elm.r.0048-01-01-00000.nc'
FREST = '/lcrc/group/e3sm2/ac.cbegeman/E3SMv3_dev/20260723.v3.SORRME3r3.CRYO2010.bluepulse-control.nucleate-ice-subgrid-1.4.chrysalis/archive/rest/0006-01-01-00000/20260723.v3.SORRME3r3.CRYO2010.bluepulse-control.nucleate-ice-subgrid-1.4.chrysalis.elm.r.0006-01-01-00000.nc'
#total column area:        1.496450e+14 m^2
#surface-water area:       2.814317e+11 m^2
#surface-water volume:     1.883782e+12 m^3
#mean depth where wet:     6.693569e+00 m
##FINIDAT = FREST

EARTH_RADIUS = 6.37122e6  # m, consistent with ELM/CIME re


def compute_column_area(ds_domain, ds_init):
    """Return column area (m^2) for each ELM column."""
    # domain area is in radians^2 on the (nj, ni) grid
    area = ds_domain['area'].values * EARTH_RADIUS**2
    frac = ds_domain['frac'].values

    # ELM 1-d indices are 1-based
    ixy = ds_init['cols1d_ixy'].values.astype(int) - 1
    jxy = ds_init['cols1d_jxy'].values.astype(int) - 1
    # weight of the column relative to the gridcell (not the landunit)
    wtgcell = ds_init['cols1d_wtxy'].values

    col_area = area[jxy, ixy] * frac[jxy, ixy] * wtgcell
    return xr.DataArray(
        col_area,
        dims=('column',),
        attrs={'units': 'm^2', 'long_name': 'column area'},
    )


def compute_column_volume(ds_domain, ds_init):
    """Return per-column surface-water volume (m^3) and column area."""
    col_area = compute_column_area(ds_domain, ds_init)
    col_volume = 1.0e-3 * ds_init['H2OSFC'].values * col_area.values
    return col_volume, col_area.values


def compute_column_lat_lon(ds_domain, ds_init):
    """Return per-column latitude and longitude (degrees)."""
    lat = ds_domain['yc'].values
    lon = ds_domain['xc'].values

    # ELM 1-d indices are 1-based
    ixy = ds_init['cols1d_ixy'].values.astype(int) - 1
    jxy = ds_init['cols1d_jxy'].values.astype(int) - 1

    return lat[jxy, ixy], lon[jxy, ixy]


def print_top_volumes(ds_domain, ds_init, num_top=10):
    """Print the largest per-column surface-water volumes in FINIDAT."""
    col_volume, col_area = compute_column_volume(ds_domain, ds_init)
    col_lat, col_lon = compute_column_lat_lon(ds_domain, ds_init)

    order = np.argsort(col_volume)[::-1][:num_top]

    print(f'top {num_top:d} per-column volumes (FINIDAT):')
    for rank, index in enumerate(order, start=1):
        print(
            f'  {rank:d}: col {int(index):d} '
            f'lat {col_lat[index]:+.4f} lon {col_lon[index]:+.4f} '
            f'V {col_volume[index]:.6e} m^3 '
            f'area {col_area[index]:.6e} m^2'
        )


def print_top_volume_changes(ds_domain, num_top=5):
    """Print the largest per-column volume changes from FINIDAT to FREST."""
    ds_early = xr.open_dataset(FINIDAT)
    ds_late = xr.open_dataset(FREST)

    vol_early, area_early = compute_column_volume(ds_domain, ds_early)
    vol_late, _ = compute_column_volume(ds_domain, ds_late)
    col_lat, col_lon = compute_column_lat_lon(ds_domain, ds_early)

    if vol_early.size != vol_late.size:
        print(
            'column counts differ between FINIDAT '
            f'({vol_early.size:d}) and FREST ({vol_late.size:d}); '
            'skipping per-column comparison'
        )
        ds_early.close()
        ds_late.close()
        return

    dvol = vol_late - vol_early
    order = np.argsort(np.abs(dvol))[::-1][:num_top]

    print(f'top {num_top:d} per-column volume changes (FREST - FINIDAT):')
    for rank, index in enumerate(order, start=1):
        print(
            f'  {rank:d}: col {int(index):d} '
            f'lat {col_lat[index]:+.4f} lon {col_lon[index]:+.4f} '
            f'dV {dvol[index]:+.6e} m^3 '
            f'V0 {vol_early[index]:.6e} m^3 '
            f'V1 {vol_late[index]:.6e} m^3 '
            f'area {area_early[index]:.6e} m^2'
        )

    ds_early.close()
    ds_late.close()


ARCHIVE_ROOT = (
    '/lcrc/group/e3sm2/ac.cbegeman/E3SMv3_dev/'
    '20260723.v3.SORRME3r3.CRYO2010.bluepulse-control.'
    'nucleate-ice-subgrid-1.4.chrysalis/archive/rest'
)
CASE_NAME = (
    '20260723.v3.SORRME3r3.CRYO2010.bluepulse-control.'
    'nucleate-ice-subgrid-1.4.chrysalis'
)
REST_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}-\d{5}$')

# attributes to report for columns entering or leaving the top list
COLUMN_ATTRS = [
    'H2OSFC',
    'FH2OSFC',
    'cols1d_ixy',
    'cols1d_jxy',
    'cols1d_itype_col',
    'cols1d_itype_lunit',
    'cols1d_wtgcell',
    'cols1d_wtlunit',
]


def find_restart_files():
    """Return (date_string, path) for each ELM restart file, time-ordered."""
    restarts = []
    for subdir in sorted(os.listdir(ARCHIVE_ROOT)):
        if not REST_PATTERN.match(subdir):
            continue
        pattern = os.path.join(
            ARCHIVE_ROOT, subdir, f'{CASE_NAME}.elm.r.{subdir}.nc'
        )
        matches = sorted(glob.glob(pattern))
        if matches:
            restarts.append((subdir, matches[0]))
    return restarts


def get_top_volume_indices(ds_domain, ds_rest, num_top=10):
    """Return the indices of the ``num_top`` largest column volumes."""
    col_volume, _ = compute_column_volume(ds_domain, ds_rest)
    return np.argsort(col_volume)[::-1][:num_top], col_volume


def print_column_attrs(ds_rest, index, col_volume, col_lat, col_lon, label):
    """Print the attributes of a single ELM column."""
    print(f'    {label} col {int(index):d}')
    print(f'      lat: {col_lat[index]:+.4f}')
    print(f'      lon: {col_lon[index]:+.4f}')
    print(f'      volume: {col_volume[index]:.6e} m^3')
    for attr in COLUMN_ATTRS:
        if attr not in ds_rest:
            continue
        value = ds_rest[attr].values[index]
        print(f'      {attr}: {value}')


def print_top_list_changes(ds_domain, num_top=10):
    """Print columns entering or leaving the top-volume list over time."""
    restarts = find_restart_files()
    if len(restarts) < 2:
        print('fewer than two restart files found; nothing to compare')
        return

    prev_date = None
    prev_top = None
    for date, path in restarts:
        ds_rest = xr.open_dataset(path)
        top, col_volume = get_top_volume_indices(ds_domain, ds_rest, num_top)
        col_lat, col_lon = compute_column_lat_lon(ds_domain, ds_rest)

        if prev_top is not None:
            added = [i for i in top if i not in prev_top]
            removed = [i for i in prev_top if i not in top]
            print(f'{prev_date} -> {date}:')
            if not added and not removed:
                print('    no change in top list')
            for index in added:
                print_column_attrs(
                    ds_rest, index, col_volume, col_lat, col_lon, 'added  '
                )
            for index in removed:
                print_column_attrs(
                    ds_rest, index, col_volume, col_lat, col_lon, 'removed'
                )

        prev_date = date
        prev_top = list(top)
        ds_rest.close()


def main():
    ds_domain = xr.open_dataset(FATMLNDFRC)
    ds_init = xr.open_dataset(FINIDAT)

    col_area = compute_column_area(ds_domain, ds_init)

    # h2osfc is a column-mean mass per unit total column area (mm),
    # so the volume must NOT be scaled by frac_h2osfc
    h2osfc = ds_init['H2OSFC']
    frac_h2osfc = ds_init['FH2OSFC']

    volume = float((h2osfc * 1.0e-3 * col_area).sum())
    sfc_water_area = float((frac_h2osfc * col_area).sum())
    total_area = float(col_area.sum())

    # mean depth over the wetted fraction only
    mean_depth = np.divide(
        volume, sfc_water_area, out=np.zeros(1), where=sfc_water_area > 0
    ).item()

    # per-column diagnostics for wet columns
    h2osfc_vals = h2osfc.values
    frac_vals = frac_h2osfc.values
    area_vals = col_area.values

    is_wet = (frac_vals > 0.0) & (h2osfc_vals > 0.0)
    num_wet = int(np.count_nonzero(is_wet))

    # depth within the wetted portion of each column (m)
    col_depth = np.zeros_like(h2osfc_vals)
    col_depth[is_wet] = 1.0e-3 * h2osfc_vals[is_wet] / frac_vals[is_wet]

    # volume of surface water in each column (m^3)
    col_volume = 1.0e-3 * h2osfc_vals * area_vals

    if num_wet > 0:
        i_max = int(np.argmax(col_volume))
        max_volume = float(col_volume[i_max])
        max_volume_depth = float(col_depth[i_max])
        max_volume_frac = float(frac_vals[i_max])
        max_volume_area = float(frac_vals[i_max] * area_vals[i_max])
    else:
        max_volume = 0.0
        max_volume_depth = 0.0
        max_volume_frac = 0.0
        max_volume_area = 0.0

    num_deep = int(np.count_nonzero(col_depth > 100.0))

    if num_wet > 0:
        frac_p10, frac_p50, frac_p90 = np.percentile(
            frac_vals[is_wet], [10.0, 50.0, 90.0]
        )
    else:
        frac_p10 = frac_p50 = frac_p90 = 0.0

    print(f'total column area:        {total_area:.6e} m^2')
    print(f'surface-water area:       {sfc_water_area:.6e} m^2')
    print(f'surface-water volume:     {volume:.6e} m^3')
    print(f'mean depth where wet:     {mean_depth:.6e} m')
    print(f'number of wet columns:    {num_wet:d}')
    print(f'columns deeper than 100m: {num_deep:d}')
    print(f'10th pct wet fraction:    {frac_p10:.6e}')
    print(f'50th pct wet fraction:    {frac_p50:.6e}')
    print(f'90th pct wet fraction:    {frac_p90:.6e}')
    print(f'max wet-column volume:    {max_volume:.6e} m^3')
    print(f'depth of that column:     {max_volume_depth:.6e} m')
    print(f'wet fraction of that col: {max_volume_frac:.6e}')
    print(f'wet area of that column:  {max_volume_area:.6e} m^2')

    print_top_volumes(ds_domain, ds_init, num_top=10)

    print_top_volume_changes(ds_domain)

    print_top_list_changes(ds_domain)

    ds_domain.close()
    ds_init.close()


if __name__ == '__main__':
    main()