import numpy as np
from netCDF4 import Dataset
from gdal_func import coords2rc, rc2coords, gdal_asarray, gdal_transform, gdal_writetiff


def nc_read(nc_file):
    nc_data = Dataset(nc_file)
    return nc_data


def nc_read_map_as_arr(dataset, map_idx, lyr_var='water_level'):
    if type(dataset) == type('0'):  # if file_name string is provided.
        dataset = nc_read(dataset)
    masked_arr = dataset.variables[lyr_var][map_idx, :, :]
    arr = masked_arr.data
    arr[masked_arr.mask] = masked_arr.fill_value
    return nc_arr2gdalarr(arr)


def nc_read_ts_as_arr(dataset, start_idx, end_idx, coords, lyr_var='water_level'):
    if type(dataset) == type('0'):  # if file_name string is provided.
        dataset = nc_read(dataset)
    r, c = coords2rc(nc_data_transform(dataset), coords)
    nc_r = dataset.variables[lyr_var].shape[0] - r - 1
    masked_arr = dataset.variables[lyr_var][start_idx:end_idx+1, nc_r, c]
    arr = masked_arr.data
    return arr


def nc_read_max_map_as_arr(dataset, lyr_var='water_level'):
    if type(dataset) == type('0'):  # if file_name string is provided.
        dataset = nc_read(dataset)
    masked_arr = dataset.variables[f'maximum_{lyr_var}'][0, :, :]
    arr = masked_arr.data
    arr[masked_arr.mask] = masked_arr.fill_value
    return nc_arr2gdalarr(arr)


def nc_get_max_of_multi_events(nc_file_ls):
    arr = nc_read_max_map_as_arr(nc_file_ls[0])
    trans = nc_data_transform(nc_file_ls[0])
    for nc_file in nc_file_ls[1:]:
        arr_c = nc_read_max_map_as_arr(nc_file)
        arr = np.fmax(arr, arr_c)
    return arr, trans


def nc_arr2gdalarr(arr, no_data=-999):
    gdalarr = arr.copy()
    gdalarr[gdalarr==no_data] = np.nan
    return np.flip(gdalarr, 0)


def nc_data_transform(dataset):
    if type(dataset) == type('0'):  # if file_name string is provided.
        dataset = nc_read(dataset)
    xres = dataset.variables['x'][1].data - dataset.variables['x'][0].data
    xmin = dataset.variables['x'][0].data - xres / 2  # calculate the original extent xmin
    yres = dataset.variables['y'][-1].data - dataset.variables['y'][-2].data
    ymax = dataset.variables['y'][-1].data + yres / 2  # calculate the original extent ymax
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    return geotransform


def get_nc_xi_yi(nc_file):
    dataset = Dataset(nc_file)
    xi = dataset.variables['x'][:].data
    yi = dataset.variables['y'][:].data
    return xi, yi


def extract_nc_rldb_wls(nc_file, t, rls_coords, save_map_to=None):
    db_rl = (433355, 5985765)
    nc_arr = nc_read_map_as_arr(nc_file, int(t*12))
    nc_trans = nc_data_transform(nc_file)
    if save_map_to is not None:
        gdal_writetiff(nc_arr, save_map_to, target_transform=nc_trans)
    db_r, db_c = coords2rc(nc_trans, db_rl)
    rls_rc = [coords2rc(nc_trans, coords) for coords in rls_coords]
    return [nc_arr[r, c] for r, c in rls_rc], [nc_arr[db_r, db_c]]


if __name__ == '__main__':
    print('import desired function!')
