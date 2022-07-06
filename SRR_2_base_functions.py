import numpy as np

from gdal_func import *
from nc_func import *
import fiona
import os
import scipy.spatial.distance as dist
import scipy.ndimage as ndimage


def read_shp_to_trajs(shpfile):
    trajs = dict()
    with fiona.open(shpfile) as copy_shp:
        for feature in copy_shp:
            geom = feature['geometry']['coordinates']
            trajs[geom[0]] = geom[1:]
    print(f'The total number trajectories loaded: {len(trajs)}')
    return trajs


def save_pts_to_shp(pts, outfile):
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outfile):
        shpDriver.DeleteDataSource(outfile)
    outDataSource = shpDriver.CreateDataSource(outfile)
    outLayer = outDataSource.CreateLayer(outfile, geom_type=ogr.wkbPoint)
    featureDefn = outLayer.GetLayerDefn()
    for coords in pts:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(coords[0], coords[1])
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(point)
        outLayer.CreateFeature(outFeature)
        del point, outFeature


def save_trajs_to_shp_lines(trajs, outfile):
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outfile):
        shpDriver.DeleteDataSource(outfile)
    outDataSource = shpDriver.CreateDataSource(outfile)
    outLayer = outDataSource.CreateLayer(outfile, geom_type=ogr.wkbMultiLineString)
    featureDefn = outLayer.GetLayerDefn()
    for key in trajs.keys():
        multiline = ogr.Geometry(ogr.wkbMultiLineString)
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(key[0], key[1])
        for pts in trajs[key]:
            line.AddPoint(pts[0], pts[1])
        multiline.AddGeometry(line)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(multiline)
        outLayer.CreateFeature(outFeature)
        del multiline, line, outFeature


def mark_inner_boundary_cells(arr, bound_mode=4, threshold=1):
    """mark up inner boundary cells in given arr"""
    arr_01 = arr.copy()
    arr_01[~np.isnan(arr_01)] = 0   # non-nan=0
    arr_01[np.isnan(arr_01)] = 1    # nan=1
    if bound_mode == 4:
        checking_arrs = [0, arr_01[0:-2, 1:-1], 2,
                         arr_01[1:-1, 0:-2], arr_01[1:-1, 1:-1], arr_01[1:-1, 2:],
                         6, arr_01[2:,   1:-1], 8]
        count_arr = checking_arrs[1] + checking_arrs[3] + checking_arrs[5] + checking_arrs[7]
    elif bound_mode == 8:
        checking_arrs = [arr_01[0:-2, 0:-2], arr_01[0:-2, 1:-1], arr_01[0:-2, 2:],
                         arr_01[1:-1, 0:-2], arr_01[1:-1, 1:-1], arr_01[1:-1, 2:],
                         arr_01[2:,   0:-2], arr_01[2:,   1:-1], arr_01[2:,   2:]]
        count_arr = checking_arrs[0] + checking_arrs[1] + checking_arrs[2] + checking_arrs[3] + checking_arrs[5] + \
                    checking_arrs[6] + checking_arrs[7] + checking_arrs[8]
    else:
        raise ValueError('Function input mark_outer_boundary_cells.bound_mode must be 4 or 8.')
    count_arr[count_arr < threshold] = 0
    count_arr[count_arr >= threshold] = 1    # eq/over threshold number of nan cell around
    count_arr[checking_arrs[4] == 1] = 0    # mask cells of nan in input
    target_arr = count_arr
    # add row & column at beginning & end
    target_arr = np.r_[[np.zeros(target_arr.shape[1])], target_arr, [np.zeros(target_arr.shape[1])]]
    target_arr = np.c_[np.zeros(target_arr.shape[0]), target_arr, np.zeros(target_arr.shape[0])]
    return target_arr   # 1 indicate inner boundary cells, 0 indicate others


def mark_outer_boundary_cells(arr, bound_mode=4, threshold=1):
    """mark up outer boundary cells in given arr"""
    arr_01 = arr.copy()
    arr_01[~np.isnan(arr_01)] = 1   # non-nan=1
    arr_01[np.isnan(arr_01)] = 0    # nan=0
    if bound_mode == 4:
        checking_arrs = [0, arr_01[0:-2, 1:-1], 2,
                         arr_01[1:-1, 0:-2], arr_01[1:-1, 1:-1], arr_01[1:-1, 2:],
                         6, arr_01[2:,   1:-1], 8]
        count_arr = checking_arrs[1] + checking_arrs[3] + checking_arrs[5] + checking_arrs[7]
    elif bound_mode == 8:
        checking_arrs = [arr_01[0:-2, 0:-2], arr_01[0:-2, 1:-1], arr_01[0:-2, 2:],
                         arr_01[1:-1, 0:-2], arr_01[1:-1, 1:-1], arr_01[1:-1, 2:],
                         arr_01[2:,   0:-2], arr_01[2:,   1:-1], arr_01[2:,   2:]]
        count_arr = checking_arrs[0] + checking_arrs[1] + checking_arrs[2] + checking_arrs[3] + checking_arrs[5] + \
                    checking_arrs[6] + checking_arrs[7] + checking_arrs[8]
    else:
        raise ValueError('Function input mark_outer_boundary_cells.bound_mode must be 4 or 8.')
    count_arr[count_arr < threshold] = 0
    count_arr[count_arr >= threshold] = 1    # non-nan around
    count_arr[checking_arrs[4] == 1] = 0    # mask cells of non-nan in input
    target_arr = count_arr
    # add row & column at beginning & end
    target_arr = np.r_[[np.zeros(target_arr.shape[1])], target_arr, [np.zeros(target_arr.shape[1])]]
    target_arr = np.c_[np.zeros(target_arr.shape[0]), target_arr, np.zeros(target_arr.shape[0])]
    return target_arr   # 1 indicate outer boundary cells, 0 indicate others


def directory_checking(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Directory ", target_dir, " Created.")
    else:
        print("Directory ", target_dir, " already exists.")


def file_checking(tif_file):
    with open(tif_file) as f:
        print(f'File {tif_file} is checked...')
    return tif_file


def coords_centering(starting_coords, dem_file):
    """ Adjust points coordinates to cell center of the dem_file. """
    dem_transform = gdal_transform(dem_file)

    for pt_idx in range(len(starting_coords)):
        starting_coords[pt_idx] = \
            rc2coords(dem_transform, coords2rc(dem_transform, starting_coords[pt_idx]))
    return starting_coords


def starting_points_sampling_from_max_extent(maxi_extent_tif_file, threshold=4):
    assert maxi_extent_tif_file[-3:] == 'tif', 'Please provide tif file for starting points sampling!'
    arr = gdal_asarray(maxi_extent_tif_file)
    arr_trans = gdal_transform(maxi_extent_tif_file)
    bounds_mark = mark_inner_boundary_cells(arr, bound_mode=8, threshold=threshold)
    rcs_bounds = np.array(np.where(bounds_mark == 1)).T
    xy_coords = [rc2coords(arr_trans, rc) for rc in rcs_bounds]
    return xy_coords


def pts_selection_DUPLEX(pts_ls, target_num, existing_pts_ls=None):
    no_of_chosen_pts = target_num
    potential_pts = np.array(pts_ls)
    if existing_pts_ls is None:
        starting_pts = np.array([])
        all_pts = potential_pts
        dist_m = dist.cdist(potential_pts, potential_pts, 'sqeuclidean')
        r1, r2 = np.argwhere(dist_m == dist_m.max())[0]
        dist_m[dist_m == 0] = np.nan
        chosen_pts_idxs = -np.ones(no_of_chosen_pts, dtype=int)
        chosen_pts_idxs[[0, 1]] = [r1, r2]
        remaining_idxs = np.array(range(potential_pts.shape[0]))
        remaining_idxs[[r1, r2]] = -1
        for i in range(no_of_chosen_pts - 2):
            r_of_remain = np.nanargmax(np.nanmin(
                dist_m[:, chosen_pts_idxs[:i + 2]][remaining_idxs[remaining_idxs != -1]], axis=1))
            target_idx = remaining_idxs[remaining_idxs != -1][r_of_remain]
            chosen_pts_idxs[i + 2] = target_idx
            remaining_idxs[target_idx] = -1
    else:
        starting_pts = np.array(existing_pts_ls)
        all_pts = np.concatenate((starting_pts, potential_pts), axis=0)
        dist_m = dist.cdist(all_pts, all_pts, 'sqeuclidean')
        dist_m[dist_m == 0] = np.nan
        chosen_pts_idxs = -np.ones(no_of_chosen_pts, dtype=int)
        chosen_pts_idxs = np.concatenate((np.array(range(starting_pts.shape[0])), chosen_pts_idxs), axis=0)
        remaining_idxs = np.array(range(all_pts.shape[0]))
        remaining_idxs[np.array(range(starting_pts.shape[0]))] = -1
        for i in range(no_of_chosen_pts):
            r_of_remain = np.nanargmax(np.nanmin(
                dist_m[:, chosen_pts_idxs[:i + starting_pts.shape[0]]][remaining_idxs[remaining_idxs != -1]],
                axis=1))
            target_idx = remaining_idxs[remaining_idxs != -1][r_of_remain]
            chosen_pts_idxs[i + starting_pts.shape[0]] = target_idx
            remaining_idxs[target_idx] = -1
    chosen_pts = all_pts[chosen_pts_idxs[starting_pts.shape[0]:], :]
    chosen_pts = [tuple(item) for item in chosen_pts]
    return chosen_pts


def inun_compare(inun_arr, ref_inun_arr):
    arr = inun_arr - ref_inun_arr
    arr[np.isnan(ref_inun_arr) & ~np.isnan(inun_arr)] = -555    # false alarm
    arr[~np.isnan(ref_inun_arr) & np.isnan(inun_arr)] = -666    # missing
    return arr


def inun_compare_report(inun_arr_compare):
    fa = np.sum(inun_arr_compare == -555)
    ms = np.sum(inun_arr_compare == -666)
    detc_inun = np.sum(~np.isnan(inun_arr_compare)) - fa - ms
    print('POD:', detc_inun/(detc_inun + ms), 'RFA:', fa/(detc_inun + fa), 'overall:', detc_inun/(detc_inun + ms + fa))
    return 0


def find_largest_cluster_in_arr(arr):
    binary_arr = arr.copy()
    binary_arr[~np.isnan(arr)] = 1
    binary_arr[np.isnan(arr)] = 0
    label_arr, n_features = ndimage.label(binary_arr, structure=np.array([[1,1,1],[1,1,1],[1,1,1]]))
    areas = ndimage.sum_labels(binary_arr, label_arr, index=range(n_features + 1))
    label_max_area = np.where(areas == np.max(areas))[0][0]
    binary_arr[label_arr != label_max_area] = 0
    filtered_arr = arr.copy()
    filtered_arr[binary_arr == 0] = np.nan
    return filtered_arr


def mask_largest_cluster_in_arr(arr):
    binary_arr = arr.copy()
    binary_arr[~np.isnan(arr)] = 1
    binary_arr[np.isnan(arr)] = 0
    label_arr, n_features = ndimage.label(binary_arr, structure=np.array([[1,1,1],[1,1,1],[1,1,1]]))
    areas = ndimage.sum_labels(binary_arr, label_arr, index=range(n_features + 1))
    label_max_area = np.where(areas == np.max(areas))[0][0]
    binary_arr[label_arr == label_max_area] = 0
    filtered_arr = arr.copy()
    filtered_arr[binary_arr == 0] = np.nan
    return filtered_arr




