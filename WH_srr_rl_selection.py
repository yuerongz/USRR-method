from SRR_2_reploc_selection import RepresentativeLoc
from SRR_2_reconstruction import SRR2InundationReconstruction
import numpy as np
from nc_func import get_nc_xi_yi
from base_func import get_tf_event_log


def test_sampling_dist(samp_dist, test_nc_file):
    rl_coords = RepresentativeLoc(work_dir).spatial_sampling(dem_tif_file, max_ext_file, sampling_dist=samp_dist,
                                                             save2shp=True, shp_output_file=f'ss_{samp_dist}.shp')
    print('Number of RLs: ', len(rl_coords))
    from nc_func import nc_read_map_as_arr, nc_data_transform
    from gdal_func import gdal_asarray, coords2rc
    t = 80 # 0-150h
    t_idx = int(t*12)   # raw data is in 5min interval
    nc_arr = nc_read_map_as_arr(test_nc_file, t_idx)
    nc_trans = nc_data_transform(test_nc_file)
    dem_arr = gdal_asarray('gis/dem_croped.tif')
    deptharr = nc_arr - dem_arr
    deptharr[np.isnan(deptharr)] = -999
    nc_arr[deptharr<=0.05] = np.nan
    all_wls = np.array([nc_arr[coords2rc(nc_trans, coords)] for coords in rl_coords])
    all_dems = np.array([dem_arr[coords2rc(nc_trans, coords)] for coords in rl_coords])
    all_wls[np.isnan(all_wls)] = all_dems[np.isnan(all_wls)]
    all_wls = list(all_wls)

    tgt_x_coords, tgt_y_coords = get_nc_xi_yi(test_nc_file)
    reco_model = SRR2InundationReconstruction(work_dir, dem_tif_file, f'{work_dir}ss_{samp_dist}.shp',
                                              tgt_x_coords, tgt_y_coords,
                                              tgt_mask_file=max_ext_file,
                                              depth_threshold=0.05)
    inun_arr, comp = reco_model.surface_rebuild(all_wls, compare_report=True, ref_arr=nc_arr)
    return 0


if __name__ == '__main__':
    work_dir = 'srr_reco_test/'
    dem_tif_file = 'gis/dem_croped.tif'
    db_tif_file = 'gis/WH_db_line.tif'
    max_ext_file = f'srr_results/maximum_inundation_extent_WH.tif'

    # # select RLs based on grid-sampling method. sampling_dist in metres.
    # rl_coords = RepresentativeLoc(work_dir).spatial_sampling(dem_tif_file, max_ext_file, sampling_dist=100,
    #                                                          save2shp=True, shp_output_file='ss_100.shp')

    # test inundation extent reconstruction accuracy based on 2D linear interpolation
    testing_dists = [300, 200, 150, 100, 90, 80]     # in metres
    tuflow_data_dir = '../TUFLOW_WH_yz/results/'
    data_indexer = get_tf_event_log()[1]
    evt = data_indexer.eventname.iloc[0]
    test_nc_file = f'{tuflow_data_dir}{evt}/WHYZ_{evt}.nc'    # TUFLOW template event
    for samp_dist in testing_dists:
        test_sampling_dist(samp_dist, test_nc_file)

