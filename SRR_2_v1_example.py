"""
This is only an example for applying the SRR-2 scripts to conduct SRR-1 tasks.
The improvements in the SRR-2 scripts include:
    1. The usage of GPU in reconstruction process.
       The speed-up rate for the reconstruction module is about 4 times compared to the SRR-1 scripts.
    2. The development of the components to be used in the USRR method,
       including spatial sampling and single-step surface reconstruction.
Author: Yuerong Zhou
Created on 7th June 2022
Last update on 7th June 2022
"""

from SRR_2_reploc_selection import RepresentativeLoc
from SRR_2_base_functions import *
from SRR_2_reconstruction import SRRInundationReconstruction

if __name__ == '__main__':
    work_dir = 'srr_results/'
    dem_tif_file = 'gis/dem_croped.tif.tif'
    iwl_tif_file = 'gis/WH_iwl_yz.tif'
    db_tif_file = 'gis/WH_db_line.tif'

    #==================================================================================================================
    #### FIND RLs
    ## 1. find mainsreams RLs
    rl_selection_module = RepresentativeLoc(work_dir)
    #  1.1 find main river RLs---------------------------------------------------------------------

    # #      option 1: using initial water condition as main river area, and do grid-based sampling
    # sampling_distance_in_m = 3000
    # RepresentativeLoc(work_dir).spatial_sampling(dem_tif_file, iwl_tif_file, sampling_dist=sampling_distance_in_m,
    #                                              save2shp=True, shp_output_file=f'rls_mcl_{sampling_distance_in_m}.shp')

    #      option 2: (SRR-1) using mainstream centroid line and do distance-based sampling
    inflow_points_file = 'gis/srr_mcl_starts.shp'
    stating_coords = coords_centering([read_shp_point(inflow_points_file)[0]], dem_tif_file)
    sampling_distance_in_m = 3000
    rls_mcl_dir = f'rls_mcl_{sampling_distance_in_m}.shp'
    rls_mss = rl_selection_module.mcl_sampling(stating_coords[0], dem_tif_file, db_tif_file,
                                               sampling_dist=sampling_distance_in_m,
                                               # mcl_shp_file=f'{work_dir}mcl.shp' # use after 1st run to save time
                                               save2shp=True, shp_output_file=rls_mcl_dir)

    # #  1.2 find concurrent inflow RLs (when there are inflow locations on the floodplain)
    # rls_cf_dir = 'cf_rls.shp'
    # cf_loc_coords = coords_centering(read_shp_point(inflow_points_file)[1:], dem_tif_file)
    # rl_selection_module.concurrent_inflow_rls(cf_loc_coords, dem_tif_file, dem_masked=False,
    #                                           # cf_traj_file=f'{work_dir}cf_trajs.shp', # use this after 1st run
    #                                           iwl_tif_file=iwl_tif_file,
    #                                           lower_boundary_tif_file=db_tif_file,
    #                                           save2shp=True, shp_output_file=rls_cf_dir)
    # rls_mss_dir = 'rls_mss.shp'
    # rl_selection_module.combine_pts_from_multi_shps([f'{work_dir}{rls_mcl_dir}', f'{work_dir}{rls_cf_dir}'],
    #                                                 save2shp=True, shp_output_file=rls_mss_dir)
    # rls_mss = read_shp_point(f'{work_dir}{rls_mss_dir}')


    ## 2. find side RLs---------------------------------------------------------------------
    #     2.1 get maximum inundation extent from available events
    save_max_ext_file = f'{work_dir}maximum_inundation_extent_WH.tif'
    # #     Example: nc format results
    # from base_func import get_tf_event_log
    # nc_dir = '../TUFLOW_WH_yz/results/'
    # event_log_file = 'tuflow_files_prep/tuflow_events_log.csv'
    # event_log_dev, event_log_vali = get_tf_event_log()
    # ncfile_ls = [f'{nc_dir}{event_log_dev.eventname[i]}/WHYZ_{event_log_dev.eventname[i]}.nc'
    #              for i in event_log_dev.index]
    # max_inun_ext_arr, trans = nc_get_max_of_multi_events(ncfile_ls)
    # gdal_writetiff(max_inun_ext_arr, save_max_ext_file, target_transform=trans)

    #     2.2 search for trajs & find RLs-side
    masked_dem_from_cf_search = f'{work_dir}dem_for_rlside_traj_search.tif'
    traj_file = f'{work_dir}trajs.shp'  # this file will be created after a successful run of side-rls
    rep_traj_file = f'{work_dir}rep_trajs.shp'  # this file will be created after a successful run of side-rls
    rl_selection_module.side_rls(masked_dem_from_cf_search, rls_mss, dem_masked=True,
                                 max_ext_tif=save_max_ext_file,
                                 selection_ratio=2,
                                 rep_trajs_selection_ratio=1/40,
                                 # traj_file=traj_file,   # use this from the second time runs to speedup
                                 # traj_file=rep_traj_file,   # use this from the second time runs to speedup
                                 save2shp=True, shp_output_file='rls_side.shp')
    #      Note: To increase rl numbers, increase selection_ratio parameter in the above function.

    #     2.3 read selected RLs from point-shapefiles
    rls_side = f'{work_dir}rls_side.shp'
    rls_mss = read_shp_point(f'{work_dir}{rls_mcl_dir}')    # point to the mainstreams point file
    rls_side_coords = read_shp_point(rls_side)
    rls_all_coords = rls_side_coords + rls_mss

    #==================================================================================================================
    #### Reconstruct water surface
    #    Note: the following example uses netcdf4-format results produced by TUFLOW model
    ## 1. retrieve NC file templates to build reco-model
    from base_func import get_tf_event_log
    import time
    data_indexer = get_tf_event_log()[1]
    evt = data_indexer.eventname.iloc[0]
    example_nc_file = f'../TUFLOW_WH_yz/results/{evt}/WHYZ_{evt}.nc'

    tgt_x_coords, tgt_y_coords = get_nc_xi_yi(f'../TUFLOW_WH_yz/results/{evt}/WHYZ_{evt}.nc')
    start_time = time.time()
    reco_model = SRRInundationReconstruction(work_dir, dem_tif_file, rls_all_coords, rep_traj_file,
                                             tgt_x_coords, tgt_y_coords, boundary_tif_file_1=db_tif_file)
    print('Building time:', time.time() - start_time)


    ## 2. test reco accuracy for several maps, adjust rls if needed
    #     2.1 extract water level from existing data
    t = 95  # hours since start of event
    rls_wls, db_wls = extract_nc_rldb_wls(example_nc_file, t, rls_all_coords)
                                          # save_map_to=f'{work_dir}rebuild_ref_{t}.tif')

    #     2.2 pass water levels to the reco-model
    start_time = time.time()
    inun_arr = reco_model.two_steps_surface_build_v1(rls_wls, db_wls,
                                                     save2tif=True, out_tif_file=f'rebuild_{t}.tif')
    print('Reconstruction time:', time.time() - start_time)

    #     2.3 compare with TUFLOW results
    t_idx = int(t*12)   # timesteps in NC file is 5min
    nc_arr = nc_read_map_as_arr(example_nc_file, t_idx)
    nc_trans = nc_data_transform(example_nc_file)
    comp_map = inun_compare(inun_arr, nc_arr)
    inun_compare_report(comp_map)

