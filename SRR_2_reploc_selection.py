from SRR_2_base_functions import *
from SRR_2_traj_searching import TrajSearch


class RepresentativeLoc:
    def __init__(self, results_dir):
        self.work_dir = results_dir
        self.rep_traj_min_pts_num = 3
        directory_checking(self.work_dir)

    def dem_iwl_db_checking(self, dem_tif_file, dem_masked, stopping_values_in_dem,
                            iwl_tif_file, lower_boundary_tif_file):
        dem_tif_file = file_checking(dem_tif_file)
        dem_arr = gdal_asarray(dem_tif_file)
        if not dem_masked:
            assert (iwl_tif_file is not None) | (lower_boundary_tif_file is not None), \
                'Please provide tif files for marking stopping zones in DEM.'
            dem_masked_file = f'{self.work_dir}masked_dem_for_cf_traj_search.tif'
            if iwl_tif_file is not None:
                iwl_tif_file = file_checking(iwl_tif_file)
                iwl_arr = gdal_asarray(iwl_tif_file)
                dem_arr[~np.isnan(iwl_arr)] = stopping_values_in_dem[0]
            if lower_boundary_tif_file is not None:
                lower_boundary_tif_file = file_checking(lower_boundary_tif_file)
                db_arr = gdal_asarray(lower_boundary_tif_file)
                dem_arr[db_arr == 1] = stopping_values_in_dem[1]
            gdal_writetiff(dem_arr, dem_masked_file, dem_tif_file)
        else:
            dem_masked_file = dem_tif_file
        return dem_arr, dem_masked_file

    def runtime_report_usr_input(self, t):
        print(f'\n          The program will need approximately {t} min.'
              f'\n          The progress of searching will be reported in output.'
              f'\n          Proceed? (y/n)')
        usr_sign = input()
        while usr_sign not in ['y', 'Y', 'n', 'N']:
            print('Please type y to proceed or n to stop:')
            usr_sign = input()
        if usr_sign in ['n', 'N']:
            raise KeyError('Program stopped with user interruption.')
        return 0

    def combine_pts_from_multi_shps(self, shp_file_ls, save2shp=False, shp_output_file=None):
        pt_ls = []
        for shp_file in shp_file_ls:
            pt_ls.extend(read_shp_point(shp_file))
        if save2shp:
            assert shp_output_file is not None, 'Please specify output shapefile name!'
            save_pts_to_shp(pt_ls, f'{self.work_dir}{shp_output_file}')
        return pt_ls

    def line_sampling(self, line_ls, sp_dist, skip_end_pts=True, start_offset=0, end_offset=0):
        if skip_end_pts & (start_offset == 0 | end_offset == 0):
            if start_offset == 0:
                start_offset = sp_dist
            if end_offset == 0:
                end_offset = sp_dist // 10
        length_by_pt = np.zeros(len(line_ls))
        for i in range(len(line_ls) - 1):
            curr_distance = np.sqrt(
                ((line_ls[i + 1][0] - line_ls[i][0]) ** 2) + ((line_ls[i + 1][1] - line_ls[i][1]) ** 2))
            length_by_pt[i + 1] = curr_distance + length_by_pt[i]
        interpolating_dists = np.arange(start_offset, length_by_pt[-1] - end_offset, sp_dist)
        xs = np.interp(interpolating_dists, length_by_pt, [coords[0] for coords in line_ls])
        ys = np.interp(interpolating_dists, length_by_pt, [coords[1] for coords in line_ls])
        sampled_pts = [(xs[i], ys[i]) for i in range(xs.shape[0])]
        return sampled_pts

    def spatial_sampling(self, dem_tif_file, potential_inun_tif_file, sampling_dist,
                         save2shp=False, shp_output_file=None):
        """ Block sampling of representative locations from the IWL map.
            Suitable for cases containing multiple mainstreams and complex IWL conditions. """
        rl_coords_ls = []
        potential_inun_arr = gdal_asarray(potential_inun_tif_file)
        dem_arr = gdal_asarray(dem_tif_file)
        num_cell_sample = int(sampling_dist / gdal_transform(potential_inun_tif_file)[1])
        num_block_0 = potential_inun_arr.shape[0] // num_cell_sample
        num_block_1 = potential_inun_arr.shape[1] // num_cell_sample
        for i_0 in range(num_block_0):
            start_0 = i_0 * num_cell_sample
            for i_1 in range(num_block_1):
                start_1 = i_1 * num_cell_sample
                curr_iwl_block_arr = potential_inun_arr[start_0:start_0 + num_cell_sample, start_1:start_1 + num_cell_sample]
                curr_dem_block_arr = dem_arr[start_0:start_0 + num_cell_sample, start_1:start_1 + num_cell_sample]
                curr_dem_block_arr[np.isnan(curr_iwl_block_arr)] = np.nan  # masked dem block
                if np.all(np.isnan(curr_dem_block_arr)):
                    continue
                else:
                    argmin_dem = np.nanargmin(curr_dem_block_arr)
                    ri, ci = argmin_dem // num_cell_sample, argmin_dem % num_cell_sample
                    rl_coords_ls.append(rc2coords(gdal_transform(potential_inun_tif_file), (start_0 + ri, start_1 + ci)))
        if save2shp:
            assert shp_output_file is not None, 'Please specify output shapefile name!'
            save_pts_to_shp(rl_coords_ls, f'{self.work_dir}{shp_output_file}')
        return rl_coords_ls

    def mcl_sampling(self, mcl_start_coords, dem_tif_file, lower_boundary_tif_file,
                     sampling_dist=3000, stopping_value=-666,
                     mcl_shp_file=None,
                     save2shp=False, shp_output_file=None):
        """
        From given starting coordinates,
        1. search for the mainstream centroid line using the given DEM; or read MCL from given shapefile (line);
        2. sample representative locations from the MCL with given sampling distance.
        :return: a list of coordinates of selected RLs.
        """
        mcl_rls = []
        dem_tif_file = file_checking(dem_tif_file)
        dem_arr = gdal_asarray(dem_tif_file)
        lower_boundary_tif_file = file_checking(lower_boundary_tif_file)
        db_arr = gdal_asarray(lower_boundary_tif_file)
        dem_arr[db_arr == 1] = stopping_value
        dem_masked_file = f'{self.work_dir}masked_dem_for_mcl_search.tif'
        if mcl_shp_file is None:
            mcl_trajs = TrajSearch(self.work_dir, [mcl_start_coords],
                                  dem_masked_file).generate_trajs(save2shp=True, shp_file='mcl.shp')
        else:
            mcl_trajs = read_shp_to_trajs(mcl_shp_file)
        for ky in mcl_trajs.keys():
            traj = mcl_trajs[ky]
            mcl_rls.extend(self.line_sampling(traj, sampling_dist))
        if save2shp:
            assert shp_output_file is not None, 'Please specify output shapefile name!'
            save_pts_to_shp(mcl_rls, f'{self.work_dir}{shp_output_file}')
        return mcl_rls

    def concurrent_inflow_rls(self, cf_loc_coords, dem_tif_file,
                              cf_traj_sampling_dist=3000,
                              stopping_values_in_dem=(-555, -666), dem_masked=True,
                              iwl_tif_file=None, lower_boundary_tif_file=None,
                              cf_traj_file = None,
                              save2shp=False, shp_output_file=None):
        cf_rls = cf_loc_coords.copy()
        dem_trans = gdal_transform(dem_tif_file)
        dem_arr, dem_masked_file = self.dem_iwl_db_checking(dem_tif_file, dem_masked, stopping_values_in_dem,
                                                            iwl_tif_file, lower_boundary_tif_file)
        filter_idx = [i for i in range(len(cf_loc_coords))
                      if dem_arr[coords2rc(dem_trans, cf_loc_coords[i])] not in stopping_values_in_dem]
        cf_traj_search_stpts = [cf_loc_coords[i] for i in filter_idx]
        if cf_traj_file is not None:
            cf_trajs = read_shp_to_trajs(cf_traj_file)
        else:
            print(f'concurrent_inflow_rls: Identified {len(cf_traj_search_stpts)} starting points '
                  f'from provided cf_loc_coords.')
            self.runtime_report_usr_input(len(cf_traj_search_stpts) / 1000 * 22)
            cf_trajs = TrajSearch(self.work_dir, cf_traj_search_stpts,
                                  dem_masked_file).generate_trajs(save2shp=True,
                                                                  shp_file='cf_trajs.shp')
            print(f'concurrent_inflow_rls: Concurrent inflow trajectories are saved in {self.work_dir}cf_trajs.shp. '
                  f'Use this file as cf_traj_file input to reduce re-run time.')
        # check trajectories overlapping
        traj_map = np.zeros(dem_arr.shape)
        for pt_coords in cf_traj_search_stpts:
            pt_check = [pt_coords in traj[1:] for traj in cf_trajs.values()]
            if any(pt_check):
                cf_trajs.pop(pt_coords, None)   # delete traj if pt_coords exists in other trajs
            else:
                for traj_pt in cf_trajs[pt_coords]:
                    if traj_map[coords2rc(dem_trans, traj_pt)] == 1:  # connected to other trajs
                        cf_trajs[pt_coords] = cf_trajs[pt_coords][:cf_trajs[pt_coords].index(traj_pt)]
                        break
                rcs = [coords2rc(dem_trans, cds) for cds in cf_trajs[pt_coords]]
                traj_map[tuple(zip(*rcs))] = 1   # update traj map
        # sampling at equal distances
        for ky in cf_trajs.keys():
            traj = cf_trajs[ky]
            cf_rls.extend(self.line_sampling(traj, cf_traj_sampling_dist))
            for pt in traj:
                dem_arr[coords2rc(dem_trans, pt)] = stopping_values_in_dem[0]
        gdal_writetiff(dem_arr, f'{self.work_dir}dem_for_rlside_traj_search.tif', dem_tif_file)
        print(f"concurrent_inflow_rls: New masked DEM file for RL-side trajectory searching is "
              f"saved as '{self.work_dir}dem_for_rlside_traj_search.tif''.")
        if save2shp:
            assert shp_output_file is not None, 'Please specify output shapefile name!'
            save_pts_to_shp(cf_rls, f'{self.work_dir}{shp_output_file}')
        return cf_rls

    def representative_trajs_selection(self, trajs, rep_trajs_ratio, dem_masked_file,
                                       stopping_values_in_dem=(-555, -666)):
        """ For each trajectories cluster (ending at the same point),
        when:
            1. more than 3 trajs over 20 pts: at least 2 trajs will be selected;
            2. only 1-2 traj over 20 pts: only this longest traj will be selected;
            3. all trajs less than 20 pts: no trajs will be selected.
        *The number of minimum points is controlled by the input min_pts_num. """
        trajs_groupping = dict()    # {end_pt1_coords:[stpt1, stpt2, ...], end_pt2_coords:[stpt1, stpt2, ...]...}
        trajs_rep_stpts = dict()
        dem_arr = gdal_asarray(dem_masked_file)
        dem_trans = gdal_transform(dem_masked_file)
        for stpt in trajs.keys():
            end_pt = trajs[stpt][-1]
            if (dem_arr[coords2rc(dem_trans, end_pt)] == stopping_values_in_dem[0]) & (len(trajs[stpt]) > self.min_pts_num):
                if end_pt in trajs_groupping.keys():
                    trajs_groupping[end_pt].append(stpt)
                else:
                    trajs_groupping[end_pt] = [stpt]
        # filter for each traj group
        rep_trajs = dict()
        for end_pt in trajs_groupping.keys():
            lengths = [len(trajs[stpt]) for stpt in trajs_groupping[end_pt]]
            longest_traj_idx = lengths.index(max(lengths))
            potential_pts = trajs_groupping[end_pt].copy()
            potential_pts.remove(trajs_groupping[end_pt][longest_traj_idx])
            if len(potential_pts) > 2:
                select_pts = pts_selection_DUPLEX(potential_pts, int(np.ceil(rep_trajs_ratio * len(potential_pts))),
                                                  existing_pts_ls=[end_pt, trajs_groupping[end_pt][longest_traj_idx]])
                trajs_rep_stpts[end_pt] = [trajs_groupping[end_pt][longest_traj_idx]] + select_pts
            else:
                trajs_rep_stpts[end_pt] = [trajs_groupping[end_pt][longest_traj_idx]]
            for stpt in trajs_rep_stpts[end_pt]:
                rep_trajs[stpt] = trajs[stpt]
        return rep_trajs

    def side_rls(self, dem_tif_file, rls_mcl_ls,
                 selection_ratio=2,
                 stopping_values_in_dem=(-555, -666), dem_masked=True,
                 iwl_tif_file=None, lower_boundary_tif_file=None,
                 traj_file = None, max_ext_tif=None,
                 rep_trajs_selection_ratio=1/40,
                 save2shp=False, shp_output_file=None):
        dem_arr, dem_masked_file = self.dem_iwl_db_checking(dem_tif_file, dem_masked, stopping_values_in_dem,
                                                            iwl_tif_file, lower_boundary_tif_file)
        assert any([traj_file, max_ext_tif]), 'Please provide a maximum inundation extent tif file or a traj_file!'
        if traj_file is None:
            stpts = starting_points_sampling_from_max_extent(max_ext_tif, threshold=4)
            print(f'side_rls: Identified {len(stpts)} starting points from the maximum inundation extent.')
            self.runtime_report_usr_input(len(stpts) / 1000 * 11)
            trajs = TrajSearch(self.work_dir, stpts, dem_masked_file).generate_trajs(save2shp=True,
                                                                                     shp_file='trajs.shp')
            print(f'side_rls: Trajectories are saved in {self.work_dir}trajs.shp. '
                  f'Use this file as traj_file input to reduce re-run time.')
        else:
            trajs = read_shp_to_trajs(traj_file)
        if bool(rep_trajs_selection_ratio):
            print(f'side_rls: Trajectories are filtered with a selection ratio of {rep_trajs_selection_ratio}...')
            rep_trajs = self.representative_trajs_selection(trajs, rep_trajs_selection_ratio, dem_masked_file)
            save_trajs_to_shp_lines(rep_trajs, f'{self.work_dir}rep_trajs.shp')
            print(f'side_rls: Representative trajectories are selected: {len(rep_trajs)}/{len(trajs)}, and'
                  f'saved in {self.work_dir}rep_trajs.shp.')
            trajs = rep_trajs
        # DUPLEX traj-end-pts selection, 2x mcl-rls
        trajs_ending_pts = list(set([traj_ls[-1] for traj_ls in trajs.values()]))
        side_rls = pts_selection_DUPLEX(trajs_ending_pts, len(rls_mcl_ls) * selection_ratio,
                                        existing_pts_ls=rls_mcl_ls)
        if save2shp:
            assert shp_output_file is not None, 'Please specify output shapefile name!'
            save_pts_to_shp(side_rls, f'{self.work_dir}{shp_output_file}')
        return side_rls


