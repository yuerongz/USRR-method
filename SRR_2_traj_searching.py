from SRR_2_base_functions import *


class TrajSearch:
    def __init__(self, result_work_dir, starting_pts, demfile,
                 no_bound_cells=4, stopping_categories=(-555, -666), search_win_size=9,
                 cluster_threshold=1, cluster_min_size=2):
        self.work_dir = result_work_dir
        directory_checking(self.work_dir)
        self.starting_pts = np.array(starting_pts)
        self.dem_dataset = gdal.Open(demfile)
        self.dem_transform = self.dem_dataset.GetGeoTransform()
        self.demarr = self.dem_dataset.GetRasterBand(1).ReadAsArray()
        self.countline = [0, 0, 0]     # ending with [stopping_Vals, nan, other-traj]
        self.bound_mode = no_bound_cells
        self.stopping_vals = stopping_categories
        self.search_win_size = search_win_size
        self.straighten_degree = cluster_threshold
        self.cluster_max_sz = cluster_min_size
        self.unique_searching_mark = 555
        self.traj_map = np.zeros(self.demarr.shape).astype(int)  # record searched cells, fill with traj idx stating at 1.

        self.stpt_standardisation()
        self.unique_searching_mark_check()
        self.stopping_vals_presence_check()
        assert self.bound_mode in [4, 8], 'Input no_bound_cells must be 4 or 8, indicating number of boundary cells.'

    def stpt_standardisation(self):
        # standardise starting points coordinates
        for pt_idx in range(len(self.starting_pts)):
            self.starting_pts[pt_idx, :] = \
                rc2coords(self.dem_transform, coords2rc(self.dem_transform, self.starting_pts[pt_idx, :]))

    def unique_searching_mark_check(self):
        """ Check default unique searching mark with dem values and stopping values. """
        if np.nanmax(self.demarr) >= self.unique_searching_mark:
            self.unique_searching_mark = int(np.nanmax(self.demarr) + 5)
        while self.unique_searching_mark in self.stopping_vals:
            self.unique_searching_mark += 5

    def stopping_vals_presence_check(self):
        existence_count = 0
        for stop_val in self.stopping_vals:
            if np.any(self.demarr == stop_val):
                existence_count += 1
        assert existence_count != 0, f'The stopping_categories {self.stopping_vals} do not present in provided demfile!'
        if existence_count < len(self.stopping_vals):
            print('Note: there are one or more stopping_categories not exist in demfile.')

    def cut_arr_with_bounds(self, arr, bounds):
        out_arr = arr
        return out_arr[bounds[0]:bounds[1], bounds[2]:bounds[3]]

    def block_corner_mask(self, arr):
        assert arr.shape == (3, 3), 'block_corner_mask is designed for 3x3 array block only!'
        if self.bound_mode == 4:
            out_arr = arr.copy()
            out_arr[0::2, 0::2] = np.nan
            return out_arr
        else:
            return arr

    def retrieve_block_boundary_rcs(self, r, c, block_sz, arrshape):
        """ Retrieve squared area rc boudnary of surrounding cells; block_sz must be odd integer. """
        step_size = block_sz // 2
        max_row = arrshape[0]
        max_col = arrshape[1]
        up, dwn = max(0, r - step_size), min(max_row, r + step_size + 1)
        lf, rit = max(0, c - step_size), min(max_col, c + step_size + 1)
        return up, dwn, lf, rit

    def find_traj_extents_rcs(self, traj_ls, buffer_sz):
        coords_arr = np.array(traj_ls)
        up, lf = coords2rc(self.dem_transform, (min(coords_arr[:, 0]), max(coords_arr[:, 1])))  # min(x), max(y)
        dwn, rit = coords2rc(self.dem_transform, (max(coords_arr[:, 0]), min(coords_arr[:, 1])))    # max(x), min(y)
        up, lf = max(0, up - buffer_sz), max(0, lf - buffer_sz)
        max_row = self.demarr.shape[0]
        max_col = self.demarr.shape[1]
        dwn, rit = min(max_row, dwn + buffer_sz + 1), min(max_col, rit + buffer_sz + 1)
        return up, dwn, lf, rit     # traj extents rows&cols with strip buffer (extra 2xbuffer_sz rows & cols around)

    def non_stopping_criteria(self, dem_arr, neib_bounds):
        blockarr = self.cut_arr_with_bounds(dem_arr, neib_bounds)
        blockarr[blockarr == self.unique_searching_mark] = np.inf     # mask current point
        # not meet with other trajs
        traj_map_block = self.cut_arr_with_bounds(self.traj_map, neib_bounds)
        criterion_1 = np.sum(traj_map_block) == 0
        criterion_2 = True  # not stopping
        for stop_val in self.stopping_vals:
            criterion_2 = criterion_2 & (~np.any(blockarr == stop_val))    # not reaching the grids with stop signs
        criterion_3 = ~np.any(np.isnan(blockarr))    # the block is hitting the boundary of model domain; Sea-side includes
        return bool(criterion_1 & criterion_2 & criterion_3)

    def single_line_search(self, st_coords):
        dem_arr = self.demarr.copy()
        line_ls = [tuple(st_coords)]
        r, c = coords2rc(self.dem_transform, st_coords)     # current row, col
        line_elevs = [self.demarr[r, c]]
        dem_arr[r, c] = self.unique_searching_mark      # mark current point
        neib_bounds = self.retrieve_block_boundary_rcs(r, c, 3, dem_arr.shape)     # 3x3 neighbourhoods
        while self.non_stopping_criteria(dem_arr, neib_bounds):
            ri, ci = self.search_next_drop(dem_arr, neib_bounds, line_ls, line_elevs)
            dem_arr[dem_arr == self.unique_searching_mark] = np.inf      # mark searched point
            # update current row, column no.
            r, c = r + ri, c + ci
            line_ls.append(rc2coords(self.dem_transform, (r, c)))
            line_elevs.append(self.demarr[r, c])
            neib_bounds = self.retrieve_block_boundary_rcs(r, c, 3, dem_arr.shape)
            dem_arr[r, c] = self.unique_searching_mark      # mark current point
        # add last point & return
        final_block = self.cut_arr_with_bounds(dem_arr, neib_bounds)
        for stop_val in self.stopping_vals:
            if np.any(final_block == stop_val):  # add the iwl cell at last if exists
                id_iwl = np.argwhere(final_block == stop_val)[0]
                line_ls.append(rc2coords(self.dem_transform, (id_iwl[0] + neib_bounds[0], id_iwl[1] + neib_bounds[2])))
                self.countline[0] += 1      # add stopping_val-end count
                return self.post_process_traj(line_ls)
        final_traj_block = self.cut_arr_with_bounds(self.traj_map, neib_bounds)
        if np.sum(final_traj_block) > 0:
            # append the point belongs to another traj to the current traj
            id_b_traj = np.argwhere(final_traj_block > 0)[0]
            line_ls.append(rc2coords(self.dem_transform, (id_b_traj[0] + neib_bounds[0],
                                                          id_b_traj[1] + neib_bounds[2])))
            self.countline[2] += 1
            return self.post_process_traj(line_ls)
        if np.any(np.isnan(final_block)):
            id_bd = np.argwhere(np.isnan(final_block))[0]
            line_ls.append(rc2coords(self.dem_transform, (id_bd[0] + neib_bounds[0], id_bd[1] + neib_bounds[2])))
            self.countline[1] += 1      # add nan-end count
            return self.post_process_traj(line_ls)
        else:
            raise ValueError('TrajSearch.single_line_search ends with error')

    def single_line_search_mark_map(self, dem_arr):
        """Mark traj cells as 1, others as nans."""
        curr_traj_mark = (dem_arr == self.unique_searching_mark) | np.isinf(dem_arr)
        curr_traj_mark = curr_traj_mark.astype(float)
        curr_traj_mark[curr_traj_mark == 0] = np.nan
        return curr_traj_mark

    def traj_outbound_dem_min(self, dem_arr):
        bounds_mask = mark_outer_boundary_cells(self.single_line_search_mark_map(dem_arr))
        bounds_dem = dem_arr.copy()
        if np.any(np.isnan(bounds_dem[bounds_mask == 1])):
            bounds_dem[bounds_mask != 1] = np.inf
            return np.nan, bounds_dem
        else:
            bounds_dem[bounds_mask != 1] = np.nan
            bound_min = np.nanmin(bounds_dem)
            return bound_min, bounds_dem

    def rc_surround_3by3_dem_min(self, dem_arr, r, c):
        # search from current r, c for outer boundary dem min
        out_block_ext = self.retrieve_block_boundary_rcs(r, c, 3, dem_arr.shape)
        out_block = self.cut_arr_with_bounds(dem_arr, out_block_ext)
        out_block = self.block_corner_mask(out_block)
        out_min = np.nanmin(out_block)
        return out_min

    def search_next_drop_bound_min_check(self, bound_min):
        criterion_1 = bound_min in self.stopping_vals
        criterion_2 = np.isnan(bound_min)
        return (criterion_1 | criterion_2)

    def search_next_drop_bound_min_stop(self, bound_min, bounds_dem, traj_bounds):
        if np.isnan(bound_min):
            target_cell_idx = np.argwhere(np.isnan(bounds_dem))[0]
            r, c = target_cell_idx[0] + traj_bounds[0], target_cell_idx[1] + traj_bounds[2]
            bounds = self.retrieve_block_boundary_rcs(r, c, 3, self.demarr.shape)
            orig_dem_block = self.cut_arr_with_bounds(self.demarr, bounds)
            min_cell_idxi = np.argwhere(orig_dem_block == np.nanmin(orig_dem_block))[0]
            r, c = r + min_cell_idxi[0] - 1, c + min_cell_idxi[1] - 1
            return r, c
        if bound_min in self.stopping_vals:
            target_cell_idx = np.argwhere(bounds_dem == bound_min)[0]
            r, c = target_cell_idx[0] + traj_bounds[0], target_cell_idx[1] + traj_bounds[2]
            bounds = self.retrieve_block_boundary_rcs(r, c, 3, self.demarr.shape)
            orig_dem_block = self.cut_arr_with_bounds(self.demarr, bounds)
            orig_dem_block[orig_dem_block == bound_min] = np.nan
            min_cell_idxi = np.argwhere(orig_dem_block == np.nanmin(orig_dem_block))[0]
            r, c = r + min_cell_idxi[0] - 1, c + min_cell_idxi[1] - 1
            return r, c

    def search_next_drop(self, dem_arr, neib_bounds, traj_ls, traj_elevs):
        curr_neibs = self.cut_arr_with_bounds(dem_arr, neib_bounds)
        curr_r = neib_bounds[0] + 1
        curr_c = neib_bounds[2] + 1
        if traj_elevs[-1] > np.nanmin(curr_neibs):  # elev descending from current cell
            direction_idxs = np.argwhere(curr_neibs == np.nanmin(curr_neibs))[0]     # take 1st one, ignore anabranches
            ri, ci = direction_idxs[0] - 1, direction_idxs[1] - 1
            return ri, ci
        else:   # cannot descend, use the new inundating strategy  ############(introduce in SRR-2)#############
            bound_min = traj_elevs[-1]
            out_min = np.nanmin(curr_neibs)
            buffer_sz_cumu = 2
            while bound_min <= out_min:     # unable to descend
                if buffer_sz_cumu == 500:
                    print(f'Warning: Searching buffer exceeding {buffer_sz_cumu} cells.')
                # cut out dem surrounding the current traj
                traj_bounds = self.find_traj_extents_rcs(traj_ls, buffer_sz_cumu)
                curr_dem = self.cut_arr_with_bounds(dem_arr, traj_bounds)
                bound_min, bounds_dem = self.traj_outbound_dem_min(curr_dem)
                if self.search_next_drop_bound_min_check(bound_min):  # hit stopping zone
                    r, c = self.search_next_drop_bound_min_stop(bound_min, bounds_dem, traj_bounds)
                    return r - curr_r, c - curr_c
                min_cell_idxs = np.argwhere(bounds_dem == bound_min)
                # check for each bound_min location whether descending is possible
                for r_ct, c_ct in min_cell_idxs:
                    # inundate current bound_min location
                    dem_arr[r_ct + traj_bounds[0], c_ct + traj_bounds[2]] = self.unique_searching_mark
                    curr_dem[r_ct, c_ct] = self.unique_searching_mark
                    out_min = self.rc_surround_3by3_dem_min(curr_dem, r_ct, c_ct)
                    if bound_min > out_min:     # able to descend from (r_ct, c_ct)
                        ri, ci = r_ct + traj_bounds[0] - curr_r, c_ct + traj_bounds[2] - curr_c
                        return ri, ci
                # no bound_min locations succeeded:
                # extend inundation of WL=bound_min by 1-cell-step-outside
                _, bounds_dem = self.traj_outbound_dem_min(curr_dem)    # new outer boundary
                min_cell_idxs = np.argwhere(bounds_dem == bound_min)
                for r_ct, c_ct in min_cell_idxs:    # will only run if bound_min exists in bounds_dem
                    # inundate current bound_min location
                    dem_arr[r_ct + traj_bounds[0], c_ct + traj_bounds[2]] = self.unique_searching_mark
                    curr_dem[r_ct, c_ct] = self.unique_searching_mark
                    out_min = self.rc_surround_3by3_dem_min(curr_dem, r_ct, c_ct)
                    if bound_min > out_min:  # able to descend from (r_ct, c_ct)
                        ri, ci = r_ct + traj_bounds[0] - curr_r, c_ct + traj_bounds[2] - curr_c
                        return ri, ci
                buffer_sz_cumu += 2

    def post_process_traj(self, line_ls):
        """ Straighten line by cutting out clusters; Also correct strange jumps. """
        rcs_ls = [list(coords2rc(self.dem_transform, coords)) for coords in line_ls]
        pt_id = len(line_ls) - 1    # start from the last pt in line_ls
        seg_jump_min_dist = self.cluster_max_sz + 1
        while pt_id > self.straighten_degree:
            r, c = rcs_ls[pt_id]
            # find dist(pt_ups, curr_pt)<self.straighten_degree
            rcs_p = np.array(rcs_ls[:pt_id - self.straighten_degree]).astype(int)  # upstream points
            block_check = rcs_p[(abs(rcs_p[:, 0] - r) <= self.cluster_max_sz) &
                                (abs(rcs_p[:, 1] - c) <= self.cluster_max_sz), :]
            if block_check.size > 0:
                next_pt_id = rcs_ls.index(list(block_check[0, :]))  # [0] most upstream, [-1] most downstream
                del line_ls[next_pt_id + 1:pt_id]
                pt_id = next_pt_id
            else:
                # check segment jump
                r_p, c_p = rcs_ls[pt_id - 1]
                rcs_p = np.array(rcs_ls[:pt_id]).astype(int)   # all upstream points
                seg_dist = np.sqrt((r_p - r)**2 + (c_p - c)**2)
                if seg_dist > seg_jump_min_dist:
                    dist_all = (rcs_p[:, 0] - r)**2 + (rcs_p[:, 1] - c)**2
                    next_pt_id = np.argwhere(dist_all == np.min(dist_all))[-1][0]  # take the most upstream one if dist==
                    del line_ls[next_pt_id + 1:pt_id]
                    pt_id = next_pt_id
                else:
                    pt_id -= 1
        return line_ls

    def search_all_trajs(self):
        traj_collection = dict()
        traj_id = 0
        for stpt in self.starting_pts:
            traj_id += 1
            curr_traj_ls = self.single_line_search(stpt)
            if self.countline[2] > 0:   # connect to another traj
                r, c = coords2rc(self.dem_transform, curr_traj_ls[-1])
                append_traj_key = self.starting_pts[self.traj_map[r, c] - 1]
                append_traj = traj_collection[tuple(append_traj_key)]
                curr_traj_ls.extend(append_traj[append_traj.index(curr_traj_ls[-1]) + 1:])
                self.countline[2] -= 1
            traj_collection[tuple(stpt)] = curr_traj_ls
            # update traj_map
            for pt_coords in curr_traj_ls:
                r, c = coords2rc(self.dem_transform, pt_coords)
                self.traj_map[r, c] = traj_id
            print(f'Searching finished for trajectory {traj_id}/{len(self.starting_pts)}.')
        return traj_collection

    def save_to_shp_lines(self, trajs, outfile):
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

    def generate_trajs(self, save2shp=False, shp_file=None):
        trajs = self.search_all_trajs()
        if save2shp:
            assert shp_file is not None, 'Please specify trajectories output shapefile dir/name!'
            self.save_to_shp_lines(trajs, f'{self.work_dir}{shp_file}')
        return trajs

