import scipy.spatial.qhull as qhull
from SRR_2_base_functions import *
from scipy.interpolate import griddata


class SRRInundationReconstruction():
    def __init__(self, work_dir, dem_tif_file, rls_coords, rep_traj_file,
                 tgt_xi_coords, tgt_yi_coords,
                 boundary_tif_file_1=None, boundary_tif_file_2=None,
                 varying_wl_boundary_coords=None, depth_threshold=0,
                 gpu_mode=True):
        self.work_dir = work_dir
        if gpu_mode:
            global torch
            import torch
            self.device = torch.device('cuda')
        else:
            self.device = None
        self.nan_fill_value = -999
        self.depth_thd = depth_threshold
        self.demarr = gdal_asarray(dem_tif_file)
        self.demtrans = gdal_transform(dem_tif_file)
        self.extent_adjust(tgt_xi_coords, tgt_yi_coords)

        self.rls_coords = rls_coords
        self.trajs = read_shp_to_trajs(rep_traj_file)
        self.tgt_shape = (len(tgt_yi_coords), len(tgt_xi_coords))
        grid_x, grid_y = np.meshgrid(tgt_xi_coords, tgt_yi_coords)
        self.tgt_xy = np.array([grid_x.flatten(), grid_y.flatten()]).T
        self.bounds_coords, self.bounds_summary = self.get_boundary_coords(boundary_tif_file_1, boundary_tif_file_2,
                                                                           varying_wl_boundary_coords)
        self.arls_coords = self.additional_rls_in_trajs()
        self.pt_bank = self.traj_pts_by_rls()
        self.edge_pt_bank = self.rls_traj_stpts()

        self.arls_simplex_vars = self.get_arl_simplex_vars()
        self.tgt_simplex_vars = self.get_tgt_simplex_vars()

    def get_arl_simplex_vars(self):
        xy = self.bounds_coords + self.rls_coords
        for rl in self.pt_bank.keys():
            if rl in self.rls_coords:
                xy.extend(self.pt_bank[rl])
        xy = np.array(xy)
        tgt_xy = np.array(self.arls_coords)
        vtx, wts = self.spatial_simplex_vars(xy, tgt_xy)
        if self.device is not None:
            return torch.from_numpy(vtx.astype('int64')).to(self.device), torch.from_numpy(wts).float().to(self.device)
        else:
            return vtx.astype('int64'), wts

    def interp_arl_wls(self, db_wls, rl_wls):
        wls = db_wls + rl_wls
        for rl in self.pt_bank.keys():
            if rl in self.rls_coords:
                wls.extend([rl_wls[self.rls_coords.index(rl)]] * len(self.pt_bank[rl]))
        arl_wls = self.linear_interpolate_2d_with_simplex_vars(np.array(wls), *self.arls_simplex_vars).cpu().numpy()
        return arl_wls

    def interp_arl_wls_cpu(self, db_wls, rl_wls):
        wls = db_wls + rl_wls
        for rl in self.pt_bank.keys():
            if rl in self.rls_coords:
                wls.extend([rl_wls[self.rls_coords.index(rl)]] * len(self.pt_bank[rl]))
        arl_wls = self.linear_interpolate_2d_with_simplex_vars_cpu(np.array(wls), *self.arls_simplex_vars)
        return arl_wls

    def get_tgt_simplex_vars(self):
        coords_all = self.bounds_coords + self.rls_coords + self.arls_coords
        for pts_ls in self.pt_bank.values():
            coords_all.extend(pts_ls)
        xy = np.array(coords_all)
        vtx, wts = self.spatial_simplex_vars(xy, self.tgt_xy)
        if self.device is not None:
            return torch.from_numpy(vtx.astype('int64')).to(self.device), torch.from_numpy(wts).float().to(self.device)
        else:
            return vtx.astype('int64'), wts

    def interp_tgt_wls(self, db_wls, rl_wls, arl_wls):
        wls = db_wls + rl_wls + arl_wls
        for pt in self.pt_bank.keys():
            if pt in self.rls_coords:
                wls.extend([rl_wls[self.rls_coords.index(pt)]] * len(self.pt_bank[pt]))
            else:
                wls.extend([arl_wls[self.arls_coords.index(pt)]] * len(self.pt_bank[pt]))
        tgt_wls = self.linear_interpolate_2d_with_simplex_vars(np.array(wls), *self.tgt_simplex_vars)
        tgt_wls = torch.flip(tgt_wls.view(self.tgt_shape), dims=[0]).cpu().numpy()
        return tgt_wls

    def interp_tgt_wls_cpu(self, db_wls, rl_wls, arl_wls):
        wls = db_wls + rl_wls + arl_wls
        for pt in self.pt_bank.keys():
            if pt in self.rls_coords:
                wls.extend([rl_wls[self.rls_coords.index(pt)]] * len(self.pt_bank[pt]))
            else:
                wls.extend([arl_wls[self.arls_coords.index(pt)]] * len(self.pt_bank[pt]))
        tgt_wls = self.linear_interpolate_2d_with_simplex_vars_cpu(np.array(wls), *self.tgt_simplex_vars)
        tgt_wls = np.flip(tgt_wls.reshape(self.tgt_shape), axis=0)
        return tgt_wls

    def extent_adjust(self, xi, yi):
        min_x = min(xi)
        max_y = max(yi)
        origin_rc = coords2rc(self.demtrans, (min_x, max_y))
        max_x = max(xi)
        min_y = min(yi)
        bt_rit_rc = coords2rc(self.demtrans, (max_x, min_y))
        self.demarr = self.demarr[origin_rc[0]:bt_rit_rc[0]+1, origin_rc[1]:bt_rit_rc[1]+1]
        self.demarr[np.isnan(self.demarr)] = self.nan_fill_value
        new_origin = rc2coords(self.demtrans, origin_rc)
        new_trans = list(self.demtrans)
        new_trans[0] = new_origin[0] - new_trans[1] / 2
        new_trans[3] = new_origin[1] + new_trans[1] / 2
        self.demtrans = tuple(new_trans)
        return 0

    def get_boundary_coords(self, boundary_tif_file_1, boundary_tif_file_2, varying_db_coords):
        db_coords = []
        db_file_summary = dict()
        for db_file in [boundary_tif_file_1, boundary_tif_file_2]:
            if db_file is not None:
                db_file_summary[db_file] = []
        for db_file in db_file_summary.keys():
            db_arr = gdal_asarray(db_file)
            db_trans = gdal_transform(db_file)
            rcs = np.array(np.where(db_arr == 1)).T
            db_curr_coords = [rc2coords(db_trans, rc) for rc in rcs]
            db_coords.extend(db_curr_coords)
            db_file_summary[db_file] = len(db_curr_coords)
        if varying_db_coords is not None:
            db_coords.extend(varying_db_coords)
            db_file_summary['varying_boundary'] = len(varying_db_coords)
        return db_coords, db_file_summary

    def additional_rls_in_trajs(self):
        end_pts_coords = [traj_ls[-1] for traj_ls in self.trajs.values()]
        arls_coords = list(dict.fromkeys(end_pts_coords))
        arls_coords = list(set(arls_coords) - set(self.rls_coords))
        return arls_coords

    def traj_pts_by_rls(self):
        pt_bank = dict()    # keys: rl_coords, values: pt_coords along trajs
        for traj_ls in self.trajs.values():
            if traj_ls[-1] in pt_bank.keys():
                pt_bank[traj_ls[-1]] = list(set(pt_bank[traj_ls[-1]] + traj_ls[:-1]))
            else:
                pt_bank[traj_ls[-1]] = traj_ls[:-1]
        return pt_bank

    def rls_traj_stpts(self):
        rls_ctl_pts = dict()
        for stpt in self.trajs.keys():
            if self.trajs[stpt][-1] in rls_ctl_pts.keys():
                rls_ctl_pts[self.trajs[stpt][-1]].append(stpt)
            else:
                rls_ctl_pts[self.trajs[stpt][-1]] = [stpt]
        return rls_ctl_pts

    def spatial_simplex_vars(self, xy, tgt_xy, d=2):
        tri = qhull.Delaunay(xy)
        simplex = tri.find_simplex(tgt_xy)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = tgt_xy - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def linear_interpolate_2d_with_simplex_vars(self, values, vtx, wts, fill_value=np.nan):
        values = torch.from_numpy(values).to(self.device)
        ret = torch.einsum('nj,nj->n', torch.take(values, vtx), wts)
        # ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
        ret[torch.any(wts < 0, axis=1)] = fill_value
        return ret

    def linear_interpolate_2d_with_simplex_vars_cpu(self, values, vtx, wts, fill_value=np.nan):
        ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
        ret[np.any(wts < 0, axis=1)] = fill_value
        return ret

    def dem_cut(self, interp_arr):
        inundation_map = interp_arr.copy()
        inundation_map[np.isnan(inundation_map)] = self.nan_fill_value
        inundation_map[self.demarr == self.nan_fill_value] = self.nan_fill_value
        inundation_map[inundation_map < (self.demarr + self.depth_thd)] = self.nan_fill_value
        inundation_map[inundation_map == self.nan_fill_value] = np.nan
        return inundation_map

    def boundary_wls_checking(self, db_wls):
        available_db_file_ls = list(self.bounds_summary.keys())
        if db_wls is None:
            assert self.bounds_coords == [], 'Boundary locations are provided, ' \
                                             'please also provide boundary water levels!'
            return db_wls
        if type(db_wls) != type([]):
            db_wls = [db_wls]
        assert (len(db_wls) == len(available_db_file_ls)) | (available_db_file_ls[0] == 'varying_boundary'), \
            f"{len(available_db_file_ls)} boundary types exist, but {len(db_wls)} boundary water level types " \
            f"are provided!"
        count_i = 0
        db_wls_out = []
        for db_file in available_db_file_ls:
            if type(db_wls[count_i]) != type([]):
                db_wls[count_i] = [db_wls[count_i]]
            db_wl_curr = db_wls[count_i]
            if db_file == 'varying_boundary':
                assert len(db_wl_curr) == self.bounds_summary[db_file], \
                    'Number of varying boundary water level values not matching with number of points!'
                db_wls_out.extend(db_wl_curr)
            elif len(db_wl_curr) == 1:
                db_wls_out.extend(db_wl_curr * self.bounds_summary[db_file])
            else:
                assert len(db_wl_curr) == self.bounds_summary[db_file], \
                    f"{db_file} has {self.bounds_summary[db_file]} points, " \
                    f"but {len(db_wl_curr)} water level values provided!"
                db_wls_out.extend(db_wl_curr)
            count_i += 1
        return db_wls_out

    def two_steps_surface_build_v1(self, rls_wls, boundary_wls=None,
                                   save2tif=False, out_tif_file=None):
        assert len(rls_wls) == len(self.rls_coords), 'The number of provided RL water levels not matching ' \
                                                     'with the number of RLs!'
        db_wls = self.boundary_wls_checking(boundary_wls)
        if self.device is not None:
            arl_wls = self.interp_arl_wls(db_wls, rls_wls)
            inun_arr = self.interp_tgt_wls(db_wls, rls_wls, list(arl_wls))
        else:
            arl_wls = self.interp_arl_wls_cpu(db_wls, rls_wls)
            inun_arr = self.interp_tgt_wls_cpu(db_wls, rls_wls, list(arl_wls))
        inun_arr = self.dem_cut(inun_arr)
        if save2tif:
            assert out_tif_file is not None, 'Please specify output geotiff file name!'
            gdal_writetiff(inun_arr, f'{self.work_dir}{out_tif_file}', target_transform=self.demtrans)
        return inun_arr

    def griddata_build(self, coords, wls, boundary_wls):
        db_wls = self.boundary_wls_checking(boundary_wls)
        inun_arr = np.flip(griddata(np.array(coords + self.bounds_coords),
                                    np.array(wls + db_wls), self.tgt_xy).reshape(self.tgt_shape), axis=0)
        inun_arr = self.dem_cut(inun_arr)
        return inun_arr


class SRR2InundationReconstruction():
    def __init__(self, work_dir, dem_tif_file, pts_file,
                 tgt_xi_coords, tgt_yi_coords, tgt_mask_file=None, inun_filter_file=None,
                 boundary_tif_file_1=None, boundary_tif_file_2=None,
                 varying_wl_boundary_coords=None, depth_threshold=0,
                 gpu_mode=True):
        self.work_dir = work_dir
        self.nan_fill_value = -999
        self.depth_thd = depth_threshold
        self.dem_file = dem_tif_file
        self.demarr = gdal_asarray(dem_tif_file)
        self.demtrans = gdal_transform(dem_tif_file)
        self.tgt_shape = (len(tgt_yi_coords), len(tgt_xi_coords))
        if self.demarr.shape != self.tgt_shape:
            self.demarr, self.demtrans = self.extent_adjust(tgt_xi_coords, tgt_yi_coords, self.demarr, self.demtrans)
        self.tgt_mask_arr = self.prep_inun_mask(tgt_xi_coords, tgt_yi_coords, tgt_mask_file)
        self.inun_filter_arr = self.prep_inun_filter(tgt_xi_coords, tgt_yi_coords, inun_filter_file)
        if gpu_mode:
            global torch
            import torch
            self.device = torch.device('cuda')
            self.demarr = torch.from_numpy(self.demarr).to(self.device)
            if self.inun_filter_arr is not None:
                self.inun_filter_arr = torch.from_numpy(self.inun_filter_arr).to(self.device)
            if self.tgt_mask_arr is not None:
                self.rebuild_temp = torch.from_numpy(self.tgt_mask_arr.copy()).to(self.device)
            else:
                self.rebuild_temp = None
        else:
            self.device = None
            if self.tgt_mask_arr is not None:
                self.rebuild_temp = self.tgt_mask_arr.copy()
            else:
                self.rebuild_temp = None

        self.pts_coords = read_shp_point(pts_file)
        if self.tgt_mask_arr is None:
            grid_x, grid_y = np.meshgrid(tgt_xi_coords, tgt_yi_coords)
            self.tgt_xy = np.array([grid_x.flatten(), grid_y.flatten()]).T
        else:
            self.tgt_xy = np.array([rc2coords(self.demtrans, (r, c)) for r, c in np.array(np.where(~np.isnan(self.tgt_mask_arr))).T])
        self.bounds_coords, self.bounds_summary = self.get_boundary_coords(boundary_tif_file_1, boundary_tif_file_2,
                                                                           varying_wl_boundary_coords)
        self.simplices_vars = self.build_simplces()

    def extent_adjust(self, xi, yi, arr, trans):
        min_x = min(xi)
        max_y = max(yi)
        origin_rc = coords2rc(trans, (min_x, max_y))
        max_x = max(xi)
        min_y = min(yi)
        bt_rit_rc = coords2rc(trans, (max_x, min_y))
        arr = arr[origin_rc[0]:bt_rit_rc[0]+1, origin_rc[1]:bt_rit_rc[1]+1]
        arr[np.isnan(arr)] = self.nan_fill_value
        new_origin = (min_x, max_y)
        new_trans = list(trans)
        new_trans[0] = new_origin[0] - new_trans[1] / 2
        new_trans[3] = new_origin[1] + new_trans[1] / 2
        trans = tuple(new_trans)
        return arr, trans

    def prep_inun_mask(self, tgt_xi_coords, tgt_yi_coords, tgt_mask_file):
        if tgt_mask_file is not None:
            tgt_arr = gdal_asarray(tgt_mask_file)
            if tgt_arr.shape != self.tgt_shape:
                tgt_arr, _ = self.extent_adjust(tgt_xi_coords, tgt_yi_coords, tgt_arr, gdal_transform(tgt_mask_file))
            return tgt_arr
        else:
            return None

    def prep_inun_filter(self, tgt_xi_coords, tgt_yi_coords, inun_filter_file, tag='WH'):
        if inun_filter_file is not None:
            tgt_arr = gdal_asarray(inun_filter_file)
            if tgt_arr.shape != self.tgt_shape:
                tgt_arr, _ = self.extent_adjust(tgt_xi_coords, tgt_yi_coords, tgt_arr, gdal_transform(inun_filter_file))
            if tag == 'WH':
                tgt_arr[(tgt_arr != -666) & (tgt_arr != -555)] = np.nan     # only clusters covering inflow locations
            tgt_arr[~np.isnan(tgt_arr)] = 0
            return tgt_arr
        else:
            return None

    def build_simplces(self):
        vtx, wts = self.spatial_simplex_vars(self.bounds_coords + self.pts_coords, self.tgt_xy)
        if self.device is not None:
            return torch.from_numpy(vtx.astype('int64')).to(self.device), torch.from_numpy(wts).float().to(self.device)
        else:
            return vtx.astype('int64'), wts

    def spatial_simplex_vars(self, xy, tgt_xy, d=2):
        tri = qhull.Delaunay(xy)
        simplex = tri.find_simplex(tgt_xy)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = tgt_xy - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def linear_interpolate_2d_with_simplex_vars(self, values, vtx, wts, fill_value=np.nan):
        values = torch.from_numpy(values).to(self.device)
        ret = torch.einsum('nj,nj->n', torch.take(values, vtx), wts)
        ret[torch.any(wts < 0, axis=1)] = fill_value
        return ret

    def linear_interpolate_2d_with_simplex_vars_cpu(self, values, vtx, wts, fill_value=np.nan):
        ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
        ret[np.any(wts < 0, axis=1)] = fill_value
        return ret

    def get_boundary_coords(self, boundary_tif_file_1, boundary_tif_file_2, varying_db_coords):
        db_coords = []
        db_file_summary = dict()
        for db_file in [boundary_tif_file_1, boundary_tif_file_2]:
            if db_file is not None:
                db_file_summary[db_file] = []
        for db_file in db_file_summary.keys():
            db_arr = gdal_asarray(db_file)
            db_trans = gdal_transform(db_file)
            rcs = np.array(np.where(db_arr == 1)).T
            db_curr_coords = [rc2coords(db_trans, rc) for rc in rcs]
            db_coords.extend(db_curr_coords)
            db_file_summary[db_file] = len(db_curr_coords)
        if varying_db_coords is not None:
            db_coords.extend(varying_db_coords)
            db_file_summary['varying_boundary'] = len(varying_db_coords)
        return db_coords, db_file_summary

    def boundary_wls_checking(self, db_wls):
        available_db_file_ls = list(self.bounds_summary.keys())
        if db_wls is None:
            assert self.bounds_coords == [], 'Boundary locations are provided, ' \
                                             'please also provide boundary water levels!'
            return []
        if type(db_wls) != type([]):
            db_wls = [db_wls]
        assert (len(db_wls) == len(available_db_file_ls)) | (available_db_file_ls[0] == 'varying_boundary'), \
            f"{len(available_db_file_ls)} boundary types exist, but {len(db_wls)} boundary water level types " \
            f"are provided!"
        count_i = 0
        db_wls_out = []
        for db_file in available_db_file_ls:
            if type(db_wls[count_i]) != type([]):
                db_wls[count_i] = [db_wls[count_i]]
            db_wl_curr = db_wls[count_i]
            if db_file == 'varying_boundary':
                assert len(db_wl_curr) == self.bounds_summary[db_file], \
                    'Number of varying boundary water level values not matching with number of points!'
                db_wls_out.extend(db_wl_curr)
            elif len(db_wl_curr) == 1:
                db_wls_out.extend(db_wl_curr * self.bounds_summary[db_file])
            else:
                assert len(db_wl_curr) == self.bounds_summary[db_file], \
                    f"{db_file} has {self.bounds_summary[db_file]} points, " \
                    f"but {len(db_wl_curr)} water level values provided!"
                db_wls_out.extend(db_wl_curr)
            count_i += 1
        return db_wls_out

    def interp_tgt_wls(self, db_wls, rl_wls):
        wls = db_wls + rl_wls
        tgt_wls = self.linear_interpolate_2d_with_simplex_vars(np.array(wls), *self.simplices_vars)
        if self.tgt_mask_arr is None:
            tgt_wls = torch.flip(tgt_wls.view(self.tgt_shape), dims=[0]).cpu().numpy()
            return tgt_wls
        else:
            out_map = self.rebuild_temp.detach().clone()
            out_map[~torch.isnan(self.rebuild_temp)] = tgt_wls
            return out_map #.detach().cpu().numpy()

    def interp_tgt_wls_cpu(self, db_wls, rl_wls):
        wls = db_wls + rl_wls
        tgt_wls = self.linear_interpolate_2d_with_simplex_vars_cpu(np.array(wls), *self.simplices_vars)
        if self.tgt_mask_arr is None:
            tgt_wls = np.flip(tgt_wls.reshape(self.tgt_shape), axis=0)
            return tgt_wls
        else:
            out_map = self.rebuild_temp.copy()
            out_map[~np.isnan(self.rebuild_temp)] = tgt_wls
            return out_map

    def dem_cut(self, interp_arr):
        inundation_map = interp_arr
        if self.device is None:
            inundation_map[np.isnan(inundation_map)] = self.nan_fill_value
        else:
            inundation_map[torch.isnan(inundation_map)] = self.nan_fill_value
        inundation_map[self.demarr == self.nan_fill_value] = self.nan_fill_value
        inundation_map[inundation_map < (self.demarr + self.depth_thd)] = self.nan_fill_value
        inundation_map[inundation_map == self.nan_fill_value] = float('nan')
        return inundation_map


    def filter_cluster_in_arr(self, arr):
        """ Filter out clusters not overlapping with (mainstream + CF thalwegs) area."""
        if self.device is None:
            binary_arr = arr.copy()
            binary_arr[~np.isnan(arr)] = 1
            binary_arr[np.isnan(arr)] = 0
            label_arr, _ = ndimage.label(binary_arr, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
            label_checks = label_arr + self.inun_filter_arr
            label_checks = np.unique(label_checks[~np.isnan(label_checks)])
            if label_checks[0] == 0:
                label_checks = label_checks[1:]
            arr[~np.isin(label_arr, label_checks)] = np.nan
            return arr
        else:
            binary_arr = (~(torch.isnan(arr))).type(torch.ByteTensor)
            label_arr, _ = ndimage.label(binary_arr, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
            label_arr = torch.from_numpy(label_arr).to(self.device).type(torch.int64)
            label_checks = label_arr + self.inun_filter_arr
            label_checks = torch.unique(label_checks[~torch.isnan(label_checks)]).type(torch.int64)
            if label_checks[0] == 0:
                label_checks = label_checks[1:]
            mask = label_checks.new_zeros((max(label_arr.max(), label_checks.max()) + 1,), dtype=torch.bool)    #todo ?
            mask[label_checks.unique()] = True
            arr[~mask[label_arr]] = float('nan')
            return arr

    def surface_rebuild(self, wls, boundary_wls=None, save2tif=False, out_tif_file=None, compare_report=True,
                        ref_arr=None, filter_if=False):
        assert len(wls) == len(self.pts_coords), 'The number of provided RL water levels not matching ' \
                                                 'with the number of RLs!'
        db_wls = self.boundary_wls_checking(boundary_wls)
        if self.device is not None:
            inun_arr = self.interp_tgt_wls(db_wls, wls)
        else:
            inun_arr = self.interp_tgt_wls_cpu(db_wls, wls)
        inun_arr = self.dem_cut(inun_arr)
        if filter_if:
            inun_arr = self.filter_cluster_in_arr(inun_arr)
        inun_arr = inun_arr.detach().cpu().numpy()
        if save2tif:
            assert out_tif_file is not None, 'Please specify output geotiff file name!'
            gdal_writetiff(inun_arr, f'{self.work_dir}{out_tif_file}', target_transform=self.demtrans)

        if compare_report & (ref_arr is not None):
            if self.device is not None:
                demarr_cpu = self.demarr.detach().cpu().numpy()
            else:
                demarr_cpu = self.demarr
            ref_arr[np.isnan(ref_arr)] = self.nan_fill_value
            deptharr = ref_arr - demarr_cpu
            deptharr[np.isnan(deptharr)] = self.nan_fill_value
            ref_arr[deptharr < self.depth_thd] = np.nan
            comp = inun_compare(inun_arr, ref_arr)
            inun_compare_report(comp)
            return inun_arr, comp
        else:
            return inun_arr

    def griddata_build(self, wls, boundary_wls, method='linear', ref_arr=None, filter_if=True):
        db_wls = self.boundary_wls_checking(boundary_wls)
        inun_arr = self.rebuild_temp.copy()
        inun_arr[~np.isnan(self.rebuild_temp)] = griddata(np.array(list(self.pts_coords) + self.bounds_coords),
                                                          np.array(wls + db_wls), self.tgt_xy, method=method)
        inun_arr = self.dem_cut(inun_arr)
        if filter_if:
            inun_arr = self.filter_cluster_in_arr(inun_arr)
        demarr_cpu = self.demarr
        ref_arr[np.isnan(ref_arr)] = self.nan_fill_value
        deptharr = ref_arr - demarr_cpu
        deptharr[np.isnan(deptharr)] = self.nan_fill_value
        ref_arr[deptharr < self.depth_thd] = np.nan
        comp = inun_compare(inun_arr, ref_arr)
        inun_compare_report(comp)
        return inun_arr, comp

