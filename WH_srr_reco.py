from SRR_2_reconstruction import SRR2InundationReconstruction
from nc_func import get_nc_xi_yi, Dataset
from gdal_func import *
from base_func import map_compare_stats, get_tf_event_log, directory_checking
import numpy as np


class reco_eval:
    def __init__(self, thd_depth=0.05):
        self.data_dir = '../TUFLOW_WH_yz/results/'
        self.work_dir = 'srr_results/'
        self.dem_tif_file = 'gis/dem_croped.tif'
        self.db_tif_file = 'gis/WH_db_line.tif'
        self.max_ext_file = f'{self.work_dir}maximum_inundation_extent_WH.tif'
        self.pts_file = self.work_dir + 'ss_100.shp'
        self.data_indexer = get_tf_event_log()[1]
        evt = self.data_indexer.eventname.iloc[0]
        tgt_x_coords, tgt_y_coords = get_nc_xi_yi(f'{self.data_dir}{evt}/WHYZ_{evt}.nc')
        self.reco_model = SRR2InundationReconstruction(self.work_dir, self.dem_tif_file, self.pts_file,
                                                       tgt_x_coords, tgt_y_coords,
                                                       tgt_mask_file=self.max_ext_file,
                                                       depth_threshold=thd_depth)
        pts_coords = read_shp_point(self.pts_file)
        trans = gdal_transform(self.dem_tif_file)   #(423770.0, 10.0, 0, 5992790.0, 0, -10.0)
        self.pts_rcs = [coords2rc(trans, coords) for coords in pts_coords]
        self.dem_map = gdal_asarray(self.dem_tif_file)
        self.dem_arr = self.dem_map[[r for r, c in self.pts_rcs], [c for r, c in self.pts_rcs]]
        self.depth_thd = thd_depth

    def get_inun_level(self, t_idx, evt_id, rls_depth):
        evt = self.data_indexer.eventname.iloc[evt_id]
        nc_file = f'{self.data_dir}{evt}/WHYZ_{evt}.nc'
        dataset = Dataset(nc_file)
        nc_arr = dataset.variables['water_level'][t_idx, :, :]
        ref_arr = nc_arr.data
        ref_arr[nc_arr.mask] = np.nan
        ref_arr = np.flip(ref_arr, axis=0)
        wls = list(rls_depth + self.dem_arr)
        wl_reco = self.reco_model.surface_rebuild(wls, compare_report=False, filter_if=False)
        return wl_reco, ref_arr

    def accuracy_test(self, time_interval=12*2):
        maxext = gdal_asarray(self.max_ext_file)
        ext_mask = (~np.isnan(maxext))
        map_per_evt = (1801 - time_interval // 2) // time_interval
        val_idxs = np.arange(0, map_per_evt)
        for evt_i in range(4):
            for map_i in val_idxs:
                t_idx = map_i * time_interval + time_interval // 2
                evt = self.data_indexer.eventname.iloc[evt_i]
                nc_file = f'{self.data_dir}{evt}/WHYZ_{evt}.nc'
                dataset = Dataset(nc_file)
                nc_arr = dataset.variables['water_level'][t_idx, :, :]
                ref_arr = nc_arr.data
                ref_arr[nc_arr.mask] = np.nan
                ref_arr = np.flip(ref_arr, axis=0)
                wls = ref_arr[[r for r, c in self.pts_rcs], [c for r, c in self.pts_rcs]]
                wls[np.isnan(wls)] = self.dem_arr[np.isnan(wls)]    # remove nan in level data, filled with dem
                wl_reco = self.reco_model.surface_rebuild(list(wls), compare_report=False, filter_if=True)
                ref_depth = ref_arr - self.dem_map
                reco_depth = wl_reco - self.dem_map
                map_compare_stats(reco_depth, ref_depth, ext_mask, self.depth_thd)
        return 0

    def lstm_2dlir_results_analysis(self, rl_preds_file, time_interval=3):
        from gdal_func import gdal_writetiff, gdal_transform
        results_dir = 'unet_results/predictions/lstm62dlir_results/'
        directory_checking(results_dir)
        pred_depths1_l = gdal_asarray(rl_preds_file).reshape(4, -1, 21133)
        pred_depths1_l[pred_depths1_l < 0] = 0

        maxext = gdal_asarray(self.max_ext_file)
        ext_mask = (~np.isnan(maxext))
        map_per_evt = (1801 - time_interval // 2) // time_interval
        val_idxs = np.arange(0, map_per_evt)
        results_se = np.zeros((4, *self.dem_map.shape))
        results_ref_sum4mean = np.zeros((4, *self.dem_map.shape))
        for evt_i in range(4):
            evt = self.data_indexer.eventname.iloc[evt_i]
            nc_file = f'{self.data_dir}{evt}/WHYZ_{evt}.nc'
            dataset = Dataset(nc_file)
            maxi_reco = np.zeros(self.dem_map.shape)
            maxi_ref = np.zeros(self.dem_map.shape)
            evolution_metrices = np.zeros((val_idxs.shape[0], 4))    # total inun area-reco&ref, pod, rfa
            for map_i in val_idxs:
                t_idx = map_i * time_interval + time_interval // 2
                nc_arr = dataset.variables['water_level'][t_idx, :, :]
                ref_map = nc_arr.data
                ref_map[nc_arr.mask] = np.nan
                ref_map = np.flip(ref_map, axis=0)
                ref_map = ref_map - self.dem_map
                wls = pred_depths1_l[evt_i, map_i] + self.dem_arr
                wl_reco = self.reco_model.surface_rebuild(list(wls), compare_report=False, filter_if=False)
                depth_map = wl_reco - self.dem_map
                depth_map[np.isnan(depth_map)] = 0
                depth_map[depth_map < self.depth_thd] = 0
                ref_map[np.isnan(ref_map)] = 0
                ref_map[ref_map < self.depth_thd] = 0
                results_ref_sum4mean[evt_i] += ref_map
                results_se[evt_i] += (depth_map - ref_map) ** 2
                maxi_reco = np.maximum(maxi_reco, depth_map)
                maxi_ref = np.maximum(maxi_ref, ref_map)

                evolution_metrices[map_i, 0] = np.sum(depth_map!=0)
                evolution_metrices[map_i, 1] = np.sum(ref_map!=0)
                evolution_metrices[map_i, 2:] = map_compare_stats(depth_map, ref_map, ext_mask)[:2]

            maxi_reco[~ext_mask] = np.nan
            maxi_ref[~ext_mask] = np.nan
            gdal_writetiff(maxi_reco, f"{results_dir}maxi_reco_map_{evt_i}.tif",
                           target_transform=gdal_transform(self.dem_tif_file))
            gdal_writetiff(maxi_ref, f"{results_dir}maxi_ref_map_{evt_i}.tif",
                           target_transform=gdal_transform(self.dem_tif_file))
            gdal_writetiff(evolution_metrices, f"{results_dir}evo_mtxs_{evt_i}.tif",
                           target_transform=(0, 1, 0, 0, 0, -1))
            gdal_writetiff(results_se[evt_i], f"{results_dir}se_{evt_i}.tif",
                           target_transform=gdal_transform(self.dem_tif_file))
            gdal_writetiff(results_ref_sum4mean[evt_i], f"{results_dir}refsum_{evt_i}.tif",
                           target_transform=gdal_transform(self.dem_tif_file))

        for evt_i in range(4):
            results_ref_sum4mean[evt_i] = gdal_asarray(f"{results_dir}refsum_{evt_i}.tif")
            results_se[evt_i] = gdal_asarray(f"{results_dir}se_{evt_i}.tif")
        ref_mean = np.mean(results_ref_sum4mean / 600, axis=0)
        rmse_map = np.sqrt(np.mean(results_se / 600, axis=0))
        rrmse_map = ref_mean.copy()
        rrmse_map[ref_mean!=0] = rmse_map[ref_mean!=0] / ref_mean[ref_mean!=0]
        rmse_map[~ext_mask] = np.nan
        rrmse_map[~ext_mask] = np.nan
        gdal_writetiff(rmse_map, f"{results_dir}rmse_map.tif",
                       target_transform=gdal_transform(self.dem_tif_file))
        gdal_writetiff(rrmse_map, f"{results_dir}rrmse_map.tif",
                       target_transform=gdal_transform(self.dem_tif_file))
        gdal_writetiff(ref_mean, f"{results_dir}refmean_map.tif",
                       target_transform=gdal_transform(self.dem_tif_file))
        for evt_i in range(4):
            evt_ref_mean = results_ref_sum4mean[evt_i] / 600
            evt_rmse = np.sqrt(results_se[evt_i] / 600)
            evt_rrmse = evt_ref_mean.copy()
            evt_rrmse[evt_ref_mean!=0] = evt_rmse[evt_ref_mean!=0] / evt_ref_mean[evt_ref_mean!=0]
            evt_rmse[~ext_mask] = np.nan
            evt_rrmse[~ext_mask] = np.nan
            gdal_writetiff(evt_rmse, f"{results_dir}rmse_map_{evt_i}.tif",
                           target_transform=gdal_transform(self.dem_tif_file))
            gdal_writetiff(evt_rrmse, f"{results_dir}rrmse_map_{evt_i}.tif",
                           target_transform=gdal_transform(self.dem_tif_file))

        for evt_i in range(4):
            maxi_reco = gdal_asarray(f"{results_dir}maxi_reco_map_{evt_i}.tif")
            maxi_ref = gdal_asarray(f"{results_dir}maxi_ref_map_{evt_i}.tif")
            rfa, pod, _ = map_compare_stats(maxi_reco, maxi_ref)
            maxi_reco[maxi_reco < 0.05] = 0
            maxi_ref[maxi_ref < 0.05] = 0
            comp = maxi_reco - maxi_ref
            comp[(maxi_ref != 0) & (maxi_reco == 0)] = -555
            comp[(maxi_ref == 0) & (maxi_reco != 0)] = -666
            comp[(maxi_ref != 0) & (maxi_reco != 0)] = 1
            comp[(maxi_ref == 0) & (maxi_reco == 0)] = np.nan
            gdal_writetiff(comp, f"{results_dir}maxi_comp_{evt_i}.tif",
                           target_transform=gdal_transform(self.dem_tif_file))
        ## rrmse correction
        # results_dir = 'unet_results/predictions/lstm62dlir_results/'
        results_dir = 'unet_results/predictions/convo3unet_results/'
        for evt_i in range(4):
            evt_ref_sum4mean = gdal_asarray(f"{results_dir}refsum_{evt_i}.tif")
            evt_ref_mean = evt_ref_sum4mean / 600
            evt_rrmse = gdal_asarray(f"{results_dir}rrmse_map_{evt_i}.tif")
            evt_rrmse[evt_ref_mean==0] = np.nan
            evt_rrmse *= 100
            gdal_writetiff(evt_rrmse, f"{results_dir}rrmse_map_{evt_i}_2.tif",
                           target_transform=gdal_transform('gis/dem_croped.tif'))


if __name__ == '__main__':
    reco_exe = reco_eval(0.05)
    reco_exe.accuracy_test()
    reco_exe.lstm_2dlir_results_analysis(rl_preds_file="lstm_results/group_test6/final_depth_predictions_unfilled.tif")

