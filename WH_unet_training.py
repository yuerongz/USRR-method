from gdal_func import read_shp_point, coords2rc, gdal_writetiff, gdal_transform
from base_func import *
from nc_func import Dataset
from WH_unet_model import UNet
import os.path
import numpy as np
import torch
from torch import nn, optim


class ImageDataBatcher:
    def __init__(self, num_of_batch_per_map=4, time_interval=12*5):
        """Note: all maps are processed in NC-style (upside-down flipped from geotiff map)."""
        self.platform_defaults()
        self.device = pytorch_cuda_check()
        self.event_log_dev, self.event_log_dev_val = get_tf_event_log()
        self.num_of_batch_per_map = num_of_batch_per_map   # use 1 for reconstruction; 8 for training

        self.map_size = 96
        self.cross_tile_dist_by_cell = 49  # overlay 500//100-1 more layers to increase sample size
        self.zero_sample_per_map = 1
        self.t_interval = time_interval

        self.temp_prep()
        self.indexing()

        # self.reconstruction_init()

    def platform_defaults(self):
        if platform == 'linux':  # on Spartan
            prj_dir = '/home/yuerongz/punim0728/WHProj/'
            self.data_dir = '/home/yuerongz/scratch_prj0728/TUFLOW_WH_yz/results/'
            self.pts_file = f'{prj_dir}srr_results/ss_100.shp'
            self.dem_file = f'{prj_dir}gis/dem_croped.tif'
            self.max_ext_file = f'{prj_dir}srr_results/maximum_inundation_extent_WH.tif'

        elif platform == 'win32':
            self.data_dir = '../TUFLOW_WH_yz/results/'
            self.pts_file = 'srr_results/ss_100.shp'
            self.dem_file = 'gis/dem_croped.tif'
            self.max_ext_file = 'srr_results/maximum_inundation_extent_WH.tif'

    def temp_prep(self):
        pts_coords = read_shp_point(self.pts_file)
        trans = gdal_transform(self.dem_file)    # (423770.0, 10.0, 0, 5992790.0, 0, -10.0)
        pts_rcs = [coords2rc(trans, coords) for coords in pts_coords]
        self.nc_pts_rcs = [(7995-r-1, c) for r, c in pts_rcs]
        self.nc_pts_rcs = np.array(self.nc_pts_rcs).T
        self.dem_map = np.flip(gdal_asarray(self.dem_file), axis=0)
        self.rl_map = np.zeros(self.dem_map.shape)
        self.rl_map[self.nc_pts_rcs[0], self.nc_pts_rcs[1]] = 1
        self.build_reshaping_func_seq(round(self.map_size / self.cross_tile_dist_by_cell))

        with torch.no_grad():
            self.dem_map[np.isnan(self.dem_map)] = np.nanmax(self.dem_map) + 1
            self.dem_map = torch.from_numpy(self.dem_map.copy()).float().to(self.device)
            self.rl_map = torch.from_numpy(self.rl_map.copy()).float().to(self.device)

            self.input_temp = self.tiles_prep_func_gpu(self.rl_map)  # prep input template
            self.batch_size = self.input_temp.size()[0] // self.num_of_batch_per_map
            assert self.input_temp.size()[0] % self.num_of_batch_per_map == 0, \
                f'Tiles ({self.input_temp.size()[0]}) of each map cannot be equally divided by the number of batches!' \
                f'Please adjust map_size or cross_tile_dist_by_cell.'

            self.shaped_dem = self.tiles_prep_func_gpu(self.dem_map)
            dem_mins = torch.amin(self.shaped_dem, dim=(1, 2))
            self.shaped_dem = self.shaped_dem - dem_mins.repeat_interleave(self.map_size ** 2).view(self.shaped_dem.size())
            self.shaped_dem[self.shaped_dem == self.shaped_dem.max()] = -1     # dem nan mark = -1
            self.dem_map[self.dem_map == self.dem_map.max()] = float('nan')

            self.shaped_dem = self.shaped_dem.detach().to('cpu').numpy()
            self.input_temp = self.input_temp.detach().to('cpu').numpy()
            self.rl_map = self.rl_map.detach().to('cpu').numpy()
        torch.cuda.empty_cache()
        return 0

    def indexing(self):
        self.map_per_evt = (1801 - self.t_interval//2) // self.t_interval

        idx_expander = lambda x, mul_i: np.repeat(x, mul_i) * mul_i + np.array(list(range(mul_i)) * len(x))
        test_evt_idxs = np.array([2, 8, 16, 23, 25, 34, 39, 43, 46, 48])
        test_map_idxs = idx_expander(test_evt_idxs, self.map_per_evt)
        self.test_idxs = idx_expander(test_map_idxs, self.num_of_batch_per_map)
        train_evt_idxs = np.delete(np.arange(50), test_evt_idxs)
        train_map_idxs = idx_expander(train_evt_idxs, self.map_per_evt)
        self.train_idxs = idx_expander(train_map_idxs, self.num_of_batch_per_map)
        rng = np.random.default_rng(321)
        rng.shuffle(self.test_idxs)
        rng.shuffle(self.train_idxs)
        self.val_idxs = np.arange(0, 4 * self.map_per_evt* self.num_of_batch_per_map)

        self.idx2evtid = lambda idx: int((idx // self.num_of_batch_per_map) // self.map_per_evt)
        self.idx2tidx = lambda idx: (idx // self.num_of_batch_per_map) % self.map_per_evt
        self.idx2batchidx = lambda idx: idx % self.num_of_batch_per_map
        return 0

    def tiles_prep_func(self, x):
        tiled_maps = self.reshaping_func_seq[0](x)
        for tiling_func in self.reshaping_func_seq[1:]:
            tiled_maps = np.append(tiled_maps, tiling_func(x), axis=0)
        return tiled_maps

    def tiles_prep_func_gpu(self, x):
        tiled_maps = [tiling_func(x) for tiling_func in self.reshaping_func_seq]
        tiled_maps = torch.cat(tiled_maps, 0)
        return tiled_maps

    def build_reshaping_func_seq(self, num_of_nodes_per_axes):
        self.reshaping_func_seq = []
        for r_i in range(num_of_nodes_per_axes):
            self.reshaping_func_seq.append(self.build_tiling_func(r_i * self.cross_tile_dist_by_cell,
                                                                  r_i * self.cross_tile_dist_by_cell))  # diagonal
            # for c_i in range(num_of_nodes_per_axes):      # horizontal & vertical & diagonal
            #     self.reshaping_func_seq.append(self.build_tiling_func(r_i * self.cross_tile_dist_by_cell,
            #                                                           c_i * self.cross_tile_dist_by_cell))
        return 0

    def build_tiling_func(self, cross_tile_size_r, cross_tile_size_c, development_mode=False):
        maxext = np.flip(gdal_asarray(self.max_ext_file), axis=0)
        ext_mask = (~np.isnan(maxext)).astype(int)
        slicing_func = lambda x: x[100+cross_tile_size_r:, 800+cross_tile_size_c:]  # adjust shaping origin
        new_shape = slicing_func(ext_mask).shape
        ax0_n = new_shape[0] // self.map_size
        ax1_n = new_shape[1] // self.map_size
        # ax0_remain = new_shape[0] % self.map_size
        # ax0_orig = new_shape[0] - ax0_remain
        reshaping_func = lambda x: slicing_func(x)[:self.map_size*ax0_n,:self.map_size*ax1_n]\
            .reshape((-1, ax1_n, self.map_size)).swapaxes(0, 1)\
            .reshape((ax1_n, -1, self.map_size, self.map_size)).swapaxes(0, 1)\
            .reshape(-1, self.map_size, self.map_size)
        filter_arr = (np.sum(reshaping_func(ext_mask), axis=(1, 2)) != 0) \
                     & (np.sum(reshaping_func(self.rl_map), axis=(1, 2)) != 0) # make sure there is RL in each tile
        if development_mode:
            return reshaping_func, filter_arr, ax0_n, ax1_n, (100+cross_tile_size_r, 800+cross_tile_size_c)
        else:
            tiling_func = lambda x: reshaping_func(x)[filter_arr,:,:]
            return tiling_func

    def nc_data_processing(self, nc_file, t_idx):
        masked_arr = Dataset(nc_file).variables['water_level'][(self.t_interval//2) + (t_idx * self.t_interval), :, :]
        arr = masked_arr.data
        arr[masked_arr.mask] = np.nan
        with torch.no_grad():
            arr_ts = torch.from_numpy(arr.copy()).float().to(self.device)
            arr_ts = arr_ts - self.dem_map
            arr_ts[torch.isnan(arr_ts)] = 0
            arr_ts[arr_ts < 0] = 0
        return arr_ts

    def validation_init(self):
        with torch.no_grad():
            self.rl_map_filled = torch.from_numpy(self.rl_map.copy()).float().to(self.device)
        return 0

    def build_back_func_lambda(self, ax1_n):
        back_func = lambda x: x.reshape(-1, ax1_n, self.map_size, self.map_size).swapaxes(0, 1).reshape(ax1_n, -1, self.map_size).swapaxes(0, 1).reshape(-1, ax1_n * self.map_size)
        return back_func

    def reconstruction_init(self):
        self.validation_init()
        self.output_maps_ls = []
        self.return_temp_ls = []
        self.filter_ls = []
        self.lyrs_separation_idxs = []
        self.map_origins = []
        self.lyr_sizes = []
        self.back_trans_funcs = []
        num_of_nodes_per_axes = round(self.map_size / self.cross_tile_dist_by_cell)
        st_idx = 0
        with torch.no_grad():
            for r_i in range(num_of_nodes_per_axes):
                c_i = r_i   # diagonal direction only
                # for c_i in range(num_of_nodes_per_axes):    # per layer
                reshaping_func, filter_arr, ax0_n, ax1_n, map_origin \
                    = self.build_tiling_func(r_i * self.cross_tile_dist_by_cell,
                                             c_i * self.cross_tile_dist_by_cell, development_mode=True)
                self.lyrs_separation_idxs.append((st_idx, st_idx + np.sum(filter_arr)))
                st_idx = self.lyrs_separation_idxs[-1][1]
                # back_idx = np.where(filter_arr)[0]
                output_map = np.zeros(self.dem_map.size())
                self.output_maps_ls.append(torch.from_numpy(output_map).float().to(self.device))
                self.return_temp_ls.append(torch.from_numpy(reshaping_func(output_map)).float().to(self.device))
                self.filter_ls.append(torch.from_numpy(filter_arr).bool().to(self.device))
                self.map_origins.append(map_origin)
                self.lyr_sizes.append((ax0_n * self.map_size, ax1_n * self.map_size))
                self.back_trans_funcs.append(self.build_back_func_lambda(ax1_n))
            if os.path.exists('unet_results/area_check.tif'):
                self.lyr_num_map = torch.from_numpy(gdal_asarray(f'unet_results/area_check.tif')).float().to(self.device)
                self.lyr_num_map = torch.fliplr(self.lyr_num_map.T).T       # convert to NC_style
            else:
                temp_tensor = torch.ones(self.input_temp.shape).float().to(self.device)
                self.lyr_num_map = self.reconstruct_full_map_return_and_sum(temp_tensor)
                gdal_writetiff(np.flip(self.lyr_num_map.detach().to('cpu').numpy(), axis=0),
                               f'unet_results/area_check.tif', ras_temp=self.dem_file)
            self.lyr_num_map[self.lyr_num_map == 0] = 1
            maxext = np.flip(gdal_asarray(self.max_ext_file), axis=0)
            ext_mask = (~np.isnan(maxext)).astype(int)
            print('All potential inundation area are covered by training layers? : ',
                  np.sum((ext_mask==1)&(self.lyr_num_map.detach().to('cpu').numpy()==0)) == 0)    # area check
        return 0

    def reconstruct_full_map_return_and_sum(self, x):
        for lyr_i in range(len(self.lyrs_separation_idxs)):
            x_st_idx = self.lyrs_separation_idxs[lyr_i][0]
            x_ed_idx = self.lyrs_separation_idxs[lyr_i][1]
            self.return_temp_ls[lyr_i][self.filter_ls[lyr_i]] = x[x_st_idx:x_ed_idx].squeeze(1)
            r_sz, c_sz = self.lyr_sizes[lyr_i]
            r_o, c_o = self.map_origins[lyr_i]
            self.output_maps_ls[lyr_i][r_o:r_o + r_sz, c_o:c_o + c_sz]\
                = self.back_trans_funcs[lyr_i](self.return_temp_ls[lyr_i])  # back transformed x
        output_sum = torch.sum(torch.stack(self.output_maps_ls), dim=0)                         # output NC-style map
        # output_sum = torch.fliplr(torch.sum(torch.stack(self.output_maps_ls), dim=0).T).T     # deprecated: output GDAL-style map
        return output_sum

    def reconstruct_full_map(self, x, saving_file_name=None):
        """ input x must be torch.Tensor() on GPU.
            Output Depth map is in NC style (row-flipped). Saved geoTiff images (if requested) are in GDAL style. """
        with torch.no_grad():
            output_final = torch.div(self.reconstruct_full_map_return_and_sum(x), self.lyr_num_map)
        # output_final[torch.isnan(self.dem_map)] = float('nan')
        if saving_file_name is not None:
            gdal_writetiff(np.flip(output_final.detach().cpu().numpy(), axis=0),
                           saving_file_name, ras_temp=self.dem_file)
        return output_final

    def batch_retriever(self, idx, validation_evts=False, validation_mode=False, rl_depth=None, need_ref=True):
        evt_id = self.idx2evtid(idx)
        t_idx = self.idx2tidx(idx)
        evt_batch_idx = self.idx2batchidx(idx)
        if validation_evts:
            data_info = self.event_log_dev_val.iloc[evt_id]
        else:
            data_info = self.event_log_dev.iloc[evt_id]
        with torch.no_grad():
            if validation_mode:
                if need_ref:
                    nc_file = f'{self.data_dir}{data_info.eventname}/WHYZ_{data_info.eventname}.nc'
                    ref_image = self.nc_data_processing(nc_file, t_idx)
                else:
                    ref_image = None
                if rl_depth is not None:
                    self.rl_map_filled[self.nc_pts_rcs[0], self.nc_pts_rcs[1]] = torch.from_numpy(rl_depth).float().to(self.device)
                    self.input_tempfilled = self.tiles_prep_func_gpu(self.rl_map_filled)  # prep new input template
                    input_filled = self.input_tempfilled[evt_batch_idx * self.batch_size:
                                                         (evt_batch_idx + 1) * self.batch_size, :, :].unsqueeze(1)
                else:
                    images = self.tiles_prep_func_gpu(ref_image)[evt_batch_idx*self.batch_size:(evt_batch_idx+1)*self.batch_size, :, :].unsqueeze(1)
                    input_filled = torch.from_numpy(self.input_temp[evt_batch_idx * self.batch_size:
                                                                    (evt_batch_idx + 1) * self.batch_size, :, :]
                                                    .copy()).float().to(self.device).unsqueeze(1)
                    input_filled[input_filled == 1] = images[input_filled == 1]
                dem_batch = torch.from_numpy(self.shaped_dem[evt_batch_idx*self.batch_size:(evt_batch_idx+1)*self.batch_size,
                                             :, :].copy()).float().to(self.device).unsqueeze(1)
                return torch.cat((input_filled, dem_batch), dim=1), ref_image
            else:
                nc_file = f'{self.data_dir}{data_info.eventname}/WHYZ_{data_info.eventname}.nc'
                images = self.tiles_prep_func_gpu(self.nc_data_processing(nc_file, t_idx))
                images = images[evt_batch_idx*self.batch_size:(evt_batch_idx+1)*self.batch_size, :, :].unsqueeze(1)
                zero_filter = torch.sum(images, dim=(1, 2, 3)) != 0
                if torch.sum(~zero_filter).item() >= (self.zero_sample_per_map * 4):
                    rnd_idxs = (np.random.uniform(0, 1, self.zero_sample_per_map) * torch.sum(~zero_filter).item()).astype(int)
                    zero_filter[torch.where(~zero_filter)[0][rnd_idxs]] = True
                images = images[zero_filter]    # used for training data filtering
                # torch.cuda.empty_cache()
                dem_batch = torch.from_numpy(self.shaped_dem[evt_batch_idx*self.batch_size:(evt_batch_idx+1)*self.batch_size,
                                             :, :].copy()).float().to(self.device).unsqueeze(1)[zero_filter]
                input_filled = torch.from_numpy(self.input_temp[evt_batch_idx*self.batch_size:
                                                                (evt_batch_idx+1)*self.batch_size, :, :]
                                                .copy()).float().to(self.device).unsqueeze(1)[zero_filter]
                input_filled[input_filled == 1] = images[input_filled == 1]
            return torch.cat((input_filled, dem_batch), dim=1), images


def training(work_dir, saving_tag, model_structure, t_interval, epoch_len=80, learning_rt=0.001):
    import time
    sttm = time.time()
    # use first 40 events for training, 10 events for testing
    training_data = ImageDataBatcher(time_interval=t_interval)
    device = pytorch_cuda_check()
    losses = []
    eval_losses = []
    val_losses = []
    eval_val_losses = []
    model = UNet(enc_chs=model_structure, dec_chs=model_structure[:0:-1]).to(device)
    model.float()
    loss_function = nn.MSELoss()
    eval_loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rt)
    for epoch in range(epoch_len):
        for idx in training_data.train_idxs:
            input_in_batch, output_in_batch = training_data.batch_retriever(idx)
            model.train()  # set model to TRAIN mode
            optimizer.zero_grad()
            pred = model(input_in_batch.float())
            out_ref = output_in_batch.float()
            loss = loss_function(pred, out_ref)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            eval_losses.append(eval_loss_function(pred, out_ref).detach().cpu().numpy())
        with torch.no_grad():
            for idx in training_data.test_idxs:
                input_in_batch, output_in_batch = training_data.batch_retriever(idx)
                model.eval()  # set model to TEST mode
                pred = model(input_in_batch.float())
                out_ref = output_in_batch.float()
                val_loss = loss_function(pred, out_ref)
                val_losses.append(val_loss.item())
                eval_val_losses.append(eval_loss_function(pred, out_ref).detach().cpu().numpy())
        print(f"Epoch {epoch}: train loss={np.mean(losses[-len(training_data.train_idxs):])}, "
              f"test loss={np.mean(val_losses[-len(training_data.test_idxs):])}; "
              f"eval_mtx={np.mean(eval_losses[-len(training_data.train_idxs):])}, "
              f"eval_test={np.mean(eval_val_losses[-len(training_data.test_idxs):])}. "
              f"time_since_start={time.time()-sttm}s.")
        if (np.mean(losses[-len(training_data.train_idxs):]) <= 0.0027) & \
                (np.mean(val_losses[-len(training_data.test_idxs):]) <= 0.001):     # early stopping
            break
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_his': losses,
        'val_loss_his': val_losses,
        'eval_losses': eval_losses,
        'eval_val_losses': eval_val_losses,
        'learning_rate': learning_rt,
        'batch_size': training_data.batch_size,
        'model_structure': (model_structure, training_data.map_size)
    }, f"{work_dir}model_{saving_tag}_ep{epoch}.pt")
    return model


if __name__ == '__main__':
    work_dir = 'unet_results/'
    saving_tag = 'unet_test9c'
    epoch_len = 15
    model_structure = [2, 32, 64, 128, 256]
    unet_model = training(work_dir, saving_tag, model_structure, t_interval=60,
                          epoch_len=epoch_len, learning_rt=0.001)
