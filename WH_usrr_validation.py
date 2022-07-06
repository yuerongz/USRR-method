from WH_unet_model import UNet
from WH_unet_training import ImageDataBatcher
from base_func import *
import numpy as np
import torch
from torch import optim


def load_saved_unet_model(model_file, epoch_needed=False):
    device = pytorch_cuda_check()
    checkpoint = torch.load(model_file)
    model = UNet(enc_chs=checkpoint['model_structure'][0], dec_chs=checkpoint['model_structure'][0][:0:-1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['loss_his']
    val_losses = checkpoint['val_loss_his']
    eval_losses = checkpoint['eval_losses']
    eval_val_losses = checkpoint['eval_val_losses']
    model.eval()
    if epoch_needed:
        return model, losses, val_losses, eval_losses, eval_val_losses, epoch
    else:
        return model, losses, val_losses, eval_losses, eval_val_losses


def single_construct(reco_model, model, rl_depth, map_idxs, ext_mask, onlyspecs=True):
    """Use rl_depth=None to conduct reconstruction accuracy test based on TUFLOW depths. """
    with torch.no_grad():
        input_batch, ref_map = reco_model.batch_retriever(map_idxs[0], validation_evts=True, validation_mode=True,
                                                          rl_depth=rl_depth)
        ref_map = ref_map.detach().cpu().numpy()
        preds = model(input_batch).detach().cpu()
        for batch_i in map_idxs[1:]:
            input_batch, _ = reco_model.batch_retriever(batch_i, validation_evts=True, validation_mode=True,
                                                        rl_depth=rl_depth)
            preds = torch.cat((preds, model(input_batch).detach().cpu()), dim=0)
        del _
        torch.cuda.empty_cache()
        depth_map = reco_model.reconstruct_full_map(preds.to(reco_model.device),
                                                    saving_file_name=None).detach().cpu().numpy()
        torch.cuda.empty_cache()
    if onlyspecs:
        rlts_specs = map_compare_stats(depth_map, ref_map, ext_mask, 0.05)
        return rlts_specs
    else:
        depth_map[np.isnan(depth_map)] = 0
        ref_map[np.isnan(ref_map)] = 0
        ref_map[ref_map < 0.05] = 0
        depth_map[depth_map < 0.05] = 0
        return depth_map, ref_map


def construct_for_speed_test(reco_model, model, rl_depth, map_idxs):
    import time
    time_used = 0
    with torch.no_grad():
        sttm = time.time()
        input_batch, _ = reco_model.batch_retriever(map_idxs[0], validation_evts=True, validation_mode=True,
                                                          rl_depth=rl_depth, need_ref=False)
        preds = model(input_batch).detach()
        time_used += time.time()-sttm
        for batch_i in map_idxs[1:]:
            sttm = time.time()
            input_batch, _ = reco_model.batch_retriever(batch_i, validation_evts=True, validation_mode=True,
                                                        rl_depth=rl_depth, need_ref=False)
            preds = torch.cat((preds, model(input_batch).detach()), dim=0)
            time_used += time.time()-sttm
        # torch.cuda.empty_cache()
        sttm = time.time()
        depth_map = reco_model.reconstruct_full_map(preds, saving_file_name=None)
        time_used += time.time()-sttm
        torch.cuda.empty_cache()
    return time_used


def speed_test(unet_model_dir, unet_model_tag, unet_model_ep, timesteps=3):
    pred_depths, refdepth = rl_loading()
    model_file = f"{unet_model_dir}model_{unet_model_tag}_ep{unet_model_ep-1}.pt"
    model_specs = load_saved_unet_model(model_file, True)
    model = model_specs[0]
    reco_model = ImageDataBatcher(num_of_batch_per_map=4, time_interval=timesteps)
    reco_model.reconstruction_init()
    map_idxs = reco_model.val_idxs.reshape((4, -1, reco_model.num_of_batch_per_map))
    val_idxs = map_idxs
    reco_t = []
    for model_i in range(2):
        for evt_i in range(4):
            for map_i in range(val_idxs.shape[1]):
                reco_t.append(construct_for_speed_test(reco_model,
                                                       model, pred_depths[evt_i, map_i, :], val_idxs[evt_i, map_i]))
    return print(np.mean(np.array(reco_t)))


def rl_loading(rl_pred_dir='unet_results/predictions/convo3/'):
    """Read predicted water depths at RLs using 1D-CNN models. """
    pred_depths = gdal_asarray(f"{rl_pred_dir}final_depth_predictions_unfilled.tif").reshape(4, -1, 21133)
    ref = gdal_asarray(f"{rl_pred_dir}final_depth_predictions_ref.tif").reshape(4, -1, 21133)
    pred_depths[pred_depths < 0] = 0
    return pred_depths, ref


def convo_unet_results_analysis(unet_model_dir, unet_model_tag, unet_model_ep, timesteps=3,
                                output_dir='unet_results/predictions/convo3unet_results/'):
    from gdal_func import gdal_writetiff, gdal_transform
    pred_depths, refdepth = rl_loading()
    model_file = f"{unet_model_dir}model_{unet_model_tag}_ep{unet_model_ep-1}.pt"
    model_specs = load_saved_unet_model(model_file, True)
    model = model_specs[0]
    reco_model = ImageDataBatcher(num_of_batch_per_map=4, time_interval=timesteps)
    reco_model.reconstruction_init()
    maxext = np.flip(gdal_asarray(reco_model.max_ext_file), axis=0)
    ext_mask = ~np.isnan(maxext)
    map_idxs = reco_model.val_idxs.reshape((4, -1, reco_model.num_of_batch_per_map))
    val_idxs = map_idxs
    results_se = np.zeros((4, *reco_model.rl_map.shape))
    results_ref_sum4mean = np.zeros((4, *reco_model.rl_map.shape))

    for evt_i in range(4):
        maxi_reco = np.zeros(reco_model.rl_map.shape)
        maxi_ref = np.zeros(reco_model.rl_map.shape)
        evolution_metrices = np.zeros((val_idxs.shape[1], 4))    # total inun area-reco&ref, pod, rfa
        for map_i in range(val_idxs.shape[1]):
            depth_map, ref_map = single_construct(reco_model, model, pred_depths[evt_i, map_i, :],
                                                  val_idxs[evt_i, map_i], ext_mask, onlyspecs=False)
            results_ref_sum4mean[evt_i] += ref_map
            results_se[evt_i] += (depth_map - ref_map) ** 2
            maxi_reco = np.maximum(maxi_reco, depth_map)
            maxi_ref = np.maximum(maxi_ref, ref_map)

            evolution_metrices[map_i, 0] = np.sum(depth_map!=0)
            evolution_metrices[map_i, 1] = np.sum(ref_map!=0)
            evolution_metrices[map_i, 2:] = map_compare_stats(depth_map, ref_map, ext_mask)[:2]

        maxi_reco[~ext_mask] = np.nan
        maxi_ref[~ext_mask] = np.nan
        gdal_writetiff(np.flip(maxi_reco, axis=0),
                       f"{output_dir}maxi_reco_map_{evt_i}.tif",
                       target_transform=gdal_transform(reco_model.dem_file))    # maximum extent map
        gdal_writetiff(np.flip(maxi_ref, axis=0),
                       f"{output_dir}maxi_ref_map_{evt_i}.tif",
                       target_transform=gdal_transform(reco_model.dem_file))    # maximum extent TUFLOW map
        gdal_writetiff(evolution_metrices,
                       f"{output_dir}evo_mtxs_{evt_i}.tif",
                       target_transform=(0, 1, 0, 0, 0, -1))    # evaluation metrics results, shape=(map_num, items)
                                                                # item 0: total inundation area
                                                                # item 1: total inundation area by TUFLOW
                                                                # item 2: RFA
                                                                # item 3: POD
        gdal_writetiff(np.flip(results_se[evt_i], axis=0),
                       f"{output_dir}se_{evt_i}.tif",
                       target_transform=gdal_transform(reco_model.dem_file))    # sum of squared errors
        gdal_writetiff(np.flip(results_ref_sum4mean[evt_i], axis=0),
                       f"{output_dir}refsum_{evt_i}.tif",
                       target_transform=gdal_transform(reco_model.dem_file))    # sum of TUFLOW water depths
    ref_mean = np.mean(results_ref_sum4mean / 600, axis=0)
    rmse_map = np.sqrt(np.mean(results_se / 600, axis=0))
    rrmse_map = ref_mean.copy()
    rrmse_map[ref_mean!=0] = rmse_map[ref_mean!=0] / ref_mean[ref_mean!=0]
    rmse_map[~ext_mask] = np.nan
    rrmse_map[~ext_mask] = np.nan
    gdal_writetiff(rmse_map, "unet_results/predictions/convo3unet_results/rmse_map.tif",
                   target_transform=gdal_transform(reco_model.dem_file))
    gdal_writetiff(rrmse_map, "unet_results/predictions/convo3unet_results/rrmse_map.tif",
                   target_transform=gdal_transform(reco_model.dem_file))
    gdal_writetiff(ref_mean, "unet_results/predictions/convo3unet_results/refmean_map.tif",
                   target_transform=gdal_transform(reco_model.dem_file))
    for evt_i in range(4):
        evt_ref_mean = results_ref_sum4mean[evt_i] / 600
        evt_rmse = np.sqrt(results_se[evt_i] / 600)
        evt_rrmse = evt_ref_mean.copy()
        evt_rrmse[evt_ref_mean!=0] = evt_rmse[evt_ref_mean!=0] / evt_ref_mean[evt_ref_mean!=0]
        evt_rmse[~ext_mask] = np.nan
        evt_rrmse[~ext_mask] = np.nan
        gdal_writetiff(evt_rmse, f"unet_results/predictions/convo3unet_results/rmse_map_{evt_i}.tif",
                       target_transform=gdal_transform(reco_model.dem_file))
        gdal_writetiff(evt_rrmse, f"unet_results/predictions/convo3unet_results/rrmse_map_{evt_i}.tif",
                       target_transform=gdal_transform(reco_model.dem_file))
    for evt_i in range(4):
        maxi_reco = gdal_asarray(f"unet_results/predictions/convo3unet_results/maxi_reco_map_{evt_i}.tif")
        maxi_ref = gdal_asarray(f"unet_results/predictions/convo3unet_results/maxi_ref_map_{evt_i}.tif")
        rfa, pod, _ = map_compare_stats(maxi_reco, maxi_ref)
        maxi_reco[maxi_reco < 0.05] = 0
        maxi_ref[maxi_ref < 0.05] = 0
        comp = maxi_reco - maxi_ref
        comp[(maxi_ref != 0) & (maxi_reco == 0)] = -555
        comp[(maxi_ref == 0) & (maxi_reco != 0)] = -666
        comp[(maxi_ref != 0) & (maxi_reco != 0)] = 1
        comp[(maxi_ref == 0) & (maxi_reco == 0)] = np.nan
        gdal_writetiff(comp, f"unet_results/predictions/convo3unet_results/maxi_comp_{evt_i}.tif",
                       target_transform=gdal_transform(reco_model.dem_file))
    return 0


if __name__ == '__main__':
    unet_dir = 'unet_results/'
    saving_tag = 'unet_test9c'
    epoch_len = 8
    # speed_test(unet_dir, saving_tag, epoch_len)
    convo_unet_results_analysis(unet_dir, saving_tag, epoch_len)


