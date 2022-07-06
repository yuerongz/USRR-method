import pandas as pd
import numpy as np
import os
import torch
from sys import platform
from gdal_func import gdal_asarray


def get_tf_event_names(event_log):
    return list(event_log.eventname)


def get_tf_event_input(data_dir, event_name):
    pd_dt = pd.read_csv(f'{data_dir}{event_name}.csv')
    return pd_dt


def get_tf_event_log():
    if platform == 'linux': # on Spartan
        work_dir = '/home/yuerongz/punim0728/WHProj/tuflow_files_prep/'
    else:
        work_dir = 'tuflow_files_prep/'
    event_log_file = f'{work_dir}tuflow_events_log.csv'
    event_log = pd.read_csv(event_log_file, index_col=0)
    validation_evt_idxs = np.array(event_log.validation)
    return event_log[validation_evt_idxs==0], event_log[validation_evt_idxs==1]


def directory_checking(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Directory ", target_dir, " Created.")
    else:
        print("Directory ", target_dir, " already exists.")


def pytorch_cuda_check():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # you can continue going on here, like cuda:1 cuda:2....etc.
        # print("Running on the GPU...")
    else:
        device = torch.device("cpu")
        print("Running on the CPU...")
    return device


def count_model_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nn_model_training_summary(model, losses, val_losses, eval_losses, eval_val_losses, epoch,
                              fig_title='MSE', fig_file=None):
    from matplotlib import pyplot as plt
    ep = epoch + 1
    loss_by_epoch = torch.mean(torch.tensor(losses).view(ep, -1), dim=1)
    val_loss_by_epoch = torch.mean(torch.tensor(val_losses).view(ep, -1), dim=1)
    eval_loss_by_epoch = np.mean(np.array(eval_losses).reshape(ep, -1), axis=1)
    eval_val_loss_by_epoch = np.mean(np.array(eval_val_losses).reshape(ep, -1), axis=1)
    # print(np.array(loss_by_epoch[-1]), np.array(val_loss_by_epoch[-1]), eval_loss_by_epoch[-1], eval_val_loss_by_epoch[-1])
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, epoch+1, (epoch+1)/len(losses)), losses, 'black', alpha=0.3)
    ax.plot(np.arange(0, epoch+1, (epoch+1)/len(val_losses)), val_losses, 'green', alpha=0.3)
    ax.plot(loss_by_epoch, 'r')
    ax.plot(val_loss_by_epoch)
    ax.set_title(fig_title)
    ax.set_ylim(0, 0.02)
    if fig_file is not None:
        plt.savefig(fig_file)
        plt.close(fig)
    else:
        plt.show()
    print('Number of Parameters:', count_model_param(model))
    return 0


class RMSE:
    def __init__(self, map_shape, mask_map, thd=0.):
        self.se_map = np.zeros(map_shape)
        self.mask_map = mask_map
        self.thd = thd
        self.counts = 0

    def se_calc(self, map1, map2):
        map1[map1 < self.thd] = 0
        map2[map2 < self.thd] = 0
        self.se_map += (map1 - map2) ** 2
        self.counts += 1
        return 0

    def rmse_final(self):
        mse_map = self.se_map / self.counts
        rmse = np.sqrt(np.mean(mse_map[self.mask_map]))
        print('The active zone RMSE of water depth in validation events:', rmse)
        return mse_map

    def example(self):
        # rmse_calc = RMSE(reco_model.dem_map.size(), ext_mask)
        # for idx in reco_model.val_idxs:
        #     input_batch, ref_map = reco_model.batch_retriever(idx, True, rl_depth)
        #     preds = model(input_batch)
        #     depth_map = reco_model.reconstruct_full_map(preds, saving_file_name=None)
        #     rmse_calc.se_calc(depth_map, ref_map)
        # mse_map = rmse_calc.rmse_final()
        return 0

def map_compare_stats(depth_map, ref_map, ext_mask=None, depth_thd=0.05):
    depth_map[np.isnan(depth_map)] = 0
    ref_map[np.isnan(ref_map)] = 0
    ref_map[ref_map < depth_thd] = 0
    depth_map[depth_map < depth_thd] = 0
    hit = np.sum((ref_map > 0) & (depth_map > 0))
    miss = np.sum((ref_map > 0) & (depth_map == 0))
    fa = np.sum((ref_map == 0) & (depth_map > 0))
    diff_map = depth_map - ref_map
    if ext_mask is not None:
        diff_map = diff_map[ext_mask]
    rmse = np.sqrt(np.mean(diff_map ** 2))
    print('RFA:', fa / (fa + hit), 'POD:', hit / (hit + miss), 'RMSE:', rmse)
    return fa / (fa + hit), hit / (hit + miss), rmse

