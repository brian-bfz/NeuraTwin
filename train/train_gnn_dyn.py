import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os, sys
import cv2
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timer import EpochTimer
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from dataset.dataset_gnn_dyn import ParticleDataset
from model.gnn_dyn import PropNetDiffDenModel
from utils import set_seed, count_trainable_parameters, get_lr, AverageMeter, load_yaml, save_yaml, YYYY_MM_DD_hh_mm_ss_ms


# from env.flex_env import FlexEnv

def collate_fn(data):
    """
    data: is a list of tuples with (example, label, length)
            where 'example' is a tensor of arbitrary shape
            and label/length are scalars
    """
    states, states_delta, attrs, particle_num = zip(*data)
    max_len = max(particle_num)
    batch_size = len(data)
    n_time, _, n_dim = states[0].shape
    states_tensor = torch.zeros((batch_size, n_time, max_len, n_dim), dtype=torch.float32)
    states_delta_tensor = torch.zeros((batch_size, n_time - 1, max_len, n_dim), dtype=torch.float32)
    attr = torch.zeros((batch_size, n_time, max_len), dtype=torch.float32)
    particle_num_tensor = torch.tensor(particle_num, dtype=torch.int32)
    # color_imgs_np = np.array(color_imgs)
    # color_imgs_tensor = torch.tensor(color_imgs_np, dtype=torch.float32)

    for i in range(len(data)):
        states_tensor[i, :, :particle_num[i], :] = states[i]
        states_delta_tensor[i, :, :particle_num[i], :] = states_delta[i]
        attr[i, :, :particle_num[i]] = attrs[i]

    return states_tensor, states_delta_tensor, attr, particle_num_tensor

def train():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train GNN dynamics model')
    parser.add_argument('--name', type=str, default=None,
                       help='Training run name. If not provided, uses timestamp (YYYY-MM-DD-hh-mm-ss-ms)')
    parser.add_argument('--profiling', action='store_true',
                       help='Enable timing profiling (logs timing breakdown each epoch)')
    args = parser.parse_args()

    config = load_yaml('config/train/gnn_dyn.yaml')
    n_rollout = config['train']['n_rollout']
    n_history = config['train']['n_history']
    ckp_per_iter = config['train']['ckp_per_iter']
    log_per_iter = config['train']['log_per_iter']
    n_epoch = config['train']['n_epoch']

    # Note: No longer need camera parameters since we work directly with world coordinates
    set_seed(config['train']['random_seed'])

    use_gpu = torch.cuda.is_available()

    ### make log dir
    TRAIN_ROOT = 'data/gnn_dyn_model'
    
    if config['train']['particle']['resume']['active']:
        TRAIN_DIR = os.path.join(TRAIN_ROOT, config['train']['particle']['resume']['folder'])
    else:
        # Determine training directory name
        if args.name:
            train_name = args.name
            TRAIN_DIR = os.path.join(TRAIN_ROOT, train_name)
            
            # Check if directory exists and is not empty
            if os.path.exists(TRAIN_DIR) and os.listdir(TRAIN_DIR):  # Directory exists and is not empty
                print(f"Error: Training directory '{TRAIN_DIR}' already exists and is not empty!")
                print(f"Please choose a different name or remove the existing directory.")
                return
        else:
            train_name = YYYY_MM_DD_hh_mm_ss_ms()
            TRAIN_DIR = os.path.join(TRAIN_ROOT, train_name)
    
    os.system('mkdir -p ' + TRAIN_DIR)
    save_yaml(config, os.path.join(TRAIN_DIR, "config.yaml"))

    print(f"Training directory: {TRAIN_DIR}")

    # Initialize profiling if enabled
    epoch_timer = EpochTimer() if args.profiling else None
    if args.profiling:
        print("Profiling enabled - will log timing breakdown each epoch")

    if not config['train']['particle']['resume']['active']:
        log_fout = open(os.path.join(TRAIN_DIR, 'log.txt'), 'w')
    else:
        log_fout = open(os.path.join(TRAIN_DIR, 'log_resume_epoch_%d_iter_%d.txt' % (
            config['train']['particle']['resume']['epoch'], config['train']['particle']['resume']['iter'])), 'w')

    ### dataloaders
    phases = ['train', 'valid']
    datasets = {phase: ParticleDataset(config['train']['data_file'], config, phase) for phase in phases}

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=config['train']['batch_size'],
        shuffle=True if phase == 'train' else False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn)
        for phase in phases}


    ### create model
    # model = PropNetModel(config, use_gpu)
    # model = PropNetNoPusherModel(config, use_gpu)
    model = PropNetDiffDenModel(config, use_gpu)
    print("model #params: %d" % count_trainable_parameters(model))


    # resume training of a saved model (if given)
    if config['train']['particle']['resume']['active']:
        model_path = os.path.join(TRAIN_DIR, 'net_epoch_%d_iter_%d.pth' % (
            config['train']['particle']['resume']['epoch'], config['train']['particle']['resume']['iter']))
        print("Loading saved ckp from %s" % model_path)

        pretrained_dict = torch.load(model_path)
        model.load_state_dict(pretrained_dict)


    if use_gpu:
        model = model.cuda()


    ### optimizer and losses
    params = model.parameters()
    optimizer = torch.optim.Adam(params,
                                lr=float(config['train']['lr']),
                                betas=(config['train']['adam_beta1'], 0.999))

    # Add lr_scheduler
    scheduler = None
    if config['train']['lr_scheduler']['enabled']:
        if config['train']['lr_scheduler']['type'] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config['train']['lr_scheduler']['factor'],
                patience=config['train']['lr_scheduler']['patience'],
                threshold_mode=config['train']['lr_scheduler']['threshold_mode'],
                cooldown=config['train']['lr_scheduler']['cooldown']
            )
        elif config['train']['lr_scheduler']['type'] == "StepLR":
            step_size = config['train']['lr_scheduler']['step_size']
            gamma = config['train']['lr_scheduler']['gamma']
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError("unknown scheduler type: %s" % config['train']['lr_scheduler']['type'])


    # start training
    st_epoch = config['train']['particle']['resume']['epoch'] if config['train']['particle']['resume']['epoch'] > 0 else 0
    best_valid_loss = np.inf

    avg_timer = EpochTimer()
    avg_timer.reset()

    for epoch in range(st_epoch, n_epoch):

        for phase in phases:
            model.train(phase == 'train')
            meter_loss = AverageMeter()
            
            # Start epoch timing
            if epoch_timer and phase == 'train':
                epoch_timer.start_epoch()

            # Time data loading
            if epoch_timer and phase == 'train':
                data_timer = epoch_timer.time_data_loading()
                data_timer.__enter__()

            for i, data in enumerate(dataloaders[phase]):

                # states: B x (n_his + n_roll) x (particles_num + pusher_num) x 3
                # attrs: B x (n_his + n_roll) x (particles_num + pusher_num)
                # states_delta: B x (n_his + n_roll - 1) x (particles_num + pusher_num) x 3
                states, states_delta, attrs, particle_nums = data

                B, length, n_obj, _ = states.size()
                assert length == n_rollout + n_history

                if use_gpu:
                    states = states.cuda()
                    attrs = attrs.cuda()
                    states_delta = states_delta.cuda()

                # End data loading timing
                if epoch_timer and phase == 'train':
                    data_timer.__exit__(None, None, None)

                loss = 0.

                with torch.set_grad_enabled(phase == 'train'):
                    
                    # Time forward pass
                    if epoch_timer and phase == 'train':
                        forward_timer = epoch_timer.time_forward()
                        forward_timer.__enter__()

                    # s_cur: B x (particles_num + pusher_num) x 3
                    s_cur = states[:, 0]
                    a_cur = attrs[:, 0]

                    for idx_step in range(n_rollout):
                        # s_nxt: B x (particles_num + pusher_num) x 3
                        s_nxt = states[:, idx_step + 1]

                        s_delta = states_delta[:, idx_step]

                        # s_pred: B x particles_num x 3
                        # s_pred = model.predict_one_step(a_cur, s_cur, s_delta)
                        # s_pred = model.predict_one_step_adj_list(a_cur, s_cur, s_delta)
                        s_pred = model.predict_one_step(a_cur, s_cur, s_delta)
                        # print('diff between s_pred and s_pred_adj: ', torch.sum(torch.abs(s_pred[0, :particle_nums[0]] - s_pred_adj[0, :particle_nums[0]])))

                        # loss += F.mse_loss(s_pred, s_nxt[:, pusher_num:])
                        for j in range(B):
                            loss += F.mse_loss(s_pred[j, :particle_nums[j]], s_nxt[j, :particle_nums[j]])
                        # loss += F.mse_loss(s_pred, s_nxt)

                        # if epoch >= 10:
                        #     print("MSE Loss: ", F.mse_loss(s_pred, s_nxt[:, pusher_num:]).item())
                        #     ax = plt.axes(projection='3d')
                        #     print("s_pred: ", s_pred.shape)
                        #     print("s_nxt: ", s_nxt[:, pusher_num:].shape)
                        #     ax.scatter3D(s_pred[0, :, 0].detach().cpu().numpy(), s_pred[0, :, 1].detach().cpu().numpy(), s_pred[0, :, 2].detach().cpu().numpy(), color='red')
                        #     ax.scatter3D(s_nxt[0, pusher_num:, 0].detach().cpu().numpy(), s_nxt[0, pusher_num:, 1].detach().cpu().numpy(), s_nxt[0, pusher_num:, 2].detach().cpu().numpy(), color='blue')
                        #     plt.show()

                        # s_cur: B x (particles_num + pusher_num) x 3
                        # s_cur = torch.cat([states[:, idx_step + 1, :pusher_num], s_pred], dim=1)
                        s_cur = s_pred

                    loss = loss / (n_rollout * B)
                    
                    # End forward pass timing
                    if epoch_timer and phase == 'train':
                        forward_timer.__exit__(None, None, None)


                meter_loss.update(loss.item(), B)

                if phase == 'train':
                    # Time backward pass
                    if epoch_timer:
                        backward_timer = epoch_timer.time_backward()
                        backward_timer.__enter__()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # End backward pass timing
                    if epoch_timer:
                        backward_timer.__exit__(None, None, None)


                ### log and save ckp
                if i % log_per_iter == 0:
                    log = '%s [%d/%d][%d/%d] LR: %.6f, Loss: %.6f (%.6f)' % (
                        phase, epoch, n_epoch, i, len(dataloaders[phase]),
                        get_lr(optimizer),
                        np.sqrt(loss.item()), np.sqrt(meter_loss.avg))

                    print()
                    print(log)
                    log_fout.write(log + '\n')
                    log_fout.flush()

                if phase == 'train' and i % ckp_per_iter == 0:
                    torch.save(model.state_dict(), '%s/net_epoch_%d_iter_%d.pth' % (TRAIN_DIR, epoch, i))
                
                                # Time data loading
                if epoch_timer and phase == 'train':
                    data_timer = epoch_timer.time_data_loading()
                    data_timer.__enter__()

            # End data loading timing
            if epoch_timer and phase == 'train':
                data_timer.__exit__(None, None, None)

            # End epoch timing and log profiling info
            if epoch_timer and phase == 'train':
                epoch_timer.end_epoch()
                if hasattr(model, '_edge_times'):
                    epoch_timer.edge_time = sum(model._edge_times)
                    model._edge_times = []
                timing_summary = epoch_timer.get_summary()
                
                # Log timing breakdown
                timing_log = f'PROFILING [Epoch {epoch}] Total: {timing_summary["total_time"]:.2f}s | ' \
                           f'Data: {timing_summary["data_loading_time"]:.2f}s ({timing_summary["data_loading_pct"]:.1f}%) | ' \
                           f'Forward: {timing_summary["forward_time"]:.2f}s ({timing_summary["forward_pct"]:.1f}%) | ' \
                           f'Edges: {timing_summary["edge_time"]:.2f}s ({timing_summary["edge_pct"]:.1f}%) | ' \
                           f'Backward: {timing_summary["backward_time"]:.2f}s ({timing_summary["backward_pct"]:.1f}%) | ' \
                           f'GPU Mem: {timing_summary["gpu_memory_peak_mb"]:.0f}MB'
                
                for attr in epoch_timer.__dict__:
                    if hasattr(avg_timer, attr):
                        avg_timer.__dict__[attr] += epoch_timer.__dict__[attr]
                timing_summary = avg_timer.get_summary()
                timing_log += f'Avg Total: {timing_summary["total_time"] / (epoch + 1) :.2f}s | ' \
                           f'Avg Data: {timing_summary["data_loading_time"] / (epoch + 1):.2f}s ({timing_summary["data_loading_pct"]:.1f}%) | ' \
                           f'Avg Forward: {timing_summary["forward_time"] / (epoch + 1):.2f}s ({timing_summary["forward_pct"]:.1f}%) | ' \
                           f'Avg Edges: {timing_summary["edge_time"] / (epoch + 1):.2f}s ({timing_summary["edge_pct"]:.1f}%) | ' \
                           f'Avg Backward: {timing_summary["backward_time"] / (epoch + 1):.2f}s ({timing_summary["backward_pct"]:.1f}%) | ' \
                           f'Avg GPU Mem: {timing_summary["gpu_memory_peak_mb"] / (epoch + 1):.0f}MB'
                
                print(timing_log)
                log_fout.write(timing_log + '\n')
                log_fout.flush()

            log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, epoch, n_epoch, np.sqrt(meter_loss.avg), np.sqrt(best_valid_loss))
            print(log)
            log_fout.write(log + '\n')
            log_fout.flush()


            if phase == 'valid':
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg
                    torch.save(model.state_dict(), '%s/net_best.pth' % (TRAIN_DIR))
                
                # Step the scheduler
                if (scheduler is not None) and (config['train']['lr_scheduler']['type'] == "StepLR"):
                    scheduler.step()
                if (scheduler is not None) and (config['train']['lr_scheduler']['type'] == "ReduceLROnPlateau"):
                    scheduler.step(meter_loss.avg)




if __name__=='__main__':
    train()
