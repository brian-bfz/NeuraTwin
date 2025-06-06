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
    Custom collation function for batching variable-sized particle data.
    Pads sequences to maximum particle count in batch and handles temporal structure.
    
    Args:
        data: list of tuples - [(states, states_delta, attrs, particle_num), ...] where:
            states: [particle_num, time, 3] - particle positions over time
            states_delta: [particle_num, time-1, 3] - particle displacements
            attrs: [particle_num, time] - particle attributes (0=object, 1=robot)
            particle_num: int - number of particles in this sample
            
    Returns:
        states_tensor: [batch_size, time, max_particles, 3] - padded position sequences
        states_delta_tensor: [batch_size, time-1, max_particles, 3] - padded displacement sequences
        attr: [batch_size, time, max_particles] - padded attribute sequences
        particle_num_tensor: [batch_size] - particle counts per sample
    """
    states, states_delta, attrs, particle_num = zip(*data)
    max_len = max(particle_num)  # Maximum particles in this batch
    batch_size = len(data)
    n_time, _, n_dim = states[0].shape
    
    # Initialize padded tensors
    states_tensor = torch.zeros((batch_size, n_time, max_len, n_dim), dtype=torch.float32)
    states_delta_tensor = torch.zeros((batch_size, n_time - 1, max_len, n_dim), dtype=torch.float32)
    attr = torch.zeros((batch_size, n_time, max_len), dtype=torch.float32)
    particle_num_tensor = torch.tensor(particle_num, dtype=torch.int32)

    # Fill tensors with actual data (rest remains zero-padded)
    for i in range(len(data)):
        states_tensor[i, :, :particle_num[i], :] = states[i]
        states_delta_tensor[i, :, :particle_num[i], :] = states_delta[i]
        attr[i, :, :particle_num[i]] = attrs[i]

    return states_tensor, states_delta_tensor, attr, particle_num_tensor

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train():
    """
    Main training loop for GNN dynamics model with autoregressive rollout.
    Handles model initialization, data loading, training with validation,
    learning rate scheduling, early stopping, and checkpoint management.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train GNN dynamics model')
    parser.add_argument('--name', type=str, default=None,
                       help='Training run name. If not provided, uses timestamp (YYYY-MM-DD-hh-mm-ss-ms)')
    parser.add_argument('--profiling', action='store_true',
                       help='Enable timing profiling (logs timing breakdown each epoch)')
    args = parser.parse_args()

    # Load training configuration
    config = load_yaml('config/train/gnn_dyn.yaml')
    n_rollout = config['train']['n_rollout']
    n_history = config['train']['n_history']
    ckp_per_iter = config['train']['ckp_per_iter']
    log_per_iter = config['train']['log_per_iter']
    n_epoch = config['train']['n_epoch']
    set_seed(config['train']['random_seed'])
    use_gpu = torch.cuda.is_available()

    # ========================================================================
    # SETUP TRAINING DIRECTORY
    # ========================================================================
    
    TRAIN_ROOT = 'data/gnn_dyn_model'
    
    if config['train']['particle']['resume']['active']:
        # Resume from existing checkpoint
        TRAIN_DIR = os.path.join(TRAIN_ROOT, config['train']['particle']['resume']['folder'])
    else:
        # Create new training directory
        if args.name:
            train_name = args.name
            TRAIN_DIR = os.path.join(TRAIN_ROOT, train_name)
            
            # Check if directory exists and is not empty
            if os.path.exists(TRAIN_DIR) and os.listdir(TRAIN_DIR):
                print(f"Error: Training directory '{TRAIN_DIR}' already exists and is not empty!")
                print(f"Please choose a different name or remove the existing directory.")
                return
        else:
            train_name = YYYY_MM_DD_hh_mm_ss_ms()
            TRAIN_DIR = os.path.join(TRAIN_ROOT, train_name)
    
    os.system('mkdir -p ' + TRAIN_DIR)
    save_yaml(config, os.path.join(TRAIN_DIR, "config.yaml"))
    print(f"Training directory: {TRAIN_DIR}")

    # Initialize profiling and logging
    epoch_timer = EpochTimer() if args.profiling else None
    if args.profiling:
        print("Profiling enabled - will log timing breakdown each epoch")

    if not config['train']['particle']['resume']['active']:
        log_fout = open(os.path.join(TRAIN_DIR, 'log.txt'), 'w')
    else:
        log_fout = open(os.path.join(TRAIN_DIR, 'log_resume_epoch_%d_iter_%d.txt' % (
            config['train']['particle']['resume']['epoch'], config['train']['particle']['resume']['iter'])), 'w')

    # ========================================================================
    # DATA LOADING SETUP
    # ========================================================================
    
    phases = ['train', 'valid']
    datasets = {phase: ParticleDataset(config['train']['data_file'], config, phase) for phase in phases}

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=config['train']['batch_size'],
        shuffle=True if phase == 'train' else False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn)
        for phase in phases}

    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================

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

    # ========================================================================
    # OPTIMIZER, TIMER, ROLLBACK, AND SCHEDULER SETUP
    # ========================================================================

    params = model.parameters()
    optimizer = torch.optim.Adam(params,
                                lr=float(config['train']['lr']),
                                betas=(config['train']['adam_beta1'], 0.999))

    # Initialize learning rate scheduler
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

    st_epoch = config['train']['particle']['resume']['epoch'] if config['train']['particle']['resume']['epoch'] > 0 else 0
    best_valid_loss = np.inf

    avg_timer = EpochTimer()
    avg_timer.reset()

    # Early stopping and rollback configuration
    if config['train']['rollback']['enabled']:
        patience_epochs = config['train']['rollback']['patience']
        rollback_threshold = config['train']['rollback']['threshold']
        validation_history = []
    best_epoch = 0

    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================

    for epoch in range(st_epoch, n_epoch):

        for phase in phases:
            model.train(phase == 'train')
            meter_loss = AverageMeter()
            
            # Start epoch timing
            if epoch_timer and phase == 'train':
                epoch_timer.start_epoch()

            # Start data loading timing
            if epoch_timer and phase == 'train':
                data_timer = epoch_timer.time_data_loading()
                data_timer.__enter__()

            for i, data in enumerate(dataloaders[phase]):
                # Data format: B x (n_his + n_roll) x particle_num x 3
                states, states_delta, attrs, particle_nums = data

                B, length, max_particles, _ = states.size()
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
                    
                    # Start forward pass timing
                    if epoch_timer and phase == 'train':
                        forward_timer = epoch_timer.time_forward()
                        forward_timer.__enter__()

                    # ============================================================
                    # AUTOREGRESSIVE ROLLOUT PREDICTION
                    # ============================================================
                    
                    # Initialize sliding window buffers with first n_history frames
                    history_buffer_states = states[:, :n_history, :, :].clone()  # B x n_history x particle_num x 3
                    history_buffer_delta = states_delta[:, :n_history, :, :].clone()  # B x n_history x particle_num x 3
                    # Assume future object deltas are already masked

                    for idx_step in range(n_rollout):
                        # Extract current history window
                        a_hist = attrs[:, idx_step:idx_step + n_history, :]  # B x n_history x particle_num # we can just use gt attrs here
                        s_hist = history_buffer_states  # B x n_history x particle_num x 3
                        s_delta_hist = history_buffer_delta  # B x n_history x particle_num x 3
                        
                        # Ground truth next state 
                        s_nxt = states[:, n_history + idx_step, :, :]  # B x particle_num x 3

                        # s_pred: B x particle_num x 3
                        s_pred = model.predict_one_step(a_hist, s_hist, s_delta_hist, particle_nums)

                        # Calculate loss only for valid particles
                        for j in range(B):
                            loss += F.mse_loss(s_pred[j, :particle_nums[j]], s_nxt[j, :particle_nums[j]])

                        # Update sliding window buffers for next rollout step
                        if idx_step < n_rollout - 1:  # Don't update on last step
                            # Update delta buffer with predicted particle motion
                            history_buffer_delta[:, -1, :, :] = s_pred - history_buffer_states[:, -1, :, :]
                            
                            # Slide delta window: remove oldest, add new robot deltas from ground truth
                            history_buffer_delta = torch.cat([
                                history_buffer_delta[:, 1:, :, :],  # Remove first frame
                                states_delta[:, idx_step + n_history, :, :].unsqueeze(1)  # Add new delta
                            ], dim=1)

                            # Slide state window: remove oldest, add prediction
                            history_buffer_states = torch.cat([
                                history_buffer_states[:, 1:, :, :],  # Remove first frame
                                s_pred.unsqueeze(1)  # Add prediction as new frame
                            ], dim=1)                                                        
                    
                    # Normalize loss by batch size and rollout steps
                    loss = loss / (n_rollout * B)
                    
                    # End forward pass timing
                    if epoch_timer and phase == 'train':
                        forward_timer.__exit__(None, None, None)

                meter_loss.update(loss.item(), B)

                # ============================================================
                # BACKPROPAGATION AND OPTIMIZATION
                # ============================================================
                
                if phase == 'train':
                    # Start backward pass timing
                    if epoch_timer:
                        backward_timer = epoch_timer.time_backward()
                        backward_timer.__enter__()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # End backward pass timing
                    if epoch_timer:
                        backward_timer.__exit__(None, None, None)

                # ============================================================
                # LOGGING AND CHECKPOINTING
                # ============================================================
                
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
                
                # Start data loading timing for next iteration
                if epoch_timer and phase == 'train':
                    data_timer = epoch_timer.time_data_loading()
                    data_timer.__enter__()

            # End data loading timing
            if epoch_timer and phase == 'train':
                data_timer.__exit__(None, None, None)

            # ============================================================
            # EPOCH-END PROFILING AND LOGGING
            # ============================================================
            
            if epoch_timer and phase == 'train':
                epoch_timer.end_epoch()
                if hasattr(model, '_edge_times'):
                    epoch_timer.edge_time = sum(model._edge_times)
                    model._edge_times = []
                timing_summary = epoch_timer.get_summary()
                
                # Log detailed timing breakdown
                timing_log = f'PROFILING [Epoch {epoch}] Total: {timing_summary["total_time"]:.2f}s | ' \
                           f'Data: {timing_summary["data_loading_time"]:.2f}s ({timing_summary["data_loading_pct"]:.1f}%) | ' \
                           f'Forward: {timing_summary["forward_time"]:.2f}s ({timing_summary["forward_pct"]:.1f}%) | ' \
                           f'Edges: {timing_summary["edge_time"]:.2f}s ({timing_summary["edge_pct"]:.1f}%) | ' \
                           f'Backward: {timing_summary["backward_time"]:.2f}s ({timing_summary["backward_pct"]:.1f}%) | ' \
                           f'GPU Mem: {timing_summary["gpu_memory_peak_mb"]:.0f}MB'
                
                # Calculate running averages
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

            # ============================================================
            # VALIDATION AND MODEL CHECKPOINTING
            # ============================================================

            if phase == 'valid':
                current_val_loss = meter_loss.avg
                if config['train']['rollback']['enabled']:
                    validation_history.append(current_val_loss)
                
                # Save best model
                if current_val_loss < best_valid_loss:
                    best_valid_loss = current_val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), '%s/net_best.pth' % (TRAIN_DIR))
                    if config['train']['rollback']['enabled']:
                        torch.save({
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        }, '%s/checkpoint_best.pth' % (TRAIN_DIR))
                
                # Check for validation loss spike and rollback
                if config['train']['rollback']['enabled'] and len(validation_history) >= patience_epochs:
                    recent_min = np.min(validation_history[-patience_epochs:])
                    if recent_min > best_valid_loss * rollback_threshold:
                        print(f"Validation loss spike detected! Rolling back to epoch {best_epoch}")
                        print(f"Current avg loss: {recent_min:.6f}, Best loss: {best_valid_loss:.6f}")
                        
                        # Load the best checkpoint
                        model.load_state_dict(torch.load('%s/net_best.pth' % (TRAIN_DIR)))
                        checkpoint = torch.load('%s/checkpoint_best.pth' % (TRAIN_DIR))
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        if scheduler and checkpoint['scheduler_state_dict']:
                            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        
                        # Reset to best epoch
                        epoch = best_epoch
                        validation_history = validation_history[:best_epoch+1]  # Trim history
                        
                        log_rollback = f"Rolled back to epoch {best_epoch} with validation loss {best_valid_loss:.6f}"
                        print(log_rollback)
                        log_fout.write(log_rollback + '\n')
                        log_fout.flush()

                # Step the scheduler
                if (scheduler is not None) and (config['train']['lr_scheduler']['type'] == "StepLR"):
                    scheduler.step()
                if (scheduler is not None) and (config['train']['lr_scheduler']['type'] == "ReduceLROnPlateau"):
                    scheduler.step(meter_loss.avg)

if __name__=='__main__':
    train()
