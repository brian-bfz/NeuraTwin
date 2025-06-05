import os
import numpy as np
import json
import cv2
import pdb
import pickle
import json
import random
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# from env.flex_env import FlexEnv
from utils import load_yaml, set_seed, fps_rad_tensor, fps_np, recenter, opengl2cam, depth2fgpcd, pcd2pix

import matplotlib.pyplot as plt
# from dgl.geometry import farthest_point_sampler

from torch.utils.data import DataLoader

np.seterr(divide='ignore', invalid='ignore')

class ParticleDataset(Dataset):
    def __init__(self, data_file, config, phase):
        self.config = config

        n_episode = config['dataset']['n_episode']
        n_timestep = config['dataset']['n_timestep']
        self.downsample_rate = config['dataset']['downsample_rate']
        self.global_scale = config['dataset']['global_scale']

        train_valid_ratio = config['train']['train_valid_ratio']

        n_train = int(n_episode * train_valid_ratio)
        n_valid = n_episode - n_train

        if phase == 'train':
            self.epi_st_idx = 0
            self.n_episode = n_train
        elif phase == 'valid':
            self.epi_st_idx = n_train
            self.n_episode = n_valid
        else:
            raise AssertionError("Unknown phase %s" % phase)

        self.n_timestep = n_timestep + 1
        self.n_his = config['train']['n_history']
        self.n_roll = config['train']['n_rollout']
        self.hdf5_path = data_file

        self.screenHeight = 720
        self.screenWidth = 720
        self.img_channel = 1

        self.add_randomness = config['dataset']['randomness']['use']
        self.state_noise = config['dataset']['randomness']['state_noise'][phase]

        self.fps_radius = config['train']['fps_radius']
        
        # File handle for efficient access - will be opened lazily per worker
        self._hdf5_file = None
        self.offset = self.n_timestep - (self.n_his + self.n_roll - 1) * self.downsample_rate

    def _get_hdf5_file(self):
        """Lazy loading of HDF5 file handle to work properly with multiprocessing."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        return self._hdf5_file

    def __del__(self):
        """Ensure HDF5 file is properly closed when dataset is destroyed."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()

    def __len__(self):
        return self.n_episode * self.offset
    
    def read_particles(self, particles_path):
        particles = np.load(particles_path).reshape(-1, 4)
        particles[:, 3] = 1.0
        opencv_T_opengl = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        opencv_T_world = np.matmul(np.linalg.inv(self.cam_extrinsic), opencv_T_opengl)
        # print('opencv_T_world', opencv_T_world)
        # print('opencv_T_world inverse', np.linalg.inv(opencv_T_world))
        particles = np.matmul(np.linalg.inv(opencv_T_world), particles.T).T[:, :3] / self.global_scale
        return particles

    def load_full_episode(self, episode_idx):
        """
        Load a complete episode for inference/visualization
        
        Args:
            episode_idx: Episode index to load
            
        Returns:
            sampled_object_trajectory: [timesteps, N_sampled_obj, 3]
            sampled_robot_trajectory: [timesteps, N_sampled_robot, 3] 
            full_object_trajectory: [timesteps, N_full_obj, 3]
            object_sample_indices: indices of sampled object particles
            robot_sample_indices: indices of sampled robot particles
        """
        
        # Get HDF5 file handle
        f = self._get_hdf5_file()
        episode_group = f[f'episode_{episode_idx:06d}']
        
        # Get metadata
        n_frames = episode_group.attrs['n_frames']
        n_obj_particles = episode_group.attrs['n_obj_particles'] 
        n_bot_particles = episode_group.attrs['n_bot_particles']
        
        # Load full episode data
        frame_indices = [i for i in range(0,n_frames, self.downsample_rate)]
        full_object_data = torch.from_numpy(episode_group['object'][frame_indices, :, :])  # [time, n_obj, 3]
        full_robot_data = torch.from_numpy(episode_group['robot'][frame_indices, :, :])    # [time, n_bot, 3]
        
        # Get first frame data for FPS sampling (consistent with training)
        first_object = full_object_data[0]  # [n_obj, 3]
        first_robot = full_robot_data[0]    # [n_bot, 3]
        
        # Apply same FPS sampling as training
        object_sample_indices = fps_rad_tensor(first_object, self.fps_radius)
        robot_sample_indices = fps_rad_tensor(first_robot, self.fps_radius)
        
        # Get sampled particle counts
        n_sampled_object = len(object_sample_indices)
        n_sampled_robot = len(robot_sample_indices)
        
        # Extract sampled trajectories using consistent indices
        sampled_object_trajectory = full_object_data[:, object_sample_indices, :]  # [time, sampled_obj, 3]
        sampled_robot_trajectory = full_robot_data[:, robot_sample_indices, :]     # [time, sampled_robot, 3]
        
        # Prepend n_history-1 copies of the first frame for history initialization
        # This allows inference to start with proper history context
        first_obj_frame = sampled_object_trajectory[:1].repeat(self.n_his - 1, 1, 1)  # [(n_his-1), sampled_obj, 3]
        first_robot_frame = sampled_robot_trajectory[:1].repeat(self.n_his - 1, 1, 1)  # [(n_his-1), sampled_robot, 3]
        
        sampled_object_trajectory = torch.cat([first_obj_frame, sampled_object_trajectory], dim=0)
        sampled_robot_trajectory = torch.cat([first_robot_frame, sampled_robot_trajectory], dim=0)
        states = torch.cat([sampled_object_trajectory, sampled_robot_trajectory], dim=1)

        states_delta = torch.zeros(states.shape[0] - 1, states.shape[1], 3)
        states_delta[:, n_sampled_object:] = sampled_robot_trajectory[1:] - sampled_robot_trajectory[:-1]
        
        attrs = torch.zeros(states.shape[0], states.shape[1])
        attrs[:, n_sampled_object:] = 1.0

        particle_num = n_sampled_object + n_sampled_robot

        print(f"Episode {episode_idx}: {n_frames} -> {len(frame_indices)} timesteps (+ {self.n_his-1} history frames padding), "
              f"Objects: {n_obj_particles} -> {n_sampled_object} sampled, "
              f"Robot: {n_bot_particles} -> {n_sampled_robot} sampled")
        
        return states.float(), states_delta.float(), attrs.float(), particle_num

    def __getitem__(self, idx):
        # Calculate which episode and timestep this idx corresponds to
        idx_episode = idx // self.offset + self.epi_st_idx
        idx_timestep = idx % self.offset

        # Get HDF5 file handle (opened lazily per worker)
        f = self._get_hdf5_file()
        episode_group = f[f'episode_{idx_episode:06d}']
                
        # Generate downsampled frame indices
        frame_indices = [idx_timestep + i * self.downsample_rate for i in range(self.n_his + self.n_roll)]
        object_data = torch.from_numpy(episode_group['object'][frame_indices, :, :])  # [time, n_obj, 3]
        robot_data = torch.from_numpy(episode_group['robot'][frame_indices, :, :])    # [time, n_bot, 3]
        
        n_frames = self.n_his + self.n_roll
        
        # Get first frame data for FPS sampling
        first_object = object_data[0]  # [n_obj, 3]
        first_robot = robot_data[0]    # [n_bot, 3]
        
        # FPS sampling using tensor indices directly 
        sampled_object_indices = fps_rad_tensor(first_object, self.fps_radius)
        sampled_robot_indices = fps_rad_tensor(first_robot, self.fps_radius)
        
        n_sampled_object = sampled_object_indices.shape[0]
        n_sampled_robot = sampled_robot_indices.shape[0]
        particle_num = n_sampled_object + n_sampled_robot
                
        # Sample particles using tensor indexing (super efficient!)
        sampled_object_states = object_data[:, sampled_object_indices, :]  # [time, sampled_object, 3]
        sampled_robot_states = robot_data[:, sampled_robot_indices, :]     # [time, sampled_robot, 3]
        
        # Combine sampled particles: [object_particles, robot_particles]
        states = torch.cat([sampled_object_states, sampled_robot_states], dim=1)  # [time, total_sampled, 3]
        if self.add_randomness:
            noise = torch.randn_like(states[:self.n_his]) * self.state_noise
            noise = torch.clamp(noise, -0.015, 0.015)
            states[:self.n_his] = states[:self.n_his] + noise
        
        # Calculate states_delta using tensor operations with new logic
        # For t < n_history - 1: s_delta = s_hist[t+1] - s_hist[t] 
        # For t >= n_history - 1: s_delta = 0 for objects, s_hist[t+1] - s_hist[t] for robot
        states_delta = torch.zeros(n_frames - 1, particle_num, 3)
        
        # For frames 0 to n_his-2: use consecutive frame differences for all particles
        states_delta[:self.n_his - 1] = states[1:self.n_his] - states[:self.n_his - 1]
        
        # For frames n_his-1 to n_frames-2: 
        # - Robot: consecutive frame differences (robot motion is given)
        states_delta[self.n_his - 1:n_frames - 1, n_sampled_object:] = states[self.n_his:n_frames, n_sampled_object:] - states[self.n_his - 1:n_frames - 1, n_sampled_object:]
        
        # Create attrs tensor
        attrs = torch.zeros(n_frames, particle_num)
        attrs[:, n_sampled_object:] = 1.0   # robot particles
        
        # Convert to float tensors
        states = states.float()
        states_delta = states_delta.float()
        attrs = attrs.float()
        
        return states, states_delta, attrs, particle_num

def dataset_test():
    config = load_yaml('config.yaml')

    cam = []
    env = FlexEnv(config)
    env.reset()
    cam.append(env.get_cam_params())
    cam.append(env.get_cam_extrinsics())
    env.close()

    dataset = ParticleDataset(config['train']['data_file'], config, 'train', cam)
    states, states_delta, attrs, particle_num, color_imgs = dataset[0]
    vid = cv2.VideoWriter('dataset_nopusher.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (720, 720))
    for i in range(states.shape[0] - 1):
        img = color_imgs[i]
        obj_pix = pcd2pix(states[i], cam[0])
        next_pix = pcd2pix(states[i] + states_delta[i], cam[0])
        for j in range(obj_pix.shape[0]):
            img = cv2.circle(img.copy(), (int(obj_pix[j, 1]), int(obj_pix[j, 0])), 5, (0, 0, 255), -1)
        for j in range(next_pix.shape[0]):
            img = cv2.arrowedLine(img.copy(), (int(obj_pix[j, 1]), int(obj_pix[j, 0])),
                                  (int(next_pix[j, 1]), int(next_pix[j, 0])), (0, 255, 0), 2)
        vid.write(img)
    vid.release()

def calibrate_res_range():
    config = load_yaml('config.yaml')
    env = FlexEnv(config)
    
    env.init_pos = 'rb_corner'
    env.reset()

    raw_obs = env.render()
    depth = raw_obs[..., -1] / config['dataset']['global_scale']

    depth_fgpcd = depth2fgpcd(depth, (depth < 0.599/0.8), env.get_cam_params()) # [N, 3]
    # sampled_pts, min_particle_r = fps_np(depth_fgpcd, 100)
    # max_particle_den = 1 / (min_particle_r ** 2)
    # print('max_particle_den: %f' % max_particle_den)
    
    env.init_pos = 'extra_small_wkspc_spread'
    env.reset()
    
    raw_obs = env.render()
    depth = raw_obs[..., -1] / config['dataset']['global_scale']
    depth_fgpcd = depth2fgpcd(depth, (depth < 0.599/0.8), env.get_cam_params()) # [N, 3]
    # sampled_pts, max_particle_r = fps_np(depth_fgpcd, 2)
    # min_particle_den = 1 / (max_particle_r ** 2)
    # print('min_particle_den: %f' % min_particle_den)
    

if __name__ == '__main__':
    # dataset_test()
    calibrate_res_range()
