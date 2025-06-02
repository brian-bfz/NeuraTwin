import os
import numpy as np
import json
import cv2
import pdb
import pickle
import json
import random
import time
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# from env.flex_env import FlexEnv

from scipy.spatial import KDTree
from utils import load_yaml, set_seed, fps_rad_tensor, fps_np, recenter, opengl2cam, depth2fgpcd, pcd2pix

import matplotlib.pyplot as plt
# from dgl.geometry import farthest_point_sampler

from torch.utils.data import DataLoader

np.seterr(divide='ignore', invalid='ignore')

class ParticleDataset(Dataset):
    def __init__(self, data_dir, config, phase):
        self.config = config

        n_episode = config['dataset']['n_episode']
        n_timestep = config['dataset']['n_timestep']
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
        self.data_dir = data_dir
        self.hdf5_path = os.path.join(data_dir, 'data.h5')

        self.screenHeight = 720
        self.screenWidth = 720
        self.img_channel = 1

        self.add_randomness = config['dataset']['randomness']['use']
        self.state_noise = config['dataset']['randomness']['state_noise'][phase]

        self.fps_radius = config['train']['fps_radius']
        
        # File handle for efficient access - will be opened lazily per worker
        self._hdf5_file = None

    def _get_hdf5_file(self):
        """Lazy loading of HDF5 file handle to work properly with multiprocessing."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file

    def __del__(self):
        """Ensure HDF5 file is properly closed when dataset is destroyed."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()

    def __len__(self):
        return self.n_episode * (self.n_timestep - self.n_his - self.n_roll + 1)
    
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
        full_object_data = torch.from_numpy(episode_group['object'][:, :, :])  # [time, n_obj, 3]
        full_robot_data = torch.from_numpy(episode_group['robot'][:, :, :])    # [time, n_bot, 3]
        
        # Get first frame data for FPS sampling (consistent with training)
        first_object = full_object_data[0]  # [n_obj, 3]
        first_robot = full_robot_data[0]    # [n_bot, 3]
        
        # Apply same FPS sampling as training
        object_sample_indices = fps_rad_tensor(first_object, self.fps_radius)
        robot_sample_indices = fps_rad_tensor(first_robot, self.fps_radius)
        
        # Extract sampled trajectories using consistent indices
        sampled_object_trajectory = full_object_data[:, object_sample_indices, :]  # [time, sampled_obj, 3]
        sampled_robot_trajectory = full_robot_data[:, robot_sample_indices, :]     # [time, sampled_robot, 3]
        
        print(f"Episode {episode_idx}: {n_frames} timesteps, "
              f"Objects: {n_obj_particles} -> {len(object_sample_indices)} sampled, "
              f"Robot: {n_bot_particles} -> {len(robot_sample_indices)} sampled")
        
        return (sampled_object_trajectory.float(), 
                sampled_robot_trajectory.float(), 
                full_object_data.float(),
                object_sample_indices, 
                robot_sample_indices)

    def __getitem__(self, idx):
        # Calculate which episode and timestep this idx corresponds to
        offset = self.n_timestep - self.n_his - self.n_roll + 1
        idx_episode = idx // offset + self.epi_st_idx
        idx_timestep = idx % offset

        # Get HDF5 file handle (opened lazily per worker)
        f = self._get_hdf5_file()
        episode_group = f[f'episode_{idx_episode:06d}']
        
        # Get metadata
        n_frames = episode_group.attrs['n_frames']
        n_obj_particles = episode_group.attrs['n_obj_particles'] 
        n_bot_particles = episode_group.attrs['n_bot_particles']
        
        # Read object and robot datasets for the required time window
        time_end = idx_timestep + self.n_his + self.n_roll
        
        # Load object and robot data as numpy arrays then convert to tensors
        object_data = torch.from_numpy(episode_group['object'][idx_timestep:time_end, :, :])  # [time, n_obj, 3]
        robot_data = torch.from_numpy(episode_group['robot'][idx_timestep:time_end, :, :])    # [time, n_bot, 3]
        
        n_frames = self.n_his + self.n_roll
        
        # Get first frame data for FPS sampling
        first_object = object_data[0]  # [n_obj, 3]
        first_robot = robot_data[0]    # [n_bot, 3]
        
        # FPS sampling using tensor indices directly (no cdist needed!)
        sampled_object_indices = fps_rad_tensor(first_object, self.fps_radius)
        sampled_robot_indices = fps_rad_tensor(first_robot, self.fps_radius)
        
        n_sampled_object = sampled_object_indices.shape[0]
        n_sampled_robot = sampled_robot_indices.shape[0]
        particle_num = n_sampled_object + n_sampled_robot
        
        # Create masks for particle types
        object_mask = torch.zeros(particle_num, dtype=torch.bool)
        robot_mask = torch.zeros(particle_num, dtype=torch.bool)
        object_mask[:n_sampled_object] = True
        robot_mask[n_sampled_object:] = True
        
        # Sample particles using tensor indexing (super efficient!)
        sampled_object_states = object_data[:, sampled_object_indices, :]  # [time, sampled_object, 3]
        sampled_robot_states = robot_data[:, sampled_robot_indices, :]     # [time, sampled_robot, 3]
        
        # Combine sampled particles: [object_particles, robot_particles]
        states = torch.cat([sampled_object_states, sampled_robot_states], dim=1)  # [time, total_sampled, 3]
        if self.add_randomness:
            states[:self.n_his] = states[:self.n_his] + torch.randn_like(states[:self.n_his]) * self.state_noise
        
        # Calculate states_delta using tensor operations
        states_delta = torch.zeros(n_frames - 1, particle_num, 3)
        
        # Efficient tensor computation for consecutive frame differences
        next_states = states[1:]      # [time-1, particles, 3]
        current_states = states[:-1]  # [time-1, particles, 3]
        
        # For object particles: delta = 0 (implicitly, since states_delta is zero-initialized)
        # For robot particles: delta = next - current
        states_delta[:, robot_mask, :] = next_states[:, robot_mask, :] - current_states[:, robot_mask, :]
        
        # Create attrs tensor
        attrs = torch.zeros(n_frames, particle_num)
        attrs[:, robot_mask] = 1.0   # robot particles
        
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

    dataset = ParticleDataset(config['train']['data_root'], config, 'train', cam)
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
