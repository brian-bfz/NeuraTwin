import numpy as np
import cv2
import h5py
import math

import torch
from torch.utils.data import Dataset

# from env.flex_env import FlexEnv
from ..utils import load_yaml, fps_rad_tensor, depth2fgpcd, pcd2pix


np.seterr(divide='ignore', invalid='ignore')

# ============================================================================
# MAIN DATASET CLASS
# ============================================================================

class ParticleDataset(Dataset):
    """
    PyTorch dataset for particle dynamics with temporal history and autoregressive rollout.
    Handles HDF5 data loading, FPS sampling, and temporal data structuring for GNN training.
    """
    
    def __init__(self, data_file, config, phase):
        """
        Initialize particle dataset with configuration and train/validation split.
        
        Args:
            data_file: str - path to HDF5 dataset file
            config: dict - training configuration containing dataset parameters
            phase: str - 'train' or 'valid' to determine data split
        """
        self.config = config

        # Extract dataset parameters
        n_episode = config['dataset']['n_episode']
        n_timestep = config['dataset']['n_timestep']
        self.downsample_rate = config['dataset']['downsample_rate']
        self.global_scale = config['dataset']['global_scale']

        # Train/validation split
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
        

        self.screenHeight = 720
        self.screenWidth = 720
        self.img_channel = 1

        self.add_randomness = config['dataset']['randomness']['use']
        self.state_noise = config['dataset']['randomness']['state_noise'][phase]

        self.fps_radius = config['train']['particle']['fps_radius']
        
        self._hdf5_file = None
        self.hdf5_path = data_file

        # Calculate valid starting timesteps (must have enough frames for history + rollout)
        self.offset = self.n_timestep - (self.n_his + self.n_roll - 1) * self.downsample_rate

    def _get_hdf5_file(self):
        """
        Lazy loading of HDF5 file handle to work properly with multiprocessing.
        Each worker process gets its own file handle to avoid conflicts.
        
        Returns:
            h5py.File - HDF5 file handle in read mode
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        return self._hdf5_file

    def __del__(self):
        """Ensure HDF5 file is properly closed when dataset is destroyed."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()

    def __len__(self):
        """
        Total number of valid training samples.
        Each episode contributes multiple samples based on valid starting timesteps.
        
        Returns:
            int - total number of samples in dataset
        """
        return self.n_episode * self.offset
    
    def read_particles(self, particles_path):
        """
        Legacy method for loading particle data from numpy files.
        Currently unused but kept for potential future compatibility.
        
        Args:
            particles_path: str - path to particle numpy file
            
        Returns:
            particles: ndarray - transformed particle positions
        """
        particles = np.load(particles_path).reshape(-1, 4)
        particles[:, 3] = 1.0
        opencv_T_opengl = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        opencv_T_world = np.matmul(np.linalg.inv(self.cam_extrinsic), opencv_T_opengl)
        particles = np.matmul(np.linalg.inv(opencv_T_world), particles.T).T[:, :3] / self.global_scale
        return particles

    def load_full_episode(self, episode_idx):
        """
        Load complete episode for inference/visualization with proper history initialization.
        
        Args:
            episode_idx: int - episode index to load
            
        Returns:
            states: [timesteps, particles, 3] - particle positions with history padding
            states_delta: [timesteps-1, particles, 3] - particle displacements
            attrs: [timesteps, particles] - particle attributes (0=object, 1=robot)
            particle_num: int - total number of sampled particles
            topological_edges: [particles, particles] - adjacency matrix for topological edges (or None)
            first_states: [particles, 3] - first frame states for topological edge computations
        """
        # Get HDF5 file handle
        f = self._get_hdf5_file()
        episode_group = f[f'episode_{episode_idx:06d}']
        
        # Get metadata
        n_frames = episode_group.attrs['n_frames']
        frame_indices = [i for i in range(0, n_frames, self.downsample_rate)]
            
        object_data = torch.from_numpy(episode_group['object'][frame_indices, :, :])  # [time, n_object, 3]
        robot_data = torch.from_numpy(episode_group['robot'][frame_indices, :, :])    # [time, n_robot, 3]
            
        n_object = object_data.shape[1]
        n_robot = robot_data.shape[1]
        particle_num = n_object + n_robot
            
        # Prepend history padding: repeat first frame (n_history-1) times
        # This enables inference to start with proper temporal context
        first_obj_frame = object_data[:1].repeat(self.n_his - 1, 1, 1)
        first_bot_frame = robot_data[:1].repeat(self.n_his - 1, 1, 1)
        
        obj_trajectory = torch.cat([first_obj_frame, object_data], dim=0)
        bot_trajectory = torch.cat([first_bot_frame, robot_data], dim=0)
        
        # Combine object and robot particles (objects first, then robots)
        states = torch.cat([obj_trajectory, bot_trajectory], dim=1)

        # Store first frame states for topological edge computations
        first_states = states[0]  # [particle_num, 3]

        states_delta = torch.zeros(states.shape[0] - 1, states.shape[1], 3)
        # Only robots get actual deltas
        states_delta[:, n_object:] = bot_trajectory[1:] - bot_trajectory[:-1]
        
        # Create attribute tensor (0=object, 1=robot)
        attrs = torch.zeros(states.shape[0], states.shape[1])
        attrs[:, n_object:] = 1.0

        topological_edges = torch.zeros(particle_num, particle_num)
        # Load topological edges if they exist and are enabled
        if self.config['train']['edges']['topological']['enabled'] and 'object_edges' in episode_group:
            object_edges = torch.from_numpy(episode_group['object_edges'][:])  # [n_object, n_object]
                
            # Create full topological edges matrix for all particles
            topological_edges[:n_object, :n_object] = object_edges

        print(f"Episode {episode_idx}: {len(frame_indices)} timesteps (+ {self.n_his-1} history frames padding), "
              f"Objects: {n_object} sampled, Robots: {n_robot} sampled")
        
        return states.float(), states_delta.float(), attrs.float(), particle_num, topological_edges.float(), first_states.float()

    def __getitem__(self, idx):
        """
        Get single training sample with temporal history and rollout structure.
        Loads pre-sampled trajectories and topological edges from preprocessed data.
        
        Args:
            idx: int - dataset index
            
        Returns:
            states: [time, particles, 3] - particle positions over history + rollout
            states_delta: [time-1, particles, 3] - particle displacements with temporal logic
            attrs: [time, particles] - particle attributes
            particle_num: int - number of particles in this sample
            topological_edges: [particles, particles] - adjacency matrix for topological edges
            first_states: [particles, 3] - first frame states for topological edge computations
        """
        # Calculate which episode and timestep this index corresponds to
        idx_episode = idx // self.offset + self.epi_st_idx
        idx_timestep = idx % self.offset

        # Get HDF5 file handle
        f = self._get_hdf5_file()
        episode_group = f[f'episode_{idx_episode:06d}']

        # Store first frame states for topological edge computations
        first_object = torch.from_numpy(episode_group['object'][0, :, :])  # [n_obj, 3]
        first_robot = torch.from_numpy(episode_group['robot'][0, :, :])    # [n_bot, 3]
        first_states = torch.cat([first_object, first_robot], dim=0)  # [particle_num, 3]
        
        # Generate downsampled frame indices for history + rollout
        n_frames = self.n_his + self.n_roll
        frame_indices = [idx_timestep + i * self.downsample_rate for i in range(n_frames)]
        object_data = torch.from_numpy(episode_group['object'][frame_indices, :, :])  # [n_frames, n_obj, 3]
        robot_data = torch.from_numpy(episode_group['robot'][frame_indices, :, :])    # [n_frames, n_bot, 3]
        
        n_object = object_data.shape[1]
        n_robot = robot_data.shape[1]
        particle_num = n_object + n_robot
        
        # Combine sampled particles: [object_particles, robot_particles]
        states = torch.cat([object_data, robot_data], dim=1)  # [n_frames, particle_num, 3]
        
        # Data augmentation: 
        # Apply random 2D rotation
        states = self._apply_random_rotation(states)        
        # Apply noise
        if self.add_randomness:
            noise = torch.randn_like(states[:self.n_his]) * self.state_noise
            noise = torch.clamp(noise, -0.015, 0.015)
            states[:self.n_his] = states[:self.n_his] + noise
        
        # Calculate displacement deltas:
        # - For t < n_history-1: consecutive frame differences for all particles
        # - For t >= n_history-1: 0 for objects, actual motion for robots
        states_delta = torch.zeros(n_frames - 1, particle_num, 3)
        
        # History frames: use consecutive differences for all particles
        states_delta[:self.n_his - 1] = states[1:self.n_his] - states[:self.n_his - 1]
        
        # Future frames: only robots get motion deltas
        states_delta[self.n_his - 1:n_frames - 1, n_object:] = states[self.n_his:n_frames, n_object:] - states[self.n_his - 1:n_frames - 1, n_object:]
        
        # Create particle attribute tensor (0=object, 1=robot)
        attrs = torch.zeros(n_frames, particle_num)
        attrs[:, n_object:] = 1.0   # Robot particles
                
        # Create full topological edges matrix for all particles
        topological_edges = torch.zeros(particle_num, particle_num)
        if self.config['train']['edges']['topological']['enabled']:
            # Load topological edges for object particles (computed from first frame)
            object_edges = torch.from_numpy(episode_group['object_edges'][:])  # [n_object, n_object]

            # Object has topological edges, robot doesn't
            topological_edges[:n_object, :n_object] = object_edges
        
        return states.float(), states_delta.float(), attrs.float(), particle_num, topological_edges.float(), first_states.float()

    def _apply_random_rotation(self, states):
        """
        Apply random 2D rotation (about z-axis) to particle positions only.
        
        Args:
            states: [time, particles, 3] - particle positions
            
        Returns:
            states_rotated: [time, particles, 3] - rotated particle positions
        """
        # Generate random rotation angle between 0 and 2π
        angle = torch.rand(1).item() * 2 * math.pi
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        # Create 2D rotation matrix (for x-y plane)
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle, 0],
            [sin_angle,  cos_angle, 0],
            [0,          0,         1]
        ], dtype=states.dtype)
        
        # Apply rotation to states: [time, particles, 3] @ [3, 3] -> [time, particles, 3]
        states_rotated = torch.matmul(states, rotation_matrix)
        
        return states_rotated

# ============================================================================
# LEGACY TESTING AND CALIBRATION FUNCTIONS
# ============================================================================

def dataset_test():
    """
    Legacy testing function for dataset visualization.
    Creates video showing particle positions and motion vectors.
    """
    config = load_yaml('config.yaml')

    cam = []
    env = FlexEnv(config)
    env.reset()
    cam.append(env.get_cam_params())
    cam.append(env.get_cam_extrinsics())
    env.close()

    dataset = ParticleDataset(config['dataset']['file'], config, 'train', cam)
    states, states_delta, attrs, particle_num, color_imgs = dataset[0]
    
    # Create visualization video
    vid = cv2.VideoWriter('dataset_nopusher.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (720, 720))
    for i in range(states.shape[0] - 1):
        img = color_imgs[i]
        obj_pix = pcd2pix(states[i], cam[0])
        next_pix = pcd2pix(states[i] + states_delta[i], cam[0])
        
        # Draw particles
        for j in range(obj_pix.shape[0]):
            img = cv2.circle(img.copy(), (int(obj_pix[j, 1]), int(obj_pix[j, 0])), 5, (0, 0, 255), -1)
        
        # Draw motion vectors
        for j in range(next_pix.shape[0]):
            img = cv2.arrowedLine(img.copy(), (int(obj_pix[j, 1]), int(obj_pix[j, 0])),
                                  (int(next_pix[j, 1]), int(next_pix[j, 0])), (0, 255, 0), 2)
        vid.write(img)
    vid.release()

def calibrate_res_range():
    """
    Legacy function for calibrating FPS sampling resolution range.
    Determines optimal particle density parameters.
    """
    config = load_yaml('config.yaml')
    env = FlexEnv(config)
    
    # Test with corner initialization
    env.init_pos = 'rb_corner'
    env.reset()

    raw_obs = env.render()
    depth = raw_obs[..., -1] / config['dataset']['global_scale']

    depth_fgpcd = depth2fgpcd(depth, (depth < 0.599/0.8), env.get_cam_params())
    
    # Test with spread initialization
    env.init_pos = 'extra_small_wkspc_spread'
    env.reset()
    
    raw_obs = env.render()
    depth = raw_obs[..., -1] / config['dataset']['global_scale']
    depth_fgpcd = depth2fgpcd(depth, (depth < 0.599/0.8), env.get_cam_params())

if __name__ == '__main__':
    # dataset_test()
    calibrate_res_range()
