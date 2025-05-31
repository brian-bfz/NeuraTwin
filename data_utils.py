"""
Utility functions for loading and working with HDF5 episode data.
"""

import h5py
import numpy as np
import os
from typing import Tuple, Dict, Optional, List
import glob

class EpisodeLoader:
    """Helper class for loading episode data from HDF5 files"""
    
    def __init__(self, episode_path: str):
        """
        Initialize episode loader
        
        Args:
            episode_path: Path to episode.h5 file or directory containing episode.h5
        """
        if os.path.isdir(episode_path):
            self.episode_file = os.path.join(episode_path, "episode.h5")
        else:
            self.episode_file = episode_path
            
        if not os.path.exists(self.episode_file):
            raise FileNotFoundError(f"Episode file not found: {self.episode_file}")
    
    def load_metadata(self) -> Dict:
        """Load episode metadata"""
        with h5py.File(self.episode_file, 'r') as f:
            if 'metadata' not in f:
                return {}
            
            metadata = {}
            for key, value in f['metadata'].attrs.items():
                metadata[key] = value
            return metadata
    
    def load_object_data(self) -> np.ndarray:
        """Load object trajectory data"""
        with h5py.File(self.episode_file, 'r') as f:
            return f['object/data'][:]
    
    def load_robot_data(self) -> np.ndarray:
        """Load robot trajectory data"""
        with h5py.File(self.episode_file, 'r') as f:
            return f['robot/data'][:]
    
    def load_gaussians_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Load gaussians data if available"""
        with h5py.File(self.episode_file, 'r') as f:
            if 'gaussians' not in f:
                return None
            
            gaussians_data = {}
            gaussians_group = f['gaussians']
            for key in gaussians_group.keys():
                gaussians_data[key] = gaussians_group[key][:]
            return gaussians_data
    
    def load_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load specific frame data"""
        with h5py.File(self.episode_file, 'r') as f:
            object_frame = f['object/data'][frame_idx]
            robot_frame = f['robot/data'][frame_idx]
            return object_frame, robot_frame
    
    def load_frame_range(self, start_frame: int, end_frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load range of frames"""
        with h5py.File(self.episode_file, 'r') as f:
            object_frames = f['object/data'][start_frame:end_frame]
            robot_frames = f['robot/data'][start_frame:end_frame]
            return object_frames, robot_frames
    
    def get_episode_info(self) -> Dict:
        """Get comprehensive episode information"""
        info = self.load_metadata()
        
        with h5py.File(self.episode_file, 'r') as f:
            # Data shapes
            if 'object/data' in f:
                info['object_shape'] = f['object/data'].shape
            if 'robot/data' in f:
                info['robot_shape'] = f['robot/data'].shape
            if 'gaussians' in f and 'xyz' in f['gaussians']:
                info['gaussians_shape'] = f['gaussians/xyz'].shape
        
        # File size
        info['file_size_mb'] = os.path.getsize(self.episode_file) / (1024 * 1024)
        
        return info

class BatchLoader:
    """Helper class for loading multiple episodes"""
    
    def __init__(self, data_dir: str):
        """
        Initialize batch loader
        
        Args:
            data_dir: Directory containing episode subdirectories
        """
        self.data_dir = data_dir
        self.episode_paths = self._find_episodes()
    
    def _find_episodes(self) -> List[str]:
        """Find all episode.h5 files in subdirectories"""
        episode_files = []
        for root, dirs, files in os.walk(self.data_dir):
            if "episode.h5" in files:
                episode_files.append(os.path.join(root, "episode.h5"))
        
        # Sort by episode number if possible
        def extract_episode_num(path):
            try:
                episode_dir = os.path.basename(os.path.dirname(path))
                return int(episode_dir)
            except:
                return 0
        
        episode_files.sort(key=extract_episode_num)
        return episode_files
    
    def get_episode_count(self) -> int:
        """Get number of episodes"""
        return len(self.episode_paths)
    
    def load_episode(self, episode_idx: int) -> EpisodeLoader:
        """Load specific episode"""
        if episode_idx >= len(self.episode_paths):
            raise IndexError(f"Episode index {episode_idx} out of range")
        return EpisodeLoader(self.episode_paths[episode_idx])
    
    def load_all_metadata(self) -> List[Dict]:
        """Load metadata for all episodes"""
        metadata_list = []
        for episode_path in self.episode_paths:
            loader = EpisodeLoader(episode_path)
            metadata_list.append(loader.load_metadata())
        return metadata_list
    
    def get_batch_info(self) -> Dict:
        """Get information about the entire batch"""
        info = {
            'num_episodes': len(self.episode_paths),
            'total_size_mb': 0,
            'episodes': []
        }
        
        for episode_path in self.episode_paths:
            loader = EpisodeLoader(episode_path)
            episode_info = loader.get_episode_info()
            info['episodes'].append(episode_info)
            info['total_size_mb'] += episode_info.get('file_size_mb', 0)
        
        return info

def load_episode_data(episode_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience function to load complete episode data
    
    Returns:
        Tuple of (object_data, robot_data, metadata)
    """
    loader = EpisodeLoader(episode_path)
    object_data = loader.load_object_data()
    robot_data = loader.load_robot_data()
    metadata = loader.load_metadata()
    return object_data, robot_data, metadata

def compare_file_sizes(npz_dir: str, h5_dir: str) -> None:
    """Compare file sizes between NPZ and HDF5 formats"""
    print("File size comparison:")
    print("-" * 50)
    
    # NPZ files
    npz_size = 0
    if os.path.exists(npz_dir):
        for file in glob.glob(os.path.join(npz_dir, "*.npz")):
            npz_size += os.path.getsize(file)
    
    # HDF5 files  
    h5_size = 0
    if os.path.exists(h5_dir):
        for file in glob.glob(os.path.join(h5_dir, "**/*.h5"), recursive=True):
            h5_size += os.path.getsize(file)
    
    print(f"NPZ total size: {npz_size / (1024*1024):.2f} MB")
    print(f"HDF5 total size: {h5_size / (1024*1024):.2f} MB")
    
    if npz_size > 0:
        compression_ratio = npz_size / h5_size if h5_size > 0 else float('inf')
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Space saved: {(npz_size - h5_size) / (1024*1024):.2f} MB ({((npz_size - h5_size) / npz_size * 100):.1f}%)") 