import time
import torch

class EpochTimer:
    """Simple timer for tracking epoch-level performance"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data_loading_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.total_time = 0.0
        self.gpu_memory_peak = 0.0
        self.edge_time = 0.0
    
    def start_epoch(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.epoch_start = time.perf_counter()
        self.reset()
    
    def end_epoch(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total_time = time.perf_counter() - self.epoch_start
        if torch.cuda.is_available():
            self.gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    def time_data_loading(self):
        return DataLoadingTimer(self)
    
    def time_forward(self):
        return ForwardTimer(self)
    
    def time_backward(self):
        return BackwardTimer(self)
    
    def get_summary(self):
        return {
            'total_time': self.total_time,
            'data_loading_time': self.data_loading_time,
            'forward_time': self.forward_time,
            'backward_time': self.backward_time,
            'gpu_memory_peak_mb': self.gpu_memory_peak,
            'edge_time': self.edge_time,
            'data_loading_pct': (self.data_loading_time / self.total_time) * 100 if self.total_time > 0 else 0,
            'forward_pct': (self.forward_time / self.total_time) * 100 if self.total_time > 0 else 0,
            'backward_pct': (self.backward_time / self.total_time) * 100 if self.total_time > 0 else 0,
            'edge_pct': (self.edge_time / self.total_time) * 100 if self.total_time > 0 else 0,
        }

class DataLoadingTimer:
    def __init__(self, epoch_timer):
        self.epoch_timer = epoch_timer
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.epoch_timer.data_loading_time += time.perf_counter() - self.start

class ForwardTimer:
    def __init__(self, epoch_timer):
        self.epoch_timer = epoch_timer
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.epoch_timer.forward_time += time.perf_counter() - self.start

class BackwardTimer:
    def __init__(self, epoch_timer):
        self.epoch_timer = epoch_timer
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.epoch_timer.backward_time += time.perf_counter() - self.start
