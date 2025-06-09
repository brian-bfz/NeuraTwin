import time
import torch

class EpochTimer:
    """Flexible timer for tracking arbitrary named operations with built-in averaging"""
    
    def __init__(self):
        self.epoch_count = 0
        self.total_timers = {}  # For averaging across epochs
        self.reset()
    
    def reset(self):
        """Reset timers for current epoch"""
        self.timers = {}  # Current epoch timers
        self.active_timers = {}  # Currently running timers
        self.total_time = 0.0
        self.gpu_memory_peak = 0.0
    
    def start_epoch(self):
        """Start timing a new epoch"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.epoch_start = time.perf_counter()
        self.reset()
    
    def end_epoch(self):
        """End current epoch and update totals for averaging"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total_time = time.perf_counter() - self.epoch_start
        if torch.cuda.is_available():
            self.gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # Update totals for averaging
        self.epoch_count += 1
        # Add total time to totals
        self.total_timers['total'] = self.total_timers.get('total', 0.0) + self.total_time
        # Add individual timer values
        for name, time_val in self.timers.items():
            self.total_timers[name] = self.total_timers.get(name, 0.0) + time_val
    
    def start_timer(self, name):
        """Start timing an operation"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active_timers[name] = time.perf_counter()
    
    def end_timer(self, name):
        """End timing an operation"""
        if name not in self.active_timers:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.active_timers[name]
        self.timers[name] = self.timers.get(name, 0.0) + elapsed
        del self.active_timers[name]
    
    def timer(self, name):
        """Get a context manager for timing an operation"""
        return TimerContext(self, name)
    
    def get_summary(self):
        """Get timing summary for current epoch with averages"""
        summary = {
            'total_time': self.total_time,
            'gpu_memory_peak_mb': self.gpu_memory_peak,
        }
        
        # Add current epoch timers
        for name, time_val in self.timers.items():
            summary[f'{name}_time'] = time_val
            summary[f'{name}_pct'] = (time_val / self.total_time) * 100 if self.total_time > 0 else 0
        
        # Add averages
        if self.epoch_count > 0:
            summary['avg_total_time'] = self.total_timers.get('total', 0.0) / self.epoch_count
            summary['avg_gpu_memory_peak_mb'] = self.gpu_memory_peak
            for name, total_time in self.total_timers.items():
                if name != 'total':  # Skip total since we already calculated it
                    avg_time = total_time / self.epoch_count
                    summary[f'avg_{name}_time'] = avg_time
                    summary[f'avg_{name}_pct'] = (avg_time / summary['avg_total_time']) * 100 if summary['avg_total_time'] > 0 else 0
        
        return summary
    
    def get_timing_log(self, epoch):
        """Generate formatted timing log string"""
        summary = self.get_summary()
        
        # Current epoch timing
        log = f'PROFILING [Epoch {epoch}] Total: {summary["total_time"]:.2f}s'
        
        # Add all timers in a consistent order
        timer_order = ['data_loading', 'forward', 'rollout', 'edge', 'backward']
        for name in timer_order:
            if f'{name}_time' in summary:
                time_val = summary[f'{name}_time']
                pct = summary[f'{name}_pct']
                display_name = name.replace('_', ' ').title()
                log += f' | {display_name}: {time_val:.2f}s ({pct:.1f}%)'
        
        # Add any additional timers not in the standard order
        for name in sorted(self.timers.keys()):
            if name not in timer_order:
                time_val = summary[f'{name}_time']
                pct = summary[f'{name}_pct']
                display_name = name.replace('_', ' ').title()
                log += f' | {display_name}: {time_val:.2f}s ({pct:.1f}%)'
        
        log += f' | GPU Mem: {summary["gpu_memory_peak_mb"]:.0f}MB\n'
        
        # Average timing
        if self.epoch_count > 0:
            log += f'Avg Total: {summary["avg_total_time"]:.2f}s'
            for name in timer_order:
                if f'avg_{name}_time' in summary:
                    avg_time = summary[f'avg_{name}_time']
                    display_name = name.replace('_', ' ').title()
                    log += f' | Avg {display_name}: {avg_time:.2f}s'
            
            # Add any additional average timers
            for name in sorted(self.timers.keys()):
                if name not in timer_order and f'avg_{name}_time' in summary:
                    avg_time = summary[f'avg_{name}_time']
                    display_name = name.replace('_', ' ').title()
                    log += f' | Avg {display_name}: {avg_time:.2f}s'
            
            log += f' | Avg GPU Mem: {summary["avg_gpu_memory_peak_mb"]:.0f}MB\n'
        
        return log


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, epoch_timer, name):
        self.epoch_timer = epoch_timer
        self.name = name
    
    def __enter__(self):
        self.epoch_timer.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.epoch_timer.end_timer(self.name)
