import torch

class Rollout:
    """
    GNN-based predictor for autoregressive inference on particle dynamics.
    Maintains history buffers and provides forward prediction functionality.
    Always forces robot positions to ground truth and works with batched data.
    Assume tensors are already on the correct device and with gradients enabled / disabled.
    """
    
    def __init__(self, model, config, initial_states, initial_deltas, initial_attrs, particle_num):
        """
        Initialize the predictor with trained model and initial history data.
        
        Args:
            model: trained GNN model
            config: dict - training configuration
            initial_states: [batch, n_history, particles, 3] - initial state history
            initial_deltas: [batch, n_history - 1, particles, 3] - initial delta history
            initial_attrs: [batch, n_history, particles] - initial attribute history
            particle_num: [batch] - total number of particles per batch
        """
        self.model = model
        self.config = config
        
        # Extract model parameters
        self.n_history = config['train']['n_history']
        
        # Initialize history buffers
        self.s_hist = initial_states    # [batch, n_history, particles, 3]
        filler = torch.zeros_like(initial_deltas[:, :1, :, :])    #  first frame gets discarded by _update_deltas 
        self.s_delta = torch.cat([filler, initial_deltas], dim=1) # [batch, n_history, particles, 3]
        self.a_hist = initial_attrs     # [batch, n_history, particles]
        self.particle_nums = particle_num # [batch]
        
    def forward(self, next_delta, next_attrs):
        """
        Predict next frame states and update history buffers.
        
        Args:
            next_delta: [batch, particles, 3] - next frame delta (robot motion)
            next_attrs: [batch, particles] - next frame attributes (for history update only)
            
        Returns:
            predicted_states: [batch, particles, 3] - predicted next frame states
        """
        # Update delta buffer with the robot's motions
        self._update_deltas(next_delta)
            
        # Predict next state using current history
        predicted_states = self.model.predict_one_step(
            self.a_hist,        # [batch, n_history, particles]
            self.s_hist,        # [batch, n_history, particles, 3]
            self.s_delta,       # [batch, n_history, particles, 3]
            self.particle_nums  # [batch]
        )  # [batch, particles, 3]
            
        # Always force robot positions to ground truth
        robot_mask = (next_attrs == 1)  # [batch, particles]
        predicted_states[robot_mask] = next_delta[robot_mask] + self.s_hist[:, -1, :, :][robot_mask]

        # Update history buffers for next iteration
        self._update_history(predicted_states, next_attrs)
            
        return predicted_states  # [batch, particles, 3]
    
    def _update_deltas(self, next_delta):
        """
        Update delta with the robot's motions.

        Args:
            next_delta: [batch, particles, 3] - next frame robot motion
        """
        # Slide windows: remove oldest, add new
        self.s_delta = torch.cat([
            self.s_delta[:, 1:, :, :],  # Remove first frame
            next_delta.unsqueeze(1)     # Add new delta [batch, 1, particles, 3]
        ], dim=1)

    def _update_history(self, predicted_states, next_attrs):
        """
        Update sliding window history with new predictions.
        
        Args:
            predicted_states: [batch, particles, 3] - predicted next frame states
            next_attrs: [batch, particles] - next frame attributes
        """
        # Update delta buffer with predicted particle motion
        self.s_delta[:, -1, :, :] = predicted_states - self.s_hist[:, -1, :, :]
        
        # Slide windows: remove oldest, add new        
        self.s_hist = torch.cat([
            self.s_hist[:, 1:, :, :],      # Remove first frame
            predicted_states.unsqueeze(1)  # Add prediction [batch, 1, particles, 3]
        ], dim=1)
        
        self.a_hist = torch.cat([
            self.a_hist[:, 1:, :],     # Remove first frame
            next_attrs.unsqueeze(1)    # Add new attrs [batch, 1, particles]
        ], dim=1)

