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
            initial_states: [batch, particles, 3] - initial position
            initial_deltas: [batch, n_history - 1, particles, 3] - initial velocity history
            initial_attrs: [batch, particles] - initial attribute
            particle_num: [batch] - total number of particles per batch
        """
        self.model = model
        self.config = config
        
        # Extract model parameters
        self.n_history = config['train']['n_history']
        
        # Initialize history buffers
        self.s_cur = initial_states    # [batch, particles, 3]
        filler = torch.zeros_like(initial_deltas[:, :1, :, :])    #  first frame gets discarded by _update_deltas 
        self.s_delta = torch.cat([filler, initial_deltas], dim=1) # [batch, n_history, particles, 3]
        self.a_cur = initial_attrs     # [batch, particles]
        self.particle_nums = particle_num # [batch]
        
    def forward(self, next_delta):
        """
        Predict next frame states and update history buffers.
        
        Args:
            next_delta: [batch, particles, 3] - next frame robot velocity
            
        Returns:
            predicted_states: [batch, particles, 3] - predicted next frame positions
        """
        # Update delta buffer with the robot's motions
        self._update_robot_delta(next_delta)
            
        # Predict next state using current history
        predicted_states = self.model.predict_one_step(
            self.a_cur,        # [batch, particles]
            self.s_cur,        # [batch, particles, 3]
            self.s_delta,       # [batch, n_history, particles, 3]
            self.particle_nums  # [batch]
        )  # [batch, particles, 3]
            
        next_attrs = self.a_cur # TODO: determine attributes with predicted positions

        # Always force robot positions to ground truth
        robot_mask = (next_attrs == 1)  # [batch, particles]
        predicted_states[robot_mask] = next_delta[robot_mask] + self.s_cur[robot_mask]

        # Update history buffers for next iteration
        self._update_history(predicted_states, next_attrs)
            
        return predicted_states  # [batch, particles, 3]
    
    def _update_robot_delta(self, next_delta):
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
            predicted_states: [batch, particles, 3] - predicted next frame positions
            next_attrs: [batch, particles] - next frame attributes
        """
        # Update delta buffer with predicted particle motion
        self.s_delta[:, -1, :, :] = predicted_states - self.s_cur
        
        self.a_cur = next_attrs
