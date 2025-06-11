import torch

class Rollout:
    """
    GNN-based predictor for autoregressive inference on particle dynamics.
    Maintains history buffers and provides forward prediction functionality.
    Always forces robot positions to ground truth and works with batched data.
    Assume tensors are already on the correct device and with gradients enabled / disabled.
    """
    
    def __init__(self, model, config, initial_states, initial_deltas, initial_attrs, particle_num, topological_edges, first_states, epoch_timer=None):
        """
        Initialize the predictor with trained model and initial history data.
        
        Args:
            model: trained GNN model
            config: dict - training configuration
            initial_states: [batch, particles, 3] - initial position
            initial_deltas: [batch, n_history - 1, particles, 3] - initial velocity history
            initial_attrs: [batch, particles] - initial attribute
            particle_num: [batch] - total number of particles per batch
            topological_edges: [batch, particles, particles] - adjacency matrix of topological edges
            first_states: [batch, particles, 3] - first frame states for topological edge computations
            epoch_timer: optional EpochTimer for profiling edge construction time
        """
        self.model = model
        self.config = config
        self.epoch_timer = epoch_timer
        
        # Extract model parameters
        self.n_history = config['train']['n_history']
        
        # Initialize history buffers
        self.s_cur = initial_states    # [batch, particles, 3]
        if initial_deltas.shape[1] == 0:
            self.s_delta = torch.zeros_like(initial_states.unsqueeze(1))
        else:
            filler = torch.zeros_like(initial_deltas[:, :1, :, :])    #  first frame gets discarded by _update_deltas 
            self.s_delta = torch.cat([filler, initial_deltas], dim=1) # [batch, n_history, particles, 3]
        self.a_cur = initial_attrs     # [batch, particles]
        self.particle_nums = particle_num # [batch]
        self.topological_edges = topological_edges  # [batch, particles, particles]
        self.first_states = first_states  # [batch, particles, 3]
        
    def forward(self, next_delta):
        """
        Predict next frame states and update history buffers.
        
        Args:
            next_delta: [batch, particles, 3] - next frame robot velocity
            
        Returns:
            predicted_states: [batch, particles, 3] - predicted next frame positions
        """
        # Update delta buffer with the robot's motions
        self.s_delta = torch.roll(self.s_delta, shifts=1, dims=1)
        self.s_delta[:, 0, :, :] = next_delta
            
        # Predict next state using current history
        predicted_states = self.model.predict_one_step(
            self.a_cur,        # [batch, particles]
            self.s_cur,        # [batch, particles, 3]
            self.s_delta,       # [batch, n_history, particles, 3]
            topological_edges=self.topological_edges,  # [batch, particles, particles]
            particle_nums=self.particle_nums,  # [batch]
            epoch_timer=self.epoch_timer,  # optional timer for profiling
            first_states=self.first_states  # [batch, particles, 3]
        )  # [batch, particles, 3]

        # Always force robot positions to ground truth
        robot_mask = (self.a_cur == 1)  # [batch, particles]
        predicted_states[robot_mask] = next_delta[robot_mask] + self.s_cur[robot_mask]

        # Update delta buffer with predicted particle motion
        self.s_delta[:, -1, :, :] = predicted_states - self.s_cur
        self.s_cur = predicted_states
        # self.a_cur = self.a_cur
            
        return predicted_states  # [batch, particles, 3]