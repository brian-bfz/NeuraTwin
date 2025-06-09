import torch
import torch.nn as nn
import time
from ..utils import construct_edges_with_attrs

# ============================================================================
# NEURAL NETWORK MODULES
# ============================================================================

class RelationEncoder(nn.Module):
    """
    Encodes relation features between connected particles in the graph.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size: int - dimension of input relation features
            hidden_size: int - dimension of hidden layers
            output_size: int - dimension of output relation embeddings
        """
        super(RelationEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_relations, input_size] - relation features
            
        Returns:
            [batch_size, n_relations, output_size] - encoded relation embeddings
        """
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class ParticleEncoder(nn.Module):
    """
    Encodes particle features including position history and attributes.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size: int - dimension of input particle features
            hidden_size: int - dimension of hidden layers  
            output_size: int - dimension of output particle embeddings
        """
        super(ParticleEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_particles, input_size] - particle features
            
        Returns:
            [batch_size, n_particles, output_size] - encoded particle embeddings
        """
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class Propagator(nn.Module):
    """
    Propagates information through the graph using message passing.
    """
    
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size: int - dimension of input features
            output_size: int - dimension of output features
        """
        super(Propagator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        """
        Args:
            x: [batch_size, n_relations/n_particles, input_size] - input features
            residual: optional residual connection tensor
            
        Returns:
            [batch_size, n_relations/n_particles, output_size] - propagated features
        """
        B, N, D = x.size()
        x = self.linear(x.view(B * N, D))

        if residual is None:
            x = self.relu(x)
        else:
            x = self.relu(x + residual.view(B * N, self.output_size))

        return x.view(B, N, self.output_size)


class ParticlePredictor(nn.Module):
    """
    Predicts particle positions based on propagated graph features.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size: int - dimension of input particle features
            hidden_size: int - dimension of hidden layers
            output_size: int - dimension of output predictions (typically 3 for 3D positions)
        """
        super(ParticlePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_particles, input_size] - particle features
            
        Returns:
            [batch_size, n_particles, output_size] - predicted particle positions
        """
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)


# ============================================================================
# MAIN GNN DYNAMICS MODULE
# ============================================================================

class PropModuleDiffDen(nn.Module):
    """
    Main GNN module for particle dynamics prediction with history-aware processing.
    Handles multiple history timesteps and different particle types (objects vs robots/tools).
    """
    
    def __init__(self, config, use_gpu=False):
        """
        Args:
            config: dict - configuration containing model hyperparameters
            use_gpu: bool - whether to use GPU acceleration
        """
        super(PropModuleDiffDen, self).__init__()

        self.config = config
        nf_effect = config['train']['particle']['nf_effect']
        self.nf_effect = nf_effect
        self.add_delta = config['train']['particle']['add_delta']
        
        # History length for temporal modeling
        self.n_history = config['train']['n_history']
        self.use_gpu = use_gpu

        # Particle encoder:
        # Input: displacement (3 * n_history) + attributes (1)
        self.particle_encoder = ParticleEncoder(
            3 * self.n_history + 1, nf_effect, nf_effect)

        # Relation encoder:
        # Input: attributes of both particles (2) + position difference (3) + edge attributes (1)
        self.relation_encoder = RelationEncoder(
            2 + 3 + 1, nf_effect, nf_effect)

        # Propagators for message passing
        self.particle_propagator = Propagator(
            2 * nf_effect, nf_effect)  # particle encoding + aggregated relation effects

        self.relation_propagator = Propagator(
            nf_effect + 2 * nf_effect, nf_effect)  # relation encoding + sender/receiver effects

        # Final position predictor
        self.particle_predictor = ParticlePredictor(
            nf_effect, nf_effect, 3)

    def forward(self, a_cur, s_cur, s_delta, Rr, Rs, edge_attrs, verbose=False):
        """
        Forward pass of the GNN dynamics model.
        
        Args:
            a_cur: (B, particle_num) - current particle attributes
            s_cur: (B, particle_num, 3) - current positions  
            s_delta: (B, particle_num, 3) - particle displacements over history
            Rr: (B, rel_num, particle_num) - receiver matrix for graph edges
            Rs: (B, rel_num, particle_num) - sender matrix for graph edges
            edge_attrs: (B, rel_num, 1) - edge attributes (1 for topological, 0 for collision)
            verbose: bool - whether to print debug information
            
        Returns:
            (B, particle_num, 3) - predicted next particle positions
        """
        B, N = a_cur.size()
        _, rel_num, _ = Rr.size()
        nf_effect = self.nf_effect
        pstep = 3  # Number of message passing steps

        # Convert from data format (B x time x particles) to model format (B x particles x time)
        s_delta = s_delta.transpose(1, 2)  # B x particle_num x n_history x 3

        # Flatten displacement history for particle encoder
        s_delta_flat = s_delta.reshape(B, N, -1)  # B x particle_num x (3 * n_history)

        Rr_t = Rr.transpose(1, 2) # TODO: add .continuous()? # B x particle_num x rel_num
        
        # Compute relation features using edge matrices
        a_cur_r = Rr.bmm(a_cur[..., None])  # B x rel_num x 1 (receiver attributes)
        a_cur_s = Rs.bmm(a_cur[..., None])  # B x rel_num x 1 (sender attributes)
        s_cur_r = Rr.bmm(s_cur)  # B x rel_num x 3 (receiver positions)
        s_cur_s = Rs.bmm(s_cur)  # B x rel_num x 3 (sender positions)

        # Encode particle features (history-aware)
        particle_encode = self.particle_encoder(
            torch.cat([s_delta_flat, a_cur[..., None]], 2))  # B x particle_num x nf_effect
        particle_effect = particle_encode

        # Encode relation features (current frame only)
        relation_encode = self.relation_encoder(
            torch.cat([a_cur_r, a_cur_s, s_cur_r - s_cur_s, edge_attrs], 2))  # B x rel_num x nf_effect

        # Message passing iterations
        for i in range(pstep):
            # Aggregate particle effects at relation endpoints
            effect_r = Rr.bmm(particle_effect)  # B x rel_num x nf_effect
            effect_s = Rs.bmm(particle_effect)  # B x rel_num x nf_effect
            
            # Update relation effects
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2))  # B x rel_num x nf_effect

            # Aggregate relation effects back to particles
            effect_rel_agg = Rr_t.bmm(effect_rel)  # B x particle_num x nf_effect
            
            # Update particle effects with residual connection
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2),
                residual=particle_effect)

        # Predict position changes
        particle_pred = self.particle_predictor(particle_effect)  # B x particle_num x 3

        # Add residual connection to current positions
        return particle_pred + s_cur

class PropNetDiffDenModel(nn.Module):
    """
    Complete GNN-based particle dynamics model with edge construction and prediction.
    Handles full pipeline from raw particle states to next-step predictions.
    """

    def __init__(self, config, use_gpu=False):
        """
        Args:
            config: dict - configuration containing model parameters
            use_gpu: bool - whether to use GPU acceleration
        """
        super(PropNetDiffDenModel, self).__init__()

        self.config = config
        self.adj_thresh = config['train']['particle']['adj_thresh']
        self.connect_tools_all = config['train']['particle']['connect_tools_all']
        self.topk = config['train']['particle']['topk']
        self.model = PropModuleDiffDen(config, use_gpu)

    def predict_one_step(self, a_cur, s_cur, s_delta, topological_edges=None, particle_nums=None):
        """
        Predict particle positions one step into the future.
        
        Args:
            a_cur: (B, particle_num) - current particle attributes
            s_cur: (B, particle_num, 3) - current positions  
            s_delta: (B, n_history, particle_num, 3) - particle displacements over history
                    For t < n_history-1: consecutive frame differences
                    For t = n_history-1: 0 for objects, actual motion for robots
            topological_edges: (B, particle_num, particle_num) - adjacency matrix of topological edges
            particle_nums: (B,) - number of valid particles per batch sample
            
        Returns:
            (B, particle_num, 3) - predicted next particle positions
        """
        assert type(a_cur) == torch.Tensor
        assert type(s_cur) == torch.Tensor  
        assert type(s_delta) == torch.Tensor
        assert a_cur.shape == s_cur.shape[:2]
        assert s_cur.shape == s_delta[:, -1].shape

        B, N = a_cur.size()

        # Create batch masks for valid particles and tools
        if particle_nums is not None:
            # Create mask for valid particles
            mask = torch.zeros((B, N), dtype=torch.bool, device=s_cur.device)
            for b in range(B):
                n_particles = particle_nums[b].item()
                mask[b, :n_particles] = True
        else:
            # All particles are valid if particle_nums not provided
            mask = torch.ones((B, N), dtype=torch.bool, device=s_cur.device)
        
        # Create tool mask (tool particles have attr=1, objects have attr=0)
        tool_mask = (a_cur > 0.5) & mask
                
        # Construct graph edges based on current particle positions
        Rr_batch, Rs_batch, edge_attrs = construct_edges_with_attrs(
            s_cur, 
            self.adj_thresh, 
            mask, 
            tool_mask, 
            topk=self.topk,
            connect_tools_all=self.connect_tools_all,
            topological_edges=topological_edges
        )

        # Forward pass through GNN
        s_pred = self.model.forward(a_cur, s_cur, s_delta, Rr_batch, Rs_batch, edge_attrs)

        return s_pred
