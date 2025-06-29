import torch
import torch.nn as nn
import time
from ..utils import construct_collision_edges, construct_topological_edges

# ============================================================================
# NEURAL NETWORK MODULES
# ============================================================================

class RelationEncoder(nn.Module):
    """
    Encodes relation features between connected particles in the graph.
    """
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        """
        Args:
            input_size: int - dimension of input relation features
            hidden_size: int - dimension of hidden layers
            output_size: int - dimension of output relation embeddings
            dropout_rate: float - percentage of dense layer neurons to drop
        """
        super(RelationEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_rate),
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
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
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
            nn.Dropout(dropout_rate),
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
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
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
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_particles, input_size] - particle features
            
        Returns:
            [batch_size, n_particles, output_size] - predicted particle positions
        """
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.dropout(self.linear_0(x))))
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
        
        # History length for temporal modeling
        self.n_history = config['train']['n_history']
        self.use_gpu = use_gpu
        self.dropout_rate = config['train']['dropout_rate']

        # Particle encoder:
        # Input: displacement (3 * n_history) + attributes (1) + z coordinate (1)
        self.particle_encoder = ParticleEncoder(
            3 * self.n_history + 2, nf_effect, nf_effect, self.dropout_rate)

        # Separate encoders for collision and topological edges
        self.collision_encoder = RelationEncoder(5, nf_effect, nf_effect, self.dropout_rate)
        self.topo_encoder = RelationEncoder(6, nf_effect, nf_effect, self.dropout_rate)

        # Propagators for message passing
        self.particle_propagator = Propagator(
            3 * nf_effect, nf_effect)  # particle encoding + collision effects + topological effects

        self.relation_propagator = Propagator(
            3 * nf_effect, nf_effect)  # relation encoding + sender effects + receiver effects

        # Final position predictor
        self.particle_predictor = ParticlePredictor(
            nf_effect, nf_effect, 3, self.dropout_rate)

    def forward(self, a_cur, s_cur, s_delta, Rr_collision, Rs_collision, Rr_topo, Rs_topo, first_edge_lengths):
        """
        Forward pass of the GNN dynamics model with separate collision and topological encoders.
        
        Args:
            a_cur: (B, particle_num) - current particle attributes
            s_cur: (B, particle_num, 3) - current positions  
            s_delta: (B, particle_num, 3) - particle displacements over history
            Rr_collision: (B, n_collision, particle_num) - receiver matrix for collision edges
            Rs_collision: (B, n_collision, particle_num) - sender matrix for collision edges
            Rr_topo: (B, n_topo, particle_num) - receiver matrix for topological edges
            Rs_topo: (B, n_topo, particle_num) - sender matrix for topological edges
            first_edge_lengths: (B, n_topo) - edge lengths in first frame for topological edges
            
        Returns:
            (B, particle_num, 3) - predicted next particle positions
        """
        B, N = a_cur.size()
        pstep = 3  # Number of message passing steps

        # Convert from data format (B x time x particles) to model format (B x particles x time)
        s_delta = s_delta.transpose(1, 2)  # B x particle_num x n_history x 3

        # Flatten displacement history for particle encoder
        s_delta_flat = s_delta.reshape(B, N, -1)  # B x particle_num x (3 * n_history)

        # Encode particle features (history-aware)
        particle_encode = self.particle_encoder(
            torch.cat([s_delta_flat, a_cur[..., None], s_cur[..., 2:3]], 2))  # B x particle_num x (3D velocity + attribute + z coordinate) -> B x particle_num x nf_effect
        particle_effect = particle_encode

        # =====================================
        # COLLISION EDGE ENCODING
        # =====================================
        # Compute collision relation features
        a_cur_r_collision = Rr_collision.bmm(a_cur[..., None])  # B x n_collision x 1
        a_cur_s_collision = Rs_collision.bmm(a_cur[..., None])  # B x n_collision x 1
        s_cur_r_collision = Rr_collision.bmm(s_cur)  # B x n_collision x 3
        s_cur_s_collision = Rs_collision.bmm(s_cur)  # B x n_collision x 3

        # Encode collision relation features: attributes (2) + position difference (3)
        collision_encode = self.collision_encoder(
            torch.cat([a_cur_r_collision, a_cur_s_collision, s_cur_r_collision - s_cur_s_collision], 2)
        )  # B x n_collision x nf_effect

        # =====================================
        # TOPOLOGICAL EDGE ENCODING
        # =====================================
        # Compute topological relation features
        a_cur_r_topo = Rr_topo.bmm(a_cur[..., None])  # B x n_topo x 1
        a_cur_s_topo = Rs_topo.bmm(a_cur[..., None])  # B x n_topo x 1
        s_cur_r_topo = Rr_topo.bmm(s_cur)  # B x n_topo x 3
        s_cur_s_topo = Rs_topo.bmm(s_cur)  # B x n_topo x 3

        # Current position difference
        current_pos_diff = s_cur_r_topo - s_cur_s_topo  # B x n_topo x 3
        
        # Normalize by first frame edge length (add small epsilon to avoid division by zero)
        first_edge_lengths_expanded = first_edge_lengths.unsqueeze(-1)  # B x n_topo x 1
        normalized_pos_diff = current_pos_diff / (first_edge_lengths_expanded + 1e-8)  # B x n_topo x 3

        # Encode topological relation features: attributes (2) + first edge length (1) + normalized position diff (3)
        topo_encode = self.topo_encoder(
            torch.cat([a_cur_r_topo, a_cur_s_topo, first_edge_lengths_expanded, normalized_pos_diff], 2)
        )  # B x n_topo x nf_effect

        # =====================================
        # MESSAGE PASSING
        # =====================================
        for i in range(pstep):
            # Process collision edges
            effect_r_collision = Rr_collision.bmm(particle_effect)  # B x n_collision x nf_effect
            effect_s_collision = Rs_collision.bmm(particle_effect)  # B x n_collision x nf_effect
            
            # Update collision edge effects
            effect_collision = self.relation_propagator(
                torch.cat([collision_encode, effect_r_collision, effect_s_collision], 2)
            )  # B x n_collision x nf_effect

            # Aggregate collision effects back to particles
            Rr_collision_t = Rr_collision.transpose(1, 2)  # B x particle_num x n_collision
            effect_collision_agg = Rr_collision_t.bmm(effect_collision)  # B x particle_num x nf_effect

            # Process topological edges
            effect_r_topo = Rr_topo.bmm(particle_effect)  # B x n_topo x nf_effect
            effect_s_topo = Rs_topo.bmm(particle_effect)  # B x n_topo x nf_effect
            
            # Update topological edge effects
            effect_topo = self.relation_propagator(
                torch.cat([topo_encode, effect_r_topo, effect_s_topo], 2)
            )  # B x n_topo x nf_effect

            # Aggregate topological effects back to particles
            Rr_topo_t = Rr_topo.transpose(1, 2)  # B x particle_num x n_topo
            effect_topo_agg = Rr_topo_t.bmm(effect_topo)  # B x particle_num x nf_effect
                        
            # Update particle effects with residual connection
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_collision_agg, effect_topo_agg], 2),
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
        self.adj_thresh = config['train']['edges']['collision']['adj_thresh']
        self.topk = config['train']['edges']['collision']['topk']
        self.connect_tools_all = config['train']['edges']['collision']['connect_tools_all']
        self.model = PropModuleDiffDen(config, use_gpu)

    def predict_one_step(self, a_cur, s_cur, s_delta, topological_edges, first_states, particle_nums=None, epoch_timer=None):
        """
        Predict particle positions one step into the future.
        
        Args:
            a_cur: (B, particle_num) - current particle attributes
            s_cur: (B, particle_num, 3) - current positions  
            s_delta: (B, n_history, particle_num, 3) - particle displacements over history
                    For t < n_history-1: consecutive frame differences
                    For t = n_history-1: 0 for objects, actual motion for robots
            topological_edges: (B, particle_num, particle_num) - adjacency matrix of topological edges
            first_states: (B, particle_num, 3) - first frame states for topological edge computations
            particle_nums: (B,) - number of valid particles per batch sample
            epoch_timer: optional EpochTimer for profiling edge construction time
            
        Returns:
            (B, particle_num, 3) - predicted next particle positions
        """
        assert type(a_cur) == torch.Tensor
        assert type(s_cur) == torch.Tensor  
        assert type(s_delta) == torch.Tensor
        assert a_cur.shape == s_cur.shape[:2]
        assert s_cur.shape == s_delta[:, -1].shape

        B, N = a_cur.size()

        # Create batch masks to exclude batch padding
        if particle_nums is not None:
            mask = torch.zeros((B, N), dtype=torch.bool, device=s_cur.device)
            for b in range(B):
                n_particles = particle_nums[b].item()
                mask[b, :n_particles] = True
        else:
            # All particles are valid if particle_nums not provided
            mask = torch.ones((B, N), dtype=torch.bool, device=s_cur.device)
        
        # Create tool mask (tool particles have attr=1, objects have attr=0)
        tool_mask = (a_cur > 0.5) & mask
                
        if epoch_timer is not None:
            epoch_timer.start_timer('edge')
        
        # Construct collision edges (excluding topological edges)
        Rr_collision, Rs_collision = construct_collision_edges(
            s_cur, 
            self.adj_thresh, 
            mask, 
            tool_mask, 
            topk=self.topk,
            connect_tools_all=self.connect_tools_all,
            topological_edges=topological_edges
        )
        
        # Construct topological edges with first frame information
        Rr_topo, Rs_topo, first_edge_lengths = construct_topological_edges(
            topological_edges, 
            first_states
        )
        
        if epoch_timer is not None:
            epoch_timer.end_timer('edge')

        # Forward pass through GNN with separate edge types
        if epoch_timer is not None:
            epoch_timer.start_timer('gnn_forward')
        
        s_pred = self.model.forward(
            a_cur, s_cur, s_delta, 
            Rr_collision, Rs_collision, 
            Rr_topo, Rs_topo, 
            first_edge_lengths
        )
        
        if epoch_timer is not None:
            epoch_timer.end_timer('gnn_forward')

        # Mask out predictions for padded particles
        s_pred[~mask] = 0.

        return s_pred
