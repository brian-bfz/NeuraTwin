import torch
import torch.nn as nn
import time

def construct_edges_from_states_batch(states, adj_thresh, mask, tool_mask, topk, connect_tools_all):
    """
    Construct edges between particles based on distance and tool connectivity rules (batch version).
    
    Args:
        states: (B, N, state_dim) torch tensor - particle positions
        adj_thresh: float or (B,) torch tensor - distance threshold for connections
        mask: (B, N) torch tensor - true when index is a valid particle
        tool_mask: (B, N) torch tensor - true when index is a valid tool particle
        topk: int - maximum number of neighbors per particle
        connect_tools_all: bool - if True, connect all tool particles to all object particles
        
    Returns:
        Rr: (B, n_rel, N) torch tensor - receiver matrix for graph edges
        Rs: (B, n_rel, N) torch tensor - sender matrix for graph edges
    """
    B, N, state_dim = states.shape
    
    # Create pairwise particle combinations for distance calculation
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)  # Receiver particles
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)   # Sender particles

    # Calculate squared distances between all particle pairs and the squared distance threshold
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # Position differences
    dis = torch.sum(s_diff ** 2, -1)  # Squared distances (B, N, N)
    
    # Create validity masks for particle connections
    mask_1 = mask[:, :, None].repeat(1, 1, N)  # Receiver validity
    mask_2 = mask[:, None, :].repeat(1, N, 1)  # Sender validity
    mask_12 = mask_1 * mask_2  # Both particles are valid
    dis[~mask_12] = 1e10  # Exclude invalid particle pairs
    
    # Prevent tool-to-tool connections
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # Avoid tool to tool relations

    # Create adjacency matrix based on distance threshold
    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # Define tool-object interaction masks
    obj_tool_mask_1 = tool_mask_1 * mask_2  # Particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # Tool sender, particle receiver
    obj_pad_tool_mask_1 = tool_mask_1 * (~tool_mask_2) # Tool receiver, non-tool sender

    # Apply topk constraint to limit connections per particle
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    # Handle tool connectivity rules
    if connect_tools_all:
        # Only connect tools to objects if there are neighboring tool - non-tool particles in batch
        batch_mask = (adj_matrix[obj_pad_tool_mask_1].reshape(B, -1).sum(-1) > 0)[:, None, None].repeat(1, N, N)
        batch_obj_tool_mask_1 = obj_tool_mask_1 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_1 = obj_tool_mask_1 * (~batch_mask)  # (B, N, N)
        batch_obj_tool_mask_2 = obj_tool_mask_2 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_2 = obj_tool_mask_2 * (~batch_mask)  # (B, N, N)

        adj_matrix[batch_obj_tool_mask_1] = 0  # Clear object-to-tool edges
        adj_matrix[batch_obj_tool_mask_2] = 1  # Add all tool-to-object edges
        adj_matrix[neg_batch_obj_tool_mask_1] = 0
        adj_matrix[neg_batch_obj_tool_mask_2] = 0
    else:
        adj_matrix[obj_tool_mask_1] = 0  # Clear object-to-tool edges

    # Convert adjacency matrix to sparse edge representation
    n_rels = adj_matrix.sum(dim=(1,2))  # Number of edges per batch
    n_rel = n_rels.max().long().item()  # Maximum edges across batch
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()  # Get edge indices
    
    # Create receiver and sender matrices
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1    
    
    return Rr, Rs

def construct_edges_from_states(states, adj_thresh, mask, tool_mask, topk, connect_tools_all):
    """
    Construct edges between particles based on distance and tool connectivity rules (NOT USED).
    
    Args:
        states: (N, state_dim) torch tensor - particle positions
        adj_thresh: float - distance threshold for connections
        mask: (N) torch tensor - true when index is a valid particle
        tool_mask: (N) torch tensor - true when index is a valid tool particle
        topk: int - maximum number of neighbors per particle
        connect_tools_all: bool - if True, connect all tool particles to all object particles
        
    Returns:
        Rr: (n_rel, N) torch tensor - receiver matrix for graph edges
        Rs: (n_rel, N) torch tensor - sender matrix for graph edges
    """
    N, state_dim = states.shape
    
    # Create pairwise particle combinations
    s_receiv = states[:, None, :].repeat(1, N, 1)
    s_sender = states[None, :, :].repeat(N, 1, 1)

    # Calculate distances and create adjacency matrix
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender
    dis = torch.sum(s_diff ** 2, -1)
    
    # Apply validity masks
    mask_1 = mask[:, None].repeat(1, N)
    mask_2 = mask[None, :].repeat(N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10
    
    # Prevent tool-to-tool connections
    tool_mask_1 = tool_mask[:, None].repeat(1, N)
    tool_mask_2 = tool_mask[None, :].repeat(N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10

    # Define interaction masks
    obj_tool_mask_1 = tool_mask_1 * mask_2
    obj_tool_mask_2 = tool_mask_2 * mask_1

    adj_matrix = ((dis - threshold) < 0).float()

    # Apply topk constraint
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    # Handle tool connectivity
    if connect_tools_all:
        adj_matrix[obj_tool_mask_1] = 0  # Clear existing tool receiver connections
        adj_matrix[obj_tool_mask_2] = 1  # Connect all object particles to all tool particles
        adj_matrix[tool_mask_12] = 0     # Avoid tool to tool relations

    # Convert to sparse representation
    n_rels = adj_matrix.sum().long().item()
    rels_idx = torch.arange(n_rels).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rr[rels_idx, rels[:, 0]] = 1
    Rs[rels_idx, rels[:, 1]] = 1
    
    return Rr, Rs

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
        # Input: displacement (3 * n_history) + attributes (1 * n_history)
        self.particle_encoder = ParticleEncoder(
            (3 + 1) * self.n_history, nf_effect, nf_effect)

        # Relation encoder:
        # Input: attributes of both particles (2) + position difference (3)
        self.relation_encoder = RelationEncoder(
            2 + 3, nf_effect, nf_effect)

        # Propagators for message passing
        self.particle_propagator = Propagator(
            2 * nf_effect, nf_effect)  # particle encoding + aggregated relation effects

        self.relation_propagator = Propagator(
            nf_effect + 2 * nf_effect, nf_effect)  # relation encoding + sender/receiver effects

        # Final position predictor
        self.particle_predictor = ParticlePredictor(
            nf_effect, nf_effect, 3)

    def forward(self, a_cur, s_cur, s_delta, Rr, Rs, verbose=False):
        """
        Forward pass of the GNN dynamics model.
        
        Args:
            a_cur: (B, particle_num) - current particle attributes
            s_cur: (B, particle_num, 3) - current positions  
            s_delta: (B, particle_num, 3) - particle displacements over history
            Rr: (B, rel_num, particle_num) - receiver matrix for graph edges
            Rs: (B, rel_num, particle_num) - sender matrix for graph edges
            verbose: bool - whether to print debug information
            
        Returns:
            (B, particle_num, 3) - predicted next particle positions
        """
        B, n_history, N = s_delta.size()
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
            torch.cat([s_delta_flat, a_cur], 2))  # B x particle_num x nf_effect
        particle_effect = particle_encode

        # Encode relation features (current frame only)
        relation_encode = self.relation_encoder(
            torch.cat([a_cur_r, a_cur_s, s_cur_r - s_cur_s], 2))  # B x rel_num x nf_effect

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

    def predict_one_step(self, a_cur, s_cur, s_delta, particle_nums=None):
        """
        Predict particle positions one step into the future.
        
        Args:
            a_cur: (B, particle_num) - current particle attributes
            s_cur: (B, particle_num, 3) - current positions  
            s_delta: (B, n_history, particle_num, 3) - particle displacements over history
                    For t < n_history-1: consecutive frame differences
                    For t = n_history-1: 0 for objects, actual motion for robots
            particle_nums: (B,) - number of valid particles per batch sample
            
        Returns:
            (B, particle_num, 3) - predicted next particle positions
        """
        assert type(a_cur) == torch.Tensor
        assert type(s_cur) == torch.Tensor  
        assert type(s_delta) == torch.Tensor
        assert a_cur.shape == s_cur.shape
        assert s_cur.shape == s_delta[:, -1].shape

        B, n_history, N = s_delta.size()

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
        Rr_batch, Rs_batch = construct_edges_from_states_batch(
            s_cur, 
            self.adj_thresh, 
            mask, 
            tool_mask, 
            topk=self.topk,
            connect_tools_all=self.connect_tools_all
        )

        # Forward pass through GNN
        s_pred = self.model.forward(a_cur, s_cur, s_delta, Rr_batch, Rs_batch)

        return s_pred
