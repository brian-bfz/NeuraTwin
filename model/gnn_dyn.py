import torch
import torch.nn as nn
import time

def construct_edges_from_states_batch(states, adj_thresh, mask, tool_mask, topk, connect_tools_all):
    """
    Construct edges between particles based on distance and tool connectivity rules (batch version)
    
    :param states: (B, N, state_dim) torch tensor
    :param adj_thresh: float or (B,) torch tensor - distance threshold for connections
    :param mask: (B, N) torch tensor, true when index is a valid particle
    :param tool_mask: (B, N) torch tensor, true when index is a valid tool particle
    :param topk: int - maximum number of neighbors per particle
    :param connect_tools_all: bool - if True, connect all tool particles to all object particles
    :return:
    - Rr: (B, n_rel, N) torch tensor - receiver matrix
    - Rs: (B, n_rel, N) torch tensor - sender matrix
    """
    B, N, state_dim = states.shape
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # (B, N, N, state_dim)
    dis = torch.sum(s_diff ** 2, -1)
    
    # Create masks for valid particle connections
    mask_1 = mask[:, :, None].repeat(1, 1, N)
    mask_2 = mask[:, None, :].repeat(1, N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    
    # Create masks for tool particles
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations

    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # add topk constraints
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    # if connect_tools_all:
    #     obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
    #     obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender
    #     adj_matrix[obj_tool_mask_1] = 0 # avoid object to tool connections
    #     adj_matrix[obj_tool_mask_2] = 1 # add all tool to object connections
    #     adj_matrix[tool_mask_12] = 0 # avoid tool to tool relations

    if connect_tools_all:
        obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
        obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender
        obj_pad_tool_mask_1 = tool_mask_1 * (~tool_mask_2)

        batch_mask = (adj_matrix[obj_pad_tool_mask_1].reshape(B, -1).sum(-1) > 0)[:, None, None].repeat(1, N, N)
        batch_obj_tool_mask_1 = obj_tool_mask_1 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_1 = obj_tool_mask_1 * (~batch_mask)  # (B, N, N)
        batch_obj_tool_mask_2 = obj_tool_mask_2 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_2 = obj_tool_mask_2 * (~batch_mask)  # (B, N, N)

        adj_matrix[batch_obj_tool_mask_1] = 0
        adj_matrix[batch_obj_tool_mask_2] = 1
        adj_matrix[neg_batch_obj_tool_mask_1] = 0
        adj_matrix[neg_batch_obj_tool_mask_2] = 0
    else:
        obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
        adj_matrix[obj_tool_mask_1] = 0

    n_rels = adj_matrix.sum(dim=(1,2))
    n_rel = n_rels.max().long().item()
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1    
    return Rr, Rs

def construct_edges_from_states(states, adj_thresh, mask, tool_mask, topk, connect_tools_all):
    """
    Construct edges between particles based on distance and tool connectivity rules
    
    :param states: (N, state_dim) torch tensor
    :param adj_thresh: float - distance threshold for connections
    :param mask: (N) torch tensor, true when index is a valid particle
    :param tool_mask: (N) torch tensor, true when index is a valid tool particle
    :param topk: int - maximum number of neighbors per particle
    :param connect_tools_all: bool - if True, connect all tool particles to all object particles
    :return:
    - Rr: (n_rel, N) torch tensor - receiver matrix
    - Rs: (n_rel, N) torch tensor - sender matrix
    """
    N, state_dim = states.shape
    s_receiv = states[:, None, :].repeat(1, N, 1)
    s_sender = states[None, :, :].repeat(N, 1, 1)

    # dis: particle_num x particle_num
    # adj_matrix: particle_num x particle_num
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # (N, N, state_dim)
    dis = torch.sum(s_diff ** 2, -1)
    
    # Create masks for valid particle connections
    mask_1 = mask[:, None].repeat(1, N)
    mask_2 = mask[None, :].repeat(N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    
    # Create masks for tool particles
    tool_mask_1 = tool_mask[:, None].repeat(1, N)
    tool_mask_2 = tool_mask[None, :].repeat(N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations

    obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender

    adj_matrix = ((dis - threshold) < 0).float()

    # add topk constraints
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    if connect_tools_all:
        adj_matrix[obj_tool_mask_1] = 0  # clear existing tool receiver connections
        adj_matrix[obj_tool_mask_2] = 1  # connect all object particles to all tool particles
        adj_matrix[tool_mask_12] = 0     # avoid tool to tool relations

    n_rels = adj_matrix.sum().long().item()
    rels_idx = torch.arange(n_rels).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rr[rels_idx, rels[:, 0]] = 1
    Rs[rels_idx, rels[:, 1]] = 1
    return Rr, Rs

### Propagation Networks

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Propagator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        '''
        Args:
            x: [batch_size, n_relations/n_particles, input_size]
        Returns:
            [batch_size, n_relations/n_particles, output_size]
        '''
        B, N, D = x.size()
        x = self.linear(x.view(B * N, D))

        if residual is None:
            x = self.relu(x)
        else:
            x = self.relu(x + residual.view(B * N, self.output_size))

        return x.view(B, N, self.output_size)

class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)

class PropModuleDiffDen(nn.Module):
    def __init__(self, config, use_gpu=False):

        super(PropModuleDiffDen, self).__init__()

        self.config = config
        nf_effect = config['train']['particle']['nf_effect']
        self.nf_effect = nf_effect
        self.add_delta = config['train']['particle']['add_delta']
        
        # Get n_history from config
        self.n_history = config['train']['n_history']
        self.use_gpu = use_gpu

        # particle encoder
        # input: pusher movement (3 * n_history), attr (1 * n_history)
        self.particle_encoder = ParticleEncoder(
            (3 + 1) * self.n_history, nf_effect, nf_effect)

        # relation encoder
        # input: attr * 2 (2), state offset (3)
        self.relation_encoder = RelationEncoder(
            2 + 3, nf_effect, nf_effect)

        # input: (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(
            2 * nf_effect, nf_effect)

        # input: (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(
            nf_effect + 2 * nf_effect, nf_effect)

        # input: (1) particle effect
        self.particle_predictor = ParticlePredictor(
            nf_effect, nf_effect, 3)

    def forward(self, a_hist, s_hist, s_delta, Rr, Rs, verbose=False):
        # a_hist: B x n_history x particle_num -- indicating the type of the objects, slider or pusher
        # s_hist: B x n_history x particle_num x 3 -- position of the objects
        # s_delta: B x n_history x particle_num x 3 -- displacement of the objects
        # Rr: B x rel_num x particle_num
        # Rs: B x rel_num x particle_num
        
        B, n_history, N = a_hist.size()
        _, rel_num, _ = Rr.size()
        nf_effect = self.nf_effect

        pstep = 3

        # Convert from data format (B x time x particle_num) to model format (B x particle_num x time)
        a_hist = a_hist.transpose(1, 2)  # B x particle_num x n_history
        s_hist = s_hist.transpose(1, 2)  # B x particle_num x n_history x 3
        s_delta = s_delta.transpose(1, 2)  # B x particle_num x n_history x 3

        Rr_t = Rr.transpose(1, 2) # TODO: add .continuous()? # B x particle_num x rel_num

        # Flatten history dimension for encoding (no transpose needed!)
        # a_hist: B x particle_num x n_history (already in the right format)
        # a_hist_flat = a_hist  # B x particle_num x n_history
        
        # s_hist: B x particle_num x n_history x 3 -> B x particle_num x (3 * n_history)
        # s_hist_flat = s_hist.reshape(B, N, -1)  # B x particle_num x (3 * n_history) Not needed because we use relative coordinates instead, but I'm keeping it in case I need absolute z-coordinates later
        
        # s_delta: B x particle_num x n_history x 3 -> B x particle_num x (3 * n_history)
        s_delta_flat = s_delta.reshape(B, N, -1)  # B x particle_num x (3 * n_history)

        # Get current frame data for relations (since edges are constructed dynamically)
        a_cur = a_hist[:, :, -1]  # B x particle_num (last time step)
        s_cur = s_hist[:, :, -1, :]  # B x particle_num x 3 (last time step)
        
        # receiver_attr, sender_attr (using current frame only)
        a_cur_r = Rr.bmm(a_cur[..., None]) # B x rel_num x 1
        a_cur_s = Rs.bmm(a_cur[..., None]) # B x rel_num x 1

        # receiver_state, sender_state
        s_cur_r = Rr.bmm(s_cur) # B x rel_num x 3
        s_cur_s = Rs.bmm(s_cur) # B x rel_num x 3

        # particle encode
        particle_encode = self.particle_encoder(
            torch.cat([s_delta_flat, a_hist], 2)) # B x particle_num x nf_effect
        particle_effect = particle_encode

        # relation encode
        relation_encode = self.relation_encoder(
            torch.cat([a_cur_r, a_cur_s, s_cur_r - s_cur_s], 2)) # B x rel_num x nf_effect

        for i in range(pstep):
            effect_r = Rr.bmm(particle_effect) # B x rel_num x nf_effect
            effect_s = Rs.bmm(particle_effect) # B x rel_num x nf_effect
            
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2)) # B x rel_num x nf_effect

            effect_rel_agg = Rr_t.bmm(effect_rel) # B x particle_num x nf_effect
            
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2),
                residual=particle_effect)
        
        # B x particle_num x 3
        particle_pred = self.particle_predictor(particle_effect)

        # Use the most recent state for residual connection
        return particle_pred + s_cur

class PropNetDiffDenModel(nn.Module):

    def __init__(self, config, use_gpu=False):
        super(PropNetDiffDenModel, self).__init__()

        self.config = config
        self.adj_thresh = config['train']['particle']['adj_thresh']
        self.connect_tools_all = config['train']['particle']['connect_tools_all']
        self.topk = config['train']['particle']['topk']
        self.model = PropModuleDiffDen(config, use_gpu)

    def predict_one_step(self, a_hist, s_hist, s_delta, particle_nums=None):
        # assume these variables have already been calculated
        # a_hist: B x n_history x particle_num (0 for objects, 1 for tools/robot)
        # s_hist: B x n_history x particle_num x 3
        # s_delta: B x n_history x particle_num x 3 (for t < n_history - 1, s_delta is s_hist[t+1] - s_hist[t]; for t = n_history - 1, s_delta is 0 for objects, s_hist[t+1] - s_hist[t] for robot)
        # particle_nums: B
        assert type(a_hist) == torch.Tensor
        assert type(s_hist) == torch.Tensor
        assert type(s_delta) == torch.Tensor
        assert a_hist.shape == s_hist.shape[:3]
        assert s_hist.shape == s_delta.shape

        B, n_history, N = a_hist.size()

        # Use the most recent state for edge construction
        a_cur = a_hist[:, -1, :]  # B x particle_num
        s_cur = s_hist[:, -1, :, :]  # B x particle_num x 3

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
        
        # Time edge construction
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # edge_start = time.perf_counter()
                
        # Construct edges using efficient batch processing (using most recent state)
        Rr_batch, Rs_batch = construct_edges_from_states_batch(
            s_cur, 
            self.adj_thresh, 
            mask, 
            tool_mask, 
            topk=self.topk,
            connect_tools_all=self.connect_tools_all
        )
        
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # edge_time = time.perf_counter() - edge_start
        
        # Store edge construction timing for profiling
        # if hasattr(self, '_edge_times'):
        #     self._edge_times.append(edge_time)
        # elif not hasattr(self, '_edge_times'):
        #     self._edge_times = [edge_time]

        # Forward pass with full history
        s_pred = self.model.forward(a_hist, s_hist, s_delta, Rr_batch, Rs_batch)

        return s_pred
