
def test_ensemble(models, config, robot_mask, topological_edges, first_states, action_seqs):
    """
    Test an ensemble of models' ability to measure uncertainty

    Args:
        models: list[PropNetDiffDenModel] - ensemble of models
        config: dict - configuration
        robot_mask: [n_particles] - boolean tensor for robot particles
        topological_edges: [n_particles, n_particles] tensor - topological edges adjacency matrix
        first_states: [n_particles, 3] - first frame particle positions
        action_seqs: [n_sample, n_look_ahead, 3] - action sequences

    Returns:
        dict containing:
    """