import torch
import numpy as np
import open3d as o3d
import sys
import os
import yaml

# Add the model directory to path to import the edge construction function
sys.path.append('model')
sys.path.append('dataset')
from .utils import construct_edges_from_states_batch
from .dataset.dataset_gnn_dyn import ParticleDataset

def load_yaml(file_path):
    """Load YAML configuration file"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def create_output_folder():
    """Create the output folder for visualizations"""
    folder_name = "visualize_edges"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def get_real_dataset_examples(config, n_examples):
    """Get n_examples from the real dataset"""
    # Create dataset instance
    dataset = ParticleDataset(config['dataset']['file'], config, 'train')
    
    examples = []
    for i in range(n_examples):
        # Get a data sample
        states, states_delta, attrs, particle_num = dataset[i * 20]
        
        # Extract first frame data
        first_frame_states = states[0]  # [particle_num, 3] 
        first_frame_attrs = attrs[0]    # [particle_num]
        
        examples.append({
            'states': first_frame_states,
            'attrs': first_frame_attrs,
            'particle_num': particle_num,
            'example_idx': i
        })
        
        print(f"Example {i}: {particle_num} particles, "
              f"{torch.sum(first_frame_attrs == 0).item()} objects, "
              f"{torch.sum(first_frame_attrs == 1).item()} tools")
    
    return examples

def prepare_batch_for_edge_construction(examples):
    """Convert list of examples into batch format for edge construction"""
    B = len(examples)
    
    # Find max particle number to create padded tensors
    max_particles = max(ex['particle_num'] for ex in examples)
    
    # Create batch tensors
    states_batch = torch.zeros(B, max_particles, 3)
    attrs_batch = torch.zeros(B, max_particles)
    mask_batch = torch.zeros(B, max_particles, dtype=torch.bool)
    
    for i, ex in enumerate(examples):
        n_particles = ex['particle_num']
        states_batch[i, :n_particles] = ex['states']
        attrs_batch[i, :n_particles] = ex['attrs']
        mask_batch[i, :n_particles] = True
    
    # Create tool mask
    tool_mask_batch = (attrs_batch > 0.5) & mask_batch
    
    return states_batch, attrs_batch, mask_batch, tool_mask_batch

def visualize_edges(states, attrs, Rr, Rs, batch_idx, connect_tools_all, output_folder, example_idx):
    """Visualize the edges for a single batch using open3d"""
    # Extract data for this batch (only valid particles)
    mask = torch.any(states[batch_idx] != 0, dim=1)  # Find non-zero particles
    valid_indices = torch.where(mask)[0]
    
    positions = states[batch_idx][valid_indices].numpy()
    attributes = attrs[batch_idx][valid_indices].numpy()
    
    print(f"  Visualizing example {example_idx}: {len(positions)} particles")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    
    # Color coding: blue for objects, red for tools
    colors = np.zeros((len(positions), 3))
    colors[attributes == 0] = [0, 0, 1]  # blue for objects
    colors[attributes == 1] = [1, 0, 0]  # red for tools
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create line set for edges
    lines = []
    edge_colors = []
    
    # Extract edges from Rr and Rs matrices
    B, n_rel, N = Rr.shape
    for rel_idx in range(n_rel):
        # Find receiver and sender for this relation
        receiver_idx = torch.where(Rr[batch_idx, rel_idx, :] == 1)[0]
        sender_idx = torch.where(Rs[batch_idx, rel_idx, :] == 1)[0]
        
        if len(receiver_idx) > 0 and len(sender_idx) > 0:
            recv_global_idx = receiver_idx[0].item()
            send_global_idx = sender_idx[0].item()
            
            # Check if both indices are in our valid set
            recv_local_idx = torch.where(valid_indices == recv_global_idx)[0]
            send_local_idx = torch.where(valid_indices == send_global_idx)[0]
            
            if len(recv_local_idx) > 0 and len(send_local_idx) > 0:
                recv_idx = recv_local_idx[0].item()
                send_idx = send_local_idx[0].item()
                
                lines.append([recv_idx, send_idx])
                
                # Color edges differently based on particle types
                recv_attr = attributes[recv_idx]
                send_attr = attributes[send_idx]
                
                if recv_attr == 0 and send_attr == 0:
                    edge_colors.append([0, 1, 0])  # green for object-object
                elif recv_attr == 1 and send_attr == 0:
                    edge_colors.append([1, 1, 0])  # yellow for object-tool (shouldn't happen)
                elif recv_attr == 0 and send_attr == 1:
                    edge_colors.append([1, 0.5, 0])  # orange for tool-object
                else:
                    edge_colors.append([1, 0, 1])  # magenta for tool-tool (shouldn't happen)
    
    # Create line set
    line_set = o3d.geometry.LineSet()
    if lines:
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(edge_colors)
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Real Dataset - Example {example_idx} - connect_tools_all={connect_tools_all}", 
                      visible=True)
    
    vis.add_geometry(pcd)
    if lines:
        vis.add_geometry(line_set)
    
    # Set view - zoom out to see all particles
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(1)  # Zoom out more to see the full scene
    
    # Update renderer and capture
    vis.poll_events()
    vis.update_renderer()
    
    # Save image
    filename = os.path.join(output_folder, f"real_dataset_example_{example_idx}_connect_tools_{connect_tools_all}.png")
    vis.capture_screen_image(filename)
    vis.destroy_window()
    
    print(f"  Saved visualization: {filename}")
    print(f"    Total edges: {len(lines)}")
    print(f"    Object particles: {np.sum(attributes == 0)}")
    print(f"    Tool particles: {np.sum(attributes == 1)}")
    
    # Count edge types
    obj_obj = sum(1 for color in edge_colors if np.allclose(color, [0, 1, 0]))
    obj_tool = sum(1 for color in edge_colors if np.allclose(color, [1, 1, 0]))
    tool_obj = sum(1 for color in edge_colors if np.allclose(color, [1, 0.5, 0]))
    tool_tool = sum(1 for color in edge_colors if np.allclose(color, [1, 0, 1]))
    
    print(f"    Object-Object edges: {obj_obj}")
    print(f"    Object-Tool edges: {obj_tool}")
    print(f"    Tool-Object edges: {tool_obj}")
    print(f"    Tool-Tool edges: {tool_tool}")
    print()

def main():
    print("Loading configuration...")
    config = load_yaml('config/train/gnn_dyn.yaml')
    
    print("Creating output folder...")
    output_folder = create_output_folder()
    
    print("Loading real dataset examples...")
    examples = get_real_dataset_examples(config, n_examples=6)
    
    print("Preparing batch for edge construction...")
    states_batch, attrs_batch, mask_batch, tool_mask_batch = prepare_batch_for_edge_construction(examples)
    
    # Get edge construction parameters from config
    adj_thresh = config['train']['particle']['adj_thresh']
    topk = config['train']['particle']['topk']
    
    print(f"Edge construction parameters:")
    print(f"  Distance threshold: {adj_thresh}")
    print(f"  Top-k neighbors: {topk}")
    print(f"  Batch shape: {states_batch.shape}")
    print()
    
    # Test both connect_tools_all settings
    for connect_tools_all in [False, True]:
        print(f"Testing with connect_tools_all = {connect_tools_all}")
        print("=" * 60)
        
        # Construct edges
        Rr, Rs = construct_edges_from_states_batch(
            states_batch, adj_thresh, mask_batch, tool_mask_batch, topk, connect_tools_all
        )
        
        print(f"Edge tensor shapes: Rr={Rr.shape}, Rs={Rs.shape}")
        
        # Visualize each example
        for i, example in enumerate(examples):
            visualize_edges(states_batch, attrs_batch, Rr, Rs, i, connect_tools_all, 
                          output_folder, example['example_idx'])
    
    print("Visualization complete!")
    print(f"All images saved to: {output_folder}/")
    print("\nLegend:")
    print("  Blue points: Object particles")
    print("  Red points: Tool particles")
    print("  Green lines: Object-Object connections")
    print("  Yellow lines: Object-Tool connections (shouldn't happen)")
    print("  Orange lines: Tool-Object connections")
    print("  Magenta lines: Tool-Tool connections (shouldn't happen)")

if __name__ == "__main__":
    main() 