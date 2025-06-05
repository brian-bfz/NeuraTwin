import torch
import numpy as np
import open3d as o3d
import sys
import os

# Add the model directory to path to import the edge construction function
sys.path.append('model')
from gnn_dyn import construct_edges_from_states_batch

def create_test_batch():
    """Create a batch of 5 test cases with ~20 objects and 5 tools each"""
    B = 5  # batch size
    N = 25  # total particles per case (20 objects + 5 tools)
    
    states = torch.zeros(B, N, 3)  # (x, y, z) positions
    attrs = torch.zeros(B, N)      # 0 for objects, 1 for tools
    mask = torch.ones(B, N, dtype=torch.bool)  # all particles are valid
    
    # Set attributes: first 20 are objects (0), last 5 are tools (1)
    attrs[:, 20:] = 1
    
    # Create different spatial configurations for each batch
    for b in range(B):
        # Objects: scattered in a roughly circular pattern
        angles = torch.linspace(0, 2*np.pi, 20, dtype=torch.float32)
        radius = 1.0 + 0.3 * torch.randn(20)  # slight randomness in radius
        object_x = radius * torch.cos(angles) + 0.1 * torch.randn(20)
        object_y = radius * torch.sin(angles) + 0.1 * torch.randn(20)
        object_z = 0.1 * torch.randn(20)  # small z variation
        
        states[b, :20, 0] = object_x
        states[b, :20, 1] = object_y
        states[b, :20, 2] = object_z
        
        # Tools: positioned strategically around objects
        if b == 0:
            # Case 0: Tools in center
            tool_positions = torch.tensor([
                [0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [-0.3, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, -0.3, 0.0]
            ], dtype=torch.float32)
        elif b == 1:
            # Case 1: Tools on one side
            tool_positions = torch.tensor([
                [2.0, 0.0, 0.0],
                [2.2, 0.3, 0.0],
                [2.2, -0.3, 0.0],
                [1.8, 0.3, 0.0],
                [1.8, -0.3, 0.0]
            ], dtype=torch.float32)
        elif b == 2:
            # Case 2: Tools scattered around perimeter
            tool_angles = torch.linspace(0, 2*np.pi, 5, dtype=torch.float32)
            tool_radius = 1.8
            tool_positions = torch.stack([
                tool_radius * torch.cos(tool_angles),
                tool_radius * torch.sin(tool_angles),
                0.1 * torch.randn(5)
            ], dim=1)
        elif b == 3:
            # Case 3: Tools in a line
            tool_positions = torch.tensor([
                [-1.5, 1.5, 0.0],
                [-0.75, 1.5, 0.0],
                [0.0, 1.5, 0.0],
                [0.75, 1.5, 0.0],
                [1.5, 1.5, 0.0]
            ], dtype=torch.float32)
        else:  # b == 4
            # Case 4: Mixed positioning
            tool_positions = torch.tensor([
                [0.5, 0.5, 0.2],
                [-0.5, -0.5, -0.2],
                [1.2, -0.8, 0.1],
                [-1.2, 0.8, -0.1],
                [0.0, 1.3, 0.0]
            ], dtype=torch.float32)
        
        states[b, 20:, :] = tool_positions
    
    return states, attrs, mask

def visualize_edges(states, attrs, Rr, Rs, batch_idx, connect_tools_all):
    """Visualize the edges for a single batch using open3d"""
    # Extract data for this batch
    positions = states[batch_idx].numpy()
    attributes = attrs[batch_idx].numpy()
    
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
            recv_idx = receiver_idx[0].item()
            send_idx = sender_idx[0].item()
            
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
    vis.create_window(window_name=f"Edges - Batch {batch_idx} - connect_tools_all={connect_tools_all}", visible=True)
    
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
    filename = f"edges_batch_{batch_idx}_connect_tools_{connect_tools_all}.png"
    vis.capture_screen_image(filename)
    vis.destroy_window()
    
    print(f"Saved visualization: {filename}")
    print(f"  Total edges: {len(lines)}")
    print(f"  Object particles: {np.sum(attributes == 0)}")
    print(f"  Tool particles: {np.sum(attributes == 1)}")
    
    # Count edge types
    obj_obj = sum(1 for color in edge_colors if np.allclose(color, [0, 1, 0]))
    obj_tool = sum(1 for color in edge_colors if np.allclose(color, [1, 1, 0]))
    tool_obj = sum(1 for color in edge_colors if np.allclose(color, [1, 0.5, 0]))
    tool_tool = sum(1 for color in edge_colors if np.allclose(color, [1, 0, 1]))
    
    print(f"  Object-Object edges: {obj_obj}")
    print(f"  Object-Tool edges: {obj_tool}")
    print(f"  Tool-Object edges: {tool_obj}")
    print(f"  Tool-Tool edges: {tool_tool}")
    print()

def main():
    print("Creating test data...")
    states, attrs, mask = create_test_batch()
    
    # Test parameters
    adj_thresh = 0.8  # distance threshold
    topk = 10        # max neighbors
    
    # Create tool mask
    tool_mask = (attrs > 0.5)
    
    print(f"Test configuration:")
    print(f"  Batch size: {states.shape[0]}")
    print(f"  Particles per batch: {states.shape[1]}")
    print(f"  Distance threshold: {adj_thresh}")
    print(f"  Top-k neighbors: {topk}")
    print()
    
    # Test both connect_tools_all settings
    for connect_tools_all in [False, True]:
        print(f"Testing with connect_tools_all = {connect_tools_all}")
        print("=" * 50)
        
        # Construct edges
        Rr, Rs = construct_edges_from_states_batch(
            states, adj_thresh, mask, tool_mask, topk, connect_tools_all
        )
        
        print(f"Edge tensor shapes: Rr={Rr.shape}, Rs={Rs.shape}")
        
        # Visualize each batch
        for batch_idx in range(states.shape[0]):
            visualize_edges(states, attrs, Rr, Rs, batch_idx, connect_tools_all)
    
    print("Visualization complete!")
    print("\nLegend:")
    print("  Blue points: Object particles")
    print("  Red points: Tool particles")
    print("  Green lines: Object-Object connections")
    print("  Yellow lines: Object-Tool connections")
    print("  Orange lines: Tool-Object connections")
    print("  Magenta lines: Tool-Tool connections (should not appear)")

if __name__ == "__main__":
    main() 