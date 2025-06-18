import os
import cv2
import h5py
import numpy as np
import open3d as o3d

from .qqtt.utils import logger, cfg
from .config_manager import PhysTwinConfig, create_common_parser
from .paths import *
from scripts.utils import parse_episodes

def video_from_data(cfg, f, episode_id, robot_loader, robot_pose, output_dir=None):
        logger.info(f"Starting video generation for episode {episode_id}")

        vis_cam_idx = 0
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        dynamic_meshes = robot_loader.get_finger_mesh(gripper_openness=0.0, transform=robot_pose)
        finger_vertex_counts = [len(mesh.vertices) for mesh in dynamic_meshes]

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        render_option = vis.get_render_option()
        render_option.point_size = 10.0

        for dynamic_mesh in dynamic_meshes:
            vis.add_geometry(dynamic_mesh)

        # Load data from shared HDF5 file
        episode_key = f'episode_{episode_id:06d}'
            
        if episode_key not in f:
            print(f"Episode {episode_id} not found in data file")
            return
                
        episode_group = f[episode_key]
        object_data = episode_group['object'][:]
        robot_data = episode_group['robot'][:]
        
        # Print episode metadata
        n_frames = episode_group.attrs['n_frames']
        n_obj_particles = episode_group.attrs['n_obj_particles']
        n_bot_particles = episode_group.attrs['n_bot_particles']
            
        print(f"Episode {episode_id}: {n_frames} frames")
        print(f"Object particles: {n_obj_particles}, Robot particles: {n_bot_particles}")

        try:
            object_type = episode_group.attrs['object_type']
            motion_type = episode_group.attrs['motion_type']
            print(f"Object type: {object_type}, Motion type: {motion_type}")
        except:
            pass


        # Initialize with first frame
        x = object_data[0]
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(x)
        object_pcd.paint_uniform_color([0, 0, 1])
        vis.add_geometry(object_pcd)

        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

        # Initialize video writer
        output_path = os.path.join(output_dir, f"{episode_id:06d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, cfg.FPS, (width, height))

        n_frames = len(object_data)
        for frame_count in range(n_frames):
            x = object_data[frame_count]
            x_robot = robot_data[frame_count]
            object_pcd.points = o3d.utility.Vector3dVector(x)
            vis.update_geometry(object_pcd) 

            cnt = 0
            for i, dynamic_mesh in enumerate(dynamic_meshes):
                vertices = x_robot[cnt : cnt + finger_vertex_counts[i]]
                dynamic_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                cnt += finger_vertex_counts[i]

            for i, dynamic_mesh in enumerate(dynamic_meshes):
                vis.update_geometry(dynamic_mesh)

            vis.poll_events()
            vis.update_renderer()
            static_image = np.asarray(
                vis.capture_screen_float_buffer(do_render=True)
            )
            static_image = (static_image * 255).astype(np.uint8)

            out.write(static_image)

            cv2.imshow("Generated Video", static_image)
            cv2.waitKey(1)

        # Release video writer
        out.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        logger.info(f"Video saved to {output_path}")


if __name__ == "__main__":
    # Create parser with common arguments
    parser = create_common_parser()
    
    # Add script-specific arguments
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(GENERATED_DATA_DIR / "data.h5"),
        help="Path to the shared HDF5 data file"
    )
    parser.add_argument("--episodes", nargs='+', type=str, default=["0-4"],
                       help="Episodes to generate videos for. Format: space-separated list (0 1 2 3 4) or range (0-4)")
    parser.add_argument("--output_dir", type=str, default=str(GENERATED_VIDEOS_DIR), help="Output directory for videos")
    args = parser.parse_args()

    # Initialize configuration - this replaces ~40 lines of setup code
    config = PhysTwinConfig(
        case_name=args.case_name,
        base_path=args.base_path,
        bg_img_path=args.bg_img_path,
        gaussian_path=args.gaussian_path
    )

    # Create robot loader and pose for video generation
    robot_loader = config.create_robot_loader()
    video_pose = config.get_robot_initial_pose("video")

    # Parse episode specification
    episode_list = parse_episodes(args.episodes)

    # Generate videos for specified episodes
    with h5py.File(args.data_file, 'r') as f:
        for episode_id in episode_list:
            video_from_data(cfg, f, episode_id, robot_loader, video_pose, args.output_dir)
