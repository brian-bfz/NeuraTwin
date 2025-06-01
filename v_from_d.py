from qqtt.utils import logger, cfg
import numpy as np
import torch
from argparse import ArgumentParser
from SampleRobot import RobotPcSampler
import os
import pickle
import json
import cv2
import open3d as o3d
import h5py

def video_from_data(cfg, data_file_path, episode_id, robot, output_dir=None):
        logger.info(f"Starting video generation for episode {episode_id}")

        vis_cam_idx = 0
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        dynamic_meshes = robot.get_finger_mesh(0.0)
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
        object_type = episode_group.attrs['object_type']
        motion_type = episode_group.attrs['motion_type']
            
        print(f"Episode {episode_id}: {n_frames} frames")
        print(f"Object particles: {n_obj_particles}, Robot particles: {n_bot_particles}")
        print(f"Object type: {object_type}, Motion type: {motion_type}")

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
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="generated_data/data.h5",
        help="Path to the shared HDF5 data file"
    )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="./data/bg.png",
    )
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--episode_ids", type=int, nargs="+", help="Episode IDs to generate videos for")
    parser.add_argument("--output_dir", type=str, default="generated_data/videos", help="Output directory for videos")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"./temp_experiments/{case_name}"

    # Load the robot finger
    urdf_path = "xarm/xarm7_with_gripper.urdf"
    R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    init_pose = np.eye(4)
    init_pose[:3, :3] = R
    init_pose[:3, 3] = [0.2, 0.0, 0.23]
    sample_robot = RobotPcSampler(
        urdf_path, link_names=["left_finger", "right_finger"], init_pose=init_pose
    )

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.bg_img_path = args.bg_img_path

    # Generate videos for specified episodes
    if args.episode_ids:
        with h5py.File(args.data_file, 'r') as f:
            for episode_id in args.episode_ids:
                video_from_data(cfg, f, episode_id, sample_robot, args.output_dir)
    else:
        # If no episodes specified, generate videos for first few episodes
        print("No episode IDs specified. Use --episode_ids to specify which episodes to process.")
        print("Example: python v_from_d.py --case_name your_case --episode_ids 0 1 2")
