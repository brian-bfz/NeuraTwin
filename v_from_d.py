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

def video_from_data(cfg, save_dir, robot):
        logger.info("Starting video generation")

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

        x = torch.load(os.path.join(save_dir, "object", "x_0.pt"), weights_only=True)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(x.cpu().numpy())
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
        # Determine case_name and timestamp from save_dir
        case_name_timestamp = os.path.basename(save_dir.rstrip('/'))
        videos_dir = os.path.join(os.path.dirname(save_dir), "videos")
        os.makedirs(videos_dir, exist_ok=True)
        output_path = os.path.join(videos_dir, f"{case_name_timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, cfg.FPS, (width, height))

        for frame_count in range(len(os.listdir(os.path.join(save_dir, "object")))):
            x = torch.load(os.path.join(save_dir, "object", f"x_{frame_count}.pt"), weights_only=True)
            x_robot = torch.load(os.path.join(save_dir, "robot", f"x_{frame_count}.pt"), weights_only=True)
            object_pcd.points = o3d.utility.Vector3dVector(x.cpu().numpy())
            vis.update_geometry(object_pcd) 

            cnt = 0
            for i, dynamic_mesh in enumerate(dynamic_meshes):
                vertices = x_robot[cnt : cnt + finger_vertex_counts[i]].cpu().numpy()
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
        logger.info(f"Video saved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    # parser.add_argument(
    #     "--gaussian_path",
    #     type=str,
    #     default="./gaussian_output",
    # )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="./data/bg.png",
    )
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--start_episode", type=int, default=0)
    # parser.add_argument("--timestamp", type=str, required=True)
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

    # exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    # gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    # logger.set_log_file(path=base_dir, name="inference_log")
    # trainer = InvPhyTrainerWarp(
    #     data_path=f"{base_path}/{case_name}/final_data.pkl",
    #     base_dir=base_dir,
    #     pure_inference_mode=True,
    #     static_meshes=[],
    #     robot=sample_robot,
    # )

    # Generate video from saved data
    for i in [1999, 1642, 743, 1526]:
        save_dir = os.path.join("generated_data", f"{i}")
        video_from_data(cfg, save_dir, sample_robot)
    # trainer.video_from_data(
    #     gaussians_path, save_dir
    # )
