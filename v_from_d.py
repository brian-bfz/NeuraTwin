# import sys
# sys.path.append("./gaussian_splatting")
# from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
# import random
import numpy as np
import torch
from argparse import ArgumentParser
from SampleRobot import RobotPcSampler
# import glob
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

        # gaussians = GaussianModel(sh_degree=3)
        # gaussians.load_ply(gs_path)
        # gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        # gaussians.isotropic = True
        # current_pos = gaussians.get_xyz
        # current_rot = gaussians.get_rotation
        # use_white_background = True  # set to True for white background
        # bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # view = self._create_gs_view(w2c, intrinsic, height, width)
        # prev_x = None
        # relations = None
        # weights = None
        # image_path = cfg.bg_img_path
        # overlay = cv2.imread(image_path)
        # overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        # overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)

        # Render mesh 
        dynamic_meshes = robot.get_finger_mesh(0.0)
        finger_vertex_counts = [len(mesh.vertices) for mesh in dynamic_meshes]

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        render_option = vis.get_render_option()
        render_option.point_size = 10.0
        # for static_mesh in self.static_meshes:
        #     vis.add_geometry(static_mesh)

        for dynamic_mesh in dynamic_meshes:
            vis.add_geometry(dynamic_mesh)

        x = torch.load(os.path.join(save_dir, "object", "x_0.pt"), weights_only=True)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(x.cpu().numpy())
        object_pcd.paint_uniform_color([0, 0, 1])
        vis.add_geometry(object_pcd)

        # x_robot = torch.load(os.path.join(save_dir, "robot", "x_0.pt"), weights_only=True)
        # robot_pcd = o3d.geometry.PointCloud()
        # robot_pcd.points = o3d.utility.Vector3dVector(x_robot.cpu().numpy())
        # robot_pcd.paint_uniform_color([1, 0, 0])
        # vis.add_geometry(robot_pcd)

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

        for frame_count in range(len(os.listdir(os.path.join(save_dir, "gaussians")))):
            # 1. Load x, gaussians, and mesh
            x = torch.load(os.path.join(save_dir, "object", f"x_{frame_count}.pt"), weights_only=True)
            x_robot = torch.load(os.path.join(save_dir, "robot", f"x_{frame_count}.pt"), weights_only=True)
            # gaussians_data = torch.load(os.path.join(save_dir, "gaussians", f"gaussians_{frame_count}.pt"))
            # gaussians._xyz = gaussians_data['xyz']
            # gaussians._rotation = gaussians_data['rotation']

            # Load saved vertices and update dynamic_meshes
            # mesh_dir = os.path.join(save_dir, "meshes")
            # for i, dynamic_mesh in enumerate(dynamic_meshes):
            #     vertices_path = os.path.join(mesh_dir, f"finger_{i}_frame_{frame_count}.npy")
            #     vertices = np.load(vertices_path)
            #     # Update vertices while preserving mesh topology
            #     dynamic_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            #     # Update normals after changing vertices
            #     dynamic_mesh.compute_vertex_normals()

            # 2. Frame initialization and setup
            # frame = overlay.clone()

            # 3. Rendering
            # render with gaussians and paste the image on top of the frame
            # results = render_gaussian(view, gaussians, None, background)
            # rendering = results["render"]  # (4, H, W)
            # image = rendering.permute(1, 2, 0).detach()

            # Continue frame compositing
            # composition code from Hanxiao
            # image = image.clamp(0, 1)
            # if use_white_background:
            #     image_mask = torch.logical_and(
            #         (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
            #     )
            # else:
            #     image_mask = torch.logical_and(
            #         (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
            #     )
            # image[..., 3].masked_fill_(~image_mask, 0.0)

            # alpha = image[..., 3:4]
            # rgb = image[..., :3] * 255
            # frame = alpha * rgb + (1 - alpha) * frame
            # frame = frame.cpu().numpy()
            # image_mask = image_mask.cpu().numpy()
            # frame = frame.astype(np.uint8)

            # render robot and object
            # x_vis = x.clone()
            object_pcd.points = o3d.utility.Vector3dVector(x.cpu().numpy())
            vis.update_geometry(object_pcd) 

            cnt = 0
            for i, dynamic_mesh in enumerate(dynamic_meshes):
                vertices = x_robot[cnt : cnt + finger_vertex_counts[i]].cpu().numpy()
                dynamic_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                cnt += finger_vertex_counts[i]
            # robot_pcd.points = o3d.utility.Vector3dVector(x_robot.cpu().numpy())
            # vis.update_geometry(robot_pcd)

            for i, dynamic_mesh in enumerate(dynamic_meshes):
                vis.update_geometry(dynamic_mesh)

            vis.poll_events()
            vis.update_renderer()
            static_image = np.asarray(
                vis.capture_screen_float_buffer(do_render=True)
            )
            static_image = (static_image * 255).astype(np.uint8)
            # static_vis_mask = np.all(static_image == [255, 255, 255], axis=-1)
            # frame[~static_vis_mask] = static_image[~static_vis_mask]

            # Add shadows
            # final_shadow = get_simple_shadow(
            #     x, intrinsic, w2c, width, height, image_mask, light_point=[0, 0, -3]
            # )
            # frame[final_shadow] = (frame[final_shadow] * 0.95).astype(np.uint8)
            # final_shadow = get_simple_shadow(
            #     x, intrinsic, w2c, width, height, image_mask, light_point=[1, 0.5, -2]
            # )
            # frame[final_shadow] = (frame[final_shadow] * 0.97).astype(np.uint8)
            # final_shadow = get_simple_shadow(
            #     x, intrinsic, w2c, width, height, image_mask, light_point=[-3, -0.5, -5]
            # )
            # frame[final_shadow] = (frame[final_shadow] * 0.98).astype(np.uint8)

            # Convert frame to BGR before drawing circles
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame to video file
            # out.write(frame)
            out.write(static_image)

            # Display frame
            # cv2.imshow("Generated Video", frame)
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
    parser.add_argument("--timestamp", type=str, required=True)
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
    save_dir = os.path.join("generated_data", f"{case_name}_{args.timestamp}")
    video_from_data(cfg, save_dir, sample_robot)
    # trainer.video_from_data(
    #     gaussians_path, save_dir
    # )
