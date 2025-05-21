from qqtt.data import RealData, SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
import open3d as o3d
import numpy as np
import torch
import wandb
import os
from tqdm import tqdm
import warp as wp
from scipy.spatial import KDTree
import pickle
import cv2
from pynput import keyboard
import pyrender
import trimesh

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from gs_render import (
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from gaussian_splatting.rotation_utils import quaternion_multiply, matrix_to_quaternion

from sklearn.cluster import KMeans
import copy
import time
import threading
import time


class InvPhyTrainerWarp:
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame=None,
        mask_path=None,
        velocity_path=None,
        pure_inference_mode=False,
        device="cuda:0",
        static_meshes=None,
        robot=None,
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.controller_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        self.static_meshes = static_meshes
        if static_meshes is not None:
            if robot is not None:
                # Extract the dynamic meshes from the robot
                self.robot = robot
                finger_meshes = self.robot.get_finger_mesh(0.0)
                self.dynamic_meshes = finger_meshes
                self.num_dynamic = len(self.dynamic_meshes)
                dynamic_vertices = [
                    np.asarray(finger_mesh.vertices) for finger_mesh in finger_meshes
                ]
                new_vertices = np.concatenate(dynamic_vertices, axis=0)
                new_vertices = torch.tensor(
                    new_vertices, dtype=torch.float32, device=cfg.device
                )
                self.dynamic_points = new_vertices
            else:
                self.dynamic_meshes = []
                self.dynamic_points = None

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

        if not pure_inference_mode:
            self.optimizer = torch.optim.Adam(
                [
                    wp.to_torch(self.simulator.wp_spring_Y),
                    wp.to_torch(self.simulator.wp_collide_elas),
                    wp.to_torch(self.simulator.wp_collide_fric),
                    wp.to_torch(self.simulator.wp_collide_object_elas),
                    wp.to_torch(self.simulator.wp_collide_object_fric),
                ],
                lr=cfg.base_lr,
                betas=(0.9, 0.99),
            )

            if "debug" not in cfg.run_name:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="phystwin_eval",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            else:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Debug",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            if not os.path.exists(f"{cfg.base_dir}/train"):
                # Create directory if it doesn't exist
                os.makedirs(f"{cfg.base_dir}/train")

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def train(self, start_epoch=-1):
        # Render the initial visualization
        video_path = f"{cfg.base_dir}/train/init.mp4"
        self.visualize_sim(save_only=True, video_path=video_path)

        best_loss = None
        best_epoch = None
        # Train the model with the physical simulator
        for i in range(start_epoch + 1, cfg.iterations):
            total_loss = 0.0
            if cfg.data_type == "real":
                total_chamfer_loss = 0.0
                total_track_loss = 0.0
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
            )
            with wp.ScopedTimer("backward"):
                for j in tqdm(range(1, cfg.train_frame)):
                    self.simulator.set_controller_target(j)
                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()

                    if cfg.use_graph:
                        wp.capture_launch(self.simulator.graph)
                    else:
                        if cfg.data_type == "real":
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_loss()
                            self.simulator.tape.backward(self.simulator.loss)
                        else:
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_simple_loss()
                            self.simulator.tape.backward(self.simulator.loss)

                    self.optimizer.step()

                    if cfg.data_type == "real":
                        chamfer_loss = wp.to_torch(
                            self.simulator.chamfer_loss, requires_grad=False
                        )
                        track_loss = wp.to_torch(
                            self.simulator.track_loss, requires_grad=False
                        )
                        total_chamfer_loss += chamfer_loss.item()
                        total_track_loss += track_loss.item()

                    loss = wp.to_torch(self.simulator.loss, requires_grad=False)
                    total_loss += loss.item()

                    if cfg.use_graph:
                        # Only need to clear the gradient, the tape is created in the graph
                        self.simulator.tape.zero()
                    else:
                        # Need to reset the compute graph and clear the gradient
                        self.simulator.tape.reset()
                    self.simulator.clear_loss()
                    # Set the intial state for the next step
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )

            total_loss /= cfg.train_frame - 1
            if cfg.data_type == "real":
                total_chamfer_loss /= cfg.train_frame - 1
                total_track_loss /= cfg.train_frame - 1
            wandb.log(
                {
                    "loss": total_loss,
                    "chamfer_loss": (
                        total_chamfer_loss if cfg.data_type == "real" else 0
                    ),
                    "track_loss": total_track_loss if cfg.data_type == "real" else 0,
                    "collide_else": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ).item(),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ).item(),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ).item(),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ).item(),
                },
                step=i,
            )

            logger.info(f"[Train]: Iteration: {i}, Loss: {total_loss}")

            if i % cfg.vis_interval == 0 or i == cfg.iterations - 1:
                video_path = f"{cfg.base_dir}/train/sim_iter{i}.mp4"
                self.visualize_sim(save_only=True, video_path=video_path)
                wandb.log(
                    {
                        "video": wandb.Video(
                            video_path,
                            format="mp4",
                            fps=cfg.FPS,
                        ),
                    },
                    step=i,
                )
                # Save the parameters
                cur_model = {
                    "epoch": i,
                    "num_object_springs": self.num_object_springs,
                    "spring_Y": torch.exp(
                        wp.to_torch(self.simulator.wp_spring_Y, requires_grad=False)
                    ),
                    "collide_elas": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                if best_loss == None or total_loss < best_loss:
                    # Remove old best model file if it exists
                    if best_loss is not None:
                        old_best_model_path = (
                            f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                        )
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)

                    # Update best loss and best epoch
                    best_loss = total_loss
                    best_epoch = i

                    # Save new best model
                    best_model_path = f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                    torch.save(cur_model, best_model_path)
                    logger.info(
                        f"Latest best model saved: epoch {best_epoch} with loss {best_loss}"
                    )

                torch.save(cur_model, f"{cfg.base_dir}/train/iter_{i}.pth")
                logger.info(
                    f"[Visualize]: Visualize the simulation at iteration {i} and save the model"
                )

        wandb.finish()

    def test(self, model_path=None):
        if model_path is not None:
            # Load the model
            logger.info(f"Load model from {model_path}")
            checkpoint = torch.load(model_path, map_location=cfg.device)

            spring_Y = checkpoint["spring_Y"]
            collide_elas = checkpoint["collide_elas"]
            collide_fric = checkpoint["collide_fric"]
            collide_object_elas = checkpoint["collide_object_elas"]
            collide_object_fric = checkpoint["collide_object_fric"]
            num_object_springs = checkpoint["num_object_springs"]

            assert (
                len(spring_Y) == self.simulator.n_springs
            ), "Check if the loaded checkpoint match the config file to connect the springs"

            self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
            self.simulator.set_collide(
                collide_elas.detach().clone(), collide_fric.detach().clone()
            )
            self.simulator.set_collide_object(
                collide_object_elas.detach().clone(),
                collide_object_fric.detach().clone(),
            )

        # Render the initial visualization
        video_path = f"{cfg.base_dir}/inference.mp4"
        save_path = f"{cfg.base_dir}/inference.pkl"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            save_trajectory=True,
            save_path=save_path,
        )

    def visualize_sim(
        self, save_only=True, video_path=None, save_trajectory=False, save_path=None
    ):
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        vertices = [
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
        ]

        with wp.ScopedTimer("simulate"):
            for i in tqdm(range(1, frame_len)):
                if cfg.data_type == "real":
                    self.simulator.set_controller_target(i, pure_inference=True)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(self.simulator.forward_graph)
                else:
                    self.simulator.step()
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

        vertices = torch.stack(vertices, dim=0)

        if save_trajectory:
            logger.info(f"Save the trajectory to {save_path}")
            vertices_to_save = vertices.cpu().numpy()
            with open(save_path, "wb") as f:
                pickle.dump(vertices_to_save, f)

        if not save_only:
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=True,
            )
        else:
            assert video_path is not None, "Please provide the video path to save"
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key.char)
        except (KeyError, AttributeError):
            try:
                self.pressed_keys.remove(str(key))
            except KeyError:
                pass

    def get_target_change(self):
        target_change = np.zeros((self.n_ctrl_parts, 3))
        for key in self.pressed_keys:
            if key in self.key_mappings:
                if (
                    key == "n"
                    or key == "m"
                    or key == "z"
                    or key == "x"
                    or key == "c"
                    or key == "v"
                ):
                    pass
                else:
                    idx, change = self.key_mappings[key]
                    target_change[idx] += change
        return target_change

    def get_finger_change(self):
        for key in self.pressed_keys:
            if key in self.key_mappings:
                if key == "n" or key == "m":
                    return self.key_mappings[key]
        return 0.0

    def get_rot_change(self):
        for key in self.pressed_keys:
            if key in self.key_mappings:
                if key == "z" or key == "x" or key == "c" or key == "v":
                    return np.array(self.key_mappings[key])
        return np.zeros(3)

    def init_control_ui(self):

        height = cfg.WH[1]
        width = cfg.WH[0]

        self.arrow_size = 30

        self.arrow_empty_orig = cv2.imread(
            "./assets/arrow_empty.png", cv2.IMREAD_UNCHANGED
        )[:, :, [2, 1, 0, 3]]
        self.arrow_1_orig = cv2.imread("./assets/arrow_1.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]
        self.arrow_2_orig = cv2.imread("./assets/arrow_2.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]

        spacing = self.arrow_size + 5

        self.bottom_margin = 25  # Margin from bottom of screen
        bottom_y = height - self.bottom_margin
        top_y = height - self.bottom_margin - spacing

        self.edge_buffer = self.bottom_margin
        set1_margin_x = self.edge_buffer  # Add buffer from left edge
        set2_margin_x = width - self.edge_buffer

        self.arrow_positions_set1 = {
            "q": (set1_margin_x + spacing * 3, top_y),  # Up
            "w": (set1_margin_x + spacing, top_y),  # Forward
            "a": (set1_margin_x, bottom_y),  # Left
            "s": (set1_margin_x + spacing, bottom_y),  # Backward
            "d": (set1_margin_x + spacing * 2, bottom_y),  # Right
            "e": (set1_margin_x + spacing * 3, bottom_y),  # Down
        }

        self.arrow_positions_set2 = {
            "u": (set2_margin_x - spacing * 3, top_y),  # Up
            "i": (set2_margin_x - spacing * 1, top_y),  # Forward
            "j": (set2_margin_x - spacing * 2, bottom_y),  # Left
            "k": (set2_margin_x - spacing * 1, bottom_y),  # Backward
            "l": (set2_margin_x, bottom_y),  # Right
            "o": (set2_margin_x - spacing * 3, bottom_y),  # Down
        }

        self.interm_size = 512
        self.rotations = {
            "w": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Forward
            "a": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1
            ),  # Left
            "s": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Backward
            "d": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270, 1
            ),  # Right
            "q": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Up
            "e": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Down
            "i": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Forward
            "j": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1
            ),  # Left
            "k": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Backward
            "l": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270, 1
            ),  # Right
            "u": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Up
            "o": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Down
        }

        self.hand_left = cv2.imread("./assets/Picture2.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]
        self.hand_right = cv2.imread("./assets/Picture1.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]

        self.hand_left_pos = torch.tensor([0.0, 0.0, 0.0], device=cfg.device)
        self.hand_right_pos = torch.tensor([0.0, 0.0, 0.0], device=cfg.device)

        # pre-compute all rotated arrows to avoid aliasing
        self.arrow_rotated_filled = {}
        self.arrow_rotated_empty = {}
        for key in self.arrow_positions_set1:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_1_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_empty_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
        for key in self.arrow_positions_set2:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_2_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_empty_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )

    def _rotate_arrow(self, arrow, key):
        rotation_matrix = self.rotations[key]
        rotated = cv2.warpAffine(
            arrow,
            rotation_matrix,
            (self.interm_size, self.interm_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        return rotated

    def _overlay_arrow(self, background, arrow, position, key, filled=True):
        x, y = position

        if filled:
            rotated_arrow = self.arrow_rotated_filled[key].copy()
        else:
            rotated_arrow = self.arrow_rotated_empty[key].copy()

        h, w = rotated_arrow.shape[:2]

        roi_x = max(0, x - w // 2)
        roi_y = max(0, y - h // 2)
        roi_w = min(w, background.shape[1] - roi_x)
        roi_h = min(h, background.shape[0] - roi_y)

        arrow_x = max(0, w // 2 - x)
        arrow_y = max(0, h // 2 - y)

        roi = background[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        arrow_roi = rotated_arrow[arrow_y : arrow_y + roi_h, arrow_x : arrow_x + roi_w]

        alpha = arrow_roi[:, :, 3] / 255.0

        for c in range(3):  # Apply for RGB channels
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + arrow_roi[:, :, c] * alpha

        background[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi

        return background

    def _overlay_hand_at_position(
        self, frame, target_points, x_axis, hand_size, hand_icon, align="center"
    ):
        result = frame.copy()

        mean_pos = target_points.cpu().numpy().mean(axis=0)

        pixel_mean = self.projection @ np.append(mean_pos, 1)
        pixel_mean = pixel_mean[:2] / pixel_mean[2]

        pos_1 = np.append(mean_pos + hand_size * x_axis, 1)
        pixel_1 = self.projection @ pos_1
        pixel_1 = pixel_1[:2] / pixel_1[2]

        pos_2 = np.append(mean_pos - hand_size * x_axis, 1)
        pixel_2 = self.projection @ pos_2
        pixel_2 = pixel_2[:2] / pixel_2[2]

        icon_size = int(np.linalg.norm(pixel_1[:2] - pixel_2[:2]) / 2)
        icon_size = max(1, min(icon_size, 100))

        resized_icon = cv2.resize(hand_icon, (icon_size, icon_size))
        h, w = resized_icon.shape[:2]
        x, y = int(pixel_mean[0]), int(pixel_mean[1])

        if align == "top-left":
            roi_x = int(max(0, x - w * 0.15))
            roi_y = int(max(0, y - h * 0.1))
        if align == "top-right":
            roi_x = int(max(0, x - w + w * 0.15))
            roi_y = int(max(0, y - h * 0.1))
        if align == "center":
            roi_x = int(max(0, x - w // 2))
            roi_y = int(max(0, y - h // 2))
        roi_w = min(w, result.shape[1] - roi_x)
        roi_h = min(h, result.shape[0] - roi_y)

        if roi_w <= 0 or roi_h <= 0:
            return result

        icon_x = max(0, w // 2 - x)
        icon_y = max(0, h // 2 - y)

        roi = result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        icon_roi = resized_icon[icon_y : icon_y + roi_h, icon_x : icon_x + roi_w]

        if icon_roi.size == 0 or roi.shape[:2] != icon_roi.shape[:2]:
            return result

        if icon_roi.shape[2] == 4:
            alpha = icon_roi[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + icon_roi[:, :, c] * alpha
            result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi
        else:
            result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = icon_roi[:, :, :3]

        return result

    def _overlay_hand_icons(self, frame):
        if self.n_ctrl_parts not in [1, 2]:
            raise ValueError("Only support 1 or 2 control parts")

        result = frame.copy()

        c2w = np.linalg.inv(self.w2c)
        x_axis = c2w[:3, 0]
        self.projection = self.intrinsic @ self.w2c[:3, :]
        hand_size = 0.1  # size in physical space (in meters)

        if self.n_ctrl_parts == 1:
            current_target = self.hand_left_pos.unsqueeze(0)
            # align = 'top-right'
            align = "center"
            result = self._overlay_hand_at_position(
                result, current_target, x_axis, hand_size, self.hand_left, align
            )
        else:
            for i in range(2):
                current_target = (
                    self.hand_left_pos.unsqueeze(0)
                    if i == 0
                    else self.hand_right_pos.unsqueeze(0)
                )
                # align = 'top-right' if i == 0 else 'top-left'
                align = "center"
                hand_icon = self.hand_left if i == 0 else self.hand_right
                result = self._overlay_hand_at_position(
                    result, current_target, x_axis, hand_size, hand_icon, align
                )

        return result

    def update_frame(self, frame, pressed_keys, overlay_hand=True):
        result = frame.copy()

        if overlay_hand:
            result = self._overlay_hand_icons(result)

        # overlay an transparent white mask on the bottom left and bottom right corners with width trans_width, and height trans_height
        trans_width = 160
        trans_height = 120
        overlay = result.copy()

        bottom_left_pt1 = (0, cfg.WH[1] - trans_height)
        bottom_left_pt2 = (trans_width, cfg.WH[1])
        cv2.rectangle(overlay, bottom_left_pt1, bottom_left_pt2, (255, 255, 255), -1)

        if self.n_ctrl_parts == 2:
            bottom_right_pt1 = (cfg.WH[0] - trans_width, cfg.WH[1] - trans_height)
            bottom_right_pt2 = (cfg.WH[0], cfg.WH[1])
            cv2.rectangle(
                overlay, bottom_right_pt1, bottom_right_pt2, (255, 255, 255), -1
            )

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        # Draw all buttons for Set 1 (left side)
        for key, pos in self.arrow_positions_set1.items():
            if key in pressed_keys:
                result = self._overlay_arrow(result, None, pos, key, filled=True)
            else:
                result = self._overlay_arrow(result, None, pos, key, filled=False)

        # Draw all buttons for Set 2 (right side)
        if self.n_ctrl_parts == 2:
            for key, pos in self.arrow_positions_set2.items():
                if key in pressed_keys:
                    result = self._overlay_arrow(result, None, pos, key, filled=True)
                else:
                    result = self._overlay_arrow(result, None, pos, key, filled=False)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        control1_x = self.edge_buffer  # hard coded for now
        control2_x = cfg.WH[0] - self.edge_buffer - 113  # hard coded for now
        text_y = (
            cfg.WH[1] - self.arrow_size * 2 - self.bottom_margin - 10
        )  # hard coded for now
        cv2.putText(
            result,
            "Left Hand",
            (control1_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        if self.n_ctrl_parts == 2:
            cv2.putText(
                result,
                "Right Hand",
                (control2_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return result

    def _find_closest_point(self, target_points):
        """Find the closest structure point to any of the target points."""
        dist_matrix = torch.sum(
            (target_points.unsqueeze(1) - self.structure_points.unsqueeze(0)) ** 2,
            dim=2,
        )
        min_dist_per_ctrl_pts, min_indices = torch.min(dist_matrix, dim=1)
        min_idx = min_indices[torch.argmin(min_dist_per_ctrl_pts)]
        return self.structure_points[min_idx].unsqueeze(0)
    
    def robot_translation(self, offset_dist, keep_off, multiplier, speed):
        """
        Generate random robot translational movement parameters.
        
        Args:
            offset_dist: Distance to offset from the selected object point
            keep_off: Minimum distance to keep from all object points
            multiplier: Multiplier for the target movement distance
            speed: Speed of the robot movement
            
        Returns:
            tuple: (initial_translation, target_changes)
                - translation: numpy array of shape (3,) for initial robot position
                - target_changes: numpy array for target movement
        """
        # Get object points from the simulator
                
        max_attempts = 100
        for _ in range(max_attempts):
            # 1. Select a random point on the object
            random_idx = torch.randint(0, len(self.init_vertices), (1,)).item()
            selected_point = self.init_vertices[random_idx]
            
            # 2. Select a random direction
            random_angle = torch.rand(1, device=self.init_vertices.device) * 2 * np.pi - np.pi
            random_direction = torch.tensor([
                torch.cos(random_angle),
                torch.sin(random_angle),
                torch.tensor(0.0, device=self.init_vertices.device)  # Assuming movement in the XY plane
            ], device=self.init_vertices.device)
            random_direction = random_direction / torch.norm(random_direction)
            
            # 3. Set starting position with offset
            start_position = selected_point + random_direction * offset_dist
            
            distances = torch.norm(self.init_vertices - start_position, dim=1)
            min_distance = torch.min(distances).item()
            
            if min_distance >= keep_off:
                # Valid position found
                break
        else:
            # If we couldn't find a valid position after max attempts
            raise RuntimeError("Could not find valid robot position after maximum attempts")
        
        # 5. Find robot's current position and calculate translation
        all_vertices = np.concatenate([vertices for vertices in self.dynamic_vertices], axis=0)
        robot_position = np.mean(all_vertices, axis=0)
        translation = start_position.cpu().numpy() - robot_position
        translation = translation.reshape(self.n_ctrl_parts, 3)
        
        # 6. Calculate total movement distance and number of frames needed
        movement_vector = -random_direction.cpu().numpy() * offset_dist * multiplier
        total_distance = np.linalg.norm(movement_vector)
        n_frames = max(1, int(np.ceil(total_distance / speed)))
        
        # 7. Divide the movement into n_frames steps
        step_movement = movement_vector / n_frames
        target_changes = np.zeros((n_frames, self.n_ctrl_parts, 3))
        for i in range(n_frames):
            target_changes[i, 0] = step_movement
                
        return translation.astype(np.float32), target_changes.astype(np.float32)
    
    def generate_data(
        self, model_path, gs_path, n_ctrl_parts=1, save_dir=None, custom_control_points=None, 
    ):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        # num_object_springs = checkpoint["num_object_springs"]

        # Initialize control parts
        self.n_ctrl_parts = n_ctrl_parts
        # if n_ctrl_parts > 1:
        #     kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
        #     cluster_labels = kmeans.fit_predict(current_target.cpu().numpy())
        #     masks_ctrl_pts = []
        #     for i in range(n_ctrl_parts):
        #         mask = cluster_labels == i
        #         masks_ctrl_pts.append(torch.from_numpy(mask))
        #     self.mask_ctrl_pts = masks_ctrl_pts
        # else:
        #     self.mask_ctrl_pts = None

        # set robot position and movement parameters
        self.dynamic_vertices = [
            np.asarray(finger_mesh.vertices) for finger_mesh in self.dynamic_meshes
        ]

        translation, target_changes = self.robot_translation(offset_dist=0.2, keep_off=0.05, multiplier=2.0, speed=0.005)
        n_frames = target_changes.shape[0]
        current_finger = 0.0
        finger_changes = np.zeros((n_frames), dtype=np.float32)
        accumulate_trans = np.zeros((n_ctrl_parts, 3), dtype=np.float32)
        accumulate_rot = torch.eye(3, dtype=torch.float32, device=cfg.device)
        rot_changes = np.zeros((n_frames, 3), dtype=np.float32)
        self.robot.change_init_pose(translation)

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        spring_Y = spring_Y[: self.num_object_springs]
        self.init_springs = self.init_springs[: self.num_object_springs]
        self.init_rest_lengths = self.init_rest_lengths[: self.num_object_springs]
        self.init_vertices = self.init_vertices[: self.num_all_points]
        self.init_masses = self.init_masses[: self.num_all_points]
        self.controller_points = None

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
            static_meshes=self.dynamic_meshes + self.static_meshes,
            dynamic_points=self.dynamic_points,
        )

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        ###########################################################################

        logger.info("Starting data generation")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        # prev_x = wp.to_torch(
        #     self.simulator.wp_states[0].wp_x, requires_grad=False
        # ).clone()

        # if custom_control_points is not None:
        #     current_target = custom_control_points
        #     # TODO: change current_target to the points selected by select_cp.py
        #     # TODO: delete old connections between controller points and object points
        #     # TODO: add new connections between the new controller points and object points using code from outdomain_inference
        # else:   
        #     current_target = self.simulator.controller_points[0]

        # prev_target = current_target

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        prev_x = None
        relations = None
        weights = None

        # Initialize key mappings for target movement
        # self.key_mappings = {
        #     # Set 1 controls
        #     "w": (0, np.array([0.005, 0, 0])),
        #     "s": (0, np.array([-0.005, 0, 0])),
        #     "a": (0, np.array([0, -0.005, 0])),
        #     "d": (0, np.array([0, 0.005, 0])),
        #     "e": (0, np.array([0, 0, 0.005])),
        #     "q": (0, np.array([0, 0, -0.005])),
        #     # Set 2 controls
        #     "i": (1, np.array([0.005, 0, 0])),
        #     "k": (1, np.array([-0.005, 0, 0])),
        #     "j": (1, np.array([0, -0.005, 0])),
        #     "l": (1, np.array([0, 0.005, 0])),
        #     "o": (1, np.array([0, 0, 0.005])),
        #     "u": (1, np.array([0, 0, -0.005])),
        #     # Set the finger
        #     "n": 0.05,
        #     "m": -0.05,
        #     # Set the rotation
        #     "z": [0, 0, 2.0 / 180 * np.pi],
        #     "x": [0, 0, -2.0 / 180 * np.pi],
        #     "c": [2.0 / 180 * np.pi, 0, 0],
        #     "v": [-2.0 / 180 * np.pi, 0, 0],
        # }
        # self.pressed_keys = set()

        frame_count = 0

        self.dynamic_vertices = [
            np.asarray(finger_mesh.vertices) for finger_mesh in self.dynamic_meshes
        ]

        origin_force_judge = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.float32, device=cfg.device
        )
        current_force_judge = origin_force_judge.clone()
        current_trans_dynamic_points = self.dynamic_points
        # close_flag = False
        # is_closing = True

        

        for i in range(n_frames):
            # self.pressed_keys.clear()
            # for key in pressed_keys:
            #     self.pressed_keys.add(key)

            # 1. Simulator step
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # collision_forces = wp.to_torch(
            #     self.simulator.collision_forces, requires_grad=False
            # )[: self.num_dynamic]
            # filter_forces = torch.einsum(
            #     "ij,ij->i", collision_forces, current_force_judge
            # )
            # if torch.all(filter_forces > 3e4):
            #     close_flag = False
            # else:
            #     close_flag = True

            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            # Save initial state at the start of each loop
            if save_dir is not None:
                # Save x tensor
                torch.save(x, os.path.join(save_dir, "x", f"x_{frame_count}.pt"))
                # Save gaussians
                torch.save({
                    'xyz': gaussians._xyz,
                    'rotation': gaussians._rotation,
                    'frame_count': frame_count
                }, os.path.join(save_dir, "gaussians", f"gaussians_{frame_count}.pt"))
                # Save dynamic vertices
                for i, vertices in enumerate(self.dynamic_vertices):
                    vertices_path = os.path.join(save_dir, f"meshes/finger_{i}_frame_{frame_count}.npy")
                    np.save(vertices_path, vertices)

            if prev_x is not None:
                with torch.no_grad():

                    prev_particle_pos = prev_x
                    cur_particle_pos = x

                    if relations is None:
                        relations = get_topk_indices(
                            prev_x, K=16
                        )  # only computed in the first iteration

                    if weights is None:
                        weights, weights_indices = knn_weights_sparse(
                            prev_particle_pos, current_pos, K=16
                        )  # only computed in the first iteration

                    weights = calc_weights_vals_from_indices(
                        prev_particle_pos, current_pos, weights_indices
                    )

                    current_pos, current_rot, _ = interpolate_motions_speedup(
                        bones=prev_particle_pos,
                        motions=cur_particle_pos - prev_particle_pos,
                        relations=relations,
                        weights=weights,
                        weights_indices=weights_indices,
                        xyz=current_pos,
                        quat=current_rot,
                    )

                    # update gaussians with the new positions and rotations
                    gaussians._xyz = current_pos
                    gaussians._rotation = current_rot

            prev_x = x.clone()
            # prev_target = current_target

            # =====================robot stuff=====================
            # Update the changes
            # target_change = self.get_target_change()
            # finger_change = self.get_finger_change()
            # rot_change = self.get_rot_change()

            # Calculate the substep vertices
            # if finger_change > 0:
            #     is_closing = False
            # elif finger_change < 0:
            #     is_closing = True
                
            # if is_closing:
            #     if close_flag == True:
            #         finger_change = -0.05
            #     else:
            #         finger_change = 0.0
            # else:
            #     finger_change = 0.05

            current_finger += finger_changes[i]
            current_finger = max(0.0, min(1.0, current_finger))

            accumulate_trans += target_changes[i]
            finger_meshes = self.robot.get_finger_mesh(current_finger)
            dynamic_vertices = torch.tensor(
                [np.asarray(finger_mesh.vertices) + accumulate_trans[0] for finger_mesh in finger_meshes],
                device=cfg.device,
                dtype=torch.float32
            ) # Directly converting from list to tensor is okay because there are only 2 fingers. 
            prev_trans_dynamic_points = current_trans_dynamic_points
            current_trans_dynamic_points = torch.reshape(dynamic_vertices, (-1, 3))

            # Caclulate the interpolated points considering finger and translation
            ratios = (
                torch.linspace(
                    1, cfg.num_substeps, cfg.num_substeps, device=cfg.device
                ).view(-1, 1, 1)
                / cfg.num_substeps
            )
            interpolated_trans_dynamic_points = (
                prev_trans_dynamic_points.unsqueeze(0)
                + (current_trans_dynamic_points - prev_trans_dynamic_points).unsqueeze(
                    0
                )
                * ratios
            )
            interpolated_center = torch.mean(interpolated_trans_dynamic_points, dim=1)

            # Do the rotation on the interpolated points
            new_rot = torch.tensor(rot_changes[i], dtype=torch.float32, device=cfg.device)
            interpolated_rot_angle =  new_rot.unsqueeze(
                0
            ) * ratios.reshape(-1, 1)
            interpolated_rot_temp = axis_angle_to_matrix(interpolated_rot_angle)
            interpolated_rot_mat = torch.matmul(accumulate_rot.unsqueeze(0), interpolated_rot_temp)
            accumulate_rot = interpolated_rot_mat[-1]

            interpolated_dynamic_points = (
                interpolated_trans_dynamic_points - interpolated_center.unsqueeze(1)
            ) @ interpolated_rot_mat.permute(0, 2, 1) + interpolated_center.unsqueeze(1)
            self.dynamic_vertices = (
                interpolated_dynamic_points[-1]
                .reshape([-1] + list(self.dynamic_vertices[0].shape))
                .cpu()
                .numpy()
            )

            # Calculate the velocity and omega for calculating relative velocity
            dynamic_velocity = torch.tensor(
                target_changes[i][0] / (2 * cfg.dt * cfg.num_substeps),
                dtype=torch.float32,
                device=cfg.device,
            )
            dynamic_omega = torch.tensor(
                rot_changes[i] / (2 * cfg.dt * cfg.num_substeps),
                dtype=torch.float32,
                device=cfg.device,
            )
            # print(dynamic_omega)

            # Update the force judge direction
            current_force_judge = origin_force_judge.clone() @ interpolated_rot_mat[-1]

            # Update the simulator with the gripper changes
            self.simulator.set_mesh_interactive(
                interpolated_dynamic_points,
                interpolated_center,
                dynamic_velocity,
                dynamic_omega,
            )

            frame_count += 1

    def video_from_data(
        self, gs_path, save_dir
    ):
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
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        render_option = vis.get_render_option()
        render_option.point_size = 10.0
        for static_mesh in self.static_meshes:
            vis.add_geometry(static_mesh)

        for dynamic_mesh in self.dynamic_meshes:
            vis.add_geometry(dynamic_mesh)

        x = torch.load(os.path.join(save_dir, "x", f"x_{frame_count}.pt"), weights_only=True)
        x_vis = x.clone()
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(x_vis.cpu().numpy())
        object_pcd.paint_uniform_color([0, 0, 1])
        # object_pcd.paint_uniform_color([1, 1, 1])
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

        for frame_count in range(len(os.listdir(os.path.join(save_dir, "gaussians")))):
            # 1. Load x, gaussians, and mesh
            x = torch.load(os.path.join(save_dir, "x", f"x_{frame_count}.pt"), weights_only=True)
            # gaussians_data = torch.load(os.path.join(save_dir, "gaussians", f"gaussians_{frame_count}.pt"))
            # gaussians._xyz = gaussians_data['xyz']
            # gaussians._rotation = gaussians_data['rotation']

            # Load saved vertices and update dynamic_meshes
            mesh_dir = os.path.join(save_dir, "meshes")
            for i, dynamic_mesh in enumerate(self.dynamic_meshes):
                vertices_path = os.path.join(mesh_dir, f"finger_{i}_frame_{frame_count}.npy")
                vertices = np.load(vertices_path)
                # Update vertices while preserving mesh topology
                dynamic_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                # Update normals after changing vertices
                dynamic_mesh.compute_vertex_normals()

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
            x_vis = x.clone()
            object_pcd.points = o3d.utility.Vector3dVector(x_vis.cpu().numpy())
            vis.update_geometry(object_pcd)

            for i, dynamic_mesh in enumerate(self.dynamic_meshes):
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
    
    def select_control_points(self, model_path, gs_path, n_ctrl_parts):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        # Get initial state
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        # Run one step to initialize the states
        self.simulator.step()
        x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
        print("Initial state shape:", x.shape)
        print("Initial state values:", x)

        # Setup visualization
        vis_cam_idx = 0
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        # Load and setup gaussians
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True

        # Setup rendering
        use_white_background = True
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view = self._create_gs_view(w2c, intrinsic, height, width)
        
        # Load background image
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Initialize control parts
        self.n_ctrl_parts = n_ctrl_parts
        self.scale_factors = 1.0
        assert self.n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        self.w2c = w2c
        self.intrinsic = intrinsic

        def generate_frame(x, control_points):
            frame = overlay.copy()
            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach().cpu().numpy()

            # Composite rendering with background
            image = image.clip(0, 1)
            if use_white_background:
                image_mask = np.logical_and(
                    (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = np.logical_and(
                    (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            image[~image_mask, 3] = 0

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.astype(np.uint8)

            # Add shadows
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[0, 0, -3]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.95).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[1, 0.5, -2]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.97).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[-3, -0.5, -5]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.98).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw control points if they exist
            for i, points in enumerate(control_points):
                points_homogeneous = np.hstack([points.cpu().numpy(), np.ones((points.shape[0], 1))])
                points_camera = (w2c @ points_homogeneous.T).T
                points_pixels = (intrinsic @ points_camera[:, :3].T).T
                points_pixels = points_pixels[:, :2] / points_pixels[:, 2:3]
                pixel_coords = points_pixels.astype(int)

                # Filter points that are within the image bounds
                valid_mask = (
                    (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
                    (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
                )
                valid_pixel_coords = pixel_coords[valid_mask]

                # Draw circles for control points (red for first set, green for second set)
                color = (0, 0, 255) if i == 0 else (0, 255, 0)
                for x, y in valid_pixel_coords:
                    cv2.circle(frame, (x, y), 3, color, -1)

            return frame, image

        # Local variables that will be modified by the click handler
        count = 0
        control_points = []
        
        def click_handler(event, x1, y1, flags, param):
            nonlocal count, control_points
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert 2D pixel coordinates to a ray in 3D camera space
                pixel_coords = np.array([x1, y1, 1])
                camera_ray = np.linalg.inv(intrinsic) @ pixel_coords
                camera_ray = camera_ray / np.linalg.norm(camera_ray)
                
                # Transform ray to world space
                world_ray = np.linalg.inv(w2c[:3, :3]) @ camera_ray
                camera_center = -np.linalg.inv(w2c[:3, :3]) @ w2c[:3, 3]
                
                # Find intersection with object points
                distances = []
                for point in x.cpu().numpy():
                    # Vector from camera center to point
                    v = point - camera_center
                    # Project this vector onto the ray direction
                    t = np.dot(v, world_ray)
                    # If t < 0, point is behind camera
                    if t < 0:
                        continue
                    # Calculate perpendicular distance from point to ray
                    perp_dist = np.linalg.norm(v - t * world_ray)
                    # If point is close enough to ray, consider it a potential intersection
                    if perp_dist < 0.01: # threshold in world units
                        distances.append((t, perp_dist))
                
                if not distances:
                    print("No valid intersection found")
                    return
                    
                # Use the closest intersection point (minimum perpendicular distance)
                depth, perp_dist = min(distances, key=lambda x: x[1])
                world_coords = camera_center + depth * world_ray
                
                count += 1
                print(f"Selected control point {count}: {world_coords}")
                
                # Convert to tensor and store
                new_point = torch.tensor([world_coords], dtype=torch.float32, device=cfg.device)
                control_points.append(new_point)

        # Create the window first
        cv2.namedWindow("Select Control Points")
        cv2.setMouseCallback("Select Control Points", click_handler)

        # Main loop for frame generation and display
        while count < self.n_ctrl_parts:
            frame, image = generate_frame(x, control_points)
            cv2.imshow("Select Control Points", frame)
            cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        
        return control_points

    def visualize_control_points(self, control_points, x):
        """Visualize object points and new control points using Open3D."""
        # Create point clouds for visualization
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(x.cpu().numpy())
        object_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for object points
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add geometries
        vis.add_geometry(object_pcd)
        vis.add_geometry(coordinate_frame)
        
        # Add control points with different colors
        for i, points in enumerate(control_points):
            ctrl_pcd = o3d.geometry.PointCloud()
            ctrl_pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            # Use red for first set, green for second set
            ctrl_pcd.paint_uniform_color([1, 0, 0] if i == 0 else [0, 1, 0])
            vis.add_geometry(ctrl_pcd)
        
        # Set default camera view
        view_control = vis.get_view_control()
        view_control.set_front([1, 0, -2])
        view_control.set_up([0, 0, -1])
        view_control.set_zoom(0.5)
        
        # Run visualization
        vis.run()
        vis.destroy_window()

    def interactive_playground(
        self, model_path, gs_path, n_ctrl_parts=1, inv_ctrl=False
    ):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        ###########################################################################

        logger.info("Party Time Start!!!!")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        current_target = self.simulator.controller_points[0]
        prev_target = current_target

        vis_controller_points = current_target.cpu().numpy()

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)

        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            N = vis_controller_points.shape[0]
            masks_ctrl_pts = []
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                masks_ctrl_pts.append(torch.from_numpy(mask))
            # project the center of the cluster to the object to the image space, those on the left will be mask 1
            center1 = np.mean(vis_controller_points[masks_ctrl_pts[0]], axis=0)
            center2 = np.mean(vis_controller_points[masks_ctrl_pts[1]], axis=0)
            center1 = np.concatenate([center1, [1]])
            center2 = np.concatenate([center2, [1]])
            proj_mat = intrinsic @ w2c[:3, :]
            center1 = proj_mat @ center1
            center2 = proj_mat @ center2
            center1 = center1 / center1[-1]
            center2 = center2 / center2[-1]
            if center1[0] > center2[0]:
                print("Switching the control parts")
                masks_ctrl_pts = [masks_ctrl_pts[1], masks_ctrl_pts[0]]
        else:
            masks_ctrl_pts = None
        self.n_ctrl_parts = n_ctrl_parts
        self.mask_ctrl_pts = masks_ctrl_pts
        self.scale_factors = 1.0
        assert n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        print("UI Controls:")
        print("- Set 1: WASD (XY movement), QE (Z movement)")
        print("- Set 2: IJKL (XY movement), UO (Z movement)")
        self.inv_ctrl = -1.0 if inv_ctrl else 1.0
        self.key_mappings = {
            # Set 1 controls
            "w": (0, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "s": (0, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "a": (0, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "d": (0, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            # Set 2 controls
            "i": (1, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "k": (1, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "j": (1, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "l": (1, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "o": (1, np.array([0, 0, 0.005])),
            "u": (1, np.array([0, 0, -0.005])),
        }
        self.pressed_keys = set()
        self.w2c = w2c
        self.intrinsic = intrinsic
        self.init_control_ui()
        if n_ctrl_parts > 1:
            hand_positions = []
            for i in range(2):
                target_points = torch.from_numpy(
                    vis_controller_points[self.mask_ctrl_pts[i]]
                ).to("cuda")
                hand_positions.append(self._find_closest_point(target_points))
            self.hand_left_pos, self.hand_right_pos = hand_positions
        else:
            target_points = torch.from_numpy(vis_controller_points).to("cuda")
            self.hand_left_pos = self._find_closest_point(target_points)

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.target_change = np.zeros((n_ctrl_parts, 3))

        ############## Temporary timer ##############
        import time

        class Timer:
            def __init__(self, name):
                self.name = name
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None
                self.use_cuda = torch.cuda.is_available()

            def start(self):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    self.cuda_start_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_end_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_start_event.record()
                self.start_time = time.time()

            def stop(self):
                if self.use_cuda:
                    self.cuda_end_event.record()
                    torch.cuda.synchronize()
                    self.elapsed = (
                        self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000
                    )  # convert ms to seconds
                else:
                    self.elapsed = time.time() - self.start_time
                return self.elapsed

            def reset(self):
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None

        sim_timer = Timer("Simulator")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        interp_timer = Timer("Full Motion Interpolation")
        total_timer = Timer("Total Loop")
        knn_weights_timer = Timer("KNN Weights")
        motion_interp_timer = Timer("Motion Interpolation")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "rendering": [],
            "frame_compositing": [],
            "full_motion_interpolation": [],
            "total": [],
            "knn_weights": [],
            "motion_interp": [],
        }

        # Number of frames to average over for stats
        STATS_WINDOW = 10
        frame_count = 0

        ############## End Temporary timer ##############
        if self.static_meshes is not None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=width, height=height)
            render_option = vis.get_render_option()
            render_option.point_size = 10.0

            for static_mesh in self.static_meshes:
                vis.add_geometry(static_mesh)

            x_vis = wp.to_torch(
                self.simulator.wp_states[0].wp_x, requires_grad=False
            ).clone()
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(x_vis.cpu().numpy())
            # object_pcd.paint_uniform_color([1, 0, 0])
            object_pcd.paint_uniform_color([1, 1, 1])
            vis.add_geometry(object_pcd)

            # o3d.visualization.draw_geometries([object_pcd] + self.static_meshes)

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

            # vis_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            # cv2.imshow("test", vis_image)
            # cv2.waitKey(0)

        while True:

            total_timer.start()

            # 1. Simulator step

            sim_timer.start()

            self.simulator.set_controller_interactive(prev_target, current_target)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            sim_time = sim_timer.stop()
            component_times["simulator"].append(sim_time)

            torch.cuda.synchronize()

            # 2. Frame initialization and setup

            frame_timer.start()

            frame = overlay.clone()

            frame_setup_time = (
                frame_timer.stop()
            )  # We'll accumulate times for frame compositing

            torch.cuda.synchronize()

            # 3. Rendering
            render_timer.start()

            # render with gaussians and paste the image on top of the frame
            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach()

            render_time = render_timer.stop()
            component_times["rendering"].append(render_time)

            torch.cuda.synchronize()

            # Continue frame compositing
            frame_timer.start()

            image = image.clamp(0, 1)
            if use_white_background:
                image_mask = torch.logical_and(
                    (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = torch.logical_and(
                    (image != 0.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
            image[..., 3].masked_fill_(~image_mask, 0.0)

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.cpu().numpy()
            image_mask = image_mask.cpu().numpy()
            frame = frame.astype(np.uint8)

            if self.static_meshes is not None:
                # Update with the visualziation of static meshes
                x_vis = x.clone()
                object_pcd.points = o3d.utility.Vector3dVector(x_vis.cpu().numpy())
                vis.update_geometry(object_pcd)
                vis.poll_events()
                vis.update_renderer()
                static_image = np.asarray(
                    vis.capture_screen_float_buffer(do_render=True)
                )
                # cv2.imshow("test", static_image)
                # cv2.waitKey(1)
                static_image = (static_image * 255).astype(np.uint8)
                static_vis_mask = np.all(static_image == [255, 255, 255], axis=-1)
                frame[~static_vis_mask] = static_image[~static_vis_mask]

            frame = self.update_frame(frame, self.pressed_keys)

            # Add shadows
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[0, 0, -3]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.95).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[1, 0.5, -2]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.97).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[-3, -0.5, -5]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.98).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)

            frame_comp_time = (
                frame_timer.stop() + frame_setup_time
            )  # Total frame compositing time
            component_times["frame_compositing"].append(frame_comp_time)

            torch.cuda.synchronize()

            if prev_x is not None:
                with torch.no_grad():

                    prev_particle_pos = prev_x
                    cur_particle_pos = x

                    if relations is None:
                        relations = get_topk_indices(
                            prev_x, K=16
                        )  # only computed in the first iteration

                    if weights is None:
                        weights, weights_indices = knn_weights_sparse(
                            prev_particle_pos, current_pos, K=16
                        )  # only computed in the first iteration

                    interp_timer.start()

                    weights = calc_weights_vals_from_indices(
                        prev_particle_pos, current_pos, weights_indices
                    )

                    current_pos, current_rot, _ = interpolate_motions_speedup(
                        bones=prev_particle_pos,
                        motions=cur_particle_pos - prev_particle_pos,
                        relations=relations,
                        weights=weights,
                        weights_indices=weights_indices,
                        xyz=current_pos,
                        quat=current_rot,
                    )

                    # update gaussians with the new positions and rotations
                    gaussians._xyz = current_pos
                    gaussians._rotation = current_rot

                interp_time = interp_timer.stop()
                component_times["full_motion_interpolation"].append(interp_time)

            torch.cuda.synchronize()

            prev_x = x.clone()

            prev_target = current_target
            target_change = self.get_target_change()
            if masks_ctrl_pts is not None:
                for i in range(n_ctrl_parts):
                    if masks_ctrl_pts[i].sum() > 0:
                        current_target[masks_ctrl_pts[i]] += torch.tensor(
                            target_change[i], dtype=torch.float32, device=cfg.device
                        )
                        if i == 0:
                            self.hand_left_pos += torch.tensor(
                                target_change[i], dtype=torch.float32, device=cfg.device
                            )
                        if i == 1:
                            self.hand_right_pos += torch.tensor(
                                target_change[i], dtype=torch.float32, device=cfg.device
                            )
            else:
                current_target += torch.tensor(
                    target_change, dtype=torch.float32, device=cfg.device
                )
                self.hand_left_pos += torch.tensor(
                    target_change, dtype=torch.float32, device=cfg.device
                )

            ############### Temporary timer ###############
            # Total loop time
            total_time = total_timer.stop()
            component_times["total"].append(total_time)

            # Calculate FPS
            fps = 1.0 / total_time
            fps_history.append(fps)

            # Display performance stats periodically
            frame_count += 1
            if frame_count % 10 == 0:
                # Limit stats to last STATS_WINDOW frames
                if len(fps_history) > STATS_WINDOW:
                    fps_history = fps_history[-STATS_WINDOW:]
                    for key in component_times:
                        component_times[key] = component_times[key][-STATS_WINDOW:]

                avg_fps = np.mean(fps_history)
                print(
                    f"\n--- Performance Stats (avg over last {len(fps_history)} frames) ---"
                )
                print(f"FPS: {avg_fps:.2f}")

                # Calculate percentages for pie chart
                total_avg = np.mean(component_times["total"])
                print(f"Total Frame Time: {total_avg*1000:.2f} ms")

                # Display individual component times
                for key in [
                    "simulator",
                    "rendering",
                    "frame_compositing",
                    "full_motion_interpolation",
                    "knn_weights",
                    "motion_interp",
                ]:
                    avg_time = np.mean(component_times[key])
                    percentage = (avg_time / total_avg) * 100
                    print(
                        f"{key.capitalize()}: {avg_time*1000:.2f} ms ({percentage:.1f}%)"
                    )
                total_energy = calculate_energy(
                    prev_x,
                    spring_Y,
                    self.init_springs,
                    self.init_rest_lengths,
                    num_object_springs,
                )
                print(f"Energy: {total_energy:.2f}")

        listener.stop()

    def interactive_robot(self, model_path, gs_path, n_ctrl_parts=1, inv_ctrl=False):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        spring_Y = spring_Y[: self.num_object_springs]
        self.init_springs = self.init_springs[: self.num_object_springs]
        self.init_rest_lengths = self.init_rest_lengths[: self.num_object_springs]
        self.init_vertices = self.init_vertices[: self.num_all_points]
        self.init_masses = self.init_masses[: self.num_all_points]
        self.controller_points = None

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
            static_meshes=self.dynamic_meshes + self.static_meshes,
            dynamic_points=self.dynamic_points,
        )

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        ###########################################################################

        logger.info("Party Time Start!!!!")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)

        self.n_ctrl_parts = n_ctrl_parts
        print("UI Controls:")
        print("- Set 1: WASD (XY movement), QE (Z movement)")
        print("- Set 2: IJKL (XY movement), UO (Z movement)")
        self.inv_ctrl = -1.0 if inv_ctrl else 1.0
        self.key_mappings = {
            # Set 1 controls
            "w": (0, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "s": (0, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "a": (0, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "d": (0, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            # Set 2 controls
            "i": (1, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "k": (1, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "j": (1, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "l": (1, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "o": (1, np.array([0, 0, 0.005])),
            "u": (1, np.array([0, 0, -0.005])),
            # Set the finger
            "n": 0.05,
            "m": -0.05,
            # Set the rotation
            "z": [0, 0, 2.0 / 180 * np.pi],
            "x": [0, 0, -2.0 / 180 * np.pi],
            "c": [2.0 / 180 * np.pi, 0, 0],
            "v": [-2.0 / 180 * np.pi, 0, 0],
        }
        self.pressed_keys = set()
        self.w2c = w2c
        self.intrinsic = intrinsic
        self.init_control_ui()

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.target_change = np.zeros((n_ctrl_parts, 3))

        ############## Temporary timer ##############
        import time

        class Timer:
            def __init__(self, name):
                self.name = name
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None
                self.use_cuda = torch.cuda.is_available()

            def start(self):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    self.cuda_start_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_end_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_start_event.record()
                self.start_time = time.time()

            def stop(self):
                if self.use_cuda:
                    self.cuda_end_event.record()
                    torch.cuda.synchronize()
                    self.elapsed = (
                        self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000
                    )  # convert ms to seconds
                else:
                    self.elapsed = time.time() - self.start_time
                return self.elapsed

            def reset(self):
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None

        sim_timer = Timer("Simulator")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        interp_timer = Timer("Full Motion Interpolation")
        total_timer = Timer("Total Loop")
        knn_weights_timer = Timer("KNN Weights")
        motion_interp_timer = Timer("Motion Interpolation")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "rendering": [],
            "frame_compositing": [],
            "full_motion_interpolation": [],
            "total": [],
            "knn_weights": [],
            "motion_interp": [],
        }

        # Number of frames to average over for stats
        STATS_WINDOW = 10
        frame_count = 0

        ############## End Temporary timer ##############
        if self.static_meshes is not None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=width, height=height)
            render_option = vis.get_render_option()
            render_option.point_size = 10.0

            for static_mesh in self.static_meshes:
                vis.add_geometry(static_mesh)

            for dynamic_mesh in self.dynamic_meshes:
                vis.add_geometry(dynamic_mesh)

            x_vis = wp.to_torch(
                self.simulator.wp_states[0].wp_x, requires_grad=False
            ).clone()
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(x_vis.cpu().numpy())
            object_pcd.paint_uniform_color([0, 0, 1])
            # object_pcd.paint_uniform_color([1, 1, 1])
            vis.add_geometry(object_pcd)

            # o3d.visualization.draw_geometries([object_pcd] + self.static_meshes)

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

            # vis_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            # cv2.imshow("test", vis_image)
            # cv2.waitKey(0)

        accumulate_trans = np.zeros((n_ctrl_parts, 3))
        accumulate_rot = torch.eye(3, dtype=torch.float32, device=cfg.device)

        self.dynamic_vertices = [
            np.asarray(finger_mesh.vertices) for finger_mesh in self.dynamic_meshes
        ]

        origin_force_judge = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.float32, device=cfg.device
        )
        current_force_judge = origin_force_judge.clone()
        current_trans_dynamic_points = self.dynamic_points
        current_finger = 1.0
        close_flag = True
        is_closing = False

        while True:

            total_timer.start()

            # 1. Simulator step

            sim_timer.start()

            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            collision_forces = wp.to_torch(
                self.simulator.collision_forces, requires_grad=False
            )[: self.num_dynamic]
            filter_forces = torch.einsum(
                "ij,ij->i", collision_forces, current_force_judge
            )
            if torch.all(filter_forces > 3e4):
                close_flag = False
            else:
                close_flag = True

            # print(filter_forces)

            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            sim_time = sim_timer.stop()
            component_times["simulator"].append(sim_time)

            torch.cuda.synchronize()

            # 2. Frame initialization and setup

            frame_timer.start()

            frame = overlay.clone()

            frame_setup_time = (
                frame_timer.stop()
            )  # We'll accumulate times for frame compositing

            torch.cuda.synchronize()

            # 3. Rendering
            render_timer.start()

            # render with gaussians and paste the image on top of the frame
            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach()

            render_time = render_timer.stop()
            component_times["rendering"].append(render_time)

            torch.cuda.synchronize()

            # Continue frame compositing
            frame_timer.start()

            image = image.clamp(0, 1)
            if use_white_background:
                image_mask = torch.logical_and(
                    (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = torch.logical_and(
                    (image != 0.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
            image[..., 3].masked_fill_(~image_mask, 0.0)

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.cpu().numpy()
            image_mask = image_mask.cpu().numpy()
            frame = frame.astype(np.uint8)

            if self.static_meshes is not None:
                # Update with the visualziation of static meshes
                x_vis = x.clone()
                object_pcd.points = o3d.utility.Vector3dVector(x_vis.cpu().numpy())
                vis.update_geometry(object_pcd)
                for i, dynamic_mesh in enumerate(self.dynamic_meshes):
                    dynamic_mesh.vertices = o3d.utility.Vector3dVector(
                        self.dynamic_vertices[i]
                    )
                    vis.update_geometry(dynamic_mesh)
                vis.poll_events()
                vis.update_renderer()
                static_image = np.asarray(
                    vis.capture_screen_float_buffer(do_render=True)
                )
                # cv2.imshow("test", static_image)
                # cv2.waitKey(1)
                static_image = (static_image * 255).astype(np.uint8)
                static_vis_mask = np.all(static_image == [255, 255, 255], axis=-1)
                frame[~static_vis_mask] = static_image[~static_vis_mask]

            frame = self.update_frame(frame, self.pressed_keys, overlay_hand=False)

            # Add shadows
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[0, 0, -3]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.95).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[1, 0.5, -2]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.97).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[-3, -0.5, -5]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.98).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)

            frame_comp_time = (
                frame_timer.stop() + frame_setup_time
            )  # Total frame compositing time
            component_times["frame_compositing"].append(frame_comp_time)

            torch.cuda.synchronize()

            if prev_x is not None:
                with torch.no_grad():

                    prev_particle_pos = prev_x
                    cur_particle_pos = x

                    if relations is None:
                        relations = get_topk_indices(
                            prev_x, K=16
                        )  # only computed in the first iteration

                    if weights is None:
                        weights, weights_indices = knn_weights_sparse(
                            prev_particle_pos, current_pos, K=16
                        )  # only computed in the first iteration

                    interp_timer.start()

                    weights = calc_weights_vals_from_indices(
                        prev_particle_pos, current_pos, weights_indices
                    )

                    current_pos, current_rot, _ = interpolate_motions_speedup(
                        bones=prev_particle_pos,
                        motions=cur_particle_pos - prev_particle_pos,
                        relations=relations,
                        weights=weights,
                        weights_indices=weights_indices,
                        xyz=current_pos,
                        quat=current_rot,
                    )

                    # update gaussians with the new positions and rotations
                    gaussians._xyz = current_pos
                    gaussians._rotation = current_rot

                interp_time = interp_timer.stop()
                component_times["full_motion_interpolation"].append(interp_time)

            torch.cuda.synchronize()

            prev_x = x.clone()

            # Update the changes
            target_change = self.get_target_change()
            finger_change = self.get_finger_change()
            rot_change = self.get_rot_change()

            # Galculate the substep vertices
            if finger_change > 0:
                is_closing = False
            elif finger_change < 0:
                is_closing = True
                
            if is_closing:
                if close_flag == True:
                    finger_change = -0.05
                else:
                    finger_change = 0.0
            else:
                finger_change = 0.05

            current_finger += finger_change
            current_finger = max(0.0, min(1.0, current_finger))

            accumulate_trans += target_change
            finger_meshes = self.robot.get_finger_mesh(current_finger)
            dynamic_vertices = torch.tensor(
                [
                    np.asarray(finger_mesh.vertices) + accumulate_trans[0]
                    for finger_mesh in finger_meshes
                ],
                dtype=torch.float32,
                device=cfg.device,
            )
            prev_trans_dynamic_points = current_trans_dynamic_points
            current_trans_dynamic_points = torch.reshape(dynamic_vertices, (-1, 3))

            # Caclulate the interpolated points considering finger and translation
            ratios = (
                torch.linspace(
                    1, cfg.num_substeps, cfg.num_substeps, device=cfg.device
                ).view(-1, 1, 1)
                / cfg.num_substeps
            )
            interpolated_trans_dynamic_points = (
                prev_trans_dynamic_points.unsqueeze(0)
                + (current_trans_dynamic_points - prev_trans_dynamic_points).unsqueeze(
                    0
                )
                * ratios
            )
            interpolated_center = torch.mean(interpolated_trans_dynamic_points, dim=1)

            # Do the rotation on the interpolated points
            new_rot = torch.tensor(rot_change, dtype=torch.float32, device=cfg.device)
            interpolated_rot_angle =  new_rot.unsqueeze(
                0
            ) * ratios.reshape(-1, 1)
            interpolated_rot_temp = axis_angle_to_matrix(interpolated_rot_angle)
            interpolated_rot_mat = torch.matmul(accumulate_rot.unsqueeze(0), interpolated_rot_temp)
            accumulate_rot = interpolated_rot_mat[-1]

            interpolated_dynamic_points = (
                interpolated_trans_dynamic_points - interpolated_center.unsqueeze(1)
            ) @ interpolated_rot_mat.permute(0, 2, 1) + interpolated_center.unsqueeze(1)
            self.dynamic_vertices = (
                interpolated_dynamic_points[-1]
                .reshape([-1] + list(self.dynamic_vertices[0].shape))
                .cpu()
                .numpy()
            )

            # Calculate the velocity and omega for calculating relative velocity
            dynamic_velocity = torch.tensor(
                target_change[0] / (2 * cfg.dt * cfg.num_substeps),
                dtype=torch.float32,
                device=cfg.device,
            )
            dynamic_omega = torch.tensor(
                rot_change / (2 * cfg.dt * cfg.num_substeps),
                dtype=torch.float32,
                device=cfg.device,
            )
            # print(dynamic_omega)

            # Update the force judge direction
            current_force_judge = origin_force_judge.clone() @ interpolated_rot_mat[-1]

            # Update the simulator with the gripper changes
            self.simulator.set_mesh_interactive(
                interpolated_dynamic_points,
                interpolated_center,
                dynamic_velocity,
                dynamic_omega,
            )

            ############### Temporary timer ###############
            # Total loop time
            total_time = total_timer.stop()
            component_times["total"].append(total_time)

            # Calculate FPS
            fps = 1.0 / total_time
            fps_history.append(fps)

            # Display performance stats periodically
            frame_count += 1
            if frame_count % 10 == 0:
                # Limit stats to last STATS_WINDOW frames
                if len(fps_history) > STATS_WINDOW:
                    fps_history = fps_history[-STATS_WINDOW:]
                    for key in component_times:
                        component_times[key] = component_times[key][-STATS_WINDOW:]

                avg_fps = np.mean(fps_history)
                print(
                    f"\n--- Performance Stats (avg over last {len(fps_history)} frames) ---"
                )
                print(f"FPS: {avg_fps:.2f}")

                # Calculate percentages for pie chart
                total_avg = np.mean(component_times["total"])
                print(f"Total Frame Time: {total_avg*1000:.2f} ms")

                # Display individual component times
                for key in [
                    "simulator",
                    "rendering",
                    "frame_compositing",
                    "full_motion_interpolation",
                    "knn_weights",
                    "motion_interp",
                ]:
                    avg_time = np.mean(component_times[key])
                    percentage = (avg_time / total_avg) * 100
                    print(
                        f"{key.capitalize()}: {avg_time*1000:.2f} ms ({percentage:.1f}%)"
                    )
                total_energy = calculate_energy(
                    prev_x,
                    spring_Y,
                    self.init_springs,
                    self.init_rest_lengths,
                    num_object_springs,
                )
                print(f"Energy: {total_energy:.2f}")

        listener.stop()

    def interactive_cube(self, cube_mesh, n_ctrl_parts=1, inv_ctrl=False):
        # Sample the points
        # Convert to trimesh mesh
        trimesh_mesh = trimesh.Trimesh(vertices=cube_mesh.vertices, faces=cube_mesh.triangles)
        # Sample the surface points
        surface_points, _ = trimesh.sample.sample_surface(
            trimesh_mesh, 1024
        )
        # Sample the interior points
        interior_points = trimesh.sample.volume_mesh(trimesh_mesh, 10000)
        all_points = np.concatenate([surface_points, interior_points], axis=0)
        
        min_bound = np.min(all_points, axis=0)
        volume_sample_size = 0.005
        index = []
        grid_flag = {}
        final_surface_points = []
        for i in range(surface_points.shape[0]):
            grid_index = tuple(
                np.floor((surface_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_surface_points.append(surface_points[i])
        final_interior_points = []
        for i in range(interior_points.shape[0]):
            grid_index = tuple(
                np.floor((interior_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_interior_points.append(interior_points[i])
        all_points = np.concatenate(
            [final_surface_points, final_interior_points],
            axis=0,
        )
        object_points = torch.tensor(
            all_points, dtype=torch.float32, device=cfg.device
        )
        # Connect the springs
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            object_points,
            None,
        )

        spring_Y = torch.ones(
            self.num_object_springs, dtype=torch.float32, device=cfg.device
        ) * 3e5
        collide_elas = torch.tensor([0.2], dtype=torch.float32, device=cfg.device)
        collide_fric = torch.tensor([2.0], dtype=torch.float32, device=cfg.device)

        self.controller_points = None

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=len(object_points),
            num_surface_points=len(object_points),
            num_original_points=len(object_points),
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
            static_meshes=self.dynamic_meshes + self.static_meshes,
            dynamic_points=self.dynamic_points,
        )

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        ###########################################################################

        logger.info("Party Time Start!!!!")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)

        self.n_ctrl_parts = n_ctrl_parts
        print("UI Controls:")
        print("- Set 1: WASD (XY movement), QE (Z movement)")
        print("- Set 2: IJKL (XY movement), UO (Z movement)")
        self.inv_ctrl = -1.0 if inv_ctrl else 1.0
        self.key_mappings = {
            # Set 1 controls
            "w": (0, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "s": (0, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "a": (0, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "d": (0, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            # Set 2 controls
            "i": (1, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "k": (1, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "j": (1, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "l": (1, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "o": (1, np.array([0, 0, 0.005])),
            "u": (1, np.array([0, 0, -0.005])),
            # Set the finger
            "n": 0.05,
            "m": -0.05,
            # Set the rotation
            "z": [0, 0, 2.0 / 180 * np.pi],
            "x": [0, 0, -2.0 / 180 * np.pi],
            "c": [2.0 / 180 * np.pi, 0, 0],
            "v": [-2.0 / 180 * np.pi, 0, 0],
        }
        self.pressed_keys = set()
        self.w2c = w2c
        self.intrinsic = intrinsic
        self.init_control_ui()

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.target_change = np.zeros((n_ctrl_parts, 3))

        ############## Temporary timer ##############
        import time

        class Timer:
            def __init__(self, name):
                self.name = name
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None
                self.use_cuda = torch.cuda.is_available()

            def start(self):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    self.cuda_start_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_end_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_start_event.record()
                self.start_time = time.time()

            def stop(self):
                if self.use_cuda:
                    self.cuda_end_event.record()
                    torch.cuda.synchronize()
                    self.elapsed = (
                        self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000
                    )  # convert ms to seconds
                else:
                    self.elapsed = time.time() - self.start_time
                return self.elapsed

            def reset(self):
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None

        sim_timer = Timer("Simulator")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        interp_timer = Timer("Full Motion Interpolation")
        total_timer = Timer("Total Loop")
        knn_weights_timer = Timer("KNN Weights")
        motion_interp_timer = Timer("Motion Interpolation")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "rendering": [],
            "frame_compositing": [],
            "full_motion_interpolation": [],
            "total": [],
            "knn_weights": [],
            "motion_interp": [],
        }

        # Number of frames to average over for stats
        STATS_WINDOW = 10
        frame_count = 0

        ############## End Temporary timer ##############
        if self.static_meshes is not None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=width, height=height)
            render_option = vis.get_render_option()
            render_option.point_size = 10.0

            for static_mesh in self.static_meshes:
                vis.add_geometry(static_mesh)

            for dynamic_mesh in self.dynamic_meshes:
                vis.add_geometry(dynamic_mesh)

            x_vis = wp.to_torch(
                self.simulator.wp_states[0].wp_x, requires_grad=False
            ).clone()
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(x_vis.cpu().numpy())
            object_pcd.paint_uniform_color([0, 0, 1])
            # object_pcd.paint_uniform_color([1, 1, 1])
            vis.add_geometry(object_pcd)

            # o3d.visualization.draw_geometries([object_pcd] + self.static_meshes)

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

            # vis_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            # cv2.imshow("test", vis_image)
            # cv2.waitKey(0)

        accumulate_trans = np.zeros((n_ctrl_parts, 3))
        accumulate_rot = torch.eye(3, dtype=torch.float32, device=cfg.device)

        self.dynamic_vertices = [
            np.asarray(finger_mesh.vertices) for finger_mesh in self.dynamic_meshes
        ]

        origin_force_judge = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.float32, device=cfg.device
        )
        current_force_judge = origin_force_judge.clone()
        current_trans_dynamic_points = self.dynamic_points
        current_finger = 1.0
        close_flag = True
        is_closing = False

        while True:

            total_timer.start()

            # 1. Simulator step

            sim_timer.start()

            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            collision_forces = wp.to_torch(
                self.simulator.collision_forces, requires_grad=False
            )[: self.num_dynamic]
            filter_forces = torch.einsum(
                "ij,ij->i", collision_forces, current_force_judge
            )
            if torch.all(filter_forces > 3e4):
                close_flag = False
            else:
                close_flag = True

            # print(filter_forces)

            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            sim_time = sim_timer.stop()
            component_times["simulator"].append(sim_time)

            torch.cuda.synchronize()

            # 2. Frame initialization and setup

            frame_timer.start()

            frame = overlay.clone()

            frame_setup_time = (
                frame_timer.stop()
            )  # We'll accumulate times for frame compositing

            torch.cuda.synchronize()

            # Continue frame compositing
            frame_timer.start()
            frame = frame.cpu().numpy()
            frame = frame.astype(np.uint8)

            if self.static_meshes is not None:
                # Update with the visualziation of static meshes
                x_vis = x.clone()
                object_pcd.points = o3d.utility.Vector3dVector(x_vis.cpu().numpy())
                vis.update_geometry(object_pcd)
                for i, dynamic_mesh in enumerate(self.dynamic_meshes):
                    dynamic_mesh.vertices = o3d.utility.Vector3dVector(
                        self.dynamic_vertices[i]
                    )
                    vis.update_geometry(dynamic_mesh)
                vis.poll_events()
                vis.update_renderer()
                static_image = np.asarray(
                    vis.capture_screen_float_buffer(do_render=True)
                )
                # cv2.imshow("test", static_image)
                # cv2.waitKey(1)
                static_image = (static_image * 255).astype(np.uint8)
                static_vis_mask = np.all(static_image == [255, 255, 255], axis=-1)
                frame[~static_vis_mask] = static_image[~static_vis_mask]

            frame = self.update_frame(frame, self.pressed_keys, overlay_hand=False)

            # # Add shadows
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
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)

            frame_comp_time = (
                frame_timer.stop() + frame_setup_time
            )  # Total frame compositing time
            component_times["frame_compositing"].append(frame_comp_time)

            torch.cuda.synchronize()
            prev_x = x.clone()

            # Update the changes
            target_change = self.get_target_change()
            finger_change = self.get_finger_change()
            rot_change = self.get_rot_change()

            # Galculate the substep vertices
            if finger_change > 0:
                is_closing = False
            elif finger_change < 0:
                is_closing = True
                
            if is_closing:
                if close_flag == True:
                    finger_change = -0.05
                else:
                    finger_change = 0.0
            else:
                finger_change = 0.05

            current_finger += finger_change
            current_finger = max(0.0, min(1.0, current_finger))

            accumulate_trans += target_change
            finger_meshes = self.robot.get_finger_mesh(current_finger)
            dynamic_vertices = torch.tensor(
                [
                    np.asarray(finger_mesh.vertices) + accumulate_trans[0]
                    for finger_mesh in finger_meshes
                ],
                dtype=torch.float32,
                device=cfg.device,
            )
            prev_trans_dynamic_points = current_trans_dynamic_points
            current_trans_dynamic_points = torch.reshape(dynamic_vertices, (-1, 3))

            # Caclulate the interpolated points considering finger and translation
            ratios = (
                torch.linspace(
                    1, cfg.num_substeps, cfg.num_substeps, device=cfg.device
                ).view(-1, 1, 1)
                / cfg.num_substeps
            )
            interpolated_trans_dynamic_points = (
                prev_trans_dynamic_points.unsqueeze(0)
                + (current_trans_dynamic_points - prev_trans_dynamic_points).unsqueeze(
                    0
                )
                * ratios
            )
            interpolated_center = torch.mean(interpolated_trans_dynamic_points, dim=1)

            # Do the rotation on the interpolated points
            new_rot = torch.tensor(rot_change, dtype=torch.float32, device=cfg.device)
            interpolated_rot_angle =  new_rot.unsqueeze(
                0
            ) * ratios.reshape(-1, 1)
            interpolated_rot_temp = axis_angle_to_matrix(interpolated_rot_angle)
            interpolated_rot_mat = torch.matmul(accumulate_rot.unsqueeze(0), interpolated_rot_temp)
            accumulate_rot = interpolated_rot_mat[-1]

            interpolated_dynamic_points = (
                interpolated_trans_dynamic_points - interpolated_center.unsqueeze(1)
            ) @ interpolated_rot_mat.permute(0, 2, 1) + interpolated_center.unsqueeze(1)
            self.dynamic_vertices = (
                interpolated_dynamic_points[-1]
                .reshape([-1] + list(self.dynamic_vertices[0].shape))
                .cpu()
                .numpy()
            )

            # Calculate the velocity and omega for calculating relative velocity
            dynamic_velocity = torch.tensor(
                target_change[0] / (2 * cfg.dt * cfg.num_substeps),
                dtype=torch.float32,
                device=cfg.device,
            )
            dynamic_omega = torch.tensor(
                rot_change / (2 * cfg.dt * cfg.num_substeps),
                dtype=torch.float32,
                device=cfg.device,
            )
            # print(dynamic_omega)

            # Update the force judge direction
            current_force_judge = origin_force_judge.clone() @ interpolated_rot_mat[-1]

            # Update the simulator with the gripper changes
            self.simulator.set_mesh_interactive(
                interpolated_dynamic_points,
                interpolated_center,
                dynamic_velocity,
                dynamic_omega,
            )

            ############### Temporary timer ###############
            # Total loop time
            total_time = total_timer.stop()
            component_times["total"].append(total_time)

            # Calculate FPS
            fps = 1.0 / total_time
            fps_history.append(fps)

            # Display performance stats periodically
            frame_count += 1
            if frame_count % 10 == 0:
                # Limit stats to last STATS_WINDOW frames
                if len(fps_history) > STATS_WINDOW:
                    fps_history = fps_history[-STATS_WINDOW:]
                    for key in component_times:
                        component_times[key] = component_times[key][-STATS_WINDOW:]

                avg_fps = np.mean(fps_history)
                print(
                    f"\n--- Performance Stats (avg over last {len(fps_history)} frames) ---"
                )
                print(f"FPS: {avg_fps:.2f}")

                # Calculate percentages for pie chart
                total_avg = np.mean(component_times["total"])
                print(f"Total Frame Time: {total_avg*1000:.2f} ms")

                # Display individual component times
                for key in [
                    "simulator",
                    "rendering",
                    "frame_compositing",
                    "full_motion_interpolation",
                    "knn_weights",
                    "motion_interp",
                ]:
                    avg_time = np.mean(component_times[key])
                    percentage = (avg_time / total_avg) * 100
                    print(
                        f"{key.capitalize()}: {avg_time*1000:.2f} ms ({percentage:.1f}%)"
                    )
                total_energy = calculate_energy(
                    prev_x,
                    spring_Y,
                    self.init_springs,
                    self.init_rest_lengths,
                    self.num_object_springs,
                )
                print(f"Energy: {total_energy:.2f}")

        listener.stop()


    def _transform_gs(self, gaussians, M, majority_scale=1):

        new_gaussians = copy.copy(gaussians)

        new_xyz = gaussians.get_xyz.clone()
        ones = torch.ones(
            (new_xyz.shape[0], 1), device=new_xyz.device, dtype=new_xyz.dtype
        )
        new_xyz = torch.cat((new_xyz, ones), dim=1)
        print("inside:", new_xyz.max(), new_xyz.min())
        new_xyz = new_xyz @ M.T
        print("outside:", new_xyz.max(), new_xyz.min())

        new_rotation = gaussians.get_rotation.clone()
        new_rotation = quaternion_multiply(
            matrix_to_quaternion(M[:3, :3]), new_rotation
        )

        new_scales = gaussians._scaling.clone()
        new_scales += torch.log(
            torch.tensor(
                majority_scale, device=new_scales.device, dtype=new_scales.dtype
            )
        )

        new_gaussians._xyz = new_xyz[:, :3]
        new_gaussians._rotation = new_rotation
        new_gaussians._scaling = new_scales

        return new_gaussians

    def _create_gs_view(self, w2c, intrinsic, height, width):
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        view = Camera(
            (width, height),
            colmap_id="0000",
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name="0000",
            uid="0000",
            data_device="cuda",
            train_test_exp=None,
            is_test_dataset=None,
            is_test_view=None,
            K=K,
            normal=None,
            depth=None,
            occ_mask=None,
        )
        return view

    def visualize_force(self, model_path, gs_path, n_ctrl_parts=2, force_scale=30000):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        video_path = f"{cfg.base_dir}/force_visualization.mp4"

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=cfg.device)
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None

        # Get the controller points index
        first_frame_controller_points = self.simulator.controller_points[0]
        force_indexes = []
        if n_ctrl_parts == 1:
            force_indexes.append(
                torch.arange(first_frame_controller_points.shape[0], device=cfg.device)
            )
        else:
            # Use kmeans to find the two set of controller points
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(
                first_frame_controller_points.cpu().numpy()
            )
            for i in range(n_ctrl_parts):
                force_indexes.append(
                    torch.tensor(np.where(cluster_labels == i)[0], device=cfg.device)
                )

        # Preprocess to get all the springs for different set of control points
        control_springs = self.init_springs[num_object_springs:]

        # Judge the springs whose left point is in the force_indexes
        force_springs = []
        force_object_points = []
        force_rest_lengths = []
        force_spring_Y = []

        for i in range(n_ctrl_parts):
            force_springs.append([])
            force_rest_lengths.append([])
            force_spring_Y.append([])
            force_object_points.append([])
            for j in range(len(control_springs)):
                if (control_springs[j][0] - self.num_all_points) in force_indexes[i]:
                    force_springs[i].append(control_springs[j])
                    force_rest_lengths[i].append(
                        self.init_rest_lengths[j + num_object_springs]
                    )
                    force_spring_Y[i].append(spring_Y[j + num_object_springs])
                    force_object_points[i].append(control_springs[j][1])
            force_springs[i] = torch.vstack(force_springs[i])
            force_springs[i][:, 0] -= self.num_all_points
            force_rest_lengths[i] = torch.tensor(
                force_rest_lengths[i], device=cfg.device
            )
            force_spring_Y[i] = torch.tensor(force_spring_Y[i], device=cfg.device)
            force_object_points[i] = torch.tensor(
                force_object_points[i], device=cfg.device
            )

        # Start to visualize the stuffs
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/0.png"
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = render_gaussian(view, gaussians, None, background)
        rendering = results["render"]  # (4, H, W)
        image = rendering.permute(1, 2, 0).detach().cpu().numpy()

        image = image.clip(0, 1)
        if use_white_background:
            image_mask = np.logical_and(
                (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        else:
            image_mask = np.logical_and(
                (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        image[~image_mask, 3] = 0

        alpha = image[..., 3:4]
        rgb = image[..., :3] * 255
        frame = alpha * rgb + (1 - alpha) * frame
        frame = frame.astype(np.uint8)

        force_arrow_meshes = []
        for j in range(n_ctrl_parts):
            # Calculate the center of the force_object_points
            force_center = (
                torch.mean(prev_x[force_object_points[j]], dim=0).cpu().numpy()
            )
            # Calculate the force vector
            force_vector = (
                self.get_force_vector(
                    prev_x,
                    force_springs[j],
                    force_rest_lengths[j],
                    force_spring_Y[j],
                    self.num_all_points,
                    self.simulator.controller_points[0],
                )
                .cpu()
                .numpy()
            )
            # Create arrow mesh in open3d
            if not (force_vector == 0).all():
                arrow_mesh = getArrowMesh(
                    origin=force_center,
                    end=force_center + force_vector / force_scale,
                    color=[1, 0, 0],
                )
                force_arrow_meshes.append(arrow_mesh)
                vis.add_geometry(force_arrow_meshes[j])
        # Adjust the viewpoint
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

        force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        force_image = (force_image * 255).astype(np.uint8)
        force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
        frame[~force_vis_mask] = force_image[~force_vis_mask]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Interactive Playground", frame)
        # cv2.waitKey(0)
        video_writer.write(frame)

        for i in tqdm(range(1, frame_len)):
            if cfg.data_type == "real":
                self.simulator.set_controller_target(i, pure_inference=True)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            torch.cuda.synchronize()

            with torch.no_grad():
                # Do LBS on the gaussian kernels
                prev_particle_pos = prev_x
                cur_particle_pos = x
                if relations is None:
                    relations = get_topk_indices(
                        prev_x, K=16
                    )  # only computed in the first iteration

                if weights is None:
                    weights, weights_indices = knn_weights_sparse(
                        prev_particle_pos, current_pos, K=16
                    )  # only computed in the first iteration

                weights = calc_weights_vals_from_indices(
                    prev_particle_pos, current_pos, weights_indices
                )

                current_pos, current_rot, _ = interpolate_motions_speedup(
                    bones=prev_particle_pos,
                    motions=cur_particle_pos - prev_particle_pos,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=current_pos,
                    quat=current_rot,
                )

                # update gaussians with the new positions and rotations
                gaussians._xyz = current_pos
                gaussians._rotation = current_rot

            prev_x = x.clone()

            frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach().cpu().numpy()

            image = image.clip(0, 1)
            if use_white_background:
                image_mask = np.logical_and(
                    (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = np.logical_and(
                    (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            image[~image_mask, 3] = 0

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.astype(np.uint8)

            for arrow_mesh in force_arrow_meshes:
                vis.remove_geometry(arrow_mesh)

            force_arrow_meshes = []
            for j in range(n_ctrl_parts):
                # Calculate the center of the force_object_points
                force_center = (
                    torch.mean(x[force_object_points[j]], dim=0).cpu().numpy()
                )
                # Calculate the force vector
                force_vector = (
                    self.get_force_vector(
                        x,
                        force_springs[j],
                        force_rest_lengths[j],
                        force_spring_Y[j],
                        self.num_all_points,
                        self.simulator.controller_points[i],
                    )
                    .cpu()
                    .numpy()
                )
                if not (force_vector == 0).all():
                    # Create arrow mesh in open3d
                    arrow_mesh = getArrowMesh(
                        origin=force_center,
                        end=force_center + force_vector / force_scale,
                        color=[1, 0, 0],
                    )
                force_arrow_meshes.append(arrow_mesh)
                vis.add_geometry(force_arrow_meshes[j])

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

            vis.poll_events()
            vis.update_renderer()

            force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            force_image = (force_image * 255).astype(np.uint8)
            force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
            frame[~force_vis_mask] = force_image[~force_vis_mask]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

            # cv2.imshow("Interactive Playground", frame)
            # cv2.waitKey(0)
        vis.destroy_window()
        video_writer.release()

    def get_force_vector(
        self, x, springs, rest_lengths, spring_Y, num_object_points, controller_points
    ):
        with torch.no_grad():
            # Calculate the force of the springs
            x1 = controller_points[springs[:, 0]]
            x2 = x[springs[:, 1]]

            dis = x2 - x1
            dis_len = torch.norm(dis, dim=1)

            d = dis / torch.clamp(dis_len, min=1e-6)[:, None]
            spring_forces = (
                torch.clamp(spring_Y, min=cfg.spring_Y_min, max=cfg.spring_Y_max)[
                    :, None
                ]
                * (dis_len / rest_lengths - 1.0)[:, None]
                * d
            )

            total_force = -spring_forces.sum(dim=0)
        return total_force

    def visualize_material(self, model_path, gs_path, relative_material=True):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        video_path = f"{cfg.base_dir}/material_visualization.mp4"

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=cfg.device)
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None

        # Start to visualize the stuffs
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/0.png"
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = render_gaussian(view, gaussians, None, background)
        rendering = results["render"]  # (4, H, W)
        image = rendering.permute(1, 2, 0).detach().cpu().numpy()

        image = image.clip(0, 1)
        if use_white_background:
            image_mask = np.logical_and(
                (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        else:
            image_mask = np.logical_and(
                (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        image[~image_mask, 3] = 0

        alpha = image[..., 3:4]
        rgb = image[..., :3] * 255
        frame = alpha * rgb + (1 - alpha) * frame
        frame = frame.astype(np.uint8)

        # Add the material visualization
        object_springs = self.init_springs[:num_object_springs]
        material_field = torch.zeros((self.num_all_points, 3), device=cfg.device)
        count_field = torch.zeros(
            self.num_all_points, dtype=torch.int32, device=cfg.device
        )
        clamp_object_spring_Y = torch.clamp(
            spring_Y[:num_object_springs], min=cfg.spring_Y_min, max=cfg.spring_Y_max
        )
        object_rest_lengths = self.init_rest_lengths[:num_object_springs]

        # idx1 = object_springs[:, 0]
        # idx2 = object_springs[:, 1]
        # x1 = prev_x[idx1]
        # x2 = prev_x[idx2]
        # dis = x2 - x1
        # dis_len = torch.norm(dis, dim=1)
        # d = dis / torch.clamp(dis_len, min=1e-6)[:, None]
        # # import pdb
        # # pdb.set_trace()
        # material_field.index_add_(
        #     0,
        #     idx1,
        #     clamp_object_spring_Y[:, None] / object_rest_lengths[:, None] * d,
        # )
        # material_field.index_add_(
        #     0,
        #     idx2,
        #     clamp_object_spring_Y[:, None] / object_rest_lengths[:, None] * d,
        # )
        # material_field = torch.norm(material_field, dim=1)
        # import pdb
        # pdb.set_trace()
        # count_field.index_add_(
        #     0, idx1, torch.ones_like(idx1, dtype=torch.int32, device=cfg.device)
        # )
        # count_field.index_add_(
        #     0, idx2, torch.ones_like(idx2, dtype=torch.int32, device=cfg.device)
        # )
        # material_field /= count_field
        # if relative_material:
        #     material_field_normalized = (material_field - material_field.min()) / (
        #         material_field.max() - material_field.min()
        #     )
        # else:
        #     material_field_normalized = (material_field - cfg.spring_Y_min) / (
        #         cfg.spring_Y_max - cfg.spring_Y_min
        #     )
        # rainbow_colors = plt.cm.rainbow(material_field_normalized.cpu().numpy())[:, :3]

        stiffness_map = compute_effective_stiffness(
            points=prev_x,
            springs=object_springs,
            Y=clamp_object_spring_Y,
            rest_lengths=object_rest_lengths,
            device=cfg.device,
        )
        normed = (stiffness_map - stiffness_map.min()) / (
            stiffness_map.max() - stiffness_map.min()
        )
        rainbow_colors = plt.cm.rainbow(normed.cpu().numpy())[:, :3]

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(prev_x.cpu().numpy())
        object_pcd.colors = o3d.utility.Vector3dVector(rainbow_colors)
        vis.add_geometry(object_pcd)

        # Adjust the viewpoint
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

        material_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        material_image = (material_image * 255).astype(np.uint8)
        material_vis_mask = np.all(material_image == [255, 255, 255], axis=-1)
        frame[~material_vis_mask] = material_image[~material_vis_mask]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Interactive Playground", frame)
        cv2.waitKey(1)
        video_writer.write(frame)

        for i in tqdm(range(1, frame_len)):
            if cfg.data_type == "real":
                self.simulator.set_controller_target(i, pure_inference=True)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            torch.cuda.synchronize()

            with torch.no_grad():
                # Do LBS on the gaussian kernels
                prev_particle_pos = prev_x
                cur_particle_pos = x
                if relations is None:
                    relations = get_topk_indices(
                        prev_x, K=16
                    )  # only computed in the first iteration

                if weights is None:
                    weights, weights_indices = knn_weights_sparse(
                        prev_particle_pos, current_pos, K=16
                    )  # only computed in the first iteration

                weights = calc_weights_vals_from_indices(
                    prev_particle_pos, current_pos, weights_indices
                )

                current_pos, current_rot, _ = interpolate_motions_speedup(
                    bones=prev_particle_pos,
                    motions=cur_particle_pos - prev_particle_pos,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=current_pos,
                    quat=current_rot,
                )

                # update gaussians with the new positions and rotations
                gaussians._xyz = current_pos
                gaussians._rotation = current_rot

            prev_x = x.clone()

            frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach().cpu().numpy()

            image = image.clip(0, 1)
            if use_white_background:
                image_mask = np.logical_and(
                    (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = np.logical_and(
                    (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            image[~image_mask, 3] = 0

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.astype(np.uint8)

            # Update the object pcd
            object_pcd.points = o3d.utility.Vector3dVector(prev_x.cpu().numpy())
            vis.update_geometry(object_pcd)

            vis.poll_events()
            vis.update_renderer()

            force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            force_image = (force_image * 255).astype(np.uint8)
            force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
            frame[~force_vis_mask] = force_image[~force_vis_mask]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)
        vis.destroy_window()
        video_writer.release()


def get_simple_shadow(
    points,
    intrinsic,
    w2c,
    width,
    height,
    image_mask,
    kernel_size=7,
    light_point=[0, 0, -3],
):
    points = points.cpu().numpy()

    t = -points[:, 2] / light_point[2]
    points_on_table = points + t[:, None] * light_point

    points_homogeneous = np.hstack(
        [points_on_table, np.ones((points_on_table.shape[0], 1))]
    )  # Convert to homogeneous coordinates
    points_camera = (w2c @ points_homogeneous.T).T

    points_pixels = (intrinsic @ points_camera[:, :3].T).T
    points_pixels /= points_pixels[:, 2:3]
    pixel_coords = points_pixels[:, :2]

    valid_mask = (
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 0] < width)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 1] < height)
    )

    valid_pixel_coords = pixel_coords[valid_mask]
    valid_pixel_coords = valid_pixel_coords.astype(int)

    shadow_image = np.zeros((height, width), dtype=np.uint8)
    shadow_image[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_1 = np.ones((3, 3), np.uint(8))
    dilated_shadow = cv2.dilate(shadow_image, kernel, iterations=1)
    dilated_shadow = cv2.dilate(dilated_shadow, kernel_1, iterations=1)
    final_shadow = cv2.erode(dilated_shadow, kernel, iterations=1)

    final_shadow[image_mask] = 0
    final_shadow = final_shadow == 255
    return final_shadow


'''
[-0.04025196  0.0259362  -0.07883434]
Selected control point 1: [ 0.1074927  -0.16420561 -0.00984779]

[-0.04025196  0.0259362  -0.07883434]
Selected control point 1: [-0.14578269 -0.19089821 -0.009117  ]

[-0.04025196  0.0259362  -0.07883434]
Selected control point 1: [-0.17029284  0.20677952 -0.00766289]

[-0.04025196  0.0259362  -0.07883434]
Selected control point 1: [ 0.06977675  0.24097862 -0.01045121]
'''

def calculate_energy(x, spring_Y, springs, rest_lengths, num_object_springs):
    object_springs = springs[:num_object_springs]
    object_rest_lengths = rest_lengths[:num_object_springs]
    clamp_spring_Y = torch.clamp(
        spring_Y[:num_object_springs], min=cfg.spring_Y_min, max=cfg.spring_Y_max
    )

    with torch.no_grad():
        # Calculate the energy of the springs
        x1 = x[object_springs[:, 0]]
        x2 = x[object_springs[:, 1]]

        dis = x2 - x1
        dis_len = torch.norm(dis, dim=1)

        # Hooke's law: 0.5 * k * (|x2 - x1| - rest_length)^2
        stretch = dis_len - object_rest_lengths
        energy = 0.5 * clamp_spring_Y * (stretch**2)

        total_energy = energy.sum()
        return total_energy


# Copy From pytorch3d implementation
def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )

'''

[-0.17972917 -0.20094689 -0.00053748]
'''