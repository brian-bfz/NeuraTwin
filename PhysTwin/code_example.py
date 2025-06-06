def outdomain_inference(self, model_path, to_case_data_path, final_points): 
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        spring_Y = spring_Y[: self.num_object_springs]

        # Loda the to_case data
        with open(to_case_data_path, "rb") as f:
            data = pickle.load(f)
        self.object_points = torch.tensor(
            data["object_points"], dtype=torch.float32, device=cfg.device
        )
        self.object_colors = self.dataset.object_colors[0].repeat(
            self.object_points.shape[0], 1, 1
        )

        self.controller_points = torch.tensor(
            data["controller_points"], dtype=torch.float32, device=cfg.device
        )
        self.dataset.frame_len = self.object_points.shape[0]

        controller_points = self.controller_points.cpu().numpy()

        assert self.num_all_points == len(
            final_points
        ), "Check the length of the final points"

        # Update the rest lengths for the springs among the object points
        springs = self.init_springs.cpu().numpy()[: self.num_object_springs]
        rest_lengths = np.linalg.norm(
            final_points[springs[:, 0]] - final_points[springs[:, 1]], axis=1
        )

        springs = springs.tolist()
        rest_lengths = rest_lengths.tolist()
        # Update the connection between the final points and the controller points
        first_frame_controller_points = controller_points[0]
        points = np.concatenate([final_points, first_frame_controller_points], axis=0)
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

        # Process to get the connection distance among the controller points and object points
        # Locate the nearest object point for each controller point
        kdtree = KDTree(final_points)
        _, idx = kdtree.query(first_frame_controller_points, k=1)
        # find the distances
        distances = np.linalg.norm(
            final_points[idx] - first_frame_controller_points, axis=1
        )
        # find the indices of the top 4 controller points that are close
        top_k = 10
        top_k_idx = np.argsort(distances)[:top_k]
        controller_radius = np.ones(first_frame_controller_points.shape[0]) * 0.01
        controller_radius[top_k_idx] = distances[top_k_idx] + 0.005

        for i in range(len(first_frame_controller_points)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                first_frame_controller_points[i],
                controller_radius[i],
                30,
            )
            for j in idx:
                springs.append([self.num_all_points + i, j])
                rest_lengths.append(
                    np.linalg.norm(first_frame_controller_points[i] - points[j])
                )

        self.init_springs = torch.tensor(
            np.array(springs), dtype=torch.int32, device=cfg.device
        )

        self.init_rest_lengths = torch.tensor(
            np.array(rest_lengths), dtype=torch.float32, device=cfg.device
        )
        self.init_masses = torch.tensor(
            np.ones(len(points)), dtype=torch.float32, device=cfg.device
        )

        self.init_vertices = torch.tensor(
            points,
            dtype=torch.float32,
            device=cfg.device,
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        cfg.dt = 5e-6
        cfg.num_substeps = round(1.0 / cfg.FPS / cfg.dt)
        cfg.collision_dist = 0.005

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

        spring_Y = torch.cat(
            [
                spring_Y,
                3e4
                * torch.ones(
                    self.simulator.n_springs - self.num_object_springs,
                    dtype=torch.float32,
                    device=cfg.device,
                ),
            ]
        )

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(), collide_object_fric.detach().clone()
        )

        # Render the final results
        video_path = f"{cfg.base_dir}/inference.mp4"
        save_path = f"{cfg.base_dir}/inference.pkl"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            save_trajectory=True,
            save_path=save_path,
        )
