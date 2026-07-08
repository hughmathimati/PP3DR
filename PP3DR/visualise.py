import numpy as np
import viser
import time
from models.PP3DR_Dino import PP3DR_Dino
import torch
from datasets.nrgbd_dataset import nrgbd_dataset


def visualize_reconstruction(pred: dict, gt: dict):
    """
    Visualizes the unprojected video frames using Viser.

    Args:
        pred: Dictionary containing predictions:
            - "XY_rays": (B, L, H, W, 2)
            - "log_depths": (B, L, H, W)
            - "relative_camera_translations": (B, L - 1, 3)
            - "relative_camera_rotations": (B, L - 1, 3, 3)
        gt: Dictionary containing ground-truth data:
            - "images": (B, L, H, W, 3)
            - "depths": (B, L, H, W)
    """
    # Start the viser server
    server = viser.ViserServer()

    # Extract dimensions
    B, L, H, W, _ = pred["XY_rays"].shape

    # -------------------------------------------------------------------------
    # GUI Elements Setup
    # -------------------------------------------------------------------------
    with server.gui.add_folder("Controls"):
        gui_batch = server.gui.add_slider(
            "Batch Index", min=0, max=B - 1, step=1, initial_value=0, visible=(B > 1)
        )
        gui_frame = server.gui.add_slider(
            "Current Frame", min=0, max=L - 1, step=1, initial_value=0
        )
        gui_accumulate = server.gui.add_checkbox(
            "Accumulate Points", initial_value=False
        )
        gui_point_size = server.gui.add_slider(
            "Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.01
        )

    # -------------------------------------------------------------------------
    # Core Geometric Computations
    # -------------------------------------------------------------------------
    def compute_absolute_poses(b_idx):
        """
        Computes transformations mapping each frame's local space to the
        coordinate system of the first camera (frame 0).
        """
        T_absolute = np.zeros((L, 4, 4))
        T_absolute[:, 3, 3] = 1.0  # Set homogeneous coordinate

        # Frame 0 is our reference origin
        current_R = np.eye(3)
        current_t = np.zeros(3)

        T_absolute[0, :3, :3] = current_R
        T_absolute[0, :3, 3] = current_t

        R_rel = pred["relative_camera_rotations"][b_idx]  # (L-1, 3, 3)
        t_rel = pred["relative_camera_translations"][b_idx]  # (L-1, 3)

        for i in range(L - 1):
            # R_rel = R_next @ R_current.T  => R_next = R_rel @ R_current
            current_R = R_rel[i] @ current_R

            # t_rel = t_next - t_current    => t_next = t_current + t_rel
            # print(current_t.shape, t_rel[i].shape) # DEBUG
            current_t = current_t + t_rel[i]

            T_absolute[i + 1, :3, :3] = current_R
            T_absolute[i + 1, :3, 3] = current_t

        return T_absolute

    def unproject_frame(b_idx, l_idx, T_world):
        """Unprojects a single frame's valid points to the frame 0 space."""
        # Convert log depth to absolute depth
        log_depth = pred["log_depths"][b_idx, l_idx]
        depth = np.exp(log_depth)

        # Valid points mask (where ground truth depth is non-zero)
        gt_depth = gt["depths"][b_idx, l_idx]
        valid_mask = gt_depth > 0

        if not np.any(valid_mask):
            return np.zeros((0, 3)), np.zeros((0, 3))

        # Filter rays, depths, and colors
        xy_rays = pred["XY_rays"][b_idx, l_idx][valid_mask]  # (N, 2)
        depth_valid = depth[valid_mask]  # (N,)

        # Pull RGB and handle scaling normalization safely
        reshaped = gt["images"][b_idx, l_idx].transpose(1, 2, 0)
        img_colors = reshaped[valid_mask]  # (N, 3)
        if img_colors.dtype == np.uint8:
            img_colors = img_colors / 255.0

        # Reconstruct 3D points in local camera space
        # P_cam = [X_ray * depth, Y_ray * depth, depth]
        pts_cam = np.stack([
            xy_rays[:, 0] * depth_valid,
            xy_rays[:, 1] * depth_valid,
            depth_valid
        ], axis=-1)

        # Transform points to frame 0 reference space
        pts_homo = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=-1)
        pts_world = (pts_homo @ T_world.T)[..., :3]

        return pts_world, img_colors

    # -------------------------------------------------------------------------
    # Render and State Update Logic
    # -------------------------------------------------------------------------
    def update_scene():
        b_idx = gui_batch.value
        current_l = gui_frame.value
        accumulate = gui_accumulate.value
        point_size = gui_point_size.value

        # Precompute absolute poses relative to frame 0 for this batch
        T_absolute = compute_absolute_poses(b_idx)

        all_pts = []
        all_colors = []

        # Determine which frames to parse based on accumulation toggle
        frames_to_render = range(current_l + 1) if accumulate else [current_l]

        for l_idx in frames_to_render:
            pts, cols = unproject_frame(b_idx, l_idx, T_absolute[l_idx])
            all_pts.append(pts)
            all_colors.append(cols)

        if len(all_pts) > 0 and max(p.shape[0] for p in all_pts) > 0:
            final_pts = np.concatenate(all_pts, axis=0)
            final_cols = np.concatenate(all_colors, axis=0)

            # Send/Update point cloud in the viser scene window
            server.scene.add_point_cloud(
                name="/reconstruction/point_cloud",
                points=final_pts,
                colors=final_cols,
                point_size=point_size,
            )
        else:
            # Clear point cloud if no valid data points exist
            server.scene.add_point_cloud(
                name="/reconstruction/point_cloud",
                points=np.zeros((0, 3)),
                colors=np.zeros((0, 3)),
                point_size=point_size,
            )

    # Bind the reactive updates to GUI changes
    gui_batch.on_update(lambda _: update_scene())
    gui_frame.on_update(lambda _: update_scene())
    gui_accumulate.on_update(lambda _: update_scene())
    gui_point_size.on_update(lambda _: update_scene())

    # Initialize scene display loop
    update_scene()

    print(f"Viser server running at: {server.get_host()}")
    while True:
        time.sleep(1.0)


# -------------------------------------------------------------------------
# Example Execution Context Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    model = PP3DR_Dino()
    loaded_state_dict = torch.load("/vulcanscratch/hughma/PP3DR/sanity/PP3DR.pth", weights_only=True)
    # The lines here are necessary if all the loaded state dict entries begin with an extra "module."
    new_state_dict = {}
    for key in loaded_state_dict:
        new_state_dict[key[7:]] = loaded_state_dict[key]
    model.load_state_dict(new_state_dict)
    model = model.eval().to("cuda")
    dataset = nrgbd_dataset()
    data = dataset[0]
    for key in data:
        data[key] = data[key].unsqueeze(0)
    with torch.amp.autocast("cuda", dtype = torch.bfloat16), torch.no_grad():
        pred = model(data['images'].to("cuda"))
    for key in pred:
        pred[key] = pred[key].to(torch.float32).cpu().numpy(force=True)
        print(f"{key}: {pred[key].shape}")
    # Since we ran point-prediction only, we need to manually add back in some fake camera poses.
    B, L = pred['log_depths'].shape[:2]
    pred["relative_camera_translations"] = np.random.uniform(-0.1, 0.1, (B, L - 1, 3))
    pred["relative_camera_rotations"] = np.tile(np.eye(3), (B, L - 1, 1, 1))

    for key in data:
        data[key] = data[key].to(torch.float32).cpu().numpy(force=True)
        print(f"{key}: {data[key].shape}")
    visualize_reconstruction(pred, data)