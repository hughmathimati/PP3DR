import viser
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def visualize_pred_sequence(pred, gt, port=8080):
    """
    Visualizes a sequence of model PREDICTIONS in 3D.
    Accumulates relative camera poses into absolute world trajectories.

    Expects `pred` dictionary to contain:
      log_depths: (..., L, H, W)
      focal_length: (..., L, 1) or scalar
      relative_camera_translations: (..., L-1, 3)
      relative_camera_rotations: (..., L-1, 3, 3)

    Expects `gt` dictionary (for images, start anchor, and principal points):
      images: (..., L, 3, H, W)
      extrinsics: (..., L, 3, 4)
      intrinsics: (..., L, 3, 3)
    """
    server = viser.ViserServer(port=port)
    print(f"Viser server running at http://localhost:{port}")

    # 1. Helper to safely extract and strip batch dimensions (Assuming B=1 if batched)
    def to_np_unbatched(x):
        arr = x.detach().cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
        # If it has a batch dimension (e.g. 5D for images, 4D for depth), squeeze the first dim
        if arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    # --- Extract GT Context ---
    images = to_np_unbatched(gt['images'])  # (L, 3, H, W)
    gt_extrinsics = to_np_unbatched(gt['extrinsics'])  # (L, 3, 4)
    gt_intrinsics = to_np_unbatched(gt['intrinsics'])  # (L, 3, 3)

    L, _, H, W = images.shape

    # --- Extract Predictions ---
    log_depths = to_np_unbatched(pred['log_depths'])  # (L, H, W)
    depths_pred = np.exp(log_depths)  # Convert log_depth back to metric Z

    focal_lengths = to_np_unbatched(pred['focal_length']).flatten()  # Extract focal lengths

    rel_T = to_np_unbatched(pred['relative_camera_translations'])  # (L-1, 3)
    rel_R = to_np_unbatched(pred['relative_camera_rotations'])  # (L-1, 3, 3)

    # --- Format Images ---
    if images.dtype in (np.float32, np.float64) and images.max() <= 1.0:
        images = (images * 255).astype(np.uint8)
    images = np.transpose(images, (0, 2, 3, 1))  # -> (L, H, W, 3)

    # --- Accumulate Relative Poses into Absolute Poses ---
    extrinsics_pred = np.zeros((L, 3, 4), dtype=np.float32)

    # Anchor Frame 0 to the exact starting position of the GT sequence
    extrinsics_pred[0] = gt_extrinsics[0]

    # Sequentially chain the relative predictions
    for i in range(1, L):
        R_prev = extrinsics_pred[i - 1, :3, :3]
        T_prev = extrinsics_pred[i - 1, :3, 3]

        # R_rel = R_next @ R_prev^T  =>  R_next = R_rel @ R_prev
        R_next = rel_R[i - 1] @ R_prev

        # T_rel = T_next - T_prev  =>  T_next = T_prev + T_rel
        T_next = T_prev + rel_T[i - 1]

        extrinsics_pred[i, :3, :3] = R_next
        extrinsics_pred[i, :3, 3] = T_next

    # --- Setup Viser UI ---
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    gui_frame = server.gui.add_slider("Frame", min=0, max=L - 1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("Play Sequence", initial_value=False)
    gui_fps = server.gui.add_slider("Playback FPS", min=1, max=60, step=1, initial_value=10)
    gui_show_all = server.gui.add_checkbox("Show All Frustums", initial_value=True)
    gui_accumulate = server.gui.add_checkbox("Accumulate Point Clouds", initial_value=False)

    # --- Draw Camera Frustums ---
    camera_nodes = []
    for i in range(L):
        R = extrinsics_pred[i, :3, :3]
        T = extrinsics_pred[i, :3, 3]

        # Handle sequence-unified vs per-frame focal length predictions
        f = focal_lengths[0] if len(focal_lengths) == 1 else focal_lengths[i]

        # Calculate Vertical FOV from the predicted focal length
        fov = 2 * np.arctan(H / (2 * f))

        # Convert Rotation Matrix to Viser Quaternion (w, x, y, z)
        quat_xyzw = Rotation.from_matrix(R).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        cam_node = server.scene.add_camera_frustum(
            f"/trajectory/cam_{i:04d}",
            fov=fov,
            aspect=W / H,
            scale=0.1,  # Physical size of the frustum
            image=images[i],  # Project the RGB image into the frustum
            position=T,
            wxyz=quat_wxyz,
        )
        camera_nodes.append(cam_node)

    # Cache for point clouds
    pc_nodes = [None] * L

    def update_frame():
        i = gui_frame.value

        # Lazy computation: Only unproject if we haven't rendered this frame yet
        if pc_nodes[i] is None:
            Z = depths_pred[i]

            # Use predicted focal length, but fallback to GT for the principal points
            f = focal_lengths[0] if len(focal_lengths) == 1 else focal_lengths[i]
            cx, cy = gt_intrinsics[i, 0, 2], gt_intrinsics[i, 1, 2]

            # Unproject to local camera coordinates
            X = (u - cx) * Z / f
            Y = (v - cy) * Z / f
            pts_cam = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

            # Flatten arrays for filtering
            Z_flat = Z.reshape(-1)
            colors_flat = images[i].reshape(-1, 3)

            # Drop invalid/empty depths (or wildly out of bounds predictions)
            valid = (Z_flat > 0) & (Z_flat < 50.0) & np.isfinite(Z_flat)
            pts_cam = pts_cam[valid]
            colors_flat = colors_flat[valid]

            # Transform local points to World coordinates using our ACCUMULATED matrices
            R = extrinsics_pred[i, :3, :3]
            T = extrinsics_pred[i, :3, 3]
            pts_world = (R @ pts_cam.T).T + T

            # Push to Viser
            pc_nodes[i] = server.scene.add_point_cloud(
                f"/point_clouds/frame_{i:04d}",
                points=pts_world,
                colors=colors_flat,
                point_size=0.05
            )

        # Update visibility states purely on the frontend
        accumulate = gui_accumulate.value
        show_all_frustums = gui_show_all.value

        for j in range(L):
            camera_nodes[j].visible = True if show_all_frustums else (j == i)
            if pc_nodes[j] is not None:
                pc_nodes[j].visible = (j == i) or (accumulate and j <= i)

    @gui_frame.on_update
    def _(_):
        update_frame()

    @gui_show_all.on_update
    def _(_):
        update_frame()

    @gui_accumulate.on_update
    def _(_):
        update_frame()

    # Draw the first frame immediately
    update_frame()

    # Main render loop
    while True:
        if gui_play.value:
            gui_frame.value = (gui_frame.value + 1) % L
        time.sleep(1.0 / gui_fps.value)


if __name__ == "__main__":
    from datasets.nrgbd_dataset import nrgbd_dataset
    # from datasets.rtmv.rtmv_test_dataset import rtmv_test_dataset
    from models.PP3DR import PP3DR

    model = PP3DR()
    dataset = nrgbd_dataset()
    # I want to see which parameters are left over after loading this.
    loaded_state_dict = torch.load("/vulcanscratch/hughma/PP3DR/16-36_changes_200-epochs/PP3DR.pth", weights_only=True, map_location="cpu")

    # 2. Fix the DDP "module." prefix trap!
    clean_state_dict = {}
    for k, v in loaded_state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    # 3. Load the fine-tuned weights ON TOP of the initialization.
    model.load_state_dict(clean_state_dict, strict=False)
    print("post load")
    model = model.eval().to("cuda")

    data = dataset[0] # 8
    print("post dataset")
    for key in data:
        try:
            data[key] = data[key].unsqueeze(0)
        except:
            data[key] = torch.tensor([data[key]])
    with torch.amp.autocast("cuda", dtype = torch.bfloat16), torch.no_grad():
        pred = model(data['images'].to("cuda"), data['rope_x'].to("cuda"), data['rope_y'].to("cuda"), data['original_height'].to('cuda'), data['original_width'].to('cuda'))
    print("post pred")

    # You can scale the translations and depths if the dataset (like RTMV) is physically tiny
    scale = 10
    pred['relative_camera_translations'] *= scale
    pred['log_depths'] = pred['log_depths'] + np.log(scale)
    data['extrinsics'][:, :, :3, 3] *= scale
    for k in pred.keys():
        pred[k] = pred[k].to(torch.float32)
    for k in data.keys():
        data[k] = data[k].to(torch.float32)

    # Launch visualizer
    visualize_pred_sequence(pred, data)