import viser
import time
import numpy as np
from scipy.spatial.transform import Rotation


def visualize_gt_sequence(gt, port=8080):
    """
    Visualizes a ground-truth sequence to verify C2W and OpenCV coordinates.
    Expects `gt` dictionary to contain:
      images:     (L, 3, H, W)
      depths:     (L, H, W)
      extrinsics: (L, 3, 4)
      intrinsics: (L, 3, 3)
    """
    server = viser.ViserServer(port=port)
    print(f"Viser server running at http://localhost:{port}")

    # 1. Safely extract Batch 0 and convert to NumPy
    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

    images = to_np(gt['images'])  # (L, 3, H, W)
    depths = to_np(gt['depths'])  # (L, H, W)
    # print(depths.shape) # DEBUG
    extrinsics = to_np(gt['extrinsics'])  # (L, 3, 4)
    intrinsics = to_np(gt['intrinsics'])  # (L, 3, 3)

    L, _, H, W = images.shape

    # Format images for Viser (H, W, 3) uint8
    if images.dtype in (np.float32, np.float64) and images.max() <= 1.0:
        images = (images * 255).astype(np.uint8)
    images = np.transpose(images, (0, 2, 3, 1))  # -> (L, H, W, 3)

    # Pre-compute meshgrid for unprojection
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    gui_frame = server.gui.add_slider("Frame", min=0, max=L - 1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("Play Sequence", initial_value=False)
    gui_fps = server.gui.add_slider("Playback FPS", min=1, max=60, step=1, initial_value=10)
    gui_show_all = server.gui.add_checkbox("Show All Frustums", initial_value=True)
    gui_accumulate = server.gui.add_checkbox("Accumulate Point Clouds", initial_value=False)

    camera_nodes = []
    for i in range(L):
        R = extrinsics[i, :3, :3]
        T = extrinsics[i, :3, 3]
        K = intrinsics[i]

        # Calculate vertical Field of View from intrinsic matrix
        fy = K[1, 1]
        fov = 2 * np.arctan(H / (2 * fy))

        # Convert Rotation Matrix to Quaternion.
        # Scipy uses (x,y,z,w). Viser expects (w,x,y,z).
        quat_xyzw = Rotation.from_matrix(R).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        cam_node = server.scene.add_camera_frustum(
            f"/trajectory/cam_{i:04d}",
            fov=fov,
            aspect=W / H,
            scale=0.1,  # Physical size of the frustum
            image=images[i],  # Renders the RGB image inside the frustum
            position=T,
            wxyz=quat_wxyz,
        )
        camera_nodes.append(cam_node)

    # Cache to store point cloud node handles so we only compute them once
    pc_nodes = [None] * L

    def update_frame():
        i = gui_frame.value

        # Lazy computation: Only unproject and send if we haven't seen this frame yet
        if pc_nodes[i] is None:
            Z = depths[i]
            K = intrinsics[i]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Unproject to local camera coordinates
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            # print(cx, cy, fx, fy)
            # print(X.max(), Y.max(), Z.max()) # DEBUG
            # print(X.shape, Y.shape, Z.shape)
            pts_cam = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

            # Flatten arrays for filtering
            Z_flat = Z.reshape(-1)
            colors_flat = images[i].reshape(-1, 3)

            # Drop invalid/empty depths
            valid = (Z_flat > 0) & np.isfinite(Z_flat)
            pts_cam = pts_cam[valid]
            colors_flat = colors_flat[valid]

            # Transform local points to World coordinates (C2W Math)
            R = extrinsics[i, :3, :3]
            T = extrinsics[i, :3, 3]
            pts_world = (R @ pts_cam.T).T + T

            # Push to Viser under a unique, permanent path for this frame
            pc_nodes[i] = server.scene.add_point_cloud(
                f"/point_clouds/frame_{i:04d}",
                points=pts_world,
                colors=colors_flat,
                point_size=0.005  # Adjust this scale based on your dataset!
            )

        # Read toggle states
        accumulate = gui_accumulate.value
        show_all_frustums = gui_show_all.value

        # Update visibility states purely on the frontend (zero data transfer overhead)
        for j in range(L):
            camera_nodes[j].visible = True if show_all_frustums else (j == i)

            if pc_nodes[j] is not None:
                # If accumulating, show the current frame and all previous frames up to this point.
                # If not accumulating, ONLY show the exact current frame.
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
            # Advance frame, loop back to 0 at the end
            gui_frame.value = (gui_frame.value + 1) % L
        time.sleep(1.0 / gui_fps.value)


# ==========================================
# Example usage with dummy data
# ==========================================
if __name__ == "__main__":
    # from datasets.nrgbd_dataset import nrgbd_dataset as dataset
    # from datasets.dtu_dataset import dtu_dataset as dataset
    # from datasets.dynamic_replica_dataset import dynamic_replica_dataset as dataset
    # from datasets.eth3d_dataset import eth3d_dataset as dataset
    # from datasets.flying_things_3d_dataset import flying_things_3d_dataset as dataset
    from datasets.interior_net_dataset import interior_net_dataset as dataset
    dataset = dataset()
    first = dataset[0]
    first['images'] = first['raw_images']
    first['depths'] = first['raw_depths']
    first['intrinsics'] = first['raw_intrinsics']

    # Change scale
    scale = 1
    print(first['extrinsics'].shape)
    for extrinsic in first['extrinsics']:
        extrinsic[:, 3] *= scale
        # print(extrinsic)
    for depth in first['depths']:
        depth *= scale

    visualize_gt_sequence(first)