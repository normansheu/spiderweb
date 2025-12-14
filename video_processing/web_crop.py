# File: web_crop.py

import open3d as o3d
import numpy as np

def interactive_crop(all_points_data, output_file_path_template, crop_step=5.0):
    """
    Launches an interactive GUI to crop a point cloud with selectable thresholds.

    Args:
        all_points_data (np.ndarray): An (N, 4) NumPy array where each row is (x, y, z, max_threshold).
        output_file_path_template (str): The base path for saving the file (e.g., ".../my_scan").
                                         The current threshold will be appended upon saving.
        crop_step (float, optional): The size of each crop step. Defaults to 5.0.

    Returns:
        None: This function handles saving interactively and does not return a final object.
    """
    # --- 1. Validate Input & Initialization ---
    if all_points_data is None or all_points_data.shape[0] == 0:
        print("🔴 Error: A valid, non-empty NumPy array of points must be provided.")
        return None

    pcd_for_vis = o3d.geometry.PointCloud()

    # State variables
    thresholds = np.unique(all_points_data[:, 3])
    thresholds.sort()
    current_threshold_idx = len(thresholds) - 1
    
    min_b = all_points_data[:, :3].min(axis=0)
    max_b = all_points_data[:, :3].max(axis=0)
    crop_bounds = {
        0: [min_b[0], max_b[0]], # X
        1: [min_b[1], max_b[1]], # Y
        2: [min_b[2], max_b[2]], # Z
    }
    
    axis_map = {0: 'X', 1: 'Y', 2: 'Z'}
    crop_sequence = [
        (2, "positive (+Z)", True), (2, "negative (-Z)", False),
        (1, "positive (+Y)", True), (1, "negative (-Y)", False),
        (0, "positive (+X)", True), (0, "negative (-X)", False),
    ]

    # --- 2. Interactive Cropping Loop (one stage at a time) ---
    for axis, direction_name, crop_from_positive in crop_sequence:
        history = [] 
        keep_cropping_this_direction = True

        def update_view(vis):
            """Filters master data based on current state and updates the visualization."""
            current_t = thresholds[current_threshold_idx]
            mask_t = all_points_data[:, 3] >= current_t
            
            cb = crop_bounds
            mask_x = (all_points_data[:, 0] >= cb[0][0]) & (all_points_data[:, 0] <= cb[0][1])
            mask_y = (all_points_data[:, 1] >= cb[1][0]) & (all_points_data[:, 1] <= cb[1][1])
            mask_z = (all_points_data[:, 2] >= cb[2][0]) & (all_points_data[:, 2] <= cb[2][1])
            
            final_mask = mask_t & mask_x & mask_y & mask_z
            visible_points = all_points_data[final_mask, :3]
            
            pcd_for_vis.points = o3d.utility.Vector3dVector(visible_points)
            vis.update_geometry(pcd_for_vis)

        def crop_action(vis, step=crop_step):
            nonlocal crop_bounds
            if crop_from_positive:
                history.append(crop_bounds[axis][1]) 
                crop_bounds[axis][1] -= step
            else:
                history.append(crop_bounds[axis][0])
                crop_bounds[axis][0] += step
            print(f"  Cropped. Points remaining: {len(pcd_for_vis.points)}")
            update_view(vis)

        def undo_action(vis):
            nonlocal crop_bounds
            if history:
                print("  ↩️ Undoing last crop step.")
                if crop_from_positive:
                    crop_bounds[axis][1] = history.pop()
                else:
                    crop_bounds[axis][0] = history.pop()
                print(f"  Restored. Points remaining: {len(pcd_for_vis.points)}")
                update_view(vis)
            else:
                print("  No more steps to undo for this direction.")
        
        def next_threshold(vis):
            nonlocal current_threshold_idx
            if current_threshold_idx < len(thresholds) - 1:
                current_threshold_idx += 1
                print(f"  Threshold ▲: {thresholds[current_threshold_idx]:.2f} | Points: {len(pcd_for_vis.points)}")
                update_view(vis)

        def prev_threshold(vis):
            nonlocal current_threshold_idx
            if current_threshold_idx > 0:
                current_threshold_idx -= 1
                print(f"  Threshold ▼: {thresholds[current_threshold_idx]:.2f} | Points: {len(pcd_for_vis.points)}")
                update_view(vis)

        def confirm_action(vis):
            nonlocal keep_cropping_this_direction
            keep_cropping_this_direction = False
            vis.close()
        
        def save_current_view(vis):
            """Saves the currently visible point cloud to a file."""
            current_threshold = thresholds[current_threshold_idx]
            output_path = f"{output_file_path_template} T{current_threshold:.2f}.pcd"
            
            if len(pcd_for_vis.points) > 0:
                o3d.io.write_point_cloud(output_path, pcd_for_vis)
                print(f"\n💾 Saved current view with {len(pcd_for_vis.points)} points to '{output_path}'")
            else:
                print("\n⚠️ Warning: No points to save.")


        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Cropping {axis_map[axis]}-Axis from {direction_name} side")
        
        vis.register_key_callback(ord('E'), crop_action)
        vis.register_key_callback(ord('R'), undo_action)
        vis.register_key_callback(ord('='), confirm_action)
        vis.register_key_callback(ord('M'), next_threshold)
        vis.register_key_callback(ord('N'), prev_threshold)
        vis.register_key_callback(ord('D'), lambda v: crop_action(v, step=50.0))
        vis.register_key_callback(ord('0'), save_current_view) # New hotkey
        
        update_view(vis)
        vis.add_geometry(pcd_for_vis)
        
        print("\n" + "="*60)
        print(f"NOW CROPPING {axis_map[axis]}-AXIS ({direction_name}) | Current Threshold: {thresholds[current_threshold_idx]:.2f}")
        print("  - Press [M] / [N] to increase/decrease threshold.")
        print(f"  - Press [E] to crop {crop_step} units.")
        print(f"  - Press [D] to crop 50.0 units.")
        print("  - Press [R] to undo the last crop step.")
        print("  - Press [0] to SAVE the current view.") # New instruction
        print("  - Press [=] to confirm and continue to the next axis.")
        print("="*60)
        
        while keep_cropping_this_direction:
            vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()

    # --- 3. Finalization ---
    # Automatic saving at the end is now removed.
    print("\n✅ Cropping complete!")
    print("   You can close the program. Any manually saved files are in the output directory.")
    
    return None # No longer returns the final point cloud object