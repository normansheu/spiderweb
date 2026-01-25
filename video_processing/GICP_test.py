import open3d as o3d
import numpy as np
import copy
import time
import itertools
import os
from tqdm import tqdm

def get_distinct_colors(n):
    """
    Generates a list of distinct RGB colors.
    """
    # Simple palette: Gold, Cyan, Red, Green, Purple, Blue, Lime, Magenta
    base_colors = [
        [1, 0.706, 0],      # Gold
        [0, 0.651, 0.929],  # Cyan
        [1.0, 0.2, 0.2],    # Red
        [0.2, 0.8, 0.2],    # Green
        [0.6, 0.2, 0.8],    # Purple
        [0.2, 0.2, 1.0],    # Blue
        [0.8, 1.0, 0.2],    # Lime
        [1.0, 0.2, 0.8]     # Magenta
    ]
    # If we have more clouds than colors, cycle through them
    return [base_colors[i % len(base_colors)] for i in range(n)]

def draw_registration_result(geometries):
    """
    Visualizes a list of geometries.
    """
    o3d.visualization.draw_geometries(geometries,
                                      window_name="Multi-Cloud Registration Result")

def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsamples the PCD and estimates normals.
    """
    # print(f"   :: Downsampling with voxel_size={voxel_size:.3f}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals (required for Point-to-Plane ICP)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
    return pcd_down

def get_90_degree_rotations():
    """
    Generates a list of all 24 unique orthogonal 3x3 rotation matrices.
    Checks every multiple of 90 degrees (0, 90, 180, 270) around X, Y, Z combinations.
    """
    rotations = []
    pi_2 = np.pi / 2.0
    
    # 0, 90, 180, 270 (which covers -90 and -180)
    angles = [0, pi_2, 2*pi_2, 3*pi_2]
    
    seen_rotations = set()
    
    # Iterate through all combinations of Euler angles (X, Y, Z)
    # This ensures we cover "face down", "sideways", etc.
    for ax_angle in itertools.product(angles, repeat=3):
        # Create rotation matrix from XYZ Euler angles
        R = o3d.geometry.get_rotation_matrix_from_xyz(ax_angle)
        
        # Use a tuple hash to avoid duplicates (there are only 24 unique matrices 
        # but 4*4*4=64 Euler combinations)
        R_hash = tuple(np.round(R.flatten(), 2))
        
        if R_hash not in seen_rotations:
            rotations.append(R)
            seen_rotations.add(R_hash)
                
    return rotations

def get_small_perturbations(current_rotation, range_deg, step_deg):
    """
    Generates rotation matrices that are slight variations of the current_rotation.
    Perturbs around X, Y, Z by +/- range_deg.
    """
    # Convert degrees to radians
    step_rad = np.deg2rad(step_deg)
    range_rad = np.deg2rad(range_deg)
    
    # Create range of angles: -range to +range
    # We use a small epsilon to include the upper bound
    angles = np.arange(-range_rad, range_rad + 0.00001, step_rad)
    
    perturbed_rotations = []
    
    # Generate combinations of small rotations (Roll, Pitch, Yaw)
    for ax_angle in itertools.product(angles, repeat=3):
        # Small rotation matrix from Euler angles
        R_perturb = o3d.geometry.get_rotation_matrix_from_xyz(ax_angle)
        
        # Apply perturbation to current rotation: R_new = R_perturb @ R_current
        R_final = np.dot(R_perturb, current_rotation)
        perturbed_rotations.append(R_final)
        
    return perturbed_rotations

def execute_90_deg_search(source, target, voxel_size, search_distance=50.0):
    """
    Coarse Search: Tries specific 90-degree rotations + Centroid alignment.
    """
    # print("   :: Executing Discrete 90-degree Rotation Search...")
    
    source_center = source.get_center()
    target_center = target.get_center()
    candidate_rotations = get_90_degree_rotations()
    
    best_fitness = -1.0
    best_transform = np.identity(4)
    
    for i, R in enumerate(candidate_rotations):
        # Align centroids
        t = target_center - np.dot(R, source_center)
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # Evaluate with specified search distance
        fitness = o3d.pipelines.registration.evaluate_registration(
            source, target, search_distance, T).fitness
            
        if fitness > best_fitness:
            best_fitness = fitness
            best_transform = T

    # print(f"   :: Best 90-deg base found. Fitness: {best_fitness:.4f}")
    return best_transform

def execute_angle_perturbation_search(source, target, current_transform, search_distance, range_deg, step_deg):
    """
    Variable Scale Search: Tries small rotations around the current transform.
    """
    # print(f"   :: Executing Angle Search (Range: ±{range_deg}°, Step: {step_deg}°)...")
    
    source_center = source.get_center()
    target_center = target.get_center()
    
    # Extract current rotation from the transform
    current_rotation = current_transform[:3, :3]
    
    perturbed_rotations = get_small_perturbations(current_rotation, range_deg, step_deg)
    
    best_transform = current_transform
    best_fitness = o3d.pipelines.registration.evaluate_registration(
            source, target, search_distance, current_transform).fitness
    
    # total_checks = len(perturbed_rotations)
    # print(f"      Checking {total_checks} perturbations...")

    # Using a simple loop as this is now much faster due to optimization
    for idx, R in tqdm(enumerate(perturbed_rotations)):
        # Re-calculate translation (t) for this specific rotation (R)
        t = target_center - np.dot(R, source_center)
        
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        fitness = o3d.pipelines.registration.evaluate_registration(
            source, target, search_distance, T).fitness
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_transform = T
            
    # print(f"      Best Fitness found: {best_fitness:.4f}")
    return best_transform


def refine_registration(source, target, voxel_size, current_transformation, range_multiplier=1.0):
    """
    GICP Refinement using Open3D registration_generalized_icp.
    """
    distance_threshold = voxel_size * range_multiplier

    # GICP benefits from normals / local geometry; ensure they exist
    if not source.has_normals():
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

    # Open3D GICP estimation object
    estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)

    result = o3d.pipelines.registration.registration_generalized_icp(
        source, target,
        distance_threshold,
        current_transformation,
        estimation_method=estimation,
        criteria=criteria
    )
    return result


def pyramid_registration(source, target, base_voxel_size=1.0):
    """
    Executes the 5-stage Pyramid Method with Hierarchical Angle Search.
    """
    
    OFFSET_SEARCH_DIST = base_voxel_size * 30.0
    
    # Multipliers for the pyramid
    multipliers = [20, 10, 5, 3, 1]
    
    current_transformation = np.identity(4)

    for scale in multipliers:
        voxel_size = base_voxel_size * scale
        # print(f"\n--- Pyramid Scale {scale}x (Voxel: {voxel_size:.2f}) ---")
        
        source_down = preprocess_point_cloud(source, voxel_size)
        target_down = preprocess_point_cloud(target, voxel_size)
        
        if scale == 20:
            current_transformation = execute_90_deg_search(
                source_down, target_down, voxel_size, search_distance=OFFSET_SEARCH_DIST)
                
        elif scale == 10:
            current_transformation = execute_angle_perturbation_search(
                source_down, target_down, current_transformation, 
                search_distance=OFFSET_SEARCH_DIST, range_deg=7.0, step_deg=0.5)

        elif scale == 5:
            current_transformation = execute_angle_perturbation_search(
                source_down, target_down, current_transformation, 
                search_distance=OFFSET_SEARCH_DIST, range_deg=2, step_deg=0.1)
            
            result = refine_registration(source_down, target_down, voxel_size, current_transformation, range_multiplier=1.5)
            current_transformation = result.transformation

        elif scale == 1:
            result = refine_registration(source_down, target_down, voxel_size, current_transformation, range_multiplier=0.3)
            current_transformation = result.transformation
            return result
            
        else:
            result = refine_registration(source_down, target_down, voxel_size, current_transformation, range_multiplier=1.0)
            current_transformation = result.transformation
            # print(f"   :: Fitness: {result.fitness:.4f}")

    return result

if __name__ == "__main__":
    print("1. Loading Point Clouds...")
    
    # --- LIST YOUR PCD FILES HERE ---
    file_list = [
        # "C:/Users/samue/Downloads/Research/Spider/Current/spiderweb/video_processing/point_clouds/tangle016u 255 2025-11-22 19-24-45 T0.30.pcd",
        "C:/Users/samue/Downloads/Research/Spider/Current/spiderweb/video_processing/point_clouds/tangle016r 255 2025-11-15 05-55-05 T0.30.pcd",
        "C:/Users/samue/Downloads/Research/Spider/Current/spiderweb/video_processing/point_clouds/tangle016 255 2025-11-15 05-31-27 T0.30.pcd"
    ]
    
    pcds = []
    for f in file_list:
        try:
            pcd = o3d.io.read_point_cloud(f)
            if pcd.is_empty():
                print(f"Warning: {f} is empty, skipping.")
                continue
            pcds.append(pcd)
            print(f"   Loaded: {f} ({len(pcd.points)} points)")
        except Exception as e:
            print(f"   Error loading {f}: {e}")

    if len(pcds) < 2:
        print("Error: Need at least 2 point clouds to perform alignment.")
        exit()

    # Base Voxel Size
    BASE_VOXEL_SIZE = 3.0
    
    # Initialize Global Map with the first PCD
    print(f"\n2. Initializing Alignment Sequence with {len(pcds)} clouds...")
    
    colors = get_distinct_colors(len(pcds))
    
    # We maintain a list of individual transformed clouds for visualization
    # And a single 'merged' cloud for the target of the next registration
    visual_pcds = []
    
    # 1. Setup Anchor (First Cloud)
    anchor_pcd = pcds[0]
    anchor_pcd.paint_uniform_color(colors[0])
    visual_pcds.append(anchor_pcd)
    
    # This is the "Growing Map" we align against
    merged_map = copy.deepcopy(anchor_pcd)

    start_time_total = time.time()

    # 2. Iterate and Align
    for i in range(1, len(pcds)):
        source_pcd = pcds[i]
        print(f"\n--- Aligning Cloud {i+1} (Source) to Merged Map (Target) ---")
        
        # Align Source -> Merged Map
        result = pyramid_registration(source_pcd, merged_map, BASE_VOXEL_SIZE)
        
        print(f"   Alignment {i+1} Finished. Fitness: {result.fitness:.4f}")
        
        # Transform the source
        source_pcd.transform(result.transformation)
        source_pcd.paint_uniform_color(colors[i])
        
        # Add to visual list
        visual_pcds.append(source_pcd)
        
        # Update the Merged Map (Target for next iteration)
        # We assume the map grows as we add pieces
        merged_map += source_pcd
        
        # Optional: Downsample the map if it gets too huge to keep speed up
        # merged_map = merged_map.voxel_down_sample(BASE_VOXEL_SIZE * 0.5)

    print(f"\nTotal Sequence Time: {time.time() - start_time_total:.3f} sec")

    # --- Step 3: Visualize ---
    print("\n3. Visualizing Final Multi-Color Alignment...")
    draw_registration_result(visual_pcds)

    # --- Step 4: Merge and Save ---
    print("\n4. Saving Combined Point Cloud...")
    
    # We already have 'merged_map', but 'visual_pcds' contains the colored versions if we want color
    # Let's combine visual_pcds to preserve the distinct colors in the saved file
    final_merged = o3d.geometry.PointCloud()
    for p in visual_pcds:
        final_merged += p
    
    # Construct output path based on the first filename
    first_path = file_list[0]
    folder, filename = os.path.split(first_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_sequence_merged{ext}"
    output_path = os.path.join(folder, output_filename)
    
    try:
        o3d.io.write_point_cloud(output_path, final_merged)
        print(f"   Successfully saved merged file to:\n   {output_path}")
    except Exception as e:
        print(f"   Failed to save file: {e}")






