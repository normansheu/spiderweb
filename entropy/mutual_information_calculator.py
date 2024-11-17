import open3d as o3d
import numpy as np

# Step 1: Load the Point Cloud Data (PCD)
def load_pcd(file_path):
    print("Loading the point cloud...")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    print(f"Loaded point cloud with {len(points)} points.")
    return points

# Step 2: Subdivide into 8000 regions (20x20x20) and count points in each region
def subdivide_and_count(points):
    print("Subdividing space into 8000 rectangular regions (20x20x20)...")
    
    # Get the bounds of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Compute the width of each bin (subdivision) along each dimension
    bin_size = (max_bound - min_bound) / 20.0  # 20 segments per dimension
    
    # Initialize a 3D array to count points in each of the 8000 subregions
    count_grid = np.zeros((20, 20, 20))
    
    # For each point, determine which subregion (bin) it belongs to
    for point in points:
        indices = np.floor((point - min_bound) / bin_size).astype(int)
        indices = np.clip(indices, 0, 19)  # Ensure indices stay within bounds
        count_grid[tuple(indices)] += 1
    
    total_points = len(points)
    print("Finished counting points in each subregion.")
    return count_grid, total_points

# Step 3: Classify density levels based on the counts
def classify_density(count_grid, total_points):
    print("Classifying density levels...")
    
    # Flatten the 3D count grid for easier processing
    flattened_grid = count_grid.flatten()
    
    # Normalize the counts by dividing by the total number of points to get the density
    densities = flattened_grid / total_points
    
    # Initialize the classification array for density levels (0-9)
    density_levels = np.zeros_like(densities, dtype=int)
    
    # Classify density into levels (0-9), where level 0 is for zero density and levels 1-9 for non-zero densities
    for i, density in enumerate(densities):
        if density == 0:
            density_levels[i] = 0  # Level 0 for density exactly 0
        elif density > 0.005:
            density_levels[i] = 9  # Level 9 for densities greater than 0.005
        else:
            density_levels[i] = int(np.ceil(density / (0.005 / 9)))  # Levels 1 to 9
    
    # Reshape density levels back to 3D for adjacency analysis
    density_levels_3d = density_levels.reshape((20, 20, 20))
    print("Density classification completed.")
    return density_levels_3d

# Step 4: Compute Mutual Information for Adjacent Regions (Skip if Self Level is 0)
def compute_mutual_information(density_levels_3d):
    print("Computing mutual information between adjacent blocks, skipping if self level is 0...")

    # Initialize a dictionary to store mutual information along each axis
    mi_values = {'x': 0, 'y': 0, 'z': 0}

    # Loop over each axis (x, y, z) to compute MI between adjacent blocks
    for axis in range(3):
        axis_name = ['x', 'y', 'z'][axis]
        
        # Initialize joint counts for pairs of density levels (0-9)
        joint_counts = np.zeros((10, 10))

        # Loop over the 3D grid to populate joint counts for adjacent pairs
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    # Get the density level of the current block
                    current_level = int(density_levels_3d[i, j, k])
                    
                    # Skip if the current (self) level is 0
                    if current_level == 0:
                        continue

                    # Determine the adjacent block based on the current axis
                    if axis == 0 and i < 19:  # X-axis (adjacent in i+1)
                        adjacent_level = int(density_levels_3d[i+1, j, k])
                    elif axis == 1 and j < 19:  # Y-axis (adjacent in j+1)
                        adjacent_level = int(density_levels_3d[i, j+1, k])
                    elif axis == 2 and k < 19:  # Z-axis (adjacent in k+1)
                        adjacent_level = int(density_levels_3d[i, j, k+1])
                    else:
                        continue
                    
                    # Update joint count for the pair (current_level, adjacent_level)
                    joint_counts[current_level, adjacent_level] += 1

        # Normalize to get joint probabilities
        joint_probabilities = joint_counts / joint_counts.sum()
        
        # Compute marginal probabilities from joint distribution
        marginal_prob_x = np.sum(joint_probabilities, axis=1)
        marginal_prob_y = np.sum(joint_probabilities, axis=0)

        # Calculate mutual information for the current axis
        mi = 0
        for x in range(10):
            for y in range(10):
                if joint_probabilities[x, y] > 0:
                    mi += joint_probabilities[x, y] * np.log2(joint_probabilities[x, y] / (marginal_prob_x[x] * marginal_prob_y[y]))

        mi_values[axis_name] = mi
        print(f"Mutual Information along {axis_name}-axis (skipping if self level is 0): {mi:.4f}")

    return mi_values

# Main function to execute the steps and compute mutual information
if __name__ == "__main__":
    # Path to your point cloud file (PCD)
    file_path = "video_processing/point_clouds/@020 255 2024-10-19 15-36-41.pcd"
    
    # Step 1: Load the PCD file
    points = load_pcd(file_path)
    
    # Step 2: Subdivide space and count points in each subregion
    count_grid, total_points = subdivide_and_count(points)
    
    # Step 3: Classify density levels based on point counts (into levels 0-9)
    density_levels_3d = classify_density(count_grid, total_points)
    
    # Step 4: Compute mutual information for adjacent blocks along each axis, skipping if self level is 0
    mutual_info = compute_mutual_information(density_levels_3d)
    
    # Print final mutual information values
    print(f"Mutual information (x-axis): {mutual_info['x']:.4f}")
    print(f"Mutual information (y-axis): {mutual_info['y']:.4f}")
    print(f"Mutual information (z-axis): {mutual_info['z']:.4f}")