import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# File path
fp = '../video_processing/point_clouds/@011 255 2024-10-04 03-20-37.pcd'
num_levels = 10  # Specify the number of density levels

# Step 1: Load the Point Cloud Data (PCD)
def load_pcd(file_path):
    print("Loading the point cloud...")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("The point cloud data is empty.")
    print(f"Loaded point cloud with {len(points)} points.")
    return points

# Step 2: Subdivide the space and calculate density levels
def calculate_density_levels(points, num_subdivisions, num_levels=10):
    print(f"Calculating density levels with {num_subdivisions} subdivisions...")
    
    # Get the bounds of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Calculate the size of each subdivision (voxel)
    subdivision_size = (max_bound - min_bound) / num_subdivisions
    voxel_volume = np.prod(subdivision_size)  # Volume of each voxel in 3D space

    # Initialize array to store density counts per voxel
    density_counts = np.zeros(num_subdivisions**3, dtype=float)
    
    # Calculate voxel indices for each point
    indices = ((points - min_bound) / subdivision_size).astype(int)
    indices = np.clip(indices, 0, num_subdivisions - 1)
    flat_indices = np.ravel_multi_index(indices.T, (num_subdivisions, num_subdivisions, num_subdivisions))
    
    # Count points in each voxel
    for idx in flat_indices:
        density_counts[idx] += 1
    
    # Convert counts to densities by dividing by voxel volume
    density_levels = density_counts / voxel_volume
    
    # Scale non-zero density levels to fit within [1, num_levels - 1]
    max_density = density_levels.max()
    print("Max density:", max_density)
    if max_density > 0:
        nonzero_indices = density_levels > 0  # Identify non-zero density voxels
        density_levels[nonzero_indices] = (
            (density_levels[nonzero_indices] * (num_levels) / max_density).astype(int) + 1
        )
    else:
        density_levels = density_levels.astype(int)  # Convert to int if all densities are zero
    
    # Ensure density levels are within [0, num_levels - 1]
    density_levels = np.clip(density_levels, 0, num_levels - 1).astype(int)
    
    return density_levels





# Step 3: Record the distribution of the density levels
def record_distribution(density_levels):
    print("Recording the distribution of density levels...")
    
    # Count how many subregions fall into each density level
    distribution = np.zeros(num_levels, dtype=int)
    for level in density_levels:
        distribution[level] += 1
    
    print(f"Distribution of density levels: {distribution}")
    return distribution

# Step 4: Compute entropy based on the distribution of density levels
def compute_entropy(distribution, exclude_zero=False):
    print(f"Computing entropy (excluding level 0: {exclude_zero})...")
    
    # Normalize the distribution to get probabilities
    if exclude_zero:
        distribution = distribution[1:]  # Exclude level 0
    probabilities = distribution / distribution.sum()
    
    # Compute entropy
    entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    
    print(f"Computed entropy: {entropy}")
    return entropy

# Step 5: Calculate and record entropies for different subdivisions
def calculate_entropies_for_subdivisions(points, min_subdivisions=100, max_subdivisions=120):
    entropies_including_zero = []
    entropies_excluding_zero = []
    averages_excluding_zero = []
    
    for num_subdivisions in range(min_subdivisions, max_subdivisions + 1):
        density_levels = calculate_density_levels(points, num_subdivisions)
        distribution = record_distribution(density_levels)
        entropy_including_zero = compute_entropy(distribution, exclude_zero=False)
        entropy_excluding_zero = compute_entropy(distribution, exclude_zero=True)
        entropies_excluding_zero.append(entropy_excluding_zero)
        # Append the sliding average of entropies of previous 5
        averages_excluding_zero.append((num_subdivisions, np.mean(entropies_excluding_zero[-20:])))
        entropies_including_zero.append((num_subdivisions, entropy_including_zero))
    
    return entropies_including_zero, averages_excluding_zero

if __name__ == "__main__":
    # Example usage
    file_path = fp 
    points = load_pcd(file_path)
    entropies_including_zero, entropies_excluding_zero,  = calculate_entropies_for_subdivisions(points)
    
    # Plot the results
    subdivisions, entropies_inc = zip(*entropies_including_zero)
    _, entropies_exc = zip(*entropies_excluding_zero)
    
    print("Final entropy avg: ", entropies_exc[-1])
