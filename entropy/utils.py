import numpy as np


def load_pcd_array(file_path):
    """load numpy array"""
    print("Loading the point cloud array...")
    points = np.load(file_path)
    points = np.unpackbits(points)
    print(f"Loaded point cloud with {sum(points)} points.")
    points = np.reshape(points, (850, 560, 850))
    return points

def calculate_density(points, M):
    """
    Calculate the density of points in a 3D grid.
    
    Parameters:
        points (numpy.ndarray): A 2D array of shape (3, N) containing (x, y, z) coordinates of the point cloud.
        num_subdivisions (int): The number of subdivisions (M) along each axis.
    
    Returns:
        numpy.ndarray: A 3D array of shape (M, M, M) where each entry (i, j, k) counts the number of points in the corresponding voxel.
    """
    assert M > 0, "Number of subdivisions must be positive"
    print(f"Calculating density with {M} subdivisions...")

    box_size = points.shape
    # density_counts = np.zeros((M, M, M), dtype=float)
    subdivision_size = np.array(box_size) / M
    voxel_volume = np.prod(subdivision_size)
    
    indices = np.nonzero(points)
    indices = np.array(indices).T  # (N, 3)
    scaled_indices = (indices / subdivision_size).astype(int)

    flat_indices = np.ravel_multi_index(scaled_indices.T, (M, M, M))
    density_counts_flat = np.bincount(flat_indices, minlength=M**3)
    density = density_counts_flat.reshape((M, M, M)) / voxel_volume
    return density


def calculate_density_levels(points, M, num_levels=10, max_density=None):
    print(f"Calculating density levels with {M} subdivisions...")

    density = calculate_density(points, M)
    density_levels = density.copy()
    
    # Scale non-zero density levels to fit within [1, num_levels - 1]
    if max_density is None:
        max_density = density.max()
    print("Normalize with max density:", max_density)
    if max_density > 0:
        nonzero_indices = density > 0  # Identify non-zero density voxels
        density_levels[nonzero_indices] = (
            (density_levels[nonzero_indices] * (num_levels) / max_density).astype(int) + 1
        )
    else:
        density_levels = density_levels.astype(int)  # Convert to int if all densities are zero
    
    # Ensure density levels are within [0, num_levels]
    density_levels = np.clip(density_levels, 0, num_levels).astype(int)
    return density, density_levels


def record_distribution(density_levels, num_levels=10):
    print("Recording the distribution of density levels...")
    
    # Count how many subregions fall into each density level
    distribution = np.zeros(num_levels + 1, dtype=int)
    for level in range(num_levels + 1):
        l_indices = density_levels == level
        distribution[level] = np.sum(l_indices)
    
    print(f"Distribution of density levels: {distribution}")
    return distribution


if __name__ == "__main__":
    points = load_pcd_array("../point_clouds/@011.npy")
    indices = np.nonzero(points)
    density, density_levels = calculate_density_levels(points, M=20, num_levels=10)
    print(density)
    print(density.shape)
    distribution = record_distribution(density_levels)
    print(distribution)
