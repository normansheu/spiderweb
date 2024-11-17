# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt

# # File path
# fp = 'video_processing/point_clouds/@011 255 2024-10-04 03-20-37.pcd'

# # Load the Point Cloud Data (PCD)
# def load_pcd(file_path):
#     print("Loading the point cloud...")
#     pcd = o3d.io.read_point_cloud(file_path)
#     points = np.asarray(pcd.points, dtype=np.float64)  # Use higher precision
#     if points.size == 0:
#         raise ValueError("The point cloud data is empty.")
#     print(f"Loaded point cloud with {len(points)} points.")
#     return points

# # Subdivide the space and calculate density levels with higher precision
# def calculate_density_levels(points, num_subdivisions, num_levels):
#     # Calculate bounds and subdivision size
#     min_bound = points.min(axis=0)
#     max_bound = points.max(axis=0)
#     subdivision_size = (max_bound - min_bound) / num_subdivisions

#     # Initialize array for density counts
#     density_counts = np.zeros(num_subdivisions**3, dtype=np.float64)  # Higher precision array
    
#     # Map points to subdivision bins
#     indices = ((points - min_bound) / subdivision_size).astype(int)
#     indices = np.clip(indices, 0, num_subdivisions - 1)
#     flat_indices = np.ravel_multi_index(indices.T, (num_subdivisions, num_subdivisions, num_subdivisions))
    
#     # Count points in each subdivision (bin)
#     for idx in flat_indices:
#         density_counts[idx] += 1.0  # Accumulate counts in high precision

#     # Apply quantization to map density counts to desired levels
#     min_density, max_density = density_counts.min(), density_counts.max()
#     if max_density > min_density:
#         # Retain higher precision in normalization before categorizing into levels
#         density_levels = ((density_counts - min_density) / (max_density - min_density) * (num_levels - 1)).astype(np.float64)
#     else:
#         density_levels = np.zeros_like(density_counts)  # All densities are the same

#     # Convert to integer levels without premature rounding for higher precision
#     density_levels = np.round(density_levels).astype(int)
#     return density_levels

# # Record the distribution of the density levels with high precision
# def record_distribution(density_levels, num_levels):
#     distribution = np.zeros(num_levels, dtype=np.float64)  # Use floating-point precision
#     for level in density_levels:
#         distribution[level] += 1.0  # Keep counts in floating-point format
#     return distribution

# # Compute entropy with high precision
# def compute_entropy(distribution, exclude_zero=False):
#     if exclude_zero:
#         distribution = distribution[1:]
#     probabilities = distribution / distribution.sum()
#     entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
#     return entropy

# # Function to run the analysis over increasing density levels and plot entropy
# def plot_entropy_vs_levels(points, num_subdivisions, max_levels):
#     levels = range(2, max_levels + 1)
#     entropies_including_zero = []
#     entropies_excluding_zero = []
#     log_growth_curve = []
#     baseline_entropy = None  # Initialize baseline_entropy

#     # Calculate the entropy for each level and record it
#     for num_levels in levels:
#         density_levels = calculate_density_levels(points, num_subdivisions, num_levels)
#         distribution = record_distribution(density_levels, num_levels)
#         entropy_incl_zero = compute_entropy(distribution, exclude_zero=False)
#         entropy_excl_zero = compute_entropy(distribution, exclude_zero=True)
#         entropies_including_zero.append(entropy_incl_zero)
#         entropies_excluding_zero.append(entropy_excl_zero)
        
#         # Set baseline_entropy when num_levels = 10
#         if num_levels == 10:
#             baseline_entropy = entropy_excl_zero
#         # Ensure baseline_entropy is set before appending log growth values
#         if baseline_entropy is not None:
#             log_growth_curve.append(baseline_entropy + np.log2(num_levels / 10))
#         else:
#             log_growth_curve.append(0)  # Placeholder value if baseline_entropy not set

#     # Plot the entropy values and the log growth curve
#     plt.plot(levels, entropies_including_zero, label="Including Level 0")
#     plt.plot(levels, entropies_excluding_zero, label="Excluding Level 0")
#     plt.plot(levels, log_growth_curve, label="Logarithmic Growth (starting from Level 10)", linestyle='--')

#     plt.xlabel("Number of Levels")
#     plt.ylabel("Entropy")
#     plt.title("Entropy vs. Number of Density Levels")
#     plt.legend()
#     plt.show()

# if __name__ == "__main__":
#     file_path = fp
#     points = load_pcd(file_path)
#     num_subdivisions = 100  # Fixed number of subdivisions
#     max_levels = 100  # Define the maximum number of density levels for the plot
#     plot_entropy_vs_levels(points, num_subdivisions, max_levels)

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# File path
fp = 'video_processing/point_clouds/@011 255 2024-10-04 03-20-37.pcd'

# Load the Point Cloud Data (PCD)
def load_pcd(file_path):
    print("Loading the point cloud data from file...")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        raise ValueError("The point cloud data is empty.")
    print(f"Loaded point cloud with {len(points)} points.")
    return points

# Subdivide the space and calculate density levels
def calculate_density_levels(points, num_subdivisions, num_levels):
    print(f"Calculating density levels with {num_subdivisions} subdivisions...")
    
    # Get the bounds of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Calculate the size of each subdivision
    subdivision_size = (max_bound - min_bound) / num_subdivisions
    
    # Initialize the density levels array
    density_levels = np.zeros(num_subdivisions**3, dtype=int)
    
    # Calculate the density levels
    indices = ((points - min_bound) / subdivision_size).astype(int)
    indices = np.clip(indices, 0, num_subdivisions - 1)  # Ensure indices are within bounds
    flat_indices = np.ravel_multi_index(indices.T, (num_subdivisions, num_subdivisions, num_subdivisions))
    for idx in flat_indices:
        density_levels[idx] += 1
    
    # Assign density levels, reserving 0 for exactly zero density and scaling non-zero densities
    max_density = density_levels.max()
    print("Max density:", max_density)
    if max_density > 0:
        nonzero_indices = density_levels > 0
        # Scale non-zero density levels to fit within [1, num_levels - 1]
        density_levels[nonzero_indices] = (density_levels[nonzero_indices] * (num_levels - 1) / max_density).astype(int) + 1
    
    # Ensure density levels are within [0, num_levels - 1]
    density_levels = np.clip(density_levels, 0, num_levels - 1)
    
    return density_levels

# Record the distribution of the density levels
def record_distribution(density_levels, num_levels):
    print("Recording the distribution of density levels...")
    
    # Count how many subregions fall into each density level
    distribution = np.zeros(num_levels, dtype=int)
    for level in density_levels:
        distribution[level] += 1
    
    print(f"Distribution of density levels: {distribution}")
    return distribution

# Compute entropy with high precision
def compute_entropy(distribution, exclude_zero=False):
    if exclude_zero:
        distribution = distribution[1:]
    probabilities = distribution / distribution.sum()
    entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    print(f"Computed entropy: {entropy}")
    return entropy

# Function to plot H(W; 2L) and H(W; L) + 1 as functions of L
def plot_entropy_functions(points, num_subdivisions, initial_levels=2, max_levels=520):
    print("Starting entropy analysis to plot H(W; 2L) and H(W; L) + 1 as functions of L.")
    entropies_L = []
    entropies_2L = []
    upper_bounds = []
    L_values = []

    levels = [initial_levels]
    while levels[-1] * 2 <= max_levels:
        levels.append(levels[-1] * 2)

    # Calculate H(W; L), H(W; 2L), and H(W; L) + 1
    for i in range(len(levels) - 1):
        num_levels_L = levels[i]
        num_levels_2L = levels[i + 1]
        
        # Calculate H(W; L)
        print(f"\nProcessing num_levels = {num_levels_L} for H(W; L)...")
        density_levels_L = calculate_density_levels(points, num_subdivisions, num_levels_L)
        distribution_L = record_distribution(density_levels_L, num_levels_L)
        entropy_L = compute_entropy(distribution_L, exclude_zero=True)
        entropies_L.append(entropy_L)
        L_values.append(num_levels_L)

        # Calculate H(W; 2L)
        print(f"Processing num_levels = {num_levels_2L} for H(W; 2L)...")
        density_levels_2L = calculate_density_levels(points, num_subdivisions, num_levels_2L)
        distribution_2L = record_distribution(density_levels_2L, num_levels_2L)
        entropy_2L = compute_entropy(distribution_2L, exclude_zero=True)
        entropies_2L.append(entropy_2L)

        # Calculate H(W; L) + 1 for the upper bound
        upper_bound = entropy_L + 1
        upper_bounds.append(upper_bound)
        print(f"Upper bound H(W; L) + 1 for level {num_levels_L}: {upper_bound}")

    # Plot H(W; 2L) and H(W; L) + 1 with respect to L
    plt.plot(L_values, entropies_2L, 'o-', label="H(W; 2L)")
    plt.plot(L_values, upper_bounds, 'x--', label="H(W; L) + 1")
    plt.xlabel("Number of Levels (L)")
    plt.ylabel("Entropy")
    plt.title("Plot of H(W; 2L) and H(W; L) + 1 as functions of L")
    plt.legend()
    plt.show()
    print("Plotting complete.")

if __name__ == "__main__":
    file_path = fp
    points = load_pcd(file_path)
    num_subdivisions = 100  # Fixed number of subdivisions
    plot_entropy_functions(points, num_subdivisions)
