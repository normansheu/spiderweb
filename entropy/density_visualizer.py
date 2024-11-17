import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

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

# Step 3: Calculate and normalize density levels
def calculate_density(count_grid, total_points):
    print("Calculating density levels...")
    
    # Normalize the counts by the total number of points to get the density in each block
    density_grid = count_grid / total_points  # This gives density as a fraction of total points
    
    print("Density calculation completed.")
    return density_grid

# # Step 4: Visualize Density Distribution in 3D
# def visualize_density(density_grid):
#     print("Visualizing density distribution...")
    
#     # Generate x, y, z coordinates for each block and corresponding density
#     x, y, z = np.indices(density_grid.shape).reshape(3, -1)
#     densities = density_grid.flatten()
    
#     # Filter out blocks with zero density for visualization
#     mask = densities > 0
#     x, y, z, densities = x[mask], y[mask], z[mask], densities[mask]
    
#     # Normalize densities for color mapping
#     norm = Normalize(vmin=densities.min(), vmax=densities.max())
    
#     # Plotting the density distribution in 3D with darker colors for higher densities
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Use a scatter plot to represent density with color
#     scatter = ax.scatter(x, y, z, c=densities, cmap='Greys', norm=norm, s=40)
    
#     # Add color bar to show density scale
#     cbar = fig.colorbar(scatter, ax=ax, shrink=1.0, aspect=5)
#     cbar.set_label('Silk Density')
    
#     # Label axes and show plot
#     ax.set_xlabel("X axis")
#     ax.set_ylabel("Y axis")
#     ax.set_zlabel("Z axis")
#     plt.title("Silk Density Distribution in 3D Space")
#     plt.show()
def visualize_density(density_grid):
    print("Visualizing density distribution...")
    
    # Generate x, y, z coordinates for each block and corresponding density
    x, y, z = np.indices(density_grid.shape).reshape(3, -1)
    densities = density_grid.flatten()
    
    # Filter out blocks with zero density for visualization
    mask = densities > 0
    x, y, z, densities = x[mask], y[mask], z[mask], densities[mask]
    
    # Normalize densities for color mapping
    norm = Normalize(vmin=densities.min(), vmax=densities.max())
    
    # Plotting the density distribution in 3D with a reversed high-contrast colormap
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use a scatter plot with the reversed colormap 'plasma_r'
    scatter = ax.scatter(x, y, z, c=densities, cmap='plasma_r', norm=norm, s=40)
    
    # Add color bar to show density scale
    cbar = fig.colorbar(scatter, ax=ax, shrink=1.0, aspect=5)
    cbar.set_label('Silk Density')
    
    # Label axes and show plot
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.title("Silk Density Distribution")
    plt.show()


# Main function to execute the steps and visualize density distribution
if __name__ == "__main__":
    # Path to your point cloud file (PCD)
    file_path = "video_processing/point_clouds/@016 255 2024-10-08 05-17-46.pcd"
    
    # Step 1: Load the PCD file
    points = load_pcd(file_path)
    
    # Step 2: Subdivide space and count points in each subregion
    count_grid, total_points = subdivide_and_count(points)
    
    # Step 3: Calculate density levels based on point counts
    density_grid = calculate_density(count_grid, total_points)
    
    # Step 4: Visualize density distribution
    visualize_density(density_grid)
