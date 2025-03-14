import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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

# Step 4: Visualize Density Distribution in 3D with labeled axes
def visualize_density_with_labels(density_grid, output_image_path):
    print("Visualizing density distribution with labeled axes...")
    
    # Generate x, y, z coordinates for each block and corresponding density
    x, y, z = np.indices(density_grid.shape).reshape(3, -1)
    densities = density_grid.flatten()
    
    # Filter out blocks with zero density for visualization
    mask = densities > 0
    x, y, z, densities = x[mask], y[mask], z[mask], densities[mask]
    
    # Normalize densities for color mapping with fixed range
    norm = Normalize(vmin=0, vmax=0.050)  # Set the range to [0, 0.050]
    
    # Plotting the density distribution in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=densities, cmap='plasma_r', norm=norm, s=40)
    
    # Add axis labels without numbers
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Save the image
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"3D density plot saved to {output_image_path}")

# Step 5: Save the color bar separately
def save_color_bar(density_grid, color_bar_path):
    print("Saving color bar...")
    
    # Normalize densities for color mapping with fixed range
    norm = Normalize(vmin=0, vmax=0.050)  # Set the range to [0, 0.050]
    
    # Create a figure for the color bar
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    # Create a color bar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma_r'), cax=ax, orientation='horizontal')
    cbar.set_label('Silk Density (fraction of total points)')

    # Save the color bar
    plt.savefig(color_bar_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Color bar saved to {color_bar_path}")


# Main function to execute the steps and visualize density distribution
if __name__ == "__main__":
    # Path to your point cloud file (PCD)
    file_path = "video_processing/point_clouds/@012 255 2024-10-04 05-06-11.pcd"
    
    # Step 1: Load the PCD file
    points = load_pcd(file_path)
    
    # Step 2: Subdivide space and count points in each subregion
    count_grid, total_points = subdivide_and_count(points)
    
    # Step 3: Calculate density levels based on point counts
    density_grid = calculate_density(count_grid, total_points)
    
    # Step 4: Visualize density distribution with labeled axes and save the plot
    visualize_density_with_labels(density_grid, "@12_density.png")
    
    # Step 5: Save the color bar
    save_color_bar(density_grid, "color_bar.png")
