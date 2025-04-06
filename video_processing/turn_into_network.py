import open3d as o3d
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize_3d
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def pcd_to_network(pcd_file, voxel_size=0.5, neighbor_dist=10, min_branch_length=3):
    # Load PCD
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    print(f"Number of points in PCD: {len(points)}")

    # Voxelization
    scaled_points = np.floor(points / voxel_size).astype(int)

    # Shift points to positive indices
    min_coords = np.min(scaled_points, axis=0)
    scaled_points -= min_coords

    # Initialize voxel grid
    grid_shape = np.max(scaled_points, axis=0) + 1
    voxel_grid = np.zeros(grid_shape, dtype=bool)
    voxel_grid[scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2]] = True

    # Skeletonization
    skeleton = skeletonize_3d(voxel_grid)

    # Find nodes (extremities and junctions)
    neighbor_kernel = np.ones((3, 3, 3))
    neighbor_count = convolve(skeleton.astype(int), neighbor_kernel, mode='constant')
    extremities_mask = ((neighbor_count == 2) | (neighbor_count >= 4)) & skeleton
    extremities = np.argwhere(extremities_mask)

    # Build graph
    G = nx.Graph()

    for idx, voxel_coord in enumerate(extremities):
        real_pos = voxel_size * voxel_coord
        G.add_node(idx, pos=tuple(real_pos))

    # Connect nodes based on proximity
    tree = cKDTree(extremities)
    for idx, point in enumerate(extremities):
        distances, indices = tree.query(point, k=5, distance_upper_bound=neighbor_dist)
        for dist, neighbor_idx in zip(distances[1:], indices[1:]):
            if neighbor_idx < len(extremities):
                G.add_edge(idx, neighbor_idx, weight=dist * voxel_size)

    # Remove short isolated branches
    short_edges = [(u, v) for u, v, attr in G.edges(data=True)
                    if attr['weight'] < min_branch_length]
    G.remove_edges_from(short_edges)

    # Keep largest connected component to remove disconnected nodes
    if nx.is_connected(G):
        largest_cc = G
    else:
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    print(f"Number of edges after cleaning: {largest_cc.number_of_edges()}")

    return largest_cc


def visualize_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        ax.plot(x, y, z, color='gray', linewidth=0.5)

    node_positions = np.array(list(pos.values()))
    ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], color='blue', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Network Visualization')
    plt.show()


if __name__ == "__main__":
    pcd_path = "video_processing/point_clouds/sparse3 255 2024-11-30 11-29-33.pcd"
    voxel_size = 0.5
    neighbor_dist = 10
    min_branch_length = 3

    G = pcd_to_network(pcd_file=pcd_path, voxel_size=voxel_size,
                       neighbor_dist=neighbor_dist,
                       min_branch_length=min_branch_length)

    visualize_graph(G)