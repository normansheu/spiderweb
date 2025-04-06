import numpy as np
import open3d as o3d
import networkx as nx
import plotly.graph_objects as go
from scipy.spatial import KDTree

def skeleton_to_graph(pcd, radius=0.1):
    """
    Convert a skeletonized point cloud into a graph representation.
    Nodes = extremity points (degree = 1), Edges = connections between them.
    """
    points = np.asarray(pcd.points)
    tree = KDTree(points)

    G = nx.Graph()
    for i, point in enumerate(points):
        G.add_node(i, pos=point)  # Add node with 3D position

        # Find neighbors within the given radius
        neighbors = tree.query_ball_point(point, radius)
        for j in neighbors:
            if i != j:  # Avoid self-loops
                G.add_edge(i, j)

    return G

def extract_extremity_points(G):
    """
    Identify extremity points (degree = 1) from the skeleton graph.
    """
    extremities = [node for node, degree in G.degree() if degree == 1]
    
    if len(extremities) == 0:
        print("Warning: No extremity points found. The skeleton may be too dense.")
    
    return extremities

def compute_average_degree(G, extremities):
    """
    Compute the average degree of the extremity points.
    """
    if len(extremities) == 0:
        return 0  # No extremity points found

    degrees = [G.degree(n) for n in extremities]
    return sum(degrees) / len(degrees)

def visualize_graph(G, extremities):
    """
    Visualize the skeleton graph with extremity points highlighted.
    """
    if len(G.nodes) == 0:
        print("Error: No nodes to visualize. Check graph connectivity.")
        return
    
    node_xyz = np.array([G.nodes[n]['pos'] for n in G.nodes()])
    
    # Handle empty extremity points case
    if len(extremities) > 0:
        extremity_xyz = np.array([G.nodes[n]['pos'] for n in extremities])
    else:
        extremity_xyz = np.empty((0, 3))  # Ensure empty case is handled

    edge_xyz = [(G.nodes[u]['pos'], G.nodes[v]['pos']) for u, v in G.edges()]

    # Create scatter plot for all nodes
    node_trace = go.Scatter3d(
        x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue'),
        name='Skeleton Nodes'
    )

    # Create scatter plot for extremity points (Red) if available
    extremity_trace = go.Scatter3d(
        x=extremity_xyz[:, 0] if len(extremities) > 0 else [],
        y=extremity_xyz[:, 1] if len(extremities) > 0 else [],
        z=extremity_xyz[:, 2] if len(extremities) > 0 else [],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Extremity Points'
    )

    # Create edges as lines
    edge_traces = []
    for edge in edge_xyz:
        edge_x, edge_y, edge_z = zip(*edge)
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=2, color='black'),
            opacity=0.7
        )
        edge_traces.append(edge_trace)

    # Create figure
    fig = go.Figure([node_trace, extremity_trace] + edge_traces)
    fig.update_layout(title="Skeleton Graph Visualization",
                      margin=dict(l=0, r=0, b=0, t=40),
                      scene=dict(aspectmode='data'))
    fig.show()

# Load skeletonized PCD
pcd_file = "video_processing/point_clouds/sparse3 255 2024-11-30 11-29-33.pcd"
skeleton_pcd = o3d.io.read_point_cloud(pcd_file)

# Print number of points in the original PCD
num_pcd_points = np.asarray(skeleton_pcd.points).shape[0]
print(f"Number of points in the PCD: {num_pcd_points}")

# Convert skeleton to graph
G = skeleton_to_graph(skeleton_pcd, radius=0.1)

# Print number of vertices in the graph
num_graph_nodes = len(G.nodes)
print(f"Number of vertices in the graph: {num_graph_nodes}")

# Identify extremity points
extremities = extract_extremity_points(G)

# Print number of extremity points
num_extremities = len(extremities)
print(f"Number of extremity points: {num_extremities}")

# Compute and print the average degree of extremity points
avg_degree = compute_average_degree(G, extremities)
print(f"Average Degree of Extremity Points: {avg_degree:.2f}")

# Visualize the graph interactively
visualize_graph(G, extremities)
