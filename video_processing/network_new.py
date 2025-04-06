import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import networkx as nx
from itertools import combinations

# 1. Load the point cloud
pcd = o3d.io.read_point_cloud("video_processing/point_clouds/sparse3 255 2024-11-30 11-29-33.pcd")
points = np.asarray(pcd.points)
print("Loaded point cloud with", points.shape[0], "points.")

# 2. Remove noise (adjust parameters if needed)
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
filtered_points = np.asarray(cl.points)
print("After noise removal:", filtered_points.shape[0], "points remain.")

# 3. Project to 2D (assuming the spiderweb is nearly planar)
points2D = filtered_points[:, :2]

# 4. Cluster points using DBSCAN to group points along the same line segment
db = DBSCAN(eps=0.05, min_samples=10).fit(points2D)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of clusters found (excluding noise):", n_clusters)

# Group points by cluster label
clusters = {}
for label in set(labels):
    if label == -1:
        continue
    clusters[label] = points2D[labels == label]

# 5. Identify extremity points and compute a direction for each cluster.
#    We store tuples: (ep1, ep2, d_orig) where d_orig is the normalized direction from ep1 to ep2.
extremity_edges = []  # list of tuples (ep1, ep2, d_orig)
extremity_points = [] # list of candidate endpoints

for label, cluster in clusters.items():
    if cluster.shape[0] < 2:
        continue
    # For exactly two points, use them directly.
    if cluster.shape[0] == 2:
        ep1, ep2 = cluster[0], cluster[1]
    else:
        try:
            # Try computing the convex hull.
            hull = ConvexHull(cluster)
            hull_pts = cluster[hull.vertices]
            if hull_pts.shape[0] == 2:
                ep1, ep2 = hull_pts[0], hull_pts[1]
            else:
                max_dist = 0
                ep1, ep2 = None, None
                for i, j in combinations(range(len(hull_pts)), 2):
                    d_tmp = np.linalg.norm(hull_pts[i] - hull_pts[j])
                    if d_tmp > max_dist:
                        max_dist = d_tmp
                        ep1, ep2 = hull_pts[i], hull_pts[j]
        except Exception as e:
            # Fallback to PCA for nearly collinear points.
            cluster_mean = np.mean(cluster, axis=0)
            U, S, Vh = np.linalg.svd(cluster - cluster_mean)
            direction = Vh[0]
            projections = np.dot(cluster - cluster_mean, direction)
            idx_min = np.argmin(projections)
            idx_max = np.argmax(projections)
            ep1, ep2 = cluster[idx_min], cluster[idx_max]
    if ep1 is not None and ep2 is not None:
        # Compute the normalized direction vector for this edge
        d = ep2 - ep1
        norm = np.linalg.norm(d)
        if norm > 0:
            d_orig = d / norm
        else:
            d_orig = np.array([0, 0])
        extremity_edges.append((ep1, ep2, d_orig))
        extremity_points.extend([ep1, ep2])
        
print("Detected extremity edges:", len(extremity_edges))

# 6. Merge endpoints that are close together.
merged_nodes = []
merge_threshold = 0.02  # adjust based on your data scale
for pt in extremity_points:
    found = False
    for i, node in enumerate(merged_nodes):
        if np.linalg.norm(pt - node) < merge_threshold:
            merged_nodes[i] = (merged_nodes[i] + pt) / 2.0  # average positions
            found = True
            break
    if not found:
        merged_nodes.append(pt)
merged_nodes = np.array(merged_nodes)
print("Number of merged nodes (graph nodes):", len(merged_nodes))

# 7. Build the graph: for each extremity edge from a cluster, connect the corresponding merged nodes.
G = nx.Graph()
for i, node in enumerate(merged_nodes):
    G.add_node(i, pos=node)

def find_nearest_node(pt, nodes, threshold=merge_threshold):
    distances = np.linalg.norm(nodes - pt, axis=1)
    idx = np.argmin(distances)
    if distances[idx] < threshold:
        return idx
    return None

# Use a relaxed cosine similarity threshold (0.5) and also allow connection if nodes are very close.
direction_cosine_threshold = 0.5
max_connection_distance = 0.05  # fallback if nodes are very close

for ep1, ep2, d_orig in extremity_edges:
    idx1 = find_nearest_node(ep1, merged_nodes)
    idx2 = find_nearest_node(ep2, merged_nodes)
    if idx1 is not None and idx2 is not None and idx1 != idx2:
        # Compute the connection vector between merged nodes
        conn_vec = merged_nodes[idx2] - merged_nodes[idx1]
        norm_conn = np.linalg.norm(conn_vec)
        if norm_conn == 0:
            continue
        conn_dir = conn_vec / norm_conn
        cosine_sim = np.abs(np.dot(conn_dir, d_orig))
        # Check conditions: either good directional match OR nodes are very close
        if cosine_sim >= direction_cosine_threshold or norm_conn < max_connection_distance:
            G.add_edge(idx1, idx2, weight=norm_conn)
        # Optionally, print debug info:
        # print(f"Edge candidate: cosine_sim={cosine_sim:.2f}, distance={norm_conn:.4f}")

print("Number of graph edges:", len(G.edges()))

# 8. Visualize the network: overlay the original 2D point cloud with graph nodes and edges.
plt.figure(figsize=(10, 10))
plt.scatter(points2D[:, 0], points2D[:, 1], s=1, color='gray', label="Point Cloud")

# Draw merged nodes
for node, data in G.nodes(data=True):
    pos = data['pos']
    plt.scatter(pos[0], pos[1], color='blue', s=50)

# Draw graph edges
for (u, v) in G.edges():
    pos_u = G.nodes[u]['pos']
    pos_v = G.nodes[v]['pos']
    plt.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], color='red', linewidth=2)

plt.title("Spiderweb Network Representation (Relaxed Directional Approach)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
