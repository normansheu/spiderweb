import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def load_data(file_path):
    data = torch.load(file_path)
    input = data[0]          # 节点和边信息 [num_samples, num_nodes, features]
    y_data = data[1]         # 图的全局属性 [num_samples, graph_properties]
    max_neighbors = data[6]  # 每个节点的最大邻居数
    return input, y_data, max_neighbors

def build_graph(sample, max_neighbors):
    node_coords = sample[:, 1:4]  # 假设第1-3列是x, y, z坐标
    neighbor_indices = sample[:, 4:4 + max_neighbors]  # 邻居索引

    G = nx.Graph()
    
    # 添加节点
    for node_idx in range(node_coords.shape[0]):
        x, y, z = node_coords[node_idx]
        G.add_node(node_idx, pos=(x, y, z))
    
    # 添加边
    for node_idx in range(neighbor_indices.shape[0]):
        for neighbor in neighbor_indices[node_idx]:
            neighbor = int(neighbor.item())
            if neighbor != 0:  # 0可能是填充值
                G.add_edge(node_idx, neighbor - 1)  # 假设邻居索引是1-based
    return G

def plot_and_save_graph(G, output_dir, sample_idx, y_data=None):
    pos = nx.get_node_attributes(G, 'pos')
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取坐标
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    
    # 绘制节点
    ax.scatter(xs, ys, zs, c='red', s=50, alpha=0.8, label='Nodes')
    
    # 绘制边
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        ax.plot([x0, x1], [y0, y1], [z0, z1], 'k-', linewidth=0.5, alpha=0.5)
    
    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    title = f"Spider Web (Sample {sample_idx})"
    if y_data is not None:
        try:
            props = y_data[sample_idx].round(2)
            title += f"\nProperties: {props}"
        except:
            pass
    ax.set_title(title)
    
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"web_{sample_idx}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Saved: {output_path}")

def analyze_graph(G, sample_idx):
    print(f"\n=== Sample {sample_idx} Properties ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    is_connected = nx.is_connected(G)
    print(f"Is connected: {is_connected}")
    
    if is_connected:
        print(f"Diameter: {nx.diameter(G)}")
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
    else:
        print("Graph is not connected - analyzing connected components")
        components = list(nx.connected_components(G))
        print(f"Number of connected components: {len(components)}")
        print(f"Size of largest component: {max(len(c) for c in components)}")

def main():
    # 配置
    file_path = "video_processing/dataset_webs_medium.pt"  # 替换为你的文件路径
    output_dir = "video_processing/real_graphs"  # 输出目录
    
    # 加载数据
    input, y_data, max_neighbors = load_data(file_path)
    num_samples = input.shape[0]
    
    # 处理每个样本

    for sample_idx in range(num_samples):
        print(f"\nProcessing sample {sample_idx + 1}/{num_samples}...")
        sample = input[sample_idx]
        
        # 构建图
        G = build_graph(sample, max_neighbors)
        
        # 分析属性
        analyze_graph(G, sample_idx)
        
        # 可视化并保存
        plot_and_save_graph(G, output_dir, sample_idx, y_data)

    print(f"\nAll visualizations saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()