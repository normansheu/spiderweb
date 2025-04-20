import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Data Preparation
def load_and_preprocess(file_path):
    data = torch.load(file_path)
    graphs = data[0]  # Shape [num_samples, 64, 16]
    properties = data[1]  # Shape [num_samples, 7]
    
    # Create valid node masks (nodes that aren't padding)
    valid_masks = torch.any(graphs != 0, dim=2)  # Shape [num_samples, 64]
    return graphs, valid_masks

# 2. Damage Creation
def damage_graph(graph, valid_mask, damage_rate=0.2):
    """Randomly remove nodes/edges from valid parts of graph"""
    valid_indices = torch.where(valid_mask)[0]
    num_to_damage = int(damage_rate * len(valid_indices))
    damage_indices = np.random.choice(valid_indices.cpu().numpy(), num_to_damage, replace=False)
    
    damaged_graph = graph.clone()
    damaged_graph[damage_indices, 1:] = 0  # Keep node ID
    
    # Remove references to damaged nodes in neighbor lists
    for node in range(graph.size(0)):
        for i in range(4, 10):  # Neighbor columns
            neighbor_idx = int(damaged_graph[node, i].item()) - 1
            if neighbor_idx in damage_indices:
                damaged_graph[node, i] = 0
                
    return damaged_graph, damage_indices

# 3. Model Definition
class WebReconstructor(nn.Module):
    def __init__(self, node_feature_dim=16, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, node_feature_dim - 1)  # Predict features except ID
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x)

# 4. Graph Visualization
def visualize_web(ax, graph, valid_mask, title, color='blue'):
    """3D visualization of a spider web graph"""
    coords = graph[:, 1:4].cpu().numpy()
    valid_nodes = np.where(valid_mask.cpu().numpy())[0]
    
    # Plot nodes
    ax.scatter(
        coords[valid_nodes, 0],
        coords[valid_nodes, 1],
        coords[valid_nodes, 2],
        c=color, s=50, alpha=0.8
    )
    
    # Plot edges
    for node in valid_nodes:
        neighbors = [int(x)-1 for x in graph[node, 4:10] if x.item() != 0]
        for nbr in neighbors:
            if nbr in valid_nodes:
                ax.plot(
                    [coords[node, 0], coords[nbr, 0]],
                    [coords[node, 1], coords[nbr, 1]],
                    [coords[node, 2], coords[nbr, 2]],
                    'k-', linewidth=0.5, alpha=0.3
                )
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# 5. Training Setup
def prepare_dataloader(graphs, valid_masks, batch_size=32):
    dataset = []
    for i in range(len(graphs)):
        original = graphs[i]
        valid_mask = valid_masks[i]
        damaged, _ = damage_graph(original, valid_mask)
        
        # Create edge index
        edge_index = []
        for node in range(original.size(0)):
            if not valid_mask[node]:
                continue
            for neighbor in damaged[node, 4:10]:
                nbr = int(neighbor.item()) - 1
                if nbr >= 0 and valid_mask[nbr]:
                    edge_index.append([node, nbr])
        
        if not edge_index:
            continue
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        x = damaged.clone()
        y = original[:, 1:]  # Targets (all features except ID)
        dataset.append(Data(x=x, edge_index=edge_index, y=y, valid_mask=valid_mask))
    return dataset

# 6. Training Loop
def train_model(dataset, epochs=50):
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    model = WebReconstructor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred[data.valid_mask], data.y[data.valid_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_data:
                data = data.to(device)
                pred = model(data)
                test_loss += criterion(pred[data.valid_mask], data.y[data.valid_mask]).item()
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_data):.4f} | '
              f'Test Loss: {test_loss/len(test_data):.4f}')
    return model

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    file_path = "video_processing/dataset_webs_medium.pt"
    graphs, valid_masks = load_and_preprocess(file_path)
    dataset = prepare_dataloader(graphs[:500], valid_masks[:500])  # Use first 100 samples
    
    # Train model
    model = train_model(dataset, epochs=30)
    
    # Visualize results for a test sample
    test_sample = dataset[0]
    with torch.no_grad():
        test_sample = test_sample.to(device)
        reconstructed_features = model(test_sample)
        
        # Reconstructed graph
        reconstructed_graph = test_sample.x.clone()
        reconstructed_graph[:, 1:] = reconstructed_features.cpu()

        # Original graph
        original_graph = test_sample.x.clone()
        original_graph[:, 1:] = test_sample.y.cpu()

        # Damaged graph (already in test_sample.x)
        damaged_graph = test_sample.x.clone().cpu()
        
        valid_mask = test_sample.valid_mask.cpu()
        
        # Create figure
        fig = plt.figure(figsize=(18, 6))
        
        # Original web
        ax1 = fig.add_subplot(131, projection='3d')
        visualize_web(ax1, original_graph, valid_mask, "Original Web", 'green')
        
        # Damaged web
        ax2 = fig.add_subplot(132, projection='3d')
        visualize_web(ax2, damaged_graph, valid_mask, "Damaged Web", 'red')
        
        # Reconstructed web
        ax3 = fig.add_subplot(133, projection='3d')
        visualize_web(ax3, reconstructed_graph, valid_mask, "Reconstructed Web", 'blue')
        
        plt.tight_layout()
        plt.show()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GATConv
# from torch_geometric.utils import negative_sampling
# import numpy as np
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 0. Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # 1. Load & preprocess
# def load_and_preprocess(path):
#     data   = torch.load(path)
#     graphs = data[0]                           # [N, 64, 16]
#     masks  = torch.any(graphs != 0, dim=2)     # [N, 64]
#     return graphs, masks

# # 2. Damage a graph and append damage signals
# def damage_graph(graph, valid_mask, damage_rate=0.2):
#     valid_idx      = torch.where(valid_mask)[0].cpu().numpy()
#     k              = int(damage_rate * len(valid_idx))
#     damaged_nodes  = np.random.choice(valid_idx, k, replace=False)
#     damaged        = graph.clone()

#     # zero out features for damaged nodes (except ID)
#     damaged[damaged_nodes, 1:] = 0

#     # prune neighbor pointers in damaged nodes
#     for u in range(graph.size(0)):
#         for c in range(4, 10):
#             v = int(damaged[u, c].item()) - 1
#             if v in damaged_nodes:
#                 damaged[u, c] = 0

#     dmg_mask = torch.zeros_like(valid_mask)
#     dmg_mask[damaged_nodes] = True

#     # append node-wise damage flag and global damage_rate
#     flag = dmg_mask.unsqueeze(-1).float()                      # [64,1]
#     rate = torch.full((graph.size(0),1), damage_rate)         # [64,1]
#     x_aug = torch.cat([damaged.float(), flag, rate], dim=1)    # [64, 16+2]

#     return x_aug, dmg_mask

# # 3. PyG Dataset\class SpiderWebDataset(Dataset):
#     def __init__(self, graphs, masks, damage_rate=0.2):
#         self.graphs = graphs
#         self.masks  = masks
#         self.rate   = damage_rate

#     def __len__(self):
#         return self.graphs.size(0)

#     def __getitem__(self, i):
#         G    = self.graphs[i]   # [64,16]
#         M    = self.masks[i]    # [64]
#         x, dmask = damage_graph(G, M, self.rate)

#         # damaged-edge index (for model input)
#         E_d = []
#         for u in range(64):
#             if not M[u]: continue
#             for c in x[u, 4:10]:
#                 v = int(c.item()) - 1
#                 if v >= 0 and M[v]:
#                     E_d.append([u, v])
#         if not E_d:
#             for u in torch.where(M)[0]: E_d.append([int(u), int(u)])
#         edge_index_d = torch.tensor(E_d, dtype=torch.long).t().contiguous()

#         # ground-truth edges
#         E_gt = []
#         for u in range(64):
#             if not M[u]: continue
#             for c in G[u, 4:10]:
#                 v = int(c.item()) - 1
#                 if v >= 0 and M[v]: E_gt.append([u, v])
#         edge_index_gt = torch.tensor(E_gt, dtype=torch.long).t().contiguous()

#         y_coords = G[:, 1:4].float()  # true (x,y,z)

#         return Data(
#             x=x,
#             edge_index=edge_index_d,
#             edge_index_gt=edge_index_gt,
#             y_coords=y_coords,
#             valid_mask=M,
#             damage_mask=dmask
#         )
# # 3. PyG Dataset
# class SpiderWebDataset(Dataset):
#     def __init__(self, graphs, masks, damage_rate=0.2):
#         self.graphs = graphs
#         self.masks  = masks
#         self.rate   = damage_rate

#     def __len__(self):
#         return self.graphs.size(0)

#     def __getitem__(self, i):
#         G    = self.graphs[i]   # [64,16]
#         M    = self.masks[i]    # [64]
#         x, dmask = damage_graph(G, M, self.rate)

#         # damaged edges
#         E_d = []
#         for u in range(64):
#             if not M[u]: continue
#             for c in x[u, 4:10]:
#                 v = int(c.item()) - 1
#                 if v >= 0 and M[v]:
#                     E_d.append([u, v])
#         if not E_d:
#             for u in torch.where(M)[0]:
#                 E_d.append([int(u), int(u)])
#         edge_index_d = torch.tensor(E_d, dtype=torch.long).t().contiguous()

#         # ground‑truth edges
#         E_gt = []
#         for u in range(64):
#             if not M[u]: continue
#             for c in G[u, 4:10]:
#                 v = int(c.item()) - 1
#                 if v >= 0 and M[v]:
#                     E_gt.append([u, v])
#         edge_index_gt = torch.tensor(E_gt, dtype=torch.long).t().contiguous()

#         y_coords = G[:, 1:4].float()  # true (x,y,z)

#         return Data(
#             x=x,
#             edge_index=edge_index_d,
#             edge_index_gt=edge_index_gt,
#             y_coords=y_coords,
#             valid_mask=M,
#             damage_mask=dmask
#         )

# # 4. Denoising GAE Model
# class WebGAE(nn.Module):
#     def __init__(self, in_dim=18, hid_dim=64, z_dim=32):
#         super().__init__()
#         self.conv1 = GATConv(in_dim, hid_dim, heads=4, concat=False)
#         self.conv2 = GATConv(hid_dim, z_dim,  heads=4, concat=False)
#         self.coord_mlp = nn.Sequential(
#             nn.Linear(z_dim, z_dim),
#             nn.ReLU(),
#             nn.Linear(z_dim, 3)
#         )

#     def encode(self, x, edge_index):
#         h = F.relu(self.conv1(x, edge_index))
#         return self.conv2(h, edge_index)

#     def decode_edges(self, z, edge_index):
#         return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

#     def forward(self, data):
#         z = self.encode(data.x, data.edge_index)
#         coords_pred = self.coord_mlp(z)
#         # preserve clean coords, overwrite only damaged ones
#         coords_clean = data.x[:,1:4]              # original coords in x
#         coords_final = torch.where(
#             data.damage_mask.unsqueeze(-1),
#             coords_pred,
#             coords_clean
#         )
#         return z, coords_pred, coords_final

# # 5. Loss: only on dropped coords & dropped edges
# def loss_fn(coords_pred, coords_true, z,
#             edge_index_gt, num_nodes,
#             valid_mask, damage_mask,
#             alpha=1.0, beta=1.0):
#     # coord MSE on damaged nodes only
#     coord_loss = F.mse_loss(
#         coords_pred[damage_mask],
#         coords_true[damage_mask]
#     )

#     # edges: only target dropped edges
#     mask_u = damage_mask[edge_index_gt[0]]
#     mask_v = damage_mask[edge_index_gt[1]]
#     drop_mask = (mask_u | mask_v)
#     pos_idx = edge_index_gt[:, drop_mask]

#     # sample negatives among all missing pairs
#     neg_idx = negative_sampling(
#         edge_index_gt,
#         num_nodes=num_nodes,
#         num_neg_samples=pos_idx.size(1)
#     )

#     pos_logits = (z[pos_idx[0]] * z[pos_idx[1]]).sum(dim=1)
#     neg_logits = (z[neg_idx[0]] * z[neg_idx[1]]).sum(dim=1)

#     labels = torch.cat([
#         torch.ones_like(pos_logits),
#         torch.zeros_like(neg_logits)
#     ], dim=0)
#     logits = torch.cat([pos_logits, neg_logits], dim=0)
#     edge_loss = F.binary_cross_entropy_with_logits(logits, labels)

#     return alpha*coord_loss + beta*edge_loss

# # 6. Training / eval loops
# def train_epoch(model, loader, optimizer, α, β):
#     model.train()
#     total = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         z, coords_pred, _ = model(data)
#         loss = loss_fn(
#             coords_pred, data.y_coords, z,
#             data.edge_index_gt, data.num_nodes,
#             data.valid_mask, data.damage_mask,
#             α, β
#         )
#         loss.backward()
#         optimizer.step()
#         total += loss.item()
#     return total / len(loader)

# def eval_epoch(model, loader, α, β):
#     model.eval()
#     total = 0
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             z, coords_pred, _ = model(data)
#             total += loss_fn(
#                 coords_pred, data.y_coords, z,
#                 data.edge_index_gt, data.num_nodes,
#                 data.valid_mask, data.damage_mask,
#                 α, β
#             ).item()
#     return total / len(loader)

# # 7. Visualization helper remains unchanged
# def visualize_graph(ax, coords, edge_index, valid_mask, title, color):
#     valid = np.where(valid_mask)[0]
#     ax.scatter(
#         coords[valid, 0], coords[valid, 1], coords[valid, 2],
#         c=color, s=50, alpha=0.8
#     )
#     for u, v in edge_index.T:
#         u, v = int(u), int(v)
#         if valid_mask[u] and valid_mask[v]:
#             ax.plot([
#                 coords[u,0], coords[v,0]
#             ], [coords[u,1], coords[v,1]], [coords[u,2], coords[v,2]],
#             'k-', linewidth=0.5, alpha=0.3)
#     ax.set_title(title)
#     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# # 8. Main
# def main():
#     graphs, masks = load_and_preprocess("video_processing/dataset_webs_medium.pt")
#     full_ds       = SpiderWebDataset(graphs, masks, damage_rate=0.2)

#     all_data   = list(full_ds)
#     train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#     test_loader  = DataLoader(test_data,  batch_size=32)

#     model = WebGAE().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     α, β = 1.0, 1.0

#     for epoch in range(1, 31):
#         t_loss = train_epoch(model, train_loader, optimizer, α, β)
#         v_loss = eval_epoch(model, test_loader,  α, β)
#         print(f"Epoch {epoch:02d}/30 — Train: {t_loss:.4f} | Val: {v_loss:.4f}")

#     # pick one test sample for visualization
#     sample = test_data[0].to(device)
#     model.eval()
#     with torch.no_grad():
#         z, coords_pred, coords_final = model(sample)
#         logits = model.decode_edges(z, sample.edge_index_gt)
#         probs  = torch.sigmoid(logits)
#         pred_mask = probs > 0.5
#         pred_edges = sample.edge_index_gt[:, pred_mask]

#     orig_coords   = sample.y_coords.cpu().numpy()
#     damaged_coords= sample.x[:,1:4].cpu().numpy()
#     recon_coords  = coords_final.cpu().numpy()
#     valid         = sample.valid_mask.cpu().numpy()
#     damaged_edges = sample.edge_index.cpu().numpy()
#     gt_edges      = sample.edge_index_gt.cpu().numpy()

#     fig = plt.figure(figsize=(18,6))
#     ax1 = fig.add_subplot(131, projection='3d')
#     visualize_graph(ax1, orig_coords, gt_edges,      valid, "Original Web",      'green')
#     ax2 = fig.add_subplot(132, projection='3d')
#     visualize_graph(ax2, damaged_coords, damaged_edges, valid, "Damaged Web",   'red')
#     ax3 = fig.add_subplot(133, projection='3d')
#     visualize_graph(ax3, recon_coords, pred_edges,    valid, "Reconstructed Web", 'blue')
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()
