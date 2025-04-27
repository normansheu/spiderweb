# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch_geometric.data import Data
# # from torch_geometric.nn import GCNConv
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# # # Set device
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print(f"Using device: {device}")

# # # 1. Data Preparation
# # def load_and_preprocess(file_path):
# #     data = torch.load(file_path)
# #     graphs = data[0]  # Shape [num_samples, 64, 16]
# #     properties = data[1]  # Shape [num_samples, 7]
    
# #     # Create valid node masks (nodes that aren't padding)
# #     valid_masks = torch.any(graphs != 0, dim=2)  # Shape [num_samples, 64]
# #     return graphs, valid_masks

# # # 2. Damage Creation
# # def damage_graph(graph, valid_mask, damage_rate=0.2):
# #     """Randomly remove nodes/edges from valid parts of graph"""
# #     valid_indices = torch.where(valid_mask)[0]
# #     num_to_damage = int(damage_rate * len(valid_indices))
# #     damage_indices = np.random.choice(valid_indices.cpu().numpy(), num_to_damage, replace=False)
    
# #     damaged_graph = graph.clone()
# #     damaged_graph[damage_indices, 1:] = 0  # Keep node ID
    
# #     # Remove references to damaged nodes in neighbor lists
# #     for node in range(graph.size(0)):
# #         for i in range(4, 10):  # Neighbor columns
# #             neighbor_idx = int(damaged_graph[node, i].item()) - 1
# #             if neighbor_idx in damage_indices:
# #                 damaged_graph[node, i] = 0
                
# #     return damaged_graph, damage_indices

# # # 3. Model Definition
# # class WebReconstructor(nn.Module):
# #     def __init__(self, node_feature_dim=16, hidden_dim=64):
# #         super().__init__()
# #         self.conv1 = GCNConv(node_feature_dim, hidden_dim)
# #         self.conv2 = GCNConv(hidden_dim, hidden_dim)
# #         self.fc = nn.Linear(hidden_dim, node_feature_dim - 1)  # Predict features except ID
    
# #     def forward(self, data):
# #         x, edge_index = data.x, data.edge_index
# #         x = F.relu(self.conv1(x, edge_index))
# #         x = F.relu(self.conv2(x, edge_index))
# #         return self.fc(x)

# # # 4. Graph Visualization
# # def visualize_web(ax, graph, valid_mask, title, color='blue'):
# #     """3D visualization of a spider web graph"""
# #     coords = graph[:, 1:4].cpu().numpy()
# #     valid_nodes = np.where(valid_mask.cpu().numpy())[0]
    
# #     # Plot nodes
# #     ax.scatter(
# #         coords[valid_nodes, 0],
# #         coords[valid_nodes, 1],
# #         coords[valid_nodes, 2],
# #         c=color, s=50, alpha=0.8
# #     )
    
# #     # Plot edges
# #     for node in valid_nodes:
# #         neighbors = [int(x)-1 for x in graph[node, 4:10] if x.item() != 0]
# #         for nbr in neighbors:
# #             if nbr in valid_nodes:
# #                 ax.plot(
# #                     [coords[node, 0], coords[nbr, 0]],
# #                     [coords[node, 1], coords[nbr, 1]],
# #                     [coords[node, 2], coords[nbr, 2]],
# #                     'k-', linewidth=0.5, alpha=0.3
# #                 )
    
# #     ax.set_title(title)
# #     ax.set_xlabel('X')
# #     ax.set_ylabel('Y')
# #     ax.set_zlabel('Z')

# # # 5. Training Setup
# # def prepare_dataloader(graphs, valid_masks, batch_size=32):
# #     dataset = []
# #     for i in range(len(graphs)):
# #         original = graphs[i]
# #         valid_mask = valid_masks[i]
# #         damaged, _ = damage_graph(original, valid_mask)
        
# #         # Create edge index
# #         edge_index = []
# #         for node in range(original.size(0)):
# #             if not valid_mask[node]:
# #                 continue
# #             for neighbor in damaged[node, 4:10]:
# #                 nbr = int(neighbor.item()) - 1
# #                 if nbr >= 0 and valid_mask[nbr]:
# #                     edge_index.append([node, nbr])
        
# #         if not edge_index:
# #             continue
        
# #         edge_index = torch.tensor(edge_index).t().contiguous()
# #         x = damaged.clone()
# #         y = original[:, 1:]  # Targets (all features except ID)
# #         dataset.append(Data(x=x, edge_index=edge_index, y=y, valid_mask=valid_mask))
# #     return dataset

# # # 6. Training Loop
# # def train_model(dataset, epochs=50):
# #     train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
# #     model = WebReconstructor().to(device)
# #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# #     criterion = nn.MSELoss()
    
# #     for epoch in range(epochs):
# #         model.train()
# #         total_loss = 0
# #         for data in train_data:
# #             data = data.to(device)
# #             optimizer.zero_grad()
# #             pred = model(data)
# #             loss = criterion(pred[data.valid_mask], data.y[data.valid_mask])
# #             loss.backward()
# #             optimizer.step()
# #             total_loss += loss.item()
        
# #         # Validation
# #         model.eval()
# #         test_loss = 0
# #         with torch.no_grad():
# #             for data in test_data:
# #                 data = data.to(device)
# #                 pred = model(data)
# #                 test_loss += criterion(pred[data.valid_mask], data.y[data.valid_mask]).item()
        
# #         print(f'Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_data):.4f} | '
# #               f'Test Loss: {test_loss/len(test_data):.4f}')
# #     return model

# # # Main execution
# # if __name__ == "__main__":
# #     # Load and prepare data
# #     file_path = "video_processing/dataset_webs_medium.pt"
# #     graphs, valid_masks = load_and_preprocess(file_path)
# #     dataset = prepare_dataloader(graphs[:500], valid_masks[:500])  # Use first 100 samples
    
# #     # Train model
# #     model = train_model(dataset, epochs=30)
    
# #     # Visualize results for a test sample
# #     test_sample = dataset[0]
# #     with torch.no_grad():
# #         test_sample = test_sample.to(device)
# #         reconstructed_features = model(test_sample)
        
# #         # Reconstructed graph
# #         reconstructed_graph = test_sample.x.clone()
# #         reconstructed_graph[:, 1:] = reconstructed_features.cpu()

# #         # Original graph
# #         original_graph = test_sample.x.clone()
# #         original_graph[:, 1:] = test_sample.y.cpu()

# #         # Damaged graph (already in test_sample.x)
# #         damaged_graph = test_sample.x.clone().cpu()
        
# #         valid_mask = test_sample.valid_mask.cpu()
        
# #         # Create figure
# #         fig = plt.figure(figsize=(18, 6))
        
# #         # Original web
# #         ax1 = fig.add_subplot(131, projection='3d')
# #         visualize_web(ax1, original_graph, valid_mask, "Original Web", 'green')
        
# #         # Damaged web
# #         ax2 = fig.add_subplot(132, projection='3d')
# #         visualize_web(ax2, damaged_graph, valid_mask, "Damaged Web", 'red')
        
# #         # Reconstructed web
# #         ax3 = fig.add_subplot(133, projection='3d')
# #         visualize_web(ax3, reconstructed_graph, valid_mask, "Reconstructed Web", 'blue')
        
# #         plt.tight_layout()
# #         plt.show()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader              # <-- use PyTorch DataLoader
# from torch_geometric.data import Data, Batch
# from torch_geometric.nn import GATConv
# from torch_geometric.utils import negative_sampling
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# # 1) Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # 2) Build adjacency from pointer‐features
# def build_full_edge_index(graph):
#     N = graph.size(0)
#     edges = []
#     for u in range(N):
#         for c in graph[u,4:10].long().tolist():
#             v = c - 1
#             if 0 <= v < N:
#                 edges.append([u, v])
#     if not edges:
#         return torch.zeros((2,0), dtype=torch.long, device=graph.device)
#     ei = torch.tensor(edges, dtype=torch.long, device=graph.device).t()
#     return torch.cat([ei, ei.flip(0)], dim=1)

# # 3) Damage function
# def damage_graph(graph, full_ei, damage_node_rate=0.2, damage_edge_rate=0.2):
#     N = graph.size(0)
#     node_mask = torch.rand(N, device=graph.device) < damage_node_rate

#     coords = graph[:,1:4].float()
#     degs   = torch.zeros(N,1, device=graph.device)
#     degs.scatter_add_(0,
#         full_ei[0:1].t(),
#         torch.ones(full_ei.size(1),1, device=graph.device)
#     )
#     x = torch.cat([coords, degs], dim=1)       # [N,4]
#     x_masked = x.clone()
#     x_masked[node_mask] = 0                    # zero out masked nodes

#     E = full_ei.size(1)
#     keep = torch.rand(E, device=graph.device) >= damage_edge_rate
#     ei_masked = full_ei[:, keep]

#     return x_masked, coords, full_ei, ei_masked, node_mask, keep

# # 4) Dataset
# class SpiderWebDataset(Dataset):
#     def __init__(self, graphs, damage_node_rate=0.2, damage_edge_rate=0.2):
#         self.graphs = graphs
#         self.node_rate = damage_node_rate
#         self.edge_rate = damage_edge_rate

#     def __len__(self):
#         return len(self.graphs)

#     def __getitem__(self, i):
#         G = self.graphs[i].to(device)
#         full_ei = build_full_edge_index(G)
#         x_mask, coords_true, full_ei, ei_masked, node_mask, edge_keep = \
#             damage_graph(G, full_ei, self.node_rate, self.edge_rate)

#         return Data(
#             x_raw       = x_mask,       # [N,4]
#             coords_true = coords_true,  # [N,3]
#             edge_index  = ei_masked,    # masked adjacency
#             full_edge   = full_ei,      # for loss
#             node_mask   = node_mask,    # [N]
#             edge_mask   = edge_keep     # [E_full]
#         )

# # 5) Custom collate_fn
# def collate_fn(data_list):
#     full_edges = [d.full_edge for d in data_list]
#     edge_masks = [d.edge_mask for d in data_list]
#     for d in data_list:
#         del d.full_edge
#         del d.edge_mask
#     batch = Batch.from_data_list(data_list)
#     offsets = batch.ptr[:-1]
#     fe_list = []
#     for off, fe in zip(offsets, full_edges):
#         fe_list.append(fe + off)
#     batch.full_edge = torch.cat(fe_list, dim=1)
#     batch.edge_mask = torch.cat(edge_masks, dim=0)
#     return batch

# # 6) GAE Model
# class WebGAE(nn.Module):
#     def __init__(self, in_dim=4, hid_dim=64, z_dim=32):
#         super().__init__()
#         self.conv1    = GATConv(in_dim, hid_dim, heads=4, concat=False)
#         self.conv2    = GATConv(hid_dim, z_dim,  heads=4, concat=False)
#         self.coord_dec= nn.Sequential(
#             nn.Linear(z_dim, z_dim),
#             nn.ReLU(),
#             nn.Linear(z_dim, 3)
#         )

#     def encode(self, x, edge_index):
#         h = F.relu(self.conv1(x, edge_index))
#         return self.conv2(h, edge_index)

#     def forward(self, data):
#         z = self.encode(data.x_raw, data.edge_index)
#         coords_pred = self.coord_dec(z)
#         return z, coords_pred

#     def decode_edges(self, z, pairs):
#         return (z[pairs[0]] * z[pairs[1]]).sum(dim=1)

# # 7) Loss
# def combined_loss(z, coords_pred, data, α=1.0, β=1.0):
#     mse = F.mse_loss(coords_pred[data.node_mask],
#                      data.coords_true[data.node_mask])
#     fe   = data.full_edge
#     keep = data.edge_mask
#     du,dv = fe[:, ~keep]
#     neg   = negative_sampling(fe, num_nodes=z.size(0), num_neg_samples=du.size(0))
#     nu,nv = neg
#     pos_logits = (z[du]*z[dv]).sum(dim=1)
#     neg_logits = (z[nu]*z[nv]).sum(dim=1)
#     logits = torch.cat([pos_logits, neg_logits], dim=0)
#     labels = torch.cat([torch.ones_like(pos_logits),
#                         torch.zeros_like(neg_logits)], dim=0)
#     edge_loss = F.binary_cross_entropy_with_logits(logits, labels)
#     return α*mse + β*edge_loss

# # 8) Train / Eval
# def train_epoch(model, loader, opt, α, β):
#     model.train(); total=0
#     for batch in loader:
#         opt.zero_grad()
#         z,cp = model(batch)
#         loss = combined_loss(z,cp,batch,α,β)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(),1.0)
#         opt.step()
#         total += loss.item()
#     return total/len(loader)

# def eval_epoch(model, loader, α, β):
#     model.eval(); total=0
#     with torch.no_grad():
#         for batch in loader:
#             z,cp = model(batch)
#             total += combined_loss(z,cp,batch,α,β).item()
#     return total/len(loader)

# # ---- Visualization helpers ----

# def plot_graph(ax, coords, edge_index, title):
#     xs, ys, zs = coords[:,0].cpu(), coords[:,1].cpu(), coords[:,2].cpu()
#     ax.scatter(xs, ys, zs, s=10)
#     for u,v in edge_index.t().long().cpu().tolist():
#         ax.plot([coords[u,0], coords[v,0]],
#                 [coords[u,1], coords[v,1]],
#                 [coords[u,2], coords[v,2]],
#                 alpha=0.5)
#     ax.set_title(title)
#     ax.set_axis_off()

# def visualize_sample(model, data):
#     model.eval()
#     data = data.to(device)
#     with torch.no_grad():
#         z, coords_pred = model(data)

#     coords_true    = data.coords_true.cpu()
#     full_ei        = data.full_edge.cpu()
#     damaged_coords = data.x_raw[:, :3].cpu()
#     damaged_ei     = data.edge_index.cpu()
#     node_mask      = data.node_mask.cpu()

#     # stitch predictions into the full coords
#     coords_recon = coords_true.clone()
#     coords_recon[node_mask] = coords_pred.cpu()[node_mask]

#     fig = plt.figure(figsize=(18,6))
#     ax1 = fig.add_subplot(1,3,1, projection='3d')
#     ax2 = fig.add_subplot(1,3,2, projection='3d')
#     ax3 = fig.add_subplot(1,3,3, projection='3d')

#     plot_graph(ax1, coords_true,    full_ei,    "Original Web")
#     plot_graph(ax2, damaged_coords, damaged_ei, "Damaged Web")
#     plot_graph(ax3, coords_recon,   full_ei,    "Reconstructed Web")

#     plt.tight_layout()
#     plt.show()

# # 9) Main
# def main():
#     graphs = torch.load("video_processing/dataset_webs_medium.pt")[0]
#     ds = SpiderWebDataset(graphs, damage_node_rate=0.3, damage_edge_rate=0.3)
#     train_ds, val_ds = torch.utils.data.random_split(
#         ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))]
#     )
#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
#                               collate_fn=collate_fn)
#     val_loader   = DataLoader(val_ds,   batch_size=32,
#                               collate_fn=collate_fn)

#     model     = WebGAE().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     α,β       = 1.0, 1.0

#     for epoch in range(1, 31):
#         t = train_epoch(model, train_loader, optimizer, α, β)
#         v = eval_epoch(model,   val_loader,   α, β)
#         print(f"Epoch {epoch:02d} — Train: {t:.4f} | Val: {v:.4f}")

#     # visualize one example from validation
#     visualize_sample(model, val_ds[0])

# if __name__=="__main__":
#     main()

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# 1. Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Data Preparation
def load_and_preprocess(file_path):
    data = torch.load(file_path)
    graphs = data[0]            # shape [num_samples, 64, 16]
    valid_masks = torch.any(graphs != 0, dim=2)  # shape [num_samples, 64]
    return graphs, valid_masks

# 3. Damage Creation
def damage_graph(graph, valid_mask, damage_rate=0.2):
    valid_indices = torch.where(valid_mask)[0]
    num_to_damage = int(damage_rate * len(valid_indices))
    damage_indices = np.random.choice(valid_indices.cpu().numpy(),
                                      num_to_damage, replace=False)

    damaged_graph = graph.clone()
    # zero out all features except ID (col 0) for damaged nodes
    damaged_graph[damage_indices, 1:] = 0

    # remove any neighbor pointers to damaged nodes
    for node in range(graph.size(0)):
        for i in range(4, 10):
            nbr = int(damaged_graph[node, i].item()) - 1
            if nbr in damage_indices:
                damaged_graph[node, i] = 0

    return damaged_graph, damage_indices

# 4. Model Definition
class WebReconstructor(nn.Module):
    def __init__(self, node_feature_dim=16, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc    = nn.Linear(hidden_dim, node_feature_dim - 1)

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.fc(h)

# 5. Visualization Utility
def visualize_web(ax, graph, valid_mask, title, color='blue'):
    coords     = graph[:, 1:4].cpu().numpy()
    valid_nodes= valid_mask.cpu().numpy().nonzero()[0]
    ax.scatter(
        coords[valid_nodes, 0],
        coords[valid_nodes, 1],
        coords[valid_nodes, 2],
        c=color, s=50, alpha=0.8
    )
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
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# 6. Build Dataset
def prepare_dataloader(graphs, valid_masks):
    dataset = []
    for graph, valid_mask in zip(graphs, valid_masks):
        damaged, _ = damage_graph(graph, valid_mask)
        edge_index = []
        # build edge_index from pointers
        for node in range(graph.size(0)):
            if not valid_mask[node]:
                continue
            for nbr_raw in damaged[node, 4:10]:
                nbr = int(nbr_raw.item()) - 1
                if nbr >= 0 and valid_mask[nbr]:
                    edge_index.append([node, nbr])

        if not edge_index:
            continue

        edge_index = torch.tensor(edge_index).t().contiguous()
        x = damaged.clone()
        y = graph[:, 1:].clone()  # ground-truth features (excluding ID)
        dataset.append(Data(
            x=x,
            edge_index=edge_index,
            y=y,
            valid_mask=valid_mask
        ))
    return dataset

# 7. Training Loop
def train_model(dataset, epochs=30):
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    model     = WebReconstructor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for data in train_data:
            optimizer.zero_grad()
            pred = model(data)
            mask = data.valid_mask.to(device)
            loss = criterion(pred[mask], data.y[mask].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in test_data:
                pred = model(data)
                mask = data.valid_mask.to(device)
                val_loss += criterion(pred[mask], data.y[mask].to(device)).item()

        print(f"Epoch {epoch}/{epochs} — Train: {total_loss/len(train_data):.4f} | "
              f"Val: {val_loss/len(test_data):.4f}")
    return model

# 8. Main & Visualization
if __name__ == "__main__":
    graphs, valid_masks = load_and_preprocess("video_processing/dataset_webs_medium.pt")
    ds = prepare_dataloader(graphs[:100], valid_masks[:100])
    model = train_model(ds, epochs=30)

    # pick one test sample
    sample = ds[0].to(device)
    with torch.no_grad():
        rec_feat = model(sample).cpu()

    orig    = sample.x.clone().cpu()
    orig[:,1:] = sample.y.clone().cpu()       # ground truth
    damaged = sample.x.clone().cpu()          # already damaged
    recon   = sample.x.clone().cpu()
    recon[:,1:] = rec_feat                    # stitched back predictions

    # plot side by side
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    visualize_web(ax1, orig,    sample.valid_mask, "Original Web",      'green')
    ax2 = fig.add_subplot(132, projection='3d')
    visualize_web(ax2, damaged, sample.valid_mask, "Damaged Web",       'red')
    ax3 = fig.add_subplot(133, projection='3d')
    visualize_web(ax3, recon,   sample.valid_mask, "Reconstructed Web", 'blue')
    plt.tight_layout()
    plt.show()
