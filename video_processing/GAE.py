import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# 1) Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

ORIG_NODE_IDX = 0  # <-- index of the 'original' node to drop

# 2) Build adjacency from pointer‐features
def build_full_edge_index(graph):
    N = graph.size(0)
    edges = []
    for u in range(N):
        # skip the original node entirely
        if u == ORIG_NODE_IDX:
            continue
        for c in graph[u,4:10].long().tolist():
            v = c - 1
            # only keep non-original nodes
            if v >= 0 and v != ORIG_NODE_IDX and v < N:
                edges.append([u, v])
    if not edges:
        return torch.zeros((2,0), dtype=torch.long, device=graph.device)
    ei = torch.tensor(edges, dtype=torch.long, device=graph.device).t()
    # make it undirected
    return torch.cat([ei, ei.flip(0)], dim=1)

# 3) Damage function (unchanged)
def damage_graph(graph, full_ei, damage_node_rate=0.2, damage_edge_rate=0.2):
    N = graph.size(0)
    node_mask = torch.rand(N, device=graph.device) < damage_node_rate

    coords = graph[:,1:4].float()
    degs   = torch.zeros(N,1, device=graph.device)
    degs.scatter_add_(0,
        full_ei[0:1].t(),
        torch.ones(full_ei.size(1),1, device=graph.device)
    )
    x = torch.cat([coords, degs], dim=1)  
    x_masked = x.clone()
    x_masked[node_mask] = 0             

    E = full_ei.size(1)
    keep = torch.rand(E, device=graph.device) >= damage_edge_rate
    ei_masked = full_ei[:, keep]

    return x_masked, coords, full_ei, ei_masked, node_mask, keep

# 4) Dataset
class SpiderWebDataset(Dataset):
    def __init__(self, graphs, damage_node_rate=0.2, damage_edge_rate=0.2):
        self.graphs = graphs
        self.node_rate = damage_node_rate
        self.edge_rate = damage_edge_rate

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        G = self.graphs[i].to(device)

        # build and damage
        full_ei = build_full_edge_index(G)
        x_mask, coords_true, full_ei, ei_masked, node_mask, edge_keep = \
            damage_graph(G, full_ei, self.node_rate, self.edge_rate)

        # ---- drop the original node at index 0 ----
        x_mask     = x_mask[1:]
        coords_true= coords_true[1:]
        node_mask  = node_mask[1:]

        # re-index edges: subtract 1 from every index
        full_ei    = full_ei - 1
        ei_masked  = ei_masked - 1

        return Data(
            x_raw       = x_mask,        # [N-1,4]
            coords_true = coords_true,   # [N-1,3]
            edge_index  = ei_masked,     # masked adjacency, reindexed
            full_edge   = full_ei,       # full adjacency, reindexed
            node_mask   = node_mask,     # [N-1]
            edge_mask   = edge_keep      # [E_full]
        )

# 5) Custom collate_fn (unchanged)
def collate_fn(data_list):
    full_edges = [d.full_edge for d in data_list]
    edge_masks = [d.edge_mask for d in data_list]
    for d in data_list:
        del d.full_edge
        del d.edge_mask
    batch = Batch.from_data_list(data_list)
    offsets = batch.ptr[:-1]
    fe_list = []
    for off, fe in zip(offsets, full_edges):
        fe_list.append(fe + off)
    batch.full_edge = torch.cat(fe_list, dim=1)
    batch.edge_mask = torch.cat(edge_masks, dim=0)
    return batch

# 6) GAE Model (unchanged)
class WebGAE(nn.Module):
    def __init__(self, in_dim=4, hid_dim=64, z_dim=32):
        super().__init__()
        self.conv1     = GATConv(in_dim,  hid_dim, heads=4, concat=False)
        self.conv2     = GATConv(hid_dim, z_dim,  heads=4, concat=False)
        self.coord_dec = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, 3)
        )
    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)
    
    def forward(self, data):
        # grab only the embedding tensor
        z = self.encode(data.x_raw, data.edge_index)
        coords_pred = self.coord_dec(z)
        return z, coords_pred

    def decode_edges(self, z, pairs):
        return (z[pairs[0]] * z[pairs[1]]).sum(dim=1)

# 7) Loss (unchanged)
def combined_loss(z, coords_pred, data, α=1.0, β=1.0):
    mse = F.mse_loss(coords_pred[data.node_mask],
                     data.coords_true[data.node_mask])
    fe   = data.full_edge
    keep = data.edge_mask
    du,dv = fe[:, ~keep]
    neg   = negative_sampling(fe, num_nodes=z.size(0),
                              num_neg_samples=du.size(0))
    nu,nv = neg
    pos_logits = (z[du]*z[dv]).sum(dim=1)
    neg_logits = (z[nu]*z[nv]).sum(dim=1)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits),
                        torch.zeros_like(neg_logits)], dim=0)
    edge_loss = F.binary_cross_entropy_with_logits(logits, labels)
    return α*mse + β*edge_loss

# 8) Train / Eval (unchanged)
def train_epoch(model, loader, opt, α, β):
    model.train(); total=0
    for batch in loader:
        opt.zero_grad()
        z,cp = model(batch)
        loss = combined_loss(z,cp,batch,α,β)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        total += loss.item()
    return total/len(loader)

def eval_epoch(model, loader, α, β):
    model.eval(); total=0
    with torch.no_grad():
        for batch in loader:
            z,cp = model(batch)
            total += combined_loss(z,cp,batch,α,β).item()
    return total/len(loader)

# 9) Visualization helpers (unchanged)
def plot_graph(ax, coords, edge_index, title):
    # build a boolean mask of 'real' nodes (not all-zero)
    real = ~torch.all(coords == 0, dim=1)

    # plot only the real nodes
    xs = coords[real,0].cpu().tolist()
    ys = coords[real,1].cpu().tolist()
    zs = coords[real,2].cpu().tolist()
    ax.scatter(xs, ys, zs, s=10)

    # plot only edges between two real nodes
    for u, v in edge_index.t().long().cpu().tolist():
        if not (real[u] and real[v]):
            continue
        ax.plot(
            [coords[u,0], coords[v,0]],
            [coords[u,1], coords[v,1]],
            [coords[u,2], coords[v,2]],
            alpha=0.5
        )

    ax.set_title(title)
    ax.set_axis_off()

def visualize_sample(model, data):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        z, coords_pred = model(data)

    coords_true    = data.coords_true.cpu()
    full_ei        = data.full_edge.cpu()
    damaged_coords = data.x_raw[:, :3].cpu()
    damaged_ei     = data.edge_index.cpu()
    node_mask      = data.node_mask.cpu()

    coords_recon = coords_true.clone()
    coords_recon[node_mask] = coords_pred.cpu()[node_mask]

    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax3 = fig.add_subplot(1,3,3, projection='3d')

    plot_graph(ax1, coords_true,    full_ei,    "Original Web")
    plot_graph(ax2, damaged_coords, damaged_ei, "Damaged Web")
    plot_graph(ax3, coords_recon,   full_ei,    "Reconstructed Web")

    plt.tight_layout()
    plt.show()

# 10) Main
def main():
    graphs = torch.load("video_processing/dataset_webs_medium.pt")[0]
    ds = SpiderWebDataset(graphs, damage_node_rate=0.3,
                                    damage_edge_rate=0.3)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))]
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=32,
                              collate_fn=collate_fn)

    model     = WebGAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    α,β       = 1.0, 1.0

    for epoch in range(1, 31):
        t = train_epoch(model, train_loader, optimizer, α, β)
        v = eval_epoch(model,   val_loader,   α, β)
        print(f"Epoch {epoch:02d} — Train: {t:.4f} | Val: {v:.4f}")

    visualize_sample(model, val_ds[0])

if __name__=="__main__":
    main()
