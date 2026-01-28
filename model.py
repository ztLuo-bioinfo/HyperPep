"""HyperPep model definition.

Components:
- HyperGLayer: message passing between hypernodes (e.g., groups / functional groups)
  and hyperedges (residues).
- Hyperpep: 2-layer hypergraph propagation + optional residue-chain GNN refinement
  (ARMAConv) + an MLP classifier head.

Implementation notes:
- Uses torch_scatter.scatter_mean if available; otherwise falls back to a deterministic
  pure-PyTorch implementation.

"""

# model.py —— 仅替换 Hyperpep 类，其余 import 保持
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, ARMAConv

try:
    from torch_scatter import scatter_mean
except Exception:
    # If torch_scatter is not installed, we provide a small, deterministic
    # scatter_mean replacement (slower but dependency-free).

    def scatter_mean(src, index, dim=0, dim_size=None):
        if src.numel() == 0:
            return src.new_zeros((dim_size,) + src.shape[1:]) if dim_size is not None else src
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out = src.new_zeros((dim_size,) + src.shape[1:])
        cnt = src.new_zeros((dim_size,), dtype=torch.long)
        out.index_add_(0, index, src)
        cnt.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
        cnt = cnt.clamp_min(1).view(-1, *([1]*(src.dim()-1)))
        return out / cnt

class HyperGLayer(nn.Module):
    """One hypergraph message passing layer.

    x_h:         [H, d_h] hypernode features
    h2_edge_*:   incidence and attributes describing residue hyperedges
    Returns:     updated hypernode features [H, out_channels]
    """
    def __init__(self, in_channels_h, edge_in_channels, out_channels):
        super().__init__()
        self.proj_in = nn.Linear(in_channels_h, out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels_h + edge_in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x_h, h2_edge_index, h2_edge_attr):
        # Message flow:
        #   hypernode -> residue (aggregate) -> residue MLP -> hypernode (aggregate)
        # with a residual connection and LayerNorm.

        if h2_edge_index.numel() == 0:
            # No incidence edges: fall back to a simple projection + activation.
            # This is a safety net for degenerate samples.

            z = self.proj_in(x_h)
            return self.norm(F.relu(z))
        # h2_edge_index stores pairs (hypernode_id, residue_id).
        fg_id = h2_edge_index[0]
        res_id = h2_edge_index[1]
        H = x_h.size(0); E = h2_edge_attr.size(0)

        # 超边上聚合其连接的超节点 (mean)
        # Aggregate hypernodes connected to each residue (mean pooling).
        edge_node_mean = scatter_mean(x_h[fg_id], res_id, dim=0, dim_size=E)
        # 边表征
        e_in = torch.cat([edge_node_mean, h2_edge_attr], dim=-1)   # [E, in+edge_in]
        edge_msg = self.edge_mlp(e_in)                              # [E, out]
        # 回传到节点
        # Send residue messages back to hypernodes (mean pooling).
        node_msg = scatter_mean(edge_msg[res_id], fg_id, dim=0, dim_size=H)
        z = self.proj_in(x_h) + self.node_mlp(node_msg)             # 残差
        return self.norm(F.relu(z))

class Hyperpep(nn.Module):
    """HyperPep classifier.

    Forward inputs (from PyG batch):
      x_h            : hypernode features [H, d]
      h2_edge_index  : incidence (hypernode_id, residue_id) [2, M]
      h2_edge_attr   : residue (hyperedge) attributes [E, d_e]
      idx_batch      : batch vector for hypernodes (PyG 'batch')
      num_hyper2edges: number of residues per sample in the batch [B]
    """
    def __init__(self, in_channels_h, edge_in_channels=20, hidden=128, drop=0.2):
        super().__init__()
        # 两层超图信息传递（超节点<->残基）
        self.hlayer1 = HyperGLayer(in_channels_h, edge_in_channels, hidden)
        self.hlayer2 = HyperGLayer(hidden,         edge_in_channels, hidden)

        # 残基级一层 GNN（只在 0-1-2-... 链上传播）
        self.res_gnn = ARMAConv(hidden, hidden, num_stacks=1, num_layers=2, shared_weights=False)
        self.res_norm = nn.LayerNorm(hidden)

        # 头部
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_h, h2_edge_index, h2_edge_attr, idx_batch, num_hyper2edges=None):
        # 1) 超图两层（在 FG <-> 残基之间传消息）
        h = self.hlayer1(x_h, h2_edge_index, h2_edge_attr)
        h = self.hlayer2(h,  h2_edge_index, h2_edge_attr)          # [H, hidden]

        # 2) 把 FG 聚合到“残基级表示” r
        if h2_edge_index.numel() == 0:
            # No incidence edges: fall back to a simple projection + activation.
            # This is a safety net for degenerate samples.

            # 极端兜底：没有隶属关系，就直接图级池化
            g = global_add_pool(h, idx_batch)
            return self.mlp(g).view(-1)

        fg_id = h2_edge_index[0]; res_id = h2_edge_index[1]
        E_total = h2_edge_attr.size(0)
        r = scatter_mean(h[fg_id], res_id, dim=0, dim_size=E_total)  # [E_total, hidden]

        # 3) 在“残基链”上传播：每条肽 0-1-2-...-E_i-1
        #    这里按 batch 的 num_hyper2edges 还原每条肽的残基数并构建链边
        if num_hyper2edges is None:
            raise RuntimeError(
                "num_hyper2edges must be provided (use batch.num_hyper2edges). "
                "This avoids nondeterministic CUDA bincount and ensures correct residue chain construction."
            )
        if not torch.is_tensor(num_hyper2edges):
            num_hyper2edges = torch.tensor(num_hyper2edges, device=r.device)
        # num_hyper2edges: [B]
        offsets = torch.cumsum(torch.cat([r.new_zeros(1, dtype=torch.long), num_hyper2edges.long()]), dim=0)
        # 构建拼接后的残基链边
        ei_src, ei_dst = [], []
        # Build a bidirectional chain graph for residues within each sample.
        # This encodes local sequential dependencies after hypergraph aggregation.
        for b in range(num_hyper2edges.numel()):
            E = int(num_hyper2edges[b].item())
            if E <= 1:
                continue
            off = int(offsets[b].item())
            idx = torch.arange(off, off+E-1, device=r.device, dtype=torch.long)
            ei_src.append(idx)
            ei_dst.append(idx+1)
        if len(ei_src) > 0:
            res_edge_index = torch.stack([torch.cat(ei_src + ei_dst), torch.cat(ei_dst + ei_src)], dim=0)  # 无向
            r = self.res_gnn(r, res_edge_index)
            r = self.res_norm(F.relu(r))

        # 4) 残基级图读出 -> 肽级
        #    残基的 batch 向量：按 num_hyper2edges 重复展开即可
        # Map each residue back to its sample id (needed for graph-level pooling).
        res_batch = torch.repeat_interleave(
            torch.arange(num_hyper2edges.numel(), device=r.device),
            repeats=num_hyper2edges.long()
        )
        g = global_add_pool(r, res_batch)  # [B, hidden]

        # 5) 分类头
        z = self.mlp(g)
        return z.view(-1)
