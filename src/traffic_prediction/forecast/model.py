from __future__ import annotations

import torch
from torch import nn


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size // 2, 0)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=(kernel_size, 1), padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.conv(x).chunk(2, dim=1)
        return torch.tanh(left) * torch.sigmoid(right)


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: B, C, T, N
        transformed = torch.einsum("nm,bctm->bctn", adjacency, x)
        transformed = transformed.permute(0, 2, 3, 1)
        return self.linear(transformed).permute(0, 3, 1, 2)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, hidden: int, dropout: float):
        super().__init__()
        self.temporal_in = TemporalConv(in_channels, hidden)
        self.spatial = GraphConv(hidden, hidden)
        self.temporal_out = TemporalConv(hidden, hidden)
        self.residual = nn.Conv2d(in_channels, hidden, kernel_size=1) if in_channels != hidden else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm2d(hidden)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        hidden = self.temporal_in(x)
        hidden = torch.relu(self.spatial(hidden, adjacency))
        hidden = self.temporal_out(hidden)
        return self.dropout(self.norm(hidden + residual))


class ExternalFusion(nn.Module):
    def __init__(self, hidden: int, external_dim: int, mode: str):
        super().__init__()
        if mode not in {"none", "concat", "gate"}:
            raise ValueError(mode)
        self.mode = mode
        self.embed = nn.Linear(external_dim, hidden)
        self.concat = nn.Linear(hidden * 2, hidden)
        self.gate = nn.Linear(hidden * 2, hidden)

    def forward(self, hidden: torch.Tensor, external: torch.Tensor) -> torch.Tensor:
        # hidden: B, N, H ; external: B, E
        if self.mode == "none":
            return hidden
        embedded = self.embed(external).unsqueeze(1).expand(-1, hidden.size(1), -1)
        joined = torch.cat([hidden, embedded], dim=-1)
        if self.mode == "concat":
            return self.concat(joined)
        gate = torch.sigmoid(self.gate(joined))
        return hidden + gate * embedded


class STGCN(nn.Module):
    def __init__(
        self,
        hidden: int,
        horizon_steps: int,
        external_dim: int,
        fusion: str = "gate",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block1 = STGCNBlock(1, hidden, dropout)
        self.block2 = STGCNBlock(hidden, hidden, dropout)
        self.fusion = ExternalFusion(hidden, external_dim, fusion)
        self.head = nn.Linear(hidden, horizon_steps)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, external: torch.Tensor) -> torch.Tensor:
        # x: B, T, N
        hidden = x.unsqueeze(1)
        hidden = self.block1(hidden, adjacency)
        hidden = self.block2(hidden, adjacency)
        pooled = hidden.mean(dim=2).permute(0, 2, 1)
        fused = self.fusion(pooled, external)
        return self.head(fused).permute(0, 2, 1)


class GRUForecaster(nn.Module):
    """Per-node GRU. It sees time and external features, not the road graph."""

    def __init__(self, hidden: int, horizon_steps: int, external_dim: int, fusion: str = "gate", dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fusion = ExternalFusion(hidden, external_dim, fusion)
        self.head = nn.Linear(hidden, horizon_steps)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, external: torch.Tensor) -> torch.Tensor:
        del adjacency
        batch, steps, nodes = x.shape
        series = x.permute(0, 2, 1).reshape(batch * nodes, steps, 1)
        _, last = self.gru(series)
        hidden = self.dropout(last[-1]).reshape(batch, nodes, -1)
        fused = self.fusion(hidden, external)
        return self.head(fused).permute(0, 2, 1)
