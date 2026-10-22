from __future__ import annotations

import numpy as np


def gaussian_adjacency(distance_m: np.ndarray, sigma_m: float = 1000.0, epsilon: float = 0.1) -> np.ndarray:
    """Distance kernel used by DCRNN-style traffic graphs. Diagonal is cleared."""
    sigma2 = sigma_m ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.exp(-(distance_m ** 2) / sigma2)
    np.fill_diagonal(weights, 0.0)
    weights[distance_m <= 0] = 0.0
    weights[weights < epsilon] = 0.0
    return weights.astype(np.float32)


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float32)
    with_self = adjacency + np.eye(adjacency.shape[0], dtype=np.float32)
    degree = with_self.sum(axis=1)
    scale = np.zeros_like(degree)
    np.divide(1.0, np.sqrt(degree), out=scale, where=degree > 0)
    return (scale[:, None] * with_self * scale[None, :]).astype(np.float32)


def connected_subset(adjacency: np.ndarray, max_nodes: int) -> np.ndarray:
    """Breadth-first subset so the kept sensors stay linked."""
    n = adjacency.shape[0]
    if max_nodes >= n:
        return np.arange(n)
    strength = (adjacency > 0).sum(axis=1)
    seed = int(np.argmax(strength))
    selected = [seed]
    frontier = [seed]
    seen = {seed}
    while frontier and len(selected) < max_nodes:
        current = frontier.pop(0)
        neighbors = np.argsort(-adjacency[current])
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor in seen or adjacency[current, neighbor] <= 0:
                continue
            seen.add(neighbor)
            selected.append(neighbor)
            frontier.append(neighbor)
            if len(selected) >= max_nodes:
                break
    if len(selected) < max_nodes:
        for index in np.argsort(-strength):
            index = int(index)
            if index not in seen:
                selected.append(index)
                seen.add(index)
            if len(selected) >= max_nodes:
                break
    return np.asarray(selected[:max_nodes], dtype=np.int64)


def congestion_level(speed: float, free_flow: float, free_ratio: float = 0.7, moderate_ratio: float = 0.4) -> str:
    if free_flow <= 1e-6:
        return "unknown"
    ratio = speed / free_flow
    if ratio >= free_ratio:
        return "free"
    if ratio >= moderate_ratio:
        return "moderate"
    return "congested"


def volume_to_speed(volume: float, volume_cap: float, free_flow: float) -> float:
    """Greenshields-style bridge from a camera count to a speed-like signal."""
    if volume_cap <= 0 or free_flow <= 0:
        return free_flow
    density_ratio = min(max(volume / volume_cap, 0.0), 1.0)
    return float(free_flow * (1.0 - density_ratio))
