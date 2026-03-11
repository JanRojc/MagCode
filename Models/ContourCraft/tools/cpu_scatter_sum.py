import numpy as np


def edges_to_csr(edges, num_nodes):
    """
    Build CSR for receiver-ordered edges.

    edges: np.ndarray [E, 2] with (receiver, sender)
    returns: row_ptr [N+1], col_idx [E], perm [E]
    """
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be [E,2] receiver/sender pairs")
    receivers = edges[:, 0]
    perm = np.argsort(receivers, kind="stable")
    receivers_sorted = receivers[perm]
    col_idx = edges[perm, 1].astype(np.int64, copy=False)

    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    counts = np.bincount(receivers_sorted, minlength=num_nodes)
    row_ptr[1:] = np.cumsum(counts)
    return row_ptr, col_idx, perm


def scatter_sum_csr(row_ptr, edge_values):
    """
    Sum edge_values per receiver using CSR.

    row_ptr: [N+1] int64
    edge_values: [E, F] float
    returns: [N, F] float
    """
    if edge_values.ndim != 2:
        raise ValueError("edge_values must be [E,F]")
    num_nodes = row_ptr.shape[0] - 1
    out = np.zeros((num_nodes, edge_values.shape[1]), dtype=edge_values.dtype)
    for i in range(num_nodes):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        if start < end:
            out[i] = edge_values[start:end].sum(axis=0)
    return out


def aggregate_edges(edges, edge_values, num_nodes):
    """
    Convenience: sort edges by receiver, reorder edge_values, then sum.

    edges: [E,2] receiver/sender
    edge_values: [E,F] per-edge features (same order as edges)
    """
    row_ptr, col_idx, perm = edges_to_csr(edges, num_nodes)
    edge_values_sorted = edge_values[perm]
    out = scatter_sum_csr(row_ptr, edge_values_sorted)
    return out, row_ptr, col_idx, perm
