import torch

def _broadcast_index(src, index, dim):
    """
    Make `index` the same shape as `src` for scatter-style ops.
    Behaves similarly to torch_scatter's broadcasting.
    """
    if index.dim() == 0:
        # scalar -> expand fully
        index = index.view(1)
    # Move dim into range if negative
    if dim < 0:
        dim = src.dim() + dim

    # Unsqueeze until dims match
    while index.dim() < src.dim():
        index = index.unsqueeze(-1)

    # Now expand along non-dim dimensions if needed
    expand_size = list(src.size())
    expand_size[dim] = index.size(dim)
    index = index.expand(*expand_size)
    return index


def _fill_value(dtype, kind: str):
    """
    kind: 'max' -> very small; 'min' -> very large for given dtype.
    Handles float, int, and bool.
    """
    if torch.is_floating_point(torch.empty((), dtype=dtype)):
        if kind == "max":   # for amax we need very small start
            return float("-inf")
        else:               # for amin we need very large start
            return float("inf")

    if dtype == torch.bool:
        # For max: start with False; for min: start with True
        return False if kind == "max" else True

    # Integer types
    info = torch.iinfo(dtype)
    return info.min if kind == "max" else info.max


def scatter_sum(src, index, dim=0, out=None):
    if dim < 0:
        dim = src.dim() + dim

    if out is None:
        out_size = list(src.size())
        out_size[dim] = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out = torch.zeros(out_size, dtype=src.dtype, device=src.device)

    index = _broadcast_index(out, index, dim)
    out = out.scatter_add(dim, index, src)
    return out


def scatter_max(src, index, dim=0, out=None):
    if dim < 0:
        dim = src.dim() + dim

    if out is None:
        out_size = list(src.size())
        out_size[dim] = int(index.max().item()) + 1 if index.numel() > 0 else 0
        fill = _fill_value(src.dtype, "max")
        out = torch.full(out_size, fill, dtype=src.dtype, device=src.device)

    index = _broadcast_index(out, index, dim)
    # include_self=False so initial fill is *not* part of the reduction
    out = out.scatter_reduce(dim, index, src, reduce="amax", include_self=False)
    return out, None  # keep API shape-compatible with real torch_scatter


def scatter_min(src, index, dim=0, out=None):
    if dim < 0:
        dim = src.dim() + dim

    if out is None:
        out_size = list(src.size())
        out_size[dim] = int(index.max().item()) + 1 if index.numel() > 0 else 0
        fill = _fill_value(src.dtype, "min")
        out = torch.full(out_size, fill, dtype=src.dtype, device=src.device)

    index = _broadcast_index(out, index, dim)
    out = out.scatter_reduce(dim, index, src, reduce="amin", include_self=False)
    return out, None


def scatter_mean(src, index, dim=0, out=None):
    """
    mean = sum / count, emulating torch_scatter.scatter_mean.
    """
    if dim < 0:
        dim = src.dim() + dim

    if out is None:
        out_size = list(src.size())
        out_size[dim] = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out = torch.zeros(out_size, dtype=src.dtype, device=src.device)

    index_b = _broadcast_index(out, index, dim)

    # sum
    out_sum = out.scatter_reduce(dim, index_b, src, reduce="sum", include_self=False)

    # count
    ones = torch.ones_like(src, dtype=torch.long)
    count = torch.zeros_like(out_sum, dtype=torch.long)
    count = count.scatter_reduce(dim, index_b, ones, reduce="sum", include_self=False)

    # avoid division by zero
    count = count.clamp_min(1)
    out_mean = out_sum / count
    return out_mean


def scatter(src, index, dim=0, out=None, reduce="sum"):
    """
    Minimal emulation of torch_scatter.scatter used by ContourCraft.
    """
    if reduce == "sum":
        return scatter_sum(src, index, dim=dim, out=out)
    elif reduce == "min":
        return scatter_min(src, index, dim=dim, out=out)[0]
    elif reduce == "max":
        return scatter_max(src, index, dim=dim, out=out)[0]
    elif reduce == "mean":
        return scatter_mean(src, index, dim=dim, out=out)
    else:
        raise NotImplementedError(f"reduce='{reduce}' not supported in local torch_scatter shim.")
