#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# 
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: gpu_neigbors.py
#       > Neighbors search functions on gpu
#


from typing import Tuple

import torch
from torch import Tensor

from pointcept.models.kpconvx.utils.batch_conversion import batch_to_pack, pack_to_batch, pack_to_list, list_to_pack


# ----------------------------------------------------------------------------------------------------------------------
#
#           Implementation of k-nn search in Keops
#       \********************************************/
#

@torch.no_grad()
def keops_radius_count(q_points: Tensor, s_points: Tensor, radius: float) -> Tensor:
    """
    Count neigbors inside radius with PyKeOps.
    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        radius (float)
    Returns:
        radius_counts (Tensor): (*, N)
    """
        
    import pykeops
    pykeops.set_verbose(False)

    num_batch_dims = q_points.dim() - 2
    xi = pykeops.torch.LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = pykeops.torch.LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    vij = (radius - dij).relu().sign()  # (*, N, M)
    radius_counts = vij.sum(dim=num_batch_dims + 1)  # (*, N)
    return radius_counts

@torch.no_grad()
def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """
    kNN with PyKeOps.
    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)
    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """

    import pykeops
    pykeops.set_verbose(False)
    
    xi = pykeops.torch.LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = pykeops.torch.LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).sqnorm2()  # (*, N, M)
    knn_d2, knn_indices = dij.Kmin_argKmin(k, dim=q_points.dim() - 1)  # (*, N, K)
    return knn_d2, knn_indices
    

@torch.no_grad()
def knn(q_points: Tensor,
        s_points: Tensor,
        k: int,
        dilation: int = 1,
        distance_limit: float = None,
        return_distance: bool = False,
        remove_nearest: bool = False,
        transposed: bool = False,
        padding_mode: str = "nearest",
        inf: float = 1e10):
    """
    Compute the kNNs of the points in `q_points` from the points in `s_points`.
    Use KeOps to accelerate computation.
    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, the neighbors are replaced according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): padding mode for neighbors further than distance radius. ('nearest', 'empty').
        inf (float=1e10): infinity value for padding.
    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)

    num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)

    knn_distances, knn_indices = keops_knn(q_points, s_points, final_k)  # (*, N, k)

    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances[knn_masks] = knn_distances[..., 0]
            knn_indices[knn_masks] = knn_indices[..., 0]
        else:
            knn_distances[knn_masks] = inf
            knn_indices[knn_masks] = num_s_points

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices

@torch.no_grad()
def radius_search_pack_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit, shadow=False, inf=1e8, return_dist=False):
    """Radius search in pack mode (fast version).
    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): radius radius.
        neighbor_limit (int): neighbor radius.
        inf (float=1e10): infinity value.
    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """

    # pack to batch
    batch_q_points, batch_q_masks = pack_to_batch(q_points, q_lengths, fill_value=inf)  # (B, M', 3)
    batch_s_points, batch_s_masks = pack_to_batch(s_points, s_lengths, fill_value=inf)  # (B, N', 3)

    # knn  (B, M', K)
    batch_knn_distances, batch_knn_indices = keops_knn(batch_q_points, batch_s_points, neighbor_limit)

    # accumulate index
    batch_start_index = torch.cumsum(s_lengths, dim=0) - s_lengths
    batch_knn_indices += batch_start_index.view(-1, 1, 1)

    # Limit for shadow neighbors
    if shadow:
        # shadow everything outside radius
        shadow_limit = radius ** 2
    else:
        # keep knns, only shadow invalid indices like when s_pts.shape < K
        shadow_limit = inf / 10

    # Fill shadow neighbors values
    batch_knn_masks = torch.gt(batch_knn_distances, shadow_limit)
    batch_knn_indices.masked_fill_(batch_knn_masks, s_points.shape[0])  # (B, M', K)

    # batch to pack
    knn_indices, _ = batch_to_pack(batch_knn_indices, batch_q_masks)  # (M, K)
    if return_dist:
        knn_distances, _ = batch_to_pack(batch_knn_distances, batch_q_masks)  # (M, K)
        return knn_indices, torch.sqrt(knn_distances)
    
    return knn_indices

@torch.no_grad()
def radius_search_list_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit, shadow=False):
    """
    Radius search in pack mode (fast version). This function is actually a knn search 
    but with option to shadow furthest neighbors (d > radius).
    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): search radius, only used for shadowing furthest neighbors.
        neighbor_limit (int): max number of neighbors, actual knn limit used for computing neighbors.
        inf (float=1e10): infinity value.
    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """

    # pack to batch
    batch_q_list = pack_to_list(q_points, q_lengths)  # (B)(?, 3)
    batch_s_list = pack_to_list(s_points, s_lengths)  # (B)(?, 3)

    # knn on each element of the list (B)[(?, K), (?, K)]
    knn_dists_inds = [keops_knn(b_q_pts, b_s_pts, neighbor_limit)
                          for b_q_pts, b_s_pts in zip(batch_q_list, batch_s_list)]

    # Accumualte indices
    b_start_ind = torch.cumsum(s_lengths, dim=0) - s_lengths
    knn_inds_list = [b_knn_inds + b_start_ind[i] for i, (_, b_knn_inds) in enumerate(knn_dists_inds)]

    # Convert list to pack (B)[(?, K) -> (M, K)
    knn_indices, _ = list_to_pack(knn_inds_list)

    # Apply shadow inds (optional because knn to far away from convolution kernel will be ignored anyway)
    if shadow:
        knn_dists_list = [b_knn_dists for b_knn_dists, _ in knn_dists_inds]
        knn_dists, _ = list_to_pack(knn_dists_list)
        knn_masks = torch.gt(knn_dists, radius**2)
        knn_indices.masked_fill_(knn_masks, s_points.shape[0])

    return knn_indices

@torch.no_grad()
def tiled_knn(q_points: Tensor, s_points: Tensor, k: int, tile_size: float, margin: float) -> Tuple[Tensor, Tensor]:
    """
    Divide the query and support in tiles and .
    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k           (int): number of neighbors
        tile_size (float): size of the square tiles
        margin    (float): margin for tiling the support (must be > max_knn_dist)
    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """

    # Get limits
    min_q, _ = torch.min(q_points, dim=-2)
    min_s, _ = torch.min(s_points, dim=-2)
    min_p = torch.minimum(min_q, min_s) - margin
    max_q, _ = torch.max(q_points, dim=-2)
    max_s, _ = torch.max(s_points, dim=-2)
    max_p = torch.maximum(max_q, max_s) + margin

    # Create tiles
    tile_N = torch.ceil((max_p - min_p) / tile_size).type(torch.long)

    # Init neighbors and dists
    knn_indices = torch.zeros((q_points.shape[0],), dtype=torch.long, device=q_points.device)
    knn_distances = torch.zeros((q_points.shape[0],), dtype=q_points.dtype, device=q_points.device) + 1e8
    s_inds = torch.arange(s_points.shape[0], dtype=torch.long, device=q_points.device)

    # Loop on tiles
    for xi in range(tile_N[0].item()):
        for yi in range(tile_N[1].item()):
            for zi in range(tile_N[2].item()):

                # Get tile limits
                tile_min = min_p + tile_size * torch.tensor([xi, yi, zi],
                                                            dtype=q_points.dtype,
                                                            device=q_points.device)
                tile_max = tile_min + tile_size + 0.1* margin 
                q_mask = torch.logical_and(torch.all(q_points > tile_min, dim=-1),
                                           torch.all(q_points <= tile_max, dim=-1))
                s_mask = torch.logical_and(torch.all(s_points > tile_min - margin, dim=-1),
                                           torch.all(s_points <= tile_max + margin, dim=-1))
                                           
                # Get points in the tile
                q_pts = q_points[q_mask]
                s_pts = s_points[s_mask]
                if q_pts.shape[0] < 1:
                    continue
                if s_pts.shape[0] < 1:
                    raise ValueError('got queries but no support points')

                # Get knn
                knn_d, knn_i = keops_knn(q_pts, s_pts, k)

                # (*, N_i),  (*, N_i) values in M_i
                knn_distances[q_mask] = knn_d.view((-1))
                knn_indices[q_mask] = s_inds[s_mask][knn_i.view((-1))]

    return knn_distances, knn_indices
