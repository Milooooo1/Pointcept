#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# 
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: cpp_funcs.py
#       > cpp functions wrapped in python
#


import numpy as np
import torch

# Subsampling extension
from pointcept.models.kpconvx.cpp_wrappers.cpp_subsampling import cpp_subsampling


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def batch_grid_partition(points, batches_len, sampleDl=0.1):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features).
    Also returns pooling and upsampling inds
    :param points: (N, 3) matrix of input points
    :param sampleDl: parameter defining the size of grid voxels
    :return: subsampled points, with features and/or labels depending of the input
    """

    if torch.is_tensor(points):
        points1 = points.cpu().numpy()
        batches_len1 = batches_len.cpu().numpy()

    s_points, s_len, pools, ups = cpp_subsampling.batch_grid_partitionning(points1,
                                                                           batches_len1,
                                                                           sampleDl=sampleDl)

    if torch.is_tensor(points):
        s_points = torch.from_numpy(s_points).to(points.device)
        s_len = torch.from_numpy(s_len).to(points.device)
        pools = torch.from_numpy(pools).to(points.device)
        ups = torch.from_numpy(ups).to(points.device)

    return s_points, s_len, pools, ups

