#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: kpnext_base.py
#       > Define the network architecture for KPNeXt
#


from copy import deepcopy
import math
import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
import numpy as np
from easydict import EasyDict

import einops
from timm.models.layers import DropPath
import pointops

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset

from pointcept.models.kpnext.kpnext_blocks import  KPConvD, KPConvX


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels, momentum=0.05)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class Block(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 radius=1.0,
                 shell_sizes=[1, 14, 28],
                 enable_checkpoint=False):
        super(Block, self).__init__()

        # Hard coded parameters
        sigma = radius
        shared_kp_data = None
        dimension = 3
        influence_mode = 'linear'
        attention_act = 'sigmoid'
        mod_grp_norm = True

        # KPConvX or KPConvD
        if groups == 0:
            self.conv = KPConvD(embed_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                Cmid=0,
                                shared_kp_data=shared_kp_data,
                                dimension=dimension,
                                influence_mode=influence_mode)
        else:
            self.conv = KPConvX(embed_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                attention_groups=groups,
                                attention_act=attention_act,
                                mod_grp_norm=mod_grp_norm,
                                shared_kp_data=shared_kp_data,
                                dimension=dimension,
                                influence_mode=influence_mode)

        # self.attn = GroupedVectorAttention(
        #     embed_channels=embed_channels,
        #     groups=groups,
        #     qkv_bias=qkv_bias,
        #     attn_drop_rate=attn_drop_rate,
        #     pe_multiplier=pe_multiplier,
        #     pe_bias=pe_bias
        # )


        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))

        # feat = self.attn(feat, coord, reference_index) \
        #     if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index)

        # Here KPConvX
        feat = self.conv(coord, coord, feat, reference_index) \
            if not self.enable_checkpoint else checkpoint(self.conv, coord, coord, feat, reference_index)


        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(self,
                 depth,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 radius=1.0,
                 shell_sizes=[1, 14, 28],
                 enable_checkpoint=False
                 ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0. for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                radius=radius,
                shell_sizes=shell_sizes,
                enable_checkpoint=enable_checkpoint
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)

        # Any index == -1 becomes the max index
        reference_index[reference_index == -1] = coord.shape[0]


        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 grid_size,
                 bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                            reduce="min") if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 bias=True,
                 skip=True,
                 backend="map"
                 ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels),
                                  nn.ReLU(inplace=True))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels, bias=bias),
                                       PointBatchNorm(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(coord, skip_coord, self.proj(feat), offset, skip_offset)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class Encoder(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 radius=1.0,
                 shell_sizes=[1, 14, 28],
                 enable_checkpoint=False,
                 ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            radius=radius,
            shell_sizes=shell_sizes,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 embed_channels,
                 groups,
                 depth,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 radius=1.0,
                 shell_sizes=[1, 14, 28],
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            radius=radius,
            shell_sizes=shell_sizes,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 radius=1.0,
                 shell_sizes=[1, 14, 28],
                 enable_checkpoint=False
                 ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            radius=radius,
            shell_sizes=shell_sizes,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])




@MODELS.register_module("kpnext")
class KPNeXt(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 patch_embed_depth=1,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=8,
                 enc_depths=(2, 2, 6, 2),
                 enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64),
                 enc_neighbours=(16, 16, 16, 16),
                 dec_depths=(1, 1, 1, 1),
                 dec_channels=(48, 96, 192, 384),
                 dec_groups=(6, 12, 24, 48),
                 dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.06, 0.12, 0.24, 0.48),
                 shell_sizes=(1, 14, 28),
                 subsample_size=0.02,
                 kp_radius=1.5,
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(KPNeXt, self).__init__()



        ############
        # Parameters
        ############

        # Parameters
        self.subsample_size = subsample_size
        self.kp_radius = kp_radius
        self.first_radius = self.subsample_size * self.kp_radius
        self.shell_sizes = shell_sizes

        # See hardcoded parameters in the Block function

        # Other KPConv parameters in original code
        # self.in_sub_mode = 'grid'
        # self.kp_sigma = 1.5
        # self.neighbor_limits = enc_neighbours
        # self.radius_scaling = 2.2
        # self.first_sigma = self.subsample_size * self.kp_sigma
        # self.layer_blocks = enc_depths
        # self.num_layers = len(self.layer_blocks)
        # self.upsample_n = 3
        # self.share_kp = True
        # self.kp_mode = 'kpconvx'
        # self.task = 'cloud_segmentation'
        # self.grid_pool = True
        # self.add_decoder_layer = True
        # cfg = EasyDict()
        # cfg.data = EasyDict()
        # cfg.model = EasyDict()
        # cfg.data.dim = 3
        # cfg.model.norm = 'batch'
        # cfg.model.bn_momentum = 0.05
        # cfg.model.kp_influence = 'linear'
        # cfg.model.kp_aggregation = 'nearest'
        # cfg.model.inv_groups = 8
        # cfg.model.conv_groups = -1
        # cfg.model.inv_act = 'sigmoid'
        # cfg.model.inv_grp_norm = True
        # cfg.model.kpx_upcut = True


        # Get radii from grid sizes
        layer_radii = [dl * self.kp_radius for dl in grid_sizes]


        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            radius=self.first_radius,
            shell_sizes=self.shell_sizes,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                radius=layer_radii[i],
                shell_sizes=self.shell_sizes,
                enable_checkpoint=enable_checkpoint
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                radius=layer_radii[i],
                shell_sizes=self.shell_sizes,
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
        coord, feat, offset = points
        seg_logits = self.seg_head(feat)
        return seg_logits




