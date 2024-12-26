#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: kpconvx_base.py
#       > Define the network architecture for KPConvX
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


from pointcept.models.builder import MODELS



from pointcept.models.kpconvx.utils.generic_blocks import LinearUpsampleBlock, NearestUpsampleBlock, UnaryBlock, local_nearest_pool, GlobalAverageBlock, MaxPoolBlock, SmoothCrossEntropyLoss
from pointcept.models.kpconvx.utils.kpconv_blocks import KPConvBlock, KPConvResidualBlock, KPConvInvertedBlock
from pointcept.models.kpconvx.utils.kpnext_blocks import KPNextResidualBlock, KPNextInvertedBlock, KPNextMultiShortcutBlock, KPNextBlock

from pointcept.models.kpconvx.utils.torch_pyramid import build_base_pyramid, fill_pyramid, build_full_pyramid



@MODELS.register_module("kpconvx_base")
class KPConvXBase(nn.Module):

    def __init__(self,
                 input_channels,
                 num_classes,
                 dim=3,
                 task='cloud_segmentation',
                 kp_mode='kpconvx',
                 kp_influence='linear',
                 kp_aggregation='nearest',
                 shell_sizes=(1, 14, 28),
                 kp_radius=2.3,
                 kp_sigma=2.3,
                 radius_scaling=2.2,
                 share_kp=False,
                 conv_groups=-1,  # Only for old KPConv blocks
                 inv_groups=8,
                 inv_act='sigmoid',
                 inv_grp_norm=True,
                 kpx_upcut=False,
                 subsample_size=0.02,
                 neighbor_limits=(12, 16, 20, 20, 20),
                 layer_blocks=(3, 3, 9, 12, 3),
                 init_channels=64,
                 channel_scaling=np.sqrt(2),
                 decoder_layer=True,
                 grid_pool=True,
                 upsample_n=3,
                 first_inv_layer=1,
                 drop_path_rate=0,
                 norm='batch',
                 bn_momentum=0.1,
                 smooth_labels=False,  # True only for classification
                 class_w=(),
                 ):
        super(KPConvXBase, self).__init__()

        ############
        # Parameters
        ############

        # Parameters
        self.dim = dim
        self.subsample_size = subsample_size    # We assume here that init_sub_size == in_sub_size > 0
        self.in_sub_mode = 'grid'
        self.neighbor_limits = neighbor_limits

        self.kp_mode = kp_mode
        self.shell_sizes = shell_sizes
        self.kp_influence = kp_influence
        self.kp_aggregation = kp_aggregation
        self.kp_sigma = kp_sigma
        self.kp_radius = kp_radius
        self.first_radius = subsample_size * kp_radius      # Actual radius of the first convolution in meters
        self.first_sigma = subsample_size * kp_sigma        # Actual sigma of the first convolution in meters
        self.radius_scaling = radius_scaling
        
        self.layer_blocks = layer_blocks
        self.num_layers = len(self.layer_blocks)
        self.upsample_n = upsample_n
        self.task = task
        self.grid_pool = grid_pool
        self.add_decoder_layer = decoder_layer
        self.first_inv_layer = first_inv_layer
        self.share_kp = share_kp
        self.num_logits = num_classes

        self.inv_act = inv_act
        self.inv_groups = inv_groups
        self.inv_grp_norm = inv_grp_norm
        self.kpx_upcut = kpx_upcut

        self.norm = norm
        self.bn_momentum = bn_momentum
        self.smooth_labels = smooth_labels
        self.class_w = class_w

        # Only for old KPConv blocks
        self.conv_groups = conv_groups  

        # Stochastic depth decay rule
        dpr_list = np.linspace(0, drop_path_rate, sum(self.layer_blocks)) 

        # Variables
        in_C = input_channels
        first_C = init_channels
        conv_r = self.first_radius
        conv_sig = self.first_sigma

        # Get channels at each layer
        layer_C = []
        for l in range(self.num_layers):
            target_C = first_C * channel_scaling ** l                   # Scale channels
            layer_C.append(int(np.ceil((target_C - 0.1) / 16)) * 16)    # Ensure it is divisible by 16 (even the first one)

        # Verify the architecture validity
        if self.layer_blocks[0] < 1:
            raise ValueError('First layer must contain at least 1 convolutional layers')
        if np.min(self.layer_blocks) < 1:
            raise ValueError('Each layer must contain at least 1 convolutional layers')
        
        #####################
        # List Encoder blocks
        #####################

        # ------ Layers 1 ------
        if share_kp:
            self.shared_kp = [{} for _ in range(self.num_layers)]
        else:
            self.shared_kp = [None for _ in range(self.num_layers)]

        # Initial convolution or MLP
        C = layer_C[0]
        self.stem = self.get_conv_block(in_C, C, conv_r, conv_sig)
        # self.stem = self.get_unary_block(in_C, C)

        # Next blocks
        self.encoder_1 = nn.ModuleList()
        use_conv = self.first_inv_layer >= 1
        for block_i in range(self.layer_blocks[0]):
            Cout = layer_C[1] if self.grid_pool and block_i == self.layer_blocks[0] - 1 else C
            self.encoder_1.append(self.get_residual_block(C, Cout, conv_r, conv_sig,
                                                          shared_kp_data=self.shared_kp[0],
                                                          conv_layer=use_conv,
                                                          drop_path=dpr_list[block_i]))

        # Pooling block
        self.pooling_1 = MaxPoolBlock()

        # ------ Layers [2, 3, 4, 5] ------
        for layer in range(2, self.num_layers + 1):
            l = layer - 1

            # Update features, radius, sigma for this layer
            C = layer_C[l]
            conv_r *= self.radius_scaling
            conv_sig *= self.radius_scaling

            # Layer blocks
            use_conv = self.first_inv_layer >= layer
            encoder_i = nn.ModuleList()
            for block_i in range(self.layer_blocks[l]):
                Cout = layer_C[l+1] if self.grid_pool and layer < self.num_layers and block_i == self.layer_blocks[l] - 1 else C
                encoder_i.append(self.get_residual_block(C, Cout, conv_r, conv_sig,
                                                         shared_kp_data=self.shared_kp[l],
                                                         conv_layer=use_conv,
                                                         drop_path=dpr_list[sum(self.layer_blocks[:l]) + block_i]))
            setattr(self, 'encoder_{:d}'.format(layer), encoder_i)

            # Pooling block (not for the last layer)
            if layer < self.num_layers:
                pooling_i = MaxPoolBlock()

                setattr(self, 'pooling_{:d}'.format(layer), pooling_i)

        #####################
        # List Decoder blocks
        #####################

        if task == 'classification':

            #  ------ Head ------

            # Global pooling
            self.global_pooling = GlobalAverageBlock()

            # New head
            self.head = nn.Sequential(self.get_unary_block(layer_C[-1], 256, norm_type='none'),
                                      nn.Dropout(0.4),
                                      nn.Linear(256, self.num_logits))

            # # Old head
            # self.head = nn.Sequential(self.get_unary_block(layer_C[-1], layer_C[-1]),
            #                           nn.Linear(layer_C[-1], self.num_logits))

        elif task == 'cloud_segmentation':

            # ------ Layers [4, 3, 2, 1] ------
            for layer in range(self.num_layers - 1, 0, -1):

                # Upsample block
                if self.grid_pool:
                    upsampling_i = NearestUpsampleBlock()
                else:
                    upsampling_i = LinearUpsampleBlock(self.upsample_n)
                setattr(self, 'upsampling_{:d}'.format(layer), upsampling_i)

                # Network layers in decoder
                C = layer_C[layer - 1]
                C1 = layer_C[layer]
                if self.grid_pool:
                    Cin = C1 + C1
                else:
                    Cin = C + C1
                decoder_unary_i = self.get_unary_block(Cin, C)
                setattr(self, 'decoder_unary_{:d}'.format(layer), decoder_unary_i)

                # Additionnal network layer (optional)
                if self.add_decoder_layer:
                    conv_r *= 1 / self.radius_scaling
                    conv_sig *= 1 / self.radius_scaling
                    decoder_layer_i = self.get_residual_block(C, C, conv_r, conv_sig,
                                                              shared_kp_data=self.shared_kp[layer - 1])
                    setattr(self, 'decoder_layer_{:d}'.format(layer), decoder_layer_i)


            #  ------ Head ------
            
            # New head
            self.head = nn.Sequential(self.get_unary_block(layer_C[0], layer_C[0]),
                                    nn.Linear(layer_C[0], self.num_logits))
            # Easy KPConv Head
            # self.head = nn.Sequential(nn.Linear(layer_C[0] * 2, layer_C[0]),
            #                           nn.GroupNorm(8, layer_C[0]),
            #                           nn.ReLU(),
            #                           nn.Linear(layer_C[0], self.num_logits))

            # My old head
            # self.head = nn.Sequential(self.get_unary_block(layer_C[0] * 2, layer_C[0], norm_type='none'),
            #                           nn.Linear(layer_C[0], self.num_logits))



        ################
        # Network Losses
        ################

        # Choose between normal cross entropy and smoothed labels
        if self.smooth_labels:
            CrossEntropy = SmoothCrossEntropyLoss
        else:
            CrossEntropy = torch.nn.CrossEntropyLoss

        if task == 'classification':
            self.criterion = CrossEntropy()
            
        elif task == 'cloud_segmentation':
            if len(self.class_w) > 0:
                class_w = torch.from_numpy(np.array(self.class_w, dtype=np.float32))
                self.criterion = CrossEntropy(weight=class_w, ignore_index=-1)
            else:
                self.criterion = CrossEntropy(ignore_index=-1)

        # self.deform_fitting_mode = config.deform_fitting_mode
        # self.deform_fitting_power = config.deform_fitting_power
        # self.deform_lr_factor = config.deform_lr_factor
        # self.repulse_extent = config.repulse_extent

        self.output_loss = 0
        self.deform_loss = 0
        self.l1 = nn.L1Loss()



        return




    def get_unary_block(self, in_C, out_C, norm_type=None):

        if norm_type is None:
            norm_type = self.norm

        return UnaryBlock(in_C,
                          out_C,
                          norm_type=norm_type,
                          bn_momentum=self.bn_momentum)

    def get_conv_block(self, in_C, out_C, radius, sigma):

        # First layer is the most simple convolution possible
        return KPConvBlock(in_C,
                           out_C,
                           self.shell_sizes,
                           radius,
                           sigma,
                           influence_mode=self.kp_influence,
                           aggregation_mode=self.kp_aggregation,
                           dimension=self.dim,
                           norm_type=self.norm,
                           bn_momentum=self.bn_momentum)
                 
    def get_residual_block(self, in_C, out_C, radius, sigma, shared_kp_data=None, conv_layer=False, drop_path=-1):

        attention_groups = self.inv_groups
        if conv_layer or 'kpconvd' in self.kp_mode:
            attention_groups = 0

        #TMP to get 
        if self.kp_mode == 'kpconv':

            return KPConvResidualBlock(in_C,
                                       out_C,
                                       self.shell_sizes,
                                       radius,
                                       sigma,
                                       groups=self.conv_groups,
                                       shared_kp_data=shared_kp_data,
                                       influence_mode=self.kp_influence,
                                       dimension=self.dim,
                                       norm_type=self.norm,
                                       bn_momentum=self.bn_momentum)
        
        elif self.kp_mode == 'kpconvtest':
            return KPNextResidualBlock(in_C,
                                       out_C,
                                       self.shell_sizes,
                                       radius,
                                       sigma,
                                       attention_groups=attention_groups,
                                       attention_act=self.inv_act,
                                       mod_grp_norm=self.inv_grp_norm,
                                       shared_kp_data=shared_kp_data,
                                       influence_mode=self.kp_influence,
                                       dimension=self.dim,
                                       norm_type=self.norm,
                                       bn_momentum=self.bn_momentum)

        else:

            return KPNextMultiShortcutBlock(in_C,
                                            out_C,
                                            self.shell_sizes,
                                            radius,
                                            sigma,
                                            attention_groups=attention_groups,
                                            attention_act=self.inv_act,
                                            mod_grp_norm=self.inv_grp_norm,
                                            expansion=4,
                                            drop_path_p=drop_path,
                                            layer_scale_init_v=-1.,
                                            use_upcut=self.kpx_upcut,
                                            shared_kp_data=shared_kp_data,
                                            influence_mode=self.kp_influence,
                                            dimension=self.dim,
                                            norm_type=self.norm,
                                            bn_momentum=self.bn_momentum)




    def forward(self, data_dict):

        #  ------ Init ------

        # coord = data_dict["coord"]
        # feat = data_dict["feat"]
        # offset = data_dict["offset"].int()
        # points = [coord, feat, offset]

        points = data_dict['coord']
        feats = data_dict["feat"]
        offset = data_dict['offset'].int()


        # Convert offsets to lengths
        offset = torch.cat([torch.zeros(1, dtype=offset.dtype, device=offset.device), offset], dim=0)
        lengths = offset[1:] - offset[:-1]

        in_dict = build_full_pyramid(points,
                                     lengths,
                                     self.num_layers,
                                     self.subsample_size,
                                     self.first_radius,
                                     self.radius_scaling,
                                     self.neighbor_limits,
                                     self.upsample_n,
                                     sub_mode=self.in_sub_mode,
                                     grid_pool_mode=self.grid_pool)
        

        #  ------ Stem ------

        feats = self.stem(in_dict.points[0], in_dict.points[0], feats, in_dict.neighbors[0])
        # feats = self.stem(feats)


        #  ------ Encoder ------

        skip_feats = []
        for layer in range(1, self.num_layers + 1):

            # Get layer blocks
            l = layer -1
            block_list = getattr(self, 'encoder_{:d}'.format(layer))
            
            # Layer blocks
            if self.kp_mode in ['kpconv', 'kpconvtest']:
                for block in block_list:
                    feats = block(in_dict.points[l], in_dict.points[l], feats, in_dict.neighbors[l])
            else:
                upcut = None
                for block in block_list:
                    feats, upcut = block(in_dict.points[l], in_dict.points[l], feats, in_dict.neighbors[l], in_dict.lengths[l], upcut=upcut)
                
            if layer < self.num_layers:

                # Skip features
                skip_feats.append(feats)

                # Pooling
                layer_pool = getattr(self, 'pooling_{:d}'.format(layer))
                if self.grid_pool:
                    if isinstance(in_dict.pools[l], tuple):
                        feats = layer_pool(feats, in_dict.pools[l][0], idx_ptr=in_dict.pools[l][1])
                    else:
                        feats = layer_pool(feats, in_dict.pools[l])
                else:
                    feats = layer_pool(in_dict.points[l+1], in_dict.points[l], feats, in_dict.pools[l])

         

        if self.task == 'classification':
            
            # Global pooling
            feats = self.global_pooling(feats, in_dict.lengths[-1])

            
        elif self.task == 'cloud_segmentation':

            #  ------ Decoder ------

            for layer in range(self.num_layers - 1, 0, -1):

                # Get layer blocks
                l = layer -1    # 3, 2, 1, 0
                upsample = getattr(self, 'upsampling_{:d}'.format(layer))

                # Upsample
                if self.grid_pool:
                    feats = upsample(feats, in_dict.upsamples[l])
                else:
                    feats = upsample(feats, in_dict.upsamples[l], in_dict.up_distances[l])

                # Concat with skip features
                feats = torch.cat([feats, skip_feats[l]], dim=1)
                
                # MLP
                unary = getattr(self, 'decoder_unary_{:d}'.format(layer))
                feats = unary(feats)

                # Optional Decoder layers
                if self.add_decoder_layer:
                    block = getattr(self, 'decoder_layer_{:d}'.format(layer))
                    if self.kp_mode in ['kpconv', 'kpconvtest']:
                        feats = block(in_dict.points[l], in_dict.points[l], feats, in_dict.neighbors[l])
                    else:
                        feats, _ = block(in_dict.points[l], in_dict.points[l], feats, in_dict.neighbors[l], in_dict.lengths[l])

        #  ------ Head ------

        logits = self.head(feats)
                


        return logits


