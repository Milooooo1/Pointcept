_base_ = ["../_base_/default_runtime.py"]

# minimal example settings
num_worker = 8  # total worker in all gpu
batch_size = 2  
mix_prob = 0.8
max_input_pts = 40000

# # Settings for multigpu training with 8 80GB gpus
# num_worker = 64  # total worker in all gpu
# batch_size = 12  
# mix_prob = 0.8
# empty_cache = False
# enable_amp = True
# sync_bn = True
# max_input_pts = 80000


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="kpconvx_base",
        input_channels=3,
        num_classes=8,
        dim=3,
        task='cloud_segmentation',
        kp_mode='kpconvx',
        shell_sizes=(1, 14, 28),
        kp_radius=2.3,
        kp_aggregation='nearest',
        kp_influence='constant',
        kp_sigma=2.3,
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
        channel_scaling=1.414,
        radius_scaling=2.2,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,  # Ignored if grid_pool is True
        first_inv_layer=1,
        drop_path_rate=0.3,
        norm='batch',
        bn_momentum=0.1,
        smooth_labels=False,  # True only for classification
        class_w=(),
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             ignore_index=-1),
        dict(type="LovaszLoss",
             mode="multiclass",
             loss_weight=1.0,
             ignore_index=-1)
    ]
)

# scheduler settings
epoch = 1000
eval_epoch = 200
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.02)
scheduler = dict(type="OneCycleLR",
                 max_lr=optimizer["lr"],
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=100.0,
                 final_div_factor=1000.0)

# dataset settings
dataset_type = "Rail3DDataset"
data_root = "data/rail3d"

data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        "Ground",
        "Vegetation",
        "Rail",
        "Poles",
        "Wires",
        "Signalling",
        "Fences",
        "Installation",
        "Building",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.0),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="GridSample", 
                 grid_size=0.02, 
                 hash_type="fnv", 
                 mode="train", 
                 keys=("coord", "segment"),
                 return_min_coord=True),
            dict(type="SphereCrop", point_max=max_input_pts, mode="random"),
            dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord"))#, "color", "normal"))
        ],
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", 
                 grid_size=0.02, 
                 hash_type="fnv", 
                 mode="train", 
                 keys=("coord", "segment"),#  keys=("coord", "color", "normal", "segment"),
                 return_min_coord=True),
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord"))#, "color", "normal"))
        ],
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample",
                          grid_size=0.02,
                          hash_type="fnv",
                          mode="test",
                          keys=("coord")#, "color", "normal")
                          ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "index"), feat_keys=("coord"))#, "color", "normal"))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1],     axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0],     axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1],     axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomFlip", p=1)]
            ]
        )
    ),
)
