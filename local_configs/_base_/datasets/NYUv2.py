# dataset settings
dataset_type = 'NYUv2Dataset'
data_root = 'data/NYUv2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (480, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(640, 480), ratio_range=(0.5, 2.0), process_clips=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, process_clips=True),
            dict(type='RandomFlip_clips'),
            dict(type='Normalize_clips', **img_norm_cfg),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=5,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images',
            ann_dir='labels40',
            split='train',
            dilation=[],
            pipeline=train_pipeline
        )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels40',
        split='val',
        dilation=[],
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels40',
        split='val',
        dilation=[],
        pipeline=test_pipeline
    ))
