# configs for REDS4
reds_data_root = 'data/REDS'

reds_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

reds_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=reds_data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=1,
        num_input_frames=100,
        fixed_seq_len=100,
        pipeline=reds_pipeline))

reds_evaluator = [
    dict(type='PSNR', prefix='REDS4-BIx4-RGB'),
    dict(type='SSIM', prefix='REDS4-BIx4-RGB')
]

# configs for vimeo90k-bd and vimeo90k-bi
vimeo_90k_data_root = 'data/vimeo90k'
vimeo_90k_file_list = [
    'im1.png', 'im2.png', 'im3.png', 'im4.png', 'im5.png', 'im6.png', 'im7.png'
]

vimeo_90k_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='MirrorSequence', keys=['img']),
    dict(type='PackInputs')
]

vimeo_90k_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vimeo90k_seq', task_name='vsr'),
        data_root=vimeo_90k_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_test_GT.txt',
        depth=2,
        num_input_frames=7,
        fixed_seq_len=7,
        load_frames_list=dict(img=vimeo_90k_file_list, gt=['im4.png']),
        pipeline=vimeo_90k_pipeline))

vimeo_90k_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vimeo90k_seq', task_name='vsr'),
        data_root=vimeo_90k_data_root,
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_test_GT.txt',
        depth=2,
        num_input_frames=7,
        fixed_seq_len=7,
        load_frames_list=dict(img=vimeo_90k_file_list, gt=['im4.png']),
        pipeline=vimeo_90k_pipeline))

vimeo_90k_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='Vimeo-90K-T-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='Vimeo-90K-T-BDx4-Y'),
]

vimeo_90k_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='Vimeo-90K-T-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='Vimeo-90K-T-BIx4-Y'),
]

# config for UDM10 (BDx4)
udm10_data_root = 'data/UDM10'

udm10_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:04d}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

udm10_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='udm10', task_name='vsr'),
        data_root=udm10_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        pipeline=udm10_pipeline))

udm10_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UDM10-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UDM10-BDx4-Y')
]

# config for vid4
vid4_data_root = 'data/Vid4'

vid4_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]
vid4_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=1,
        pipeline=vid4_pipeline))

vid4_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=1,
        pipeline=vid4_pipeline))

vid4_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='VID4-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='VID4-BDx4-Y'),
]
vid4_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='VID4-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='VID4-BIx4-Y'),
]

# config for uvg
uvg_data_root = 'data/UVG'

uvg_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:03d}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

uvg_beauty_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Beauty/BDx4', gt='Beauty/GT'),
        pipeline=uvg_pipeline))

uvg_beauty_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Beauty-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Beauty-BDx4-Y')
]

uvg_beauty_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Beauty/BIx4', gt='Beauty/GT'),
        pipeline=uvg_pipeline))

uvg_beauty_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Beauty-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Beauty-BIx4-Y')
]

uvg_bosphorus_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Bosphorus/BDx4', gt='Bosphorus/GT'),
        pipeline=uvg_pipeline))

uvg_bosphorus_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Bosphorus-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Bosphorus-BDx4-Y')
]

uvg_bosphorus_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Bosphorus/BIx4', gt='Bosphorus/GT'),
        pipeline=uvg_pipeline))

uvg_bosphorus_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Bosphorus-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Bosphorus-BIx4-Y')
]

uvg_honeybee_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='HoneyBee/BDx4', gt='HoneyBee/GT'),
        pipeline=uvg_pipeline))

uvg_honeybee_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-HoneyBee-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-HoneyBee-BDx4-Y')
]

uvg_honeybee_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='HoneyBee/BIx4', gt='HoneyBee/GT'),
        pipeline=uvg_pipeline))

uvg_honeybee_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-HoneyBee-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-HoneyBee-BIx4-Y')
]
uvg_jockey_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Jockey/BDx4', gt='Jockey/GT'),
        pipeline=uvg_pipeline))

uvg_jockey_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Jockey-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Jockey-BDx4-Y')
]

uvg_jockey_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Jockey/BIx4', gt='Jockey/GT'),
        pipeline=uvg_pipeline))

uvg_jockey_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Jockey-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Jockey-BIx4-Y')
]

uvg_readysetgo_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='ReadySetGo/BDx4', gt='ReadySetGo/GT'),
        pipeline=uvg_pipeline))

uvg_readysetgo_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-ReadySetGo-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-ReadySetGo-BDx4-Y')
]

uvg_readysetgo_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='ReadySetGo/BIx4', gt='ReadySetGo/GT'),
        pipeline=uvg_pipeline))

uvg_readysetgo_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-ReadySetGo-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-ReadySetGo-BIx4-Y')
]

uvg_shakendry_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='ShakeNDry/BDx4', gt='ShakeNDry/GT'),
        pipeline=uvg_pipeline))

uvg_shakendry_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-ShakeNDry-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-ShakeNDry-BDx4-Y')
]

uvg_shakendry_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='ShakeNDry/BIx4', gt='ShakeNDry/GT'),
        pipeline=uvg_pipeline))

uvg_shakendry_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-ShakeNDry-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-ShakeNDry-BIx4-Y')
]


uvg_yachtride_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='YachtRide/BDx4', gt='YachtRide/GT'),
        pipeline=uvg_pipeline))

uvg_yachtride_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-YachtRide-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-YachtRide-BDx4-Y')
]

uvg_yachtride_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='YachtRide/BIx4', gt='YachtRide/GT'),
        pipeline=uvg_pipeline))

uvg_yachtride_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-YachtRide-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-YachtRide-BIx4-Y')
]

uvg_cityalley_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='CityAlley/BDx4', gt='CityAlley/GT'),
        pipeline=uvg_pipeline))

uvg_cityalley_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-CityAlley-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-CityAlley-BDx4-Y')
]

uvg_cityalley_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='CityAlley/BIx4', gt='CityAlley/GT'),
        pipeline=uvg_pipeline))

uvg_cityalley_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-CityAlley-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-CityAlley-BIx4-Y')
]

uvg_flowerfocus_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='FlowerFocus/BDx4', gt='FlowerFocus/GT'),
        pipeline=uvg_pipeline))

uvg_flowerfocus_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-FlowerFocus-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-FlowerFocus-BDx4-Y')
]

uvg_flowerfocus_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='FlowerFocus/BIx4', gt='FlowerFocus/GT'),
        pipeline=uvg_pipeline))

uvg_flowerfocus_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-FlowerFocus-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-FlowerFocus-BIx4-Y')
]

uvg_flowerkids_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='FlowerKids/BDx4', gt='FlowerKids/GT'),
        pipeline=uvg_pipeline))

uvg_flowerkids_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-FlowerKids-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-FlowerKids-BDx4-Y')
]

uvg_flowerkids_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='FlowerKids/BIx4', gt='FlowerKids/GT'),
        pipeline=uvg_pipeline))

uvg_flowerkids_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-FlowerKids-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-FlowerKids-BIx4-Y')
]

uvg_flowerpan_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='FlowerPan/BDx4', gt='FlowerPan/GT'),
        pipeline=uvg_pipeline))

uvg_flowerpan_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-FlowerPan-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-FlowerPan-BDx4-Y')
]

uvg_flowerpan_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='FlowerPan/BIx4', gt='FlowerPan/GT'),
        pipeline=uvg_pipeline))

uvg_flowerpan_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-FlowerPan-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-FlowerPan-BIx4-Y')
]

uvg_lips_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Lips/BDx4', gt='Lips/GT'),
        pipeline=uvg_pipeline))

uvg_lips_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Lips-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Lips-BDx4-Y')
]

uvg_lips_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Lips/BIx4', gt='Lips/GT'),
        pipeline=uvg_pipeline))

uvg_lips_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Lips-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Lips-BIx4-Y')
]

uvg_racenight_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='RaceNight/BDx4', gt='RaceNight/GT'),
        pipeline=uvg_pipeline))

uvg_racenight_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-RaceNight-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-RaceNight-BDx4-Y')
]

uvg_racenight_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='RaceNight/BIx4', gt='RaceNight/GT'),
        pipeline=uvg_pipeline))

uvg_racenight_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-RaceNight-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-RaceNight-BIx4-Y')
]


uvg_riverbank_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='RiverBank/BDx4', gt='RiverBank/GT'),
        pipeline=uvg_pipeline))

uvg_riverbank_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-RiverBank-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-RiverBank-BDx4-Y')
]

uvg_riverbank_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='RiverBank/BIx4', gt='RiverBank/GT'),
        pipeline=uvg_pipeline))

uvg_riverbank_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-RiverBank-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-RiverBank-BIx4-Y')
]

uvg_sunbath_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='SunBath/BDx4', gt='SunBath/GT'),
        pipeline=uvg_pipeline))

uvg_sunbath_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-SunBath-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-SunBath-BDx4-Y')
]

uvg_sunbath_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='SunBath/BIx4', gt='SunBath/GT'),
        pipeline=uvg_pipeline))

uvg_sunbath_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-SunBath-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-SunBath-BIx4-Y')
]

uvg_twilight_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Twilight/BDx4', gt='Twilight/GT'),
        pipeline=uvg_pipeline))

uvg_twilight_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Twilight-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Twilight-BDx4-Y')
]

uvg_twilight_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='uvg', task_name='vsr'),
        data_root=uvg_data_root,
        data_prefix=dict(img='Twilight/BIx4', gt='Twilight/GT'),
        pipeline=uvg_pipeline))

uvg_twilight_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UVG-Twilight-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UVG-Twilight-BIx4-Y')
]

# config for test
test_cfg = dict(type='MultiTestLoop')
test_dataloader = [
    #reds_dataloader,
    #vimeo_90k_bd_dataloader,
    #vimeo_90k_bi_dataloader,
    #udm10_dataloader,
    #vid4_bd_dataloader,
    #vid4_bi_dataloader,

    #uvg_beauty_bd_dataloader,
    #uvg_bosphorus_bd_dataloader,
    #uvg_honeybee_bd_dataloader,
    #uvg_jockey_bd_dataloader,
    #uvg_readysetgo_bd_dataloader,
    #uvg_shakendry_bd_dataloader,
    #uvg_yachtride_bd_dataloader,
    #uvg_cityalley_bd_dataloader,
    #uvg_flowerfocus_bd_dataloader,
    #uvg_flowerkids_bd_dataloader,
    #uvg_flowerpan_bd_dataloader,
    #uvg_lips_bd_dataloader,
    #uvg_racenight_bd_dataloader,
    #uvg_riverbank_bd_dataloader,
    #uvg_sunbath_bd_dataloader,
    #uvg_twilight_bd_dataloader,

    uvg_beauty_bi_dataloader,
    uvg_bosphorus_bi_dataloader,
    uvg_honeybee_bi_dataloader,
    uvg_jockey_bi_dataloader,
    uvg_readysetgo_bi_dataloader,
    uvg_shakendry_bi_dataloader,
    uvg_yachtride_bi_dataloader,
    uvg_cityalley_bi_dataloader,
    uvg_flowerfocus_bi_dataloader,
    uvg_flowerkids_bi_dataloader,
    uvg_flowerpan_bi_dataloader,
    uvg_lips_bi_dataloader,
    uvg_racenight_bi_dataloader,
    uvg_riverbank_bi_dataloader,
    uvg_sunbath_bi_dataloader,
    uvg_twilight_bi_dataloader,
]
test_evaluator = [
    #reds_evaluator,
    #vimeo_90k_bd_evaluator,
    #vimeo_90k_bi_evaluator,
    #udm10_evaluator,
    #vid4_bd_evaluator,
    #vid4_bi_evaluator,

    #uvg_beauty_bd_evaluator,
    #uvg_bosphorus_bd_evaluator,
    #uvg_honeybee_bd_evaluator,
    #uvg_jockey_bd_evaluator,
    #uvg_readysetgo_bd_evaluator,
    #uvg_shakendry_bd_evaluator,
    #uvg_yachtride_bd_evaluator,
    #uvg_cityalley_bd_evaluator,
    #uvg_flowerfocus_bd_evaluator,
    #uvg_flowerkids_bd_evaluator,
    #uvg_flowerpan_bd_evaluator,
    #uvg_lips_bd_evaluator,
    #uvg_racenight_bd_evaluator,
    #uvg_riverbank_bd_evaluator,
    #uvg_sunbath_bd_evaluator,
    #uvg_twilight_bd_evaluator,

    uvg_beauty_bi_evaluator,
    uvg_bosphorus_bi_evaluator,
    uvg_honeybee_bi_evaluator,
    uvg_jockey_bi_evaluator,
    uvg_readysetgo_bi_evaluator,
    uvg_shakendry_bi_evaluator,
    uvg_yachtride_bi_evaluator,
    uvg_cityalley_bi_evaluator,
    uvg_flowerfocus_bi_evaluator,
    uvg_flowerkids_bi_evaluator,
    uvg_flowerpan_bi_evaluator,
    uvg_lips_bi_evaluator,
    uvg_racenight_bi_evaluator,
    uvg_riverbank_bi_evaluator,
    uvg_sunbath_bi_evaluator,
    uvg_twilight_bi_evaluator,
]
