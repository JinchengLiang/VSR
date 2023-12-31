# VSR

## Files Location
### Installation
All related information is available at [mmagic/README.md](https://github.com/JinchengLiang/VSR/blob/main/mmagic/README.md).

### Dadasets
All datasets are in [toos/dataset_converters](https://github.com/JinchengLiang/VSR/tree/main/tools/dataset_converters).

### Analysis
All analysis tools are in [tools/analysis_tools](https://github.com/JinchengLiang/VSR/tree/Shaomin/tools/analysis_tools).

### Outputs
All outputs are saved in [outputs](https://github.com/JinchengLiang/VSR/tree/main/outputs).

## Quick Start
We write IconVSR+ model in [iconvsr_net.py](https://github.com/JinchengLiang/VSR/tree/main/mmagic/models/editors/iconvsr) to avoid customer model registry problem.  
To use the wavelet-based loss, modify the `type` of `pixel_loss` in the config to `WCPatchLoss`.

### Train
You can use the following commands to train a model with cpu or single/multiple GPUs.
```
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/iconvsr/iconvsr_2xb4_reds4.py

# single-gpu train
python tools/train.py configs/iconvsr/iconvsr_2xb4_reds4.py

# multi-gpu train
./tools/dist_train.sh configs/iconvsr/iconvsr_2xb4_reds4.py 8
```

### Test
You can use the following commands to test a model with cpu or single/multiple GPUs.

Download [best_PSNR.pth](https://drive.google.com/file/d/1JXyJEICXPT2AGG-F8PsS7g0bTkq2PO4v/view?usp=drive_link) first, and then save this path in directory `work_dirs/iconvsr_2xb4_reds4`.
```
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/iconvsr/iconvsr_2xb4_reds4.py work_dirs/iconvsr_2xb4_reds4/best_PSNR_iter_29000.pth

# single-gpu test
python tools/test.py configs/iconvsr/iconvsr_2xb4_reds4.py work_dirs/iconvsr_2xb4_reds4/best_PSNR_iter_29000.pth

# multi-gpu test
./tools/dist_test.sh configs/iconvsr/iconvsr_2xb4_reds4.py work_dirs/iconvsr_2xb4_reds4/best_PSNR_iter_29000.pth 8
```


## Citation
```
@misc{mmagic2023,
    title = {{MMagic}: {OpenMMLab} Multimodal Advanced, Generative, and Intelligent Creation Toolbox},
    author = {{MMagic Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmagic}},
    year = {2023}
}
```
```
@misc{mmediting2022,
    title = {{MMEditing}: {OpenMMLab} Image and Video Editing Toolbox},
    author = {{MMEditing Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year = {2022}
}
```
