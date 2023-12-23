# VSR

## Installation
All related information is available at [mmgic/README.md](https://github.com/JinchengLiang/VSR/blob/main/mmagic/README.md).

## Dadasets
All datasets are in [toos/dataset_converters]([https://github.com/JinchengLiang/VSR/tree/main/tools](https://github.com/JinchengLiang/VSR/tree/main/tools/dataset_converters)).

## Quick Start
We write IconVSR+ model in [iconvsr_net.py](https://github.com/JinchengLiang/VSR/tree/main/mmagic/models/editors/iconvsr) to avoid customer model registry problem.

### Train
<details>
  <summary>Train Instructions</summary>
You can use the following commands to train a model with cpu or single/multiple GPUs.
```
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/iconvsr/iconvsr_2xb4_reds4.py

# single-gpu train
python tools/train.py configs/iconvsr/iconvsr_2xb4_reds4.py

# multi-gpu train
./tools/dist_train.sh configs/iconvsr/iconvsr_2xb4_reds4.py 8
```
  
</details>
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
```
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/iconvsr/iconvsr_2xb4_reds4.py work/work_dirs/iconvsr_2xb4_reds4/best_PSNR_iter_29000.pth

# single-gpu test
python tools/test.py configs/iconvsr/iconvsr_2xb4_reds4.pywork_dirs/iconvsr_2xb4_reds4/best_PSNR_iter_29000.pth

# multi-gpu test
./tools/dist_test.sh configs/iconvsr/iconvsr_2xb4_reds4.py work_dirs/iconvsr_2xb4_reds4/best_PSNR_iter_29000.pth 8
```

## Outputs

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
