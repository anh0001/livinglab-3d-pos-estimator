# Object DGCNN & DETR3D

This repository implements Object DGCNN ([paper](https://arxiv.org/abs/2110.06923)) and DETR3D ([paper](https://arxiv.org/abs/2110.06922)) for 3D object detection. The implementations are built on top of MMDetection3D.

## Features
- Implementation of Object DGCNN for 3D object detection using dynamic graphs
- Implementation of DETR3D for multi-view 3D object detection
- Support for multiple backbones including ResNet101 and VoVNet
- Integration with MMDetection3D ecosystem

## Prerequisites

The following OpenMMLab packages are required:
- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

## Installation

1. Create and activate conda environment:
```bash
conda create --prefix ./detr3d_env python=3.8 -y
conda activate ./detr3d_env

# Install PyTorch
pip install --upgrade pip
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

2. Install MMCV using MIM (OpenMMLab Model Installer):
```bash
pip install openmim fsspec
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
mim install mmdet3d
```

# Data Preparation

This section describes how to prepare datasets for training and evaluation.

## NuScenes Dataset

### Download and Extract
1. Create the data directory:
```bash
mkdir -p data/nuscenes
cd data/nuscenes
```

2. Download the NuScenes dataset:
   - Full dataset: Download from [NuScenes official website](https://www.nuscenes.org/download)
   - Mini dataset (for testing): 
     ```bash
     wget https://xxx.cloudfront.net/public/v1.0/v1.0-mini.tgz
     ```

3. Extract the downloaded files:
```bash
# For full dataset
tar -xzf v1.0-trainval.tgz
tar -xzf v1.0-test.tgz
# For mini dataset
tar -xzf v1.0-mini.tgz
```

### Directory Structure
After extraction, ensure your directory structure looks like this:
```
data/nuscenes
├── maps
├── samples
├── sweeps
├── v1.0-mini
│   ├── attribute.json
│   ├── category.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── sample.json
│   └── scene.json
└── v1.0-trainval
    ├── attribute.json
    ├── category.json
    ├── instance.json
    ├── log.json
    ├── map.json
    ├── sample_annotation.json
    ├── sample_data.json
    ├── sample.json
    └── scene.json
```

### Create Symbolic Link
Create a symbolic link to the dataset in the MMDetection3D directory:
```bash
ln -s ../../data/nuscenes ./mmdetection3d/data/nuscenes
```

### Data Processing
1. Install required dependencies:
```bash
pip install nuscenes-devkit
```

2. Create the data info files:
```bash
cd mmdetection3d
python ./tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini
```

### Validation
To verify your data preparation:
```bash
# Check the symbolic link
ls -l ./mmdetection3d/data/nuscenes

# Verify data info files exist
ls ./mmdetection3d/data/nuscenes/nuscenes_infos_*.pkl

# Optional: Preview sample data
python tools/misc/browse_dataset.py \
    configs/_base_/datasets/nus-3d.py \
    --output-dir ./data_preview
```

## Common Issues and Solutions

1. **Permission Denied**: If you encounter permission issues during download or extraction:
```bash
chmod +x tools/create_data.py
sudo chown -R $USER:$USER data/nuscenes
```

2. **Disk Space**: Ensure you have sufficient disk space:
   - Full dataset: ~400GB
   - Mini dataset: ~5GB

3. **Download Interruption**: If download is interrupted, use `wget -c` to resume:
```bash
wget -c https://xxx.cloudfront.net/public/v1.0/v1.0-mini.tgz
```

## Additional Datasets

For other supported datasets (KITTI, Waymo, etc.), please refer to the [MMDetection3D Data Preparation Guide](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/index.html).

## Training

1. Download the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) and place them in the `pretrained/` directory.

2. Start training:
```bash
# Example: Train Object-DGCNN with pillar backbone on 8 GPUs
tools/dist_train.sh projects/configs/obj_dgcnn/pillar.py 8
```

## Pretrained Models and Results

### DETR3D Models

| Backbone | mAP | NDS | Config | Download |
|:--------:|:---:|:---:|:------:|:--------:|
| ResNet101 w/ DCN | 34.7 | 42.2 | [config](./projects/configs/detr3d/detr3d_res101_gridmask.py) | [model](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) \| [log](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XjrdGUZ2Qbykid/view?usp=sharing) |
| ResNet101 w/ DCN + CBGS | 34.9 | 43.4 | [config](./projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py) | [model](https://drive.google.com/file/d/1sXPFiA18K9OMh48wkk9dF1MxvBDUCj2t/view?usp=sharing) \| [log](https://drive.google.com/file/d/1NJNggvFGqA423usKanqbsZVE_CzF4ltT/view?usp=sharing) |
| VoVNet (trainval) | 41.2 | 47.9 | [config](./projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py) | [model](https://drive.google.com/file/d/1d5FaqoBdUH6dQC3hBKEZLcqbvWK0p9Zv/view?usp=sharing) \| [log](https://drive.google.com/file/d/1ONEMm_2W9MZAutjQk1UzaqRywz5PMk3p/view?usp=sharing) |

### Object DGCNN Models

| Backbone | mAP | NDS | Config | Download |
|:--------:|:---:|:---:|:------:|:--------:|
| Pillar | 53.2 | 62.8 | [config](./projects/configs/obj_dgcnn/pillar.py) | [model](https://drive.google.com/file/d/1nd6-PPgdb2b2Bi3W8XPsXPIo2aXn5SO8/view?usp=sharing) \| [log](https://drive.google.com/file/d/1A98dWp7SBOdMpo1fHtirwfARvpE38KOn/view?usp=sharing) |
| Voxel | 58.6 | 66.0 | [config](./projects/configs/obj_dgcnn/voxel.py) | [model](https://drive.google.com/file/d/1zwUue39W0cAP6lrPxC1Dbq_gqWoSiJUX/view?usp=sharing) \| [log](https://drive.google.com/file/d/1pjRMW2ffYdtL_vOYGFcyg4xJImbT7M2p/view?usp=sharing) |

## Evaluation

To evaluate a pretrained model:
```bash
tools/dist_test.sh projects/configs/obj_dgcnn/pillar_cosine.py /path/to/checkpoint 8 --eval=bbox
```

## Citation

If you find this work useful, please cite our papers:

```bibtex
@inproceedings{obj-dgcnn,
   title={Object DGCNN: 3D Object Detection using Dynamic Graphs},
   author={Wang, Yue and Solomon, Justin M.},
   booktitle={2021 Conference on Neural Information Processing Systems ({NeurIPS})},
   year={2021}
}

@inproceedings{detr3d,
   title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
   author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle={The Conference on Robot Learning ({CoRL})},
   year={2021}
}
```

## Acknowledgements

This project is built on top of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). We thank the OpenMMLab team for their excellent work.