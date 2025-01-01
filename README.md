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
```

2. Verify CUDA installation:
```bash
# Check NVIDIA driver and CUDA version
nvidia-smi

# Alternative CUDA version check
nvcc --version
```

3. Install PyTorch with CUDA support:
```bash
# Install PyTorch and related libraries (CUDA 11.1)
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

4. Install OpenMMLab libraries:
```bash
# Install MMCV with CUDA support
pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html

# Install other OpenMMLab dependencies
pip install mmdet==2.16.0
pip install mmsegmentation==0.17.0

# Install MMDetection3D
cd mmdetection3d
pip install -v -e .
```

5. Verify installation:
```bash
# Verify CUDA recognition in PyTorch
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"

# Verify OpenMMLab versions
python -c "import mmcv; import mmdet; import mmseg; import mmdet3d; print(f'MMCV: {mmcv.__version__}\nMMDet: {mmdet.__version__}\nMMSeg: {mmseg.__version__}\nMMDet3D: {mmdet3d.__version__}')"
```

Note: If you encounter CUDA-related errors, ensure that:
- CUDA 11.1 is properly installed on your system
- Your NVIDIA drivers are compatible with CUDA 11.1
- Environment variables CUDA_HOME or CUDA_PATH are set correctly

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