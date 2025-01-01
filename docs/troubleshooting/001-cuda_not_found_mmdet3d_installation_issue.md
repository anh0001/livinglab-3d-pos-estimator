I'll help create a clear GitHub issue description based on the error logs:

```markdown
# Failed to install mmdetection3d: CUDA not found during setup

## Description
While trying to install mmdetection3d using `pip install -v -e .`, the installation failed during the build process due to CUDA not being found on the system.

## Environment
- Python version: 3.8
- Operating System: Linux (Ubuntu)
- Installation method: pip install -v -e .
- Virtual environment: Yes (detr3d_env)

## Error Message
```
RuntimeError:
CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH
environment variable or add NVCC to your system PATH. The extension compilation will fail.
```

## Full Error Context
- The installation process started successfully and managed to install/verify all dependencies
- The error occurred during the `build_ext` phase when attempting to compile CUDA extensions
- Several CUDA-related components were attempting to compile:
  - sparse_conv_ext
  - iou3d_cuda
  - voxel_layer
  - roiaware_pool3d_ext
  - ball_query_ext
  - and others

## Steps to Reproduce
1. Created a virtual environment
2. Attempted to install mmdetection3d using `pip install -v -e .`

## Possible Solutions Needed
1. Set up CUDA environment variables (CUDA_HOME or CUDA_PATH)
2. Add NVCC to system PATH
3. Verify CUDA installation on the system

## Additional Notes
The system appears to be missing CUDA configuration. This needs to be addressed before the installation can proceed successfully.

## Solution

The key is to install the specific versions required for compatibility. These version requirements are based on the official DETR3D ResNet101 w/ DCN model log file ([available here](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XJrdGUZ2Qbykid/view?usp=sharing)).

### 1. Check Current CUDA Version

First, check if CUDA is installed and its version:

```bash
# Check NVIDIA driver and CUDA version
nvidia-smi

# Alternative CUDA version check
nvcc --version
```

### 2. Install Specific PyTorch Version with CUDA 11.1

```bash
# Install PyTorch and related libraries with specific versions
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install OpenMMLab Libraries with Specific Versions

```bash
# Install MMCV
pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html

# Install MMDetection
pip install mmdet==2.16.0

# Install MMSegmentation
pip install mmsegmentation==0.17.0

# Install MMDetection3D
cd mmdetection3d
pip install -v -e .
```

### 4. Verify Installation

```bash
# Verify CUDA recognition in PyTorch
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"

# Verify OpenMMLab versions
python -c "import mmcv; import mmdet; import mmseg; import mmdet3d; print(f'MMCV: {mmcv.__version__}\nMMDet: {mmdet.__version__}\nMMSeg: {mmseg.__version__}\nMMDet3D: {mmdet3d.__version__}')"
```

### Important Notes

- Make sure CUDA 11.1 is installed on your system
- The versions specified above are interdependent and tested to work together
- If you have other versions installed, remove them first:
  ```bash
  pip uninstall torch torchvision torchaudio mmcv-full mmdet mmsegmentation
  ```
- If you encounter any CUDA errors, ensure your NVIDIA drivers are compatible with CUDA 11.1

After ensuring CUDA compatibility, proceed with mmdetection3d installation:

```bash
pip install -v -e .
```