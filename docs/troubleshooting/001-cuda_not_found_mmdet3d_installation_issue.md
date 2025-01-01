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
```