"""Library versions behind a run, saved with every training and evaluation."""
from __future__ import annotations

from importlib import metadata
import platform


def runtime_versions(device) -> dict:
    """All installed OpenCV packages are listed, because two packages that provide cv2 overwrite each other."""
    import cv2
    import numpy as np
    import torch
    import ultralytics
    installed = {dist.metadata['Name'].lower(): dist.version for dist in metadata.distributions() if dist.metadata['Name']}
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'cv2': cv2.__version__,
                'opencv_distributions': sorted(f'{name}=={version}' for name, version in installed.items()
                                               if name.startswith('opencv')),
                'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                'torchvision': installed.get('torchvision'), 'ultralytics': ultralytics.__version__}
    if str(device).lower() != 'cpu':
        versions.update(cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
    return versions
