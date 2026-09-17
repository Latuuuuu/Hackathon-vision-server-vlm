import ctypes
import json
import os
import platform
import subprocess
from pathlib import Path

import pyrealsense2 as rs
import torch
from app.gpu_check import check_gpu

failed = False

print('Platform:', platform.platform())
print('Torch:', torch.__version__, 'HIP:', torch.version.hip, 'requested:', os.getenv('SAM_DEVICE'))
print('CUDA/HIP available:', torch.cuda.is_available())
if os.getenv('SAM_DEVICE') == 'cuda':
    try:
        print('GPU preflight:', json.dumps(check_gpu(), indent=2), flush=True)
    except Exception as exc:
        failed = True
        print('GPU KERNEL FAILED:', str(exc), flush=True)
for path in ['/dev/dri', '/dev/kfd', '/dev/bus/usb', os.getenv('LOCATE_MODEL', ''), os.getenv('SAM_CHECKPOINT', '')]:
    print(path, 'exists:', bool(path) and Path(path).exists())
    if not path or not Path(path).exists():
        if path != '/dev/kfd' or os.getenv('SAM_DEVICE') == 'cuda':
            failed = True
try:
    lib = ctypes.CDLL(os.environ['LOCATE_LIBRARY'])
    print('Locate C ABI:', lib.la_capi_abi_version())
except Exception as exc:
    failed = True
    print('LOCATE LIBRARY FAILED:', repr(exc))
devices = rs.context().query_devices()
print('RealSense:', [d.get_info(rs.camera_info.name) for d in devices])
failed |= len(devices) == 0
result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
print('Vulkan:', result.stdout, result.stderr)
print('SAM commit:', subprocess.run(['git', '-C', '/opt/sam2', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout)
print('Run smoke test to verify complete SAM kernels; this report alone is not a pass.')
raise SystemExit(1 if failed else 0)
