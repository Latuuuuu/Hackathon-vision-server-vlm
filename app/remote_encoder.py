import io
import json
import os
import types
import urllib.request
import numpy as np
from npu.contract import CONTRACT, sha256, validate_arrays

def attach_encoder(predictor, url):
    import torch
    with urllib.request.urlopen(url.rstrip('/')+'/health', timeout=10) as response:
        info = json.load(response)
    manifest = info['manifest']
    if not info.get('ready') or manifest['contract'] != CONTRACT or info.get('npu_nodes',0) <= 0:
        raise RuntimeError('NPU encoder has no verified NPU assignment')
    if manifest['checkpoint_sha256'] != sha256(os.environ['SAM_CHECKPOINT']):
        raise RuntimeError('NPU encoder checkpoint differs from tracking checkpoint')
    device = predictor.device
    def forward_image(self, image):
        buf = io.BytesIO()
        np.save(buf, image.detach().float().cpu().numpy(), allow_pickle=False)
        req = urllib.request.Request(url.rstrip('/')+'/encode', data=buf.getvalue(),
                                     headers={'Content-Type':'application/octet-stream'})
        with urllib.request.urlopen(req, timeout=float(os.getenv('NPU_REQUEST_TIMEOUT','120'))) as response:
            payload = response.read(128*1024*1024+1)
        if len(payload) > 128*1024*1024:
            raise RuntimeError('Encoder response too large')
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name:archive[name] for name in archive.files}
        validate_arrays(arrays, manifest['outputs'])
        tensors = {name:torch.from_numpy(value).to(device) for name,value in arrays.items()}
        levels = manifest['levels']
        return dict(vision_features=tensors['vision_features'],
                    backbone_fpn=[tensors[f'fpn_{i}'] for i in range(levels)],
                    vision_pos_enc=[tensors[f'pos_{i}'] for i in range(levels)])
    predictor.forward_image = types.MethodType(forward_image,predictor)
    predictor.image_encoder.cpu()
    return dict(kind='experimental_npu_encoder',npu_nodes=info['npu_nodes'],cpu_nodes=info.get('cpu_nodes'),
                url=url,onnx_sha256=manifest['onnx_sha256'])
