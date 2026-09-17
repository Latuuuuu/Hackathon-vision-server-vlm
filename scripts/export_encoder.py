"""Export static forward_image; compare ONNX CPU against PyTorch."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from npu.contract import CONTRACT, sha256

class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, image):
        out = self.model.forward_image(image)
        return (out['vision_features'], *out['backbone_fpn'], *out['vision_pos_enc'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/models/sam2.1_hiera_tiny.pt')
    parser.add_argument('--output', default='/output/encoder')
    parser.add_argument('--image', help='Optional real frame for export/parity')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.manual_seed(0)
    model = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml', args.checkpoint,
                                      device='cpu', apply_postprocessing=False)
    wrapper = Encoder(model).eval()
    if args.image:
        rgb = np.asarray(Image.open(args.image).convert('RGB').resize((1024, 1024)), dtype=np.float32) / 255
        x = torch.from_numpy(((rgb - np.array([.485,.456,.406], np.float32)) /
                              np.array([.229,.224,.225], np.float32)).transpose(2,0,1).copy())[None]
    else:
        x = torch.randn(1, 3, 1024, 1024)
    path = root / 'encoder.onnx'
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = wrapper(x)
        levels = (len(expected) - 1) // 2
        names = ['vision_features'] + [f'fpn_{i}' for i in range(levels)] + [f'pos_{i}' for i in range(levels)]
        for module in model.modules():
            if isinstance(getattr(module, 'cache', None), dict):
                module.cache.clear()
        torch.onnx.export(wrapper, x, str(path), input_names=['image'], output_names=names,
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(path))
    actual = ort.InferenceSession(str(path), providers=['CPUExecutionProvider']).run(names, {'image':x.numpy()})
    errors = {}
    for name, reference, value in zip(names, expected, actual):
        reference = reference.numpy()
        errors[name] = float(np.max(np.abs(reference-value)))
        np.testing.assert_allclose(value, reference, rtol=1e-3, atol=2e-3)
    np.save(root / 'sample.npy', x.numpy())
    np.savez(root / 'reference.npz', **{name:y.numpy() for name,y in zip(names,expected)})
    manifest = dict(contract=CONTRACT, checkpoint_sha256=sha256(args.checkpoint),
                    onnx_sha256=sha256(path), levels=levels, input_shape=list(x.shape),
                    outputs=[dict(name=name,shape=list(value.shape)) for name,value in zip(names,actual)],
                    cpu_parity_max_abs=errors, torch=torch.__version__, opset=17)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('EXPORT_CPU_PARITY_PASS', json.dumps(errors), flush=True)
    print('NPU compilation, accuracy and speed are not yet verified.', flush=True)

if __name__ == '__main__':
    main()
