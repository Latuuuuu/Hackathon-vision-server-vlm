"""LocateAnything boxes, optionally refined by a single-image SAM-2 mask."""
import io
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.core import valid_box


@dataclass
class Detection:
    bbox: list                 # [x1, y1, x2, y2] int, x2/y2 exclusive
    score: float = -1.0        # SAM predicted IoU; -1 when no mask was computed
    num_candidates: int = 1
    mask_png: bytes = None     # single-channel 0/255 PNG cropped to bbox
    timings: dict = field(default_factory=dict)


def to_pixel_box(box, width, height):
    """Float xyxy -> int xyxy covering the box, x2/y2 exclusive, clipped to the image."""
    x1, y1, x2, y2 = valid_box(box, width, height)
    return [int(math.floor(x1)), int(math.floor(y1)),
            min(width, int(math.floor(x2)) + 1), min(height, int(math.floor(y2)) + 1)]


def encode_mask_crop(mask, bbox):
    x1, y1, x2, y2 = bbox
    crop = np.where(mask[y1:y2, x1:x2], 255, 0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(crop, mode='L').save(stream, format='PNG')
    return stream.getvalue()


class MaskPredictor:
    """SAM-2.1 image predictor; no video memory, every request is independent."""
    def __init__(self):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.device = os.getenv('SAM_DEVICE', 'cpu')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('SAM_DEVICE must be cpu or cuda (HIP uses cuda API)')
        if self.device == 'cuda':
            from app.gpu_check import check_gpu
            check_gpu()
        torch.set_num_threads(int(os.getenv('CPU_THREADS', '6')))
        ckpt = os.getenv('SAM_CHECKPOINT', '/models/sam2.1_hiera_tiny.pt')
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f'Missing SAM checkpoint: {ckpt}')
        size = int(os.getenv('VLM_SAM_IMAGE_SIZE', '1024'))
        if size not in (384, 512, 768, 1024):
            raise ValueError('VLM_SAM_IMAGE_SIZE must be 384, 512, 768 or 1024')
        model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', ckpt, device=self.device,
                           apply_postprocessing=False,
                           hydra_overrides_extra=[f'++model.image_size={size}'])
        self.predictor = SAM2ImagePredictor(model)
        # The predictor hardcodes backbone feature sizes for 1024 input.
        self.predictor._bb_feat_sizes = [(size // 4,) * 2, (size // 8,) * 2, (size // 16,) * 2]
        self.image_size = size

    def context(self):
        return self.torch.autocast('cuda', dtype=self.torch.float16) if self.device == 'cuda' else nullcontext()

    def best(self, rgb, boxes):
        """-> (index of best box, bool mask HxW, score)."""
        with self.torch.inference_mode(), self.context():
            self.predictor.set_image(rgb)
            masks, scores, _ = self.predictor.predict(box=np.asarray(boxes, dtype=np.float32),
                                                      multimask_output=False)
        masks = np.asarray(masks).reshape(len(boxes), *rgb.shape[:2])
        scores = np.asarray(scores, dtype=np.float32).reshape(len(boxes))
        index = int(np.argmax(scores))
        return index, masks[index] > 0, float(scores[index])


class TargetFinder:
    def __init__(self, return_mask=False):
        from app.models import Locator
        self.locator = Locator()
        self.masker = MaskPredictor() if return_mask else None

    @property
    def info(self):
        return dict(return_mask=self.masker is not None,
                    sam_image_size=self.masker.image_size if self.masker else None,
                    la_mode=os.getenv('LA_MODE', 'slow'))

    def find(self, jpeg, text):
        """-> Detection, or None when Locate returns no valid box."""
        timings = {}
        start = time.monotonic()
        boxes, _raw = self.locator.locate(jpeg, text)
        timings['locate_ms'] = (time.monotonic() - start) * 1000
        if not boxes:
            return None
        image = Image.open(io.BytesIO(jpeg)).convert('RGB')
        width, height = image.size
        if self.masker is None:
            # Locate exposes no confidence; keep its first box.
            return Detection(to_pixel_box(boxes[0], width, height), num_candidates=len(boxes), timings=timings)
        start = time.monotonic()
        index, mask, score = self.masker.best(np.asarray(image), boxes)
        timings['sam_ms'] = (time.monotonic() - start) * 1000
        bbox = to_pixel_box(boxes[index], width, height)
        return Detection(bbox, score, len(boxes), encode_mask_crop(mask, bbox), timings)
