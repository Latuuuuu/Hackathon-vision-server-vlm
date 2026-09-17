import ctypes as C
import io
import os
import tempfile
import time
from contextlib import nullcontext

import numpy as np
from PIL import Image

from .core import prune_state, valid_box


class Locator:
    def __init__(self):
        path = os.environ.get('LOCATE_MODEL', '/models/locate-anything-q8_0.gguf')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing LocateAnything model: {path}')
        self.lib = C.CDLL(os.environ['LOCATE_LIBRARY'])
        signatures = {
            'la_capi_load': ([C.c_char_p, C.c_int], C.c_void_p),
            'la_capi_free': ([C.c_void_p], None),
            'la_capi_locate_buffer': ([C.c_void_p, C.POINTER(C.c_ubyte), C.c_size_t, C.c_char_p, C.c_int], C.c_void_p),
            'la_capi_get_n_detections': ([C.c_void_p], C.c_int),
            'la_capi_get_detection_box': ([C.c_void_p, C.c_int, C.POINTER(C.c_float)], C.c_int),
            'la_capi_free_string': ([C.c_void_p], None),
            'la_capi_last_error': ([C.c_void_p], C.c_char_p),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.ctx = self.lib.la_capi_load(path.encode(), int(os.getenv('CPU_THREADS', '6')))
        if not self.ctx:
            raise RuntimeError('LocateAnything load failed; inspect Vulkan/model logs')

    def locate(self, jpeg, description):
        data = (C.c_ubyte * len(jpeg)).from_buffer_copy(jpeg)
        prompt = f'Locate all the instances that matches the following description: {description}.'
        mode = {'hybrid': 0, 'slow': 1, 'fast': 2}[os.getenv('LA_MODE', 'slow')]
        ptr = self.lib.la_capi_locate_buffer(self.ctx, data, len(jpeg), prompt.encode(), mode)
        if not ptr:
            raise RuntimeError(str(self.lib.la_capi_last_error(self.ctx)))
        try:
            raw = C.string_at(ptr).decode(errors='replace')
        finally:
            self.lib.la_capi_free_string(ptr)
        width, height = Image.open(io.BytesIO(jpeg)).size
        boxes = []
        for i in range(self.lib.la_capi_get_n_detections(self.ctx)):
            box = (C.c_float * 4)()
            if self.lib.la_capi_get_detection_box(self.ctx, i, box) == 0:
                try:
                    boxes.append(valid_box(list(box), width, height))
                except ValueError:
                    pass
        return boxes, raw


class SegmentTracker:
    """Bounded live adapter for the pinned official SAM2VideoPredictor.

    Each accepted camera image is a real SAM-2 video step (with memory encoder).
    Skipped camera frames are explicit; no flow/rectangle substitutes.
    """
    def __init__(self):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        self.torch = torch
        torch.set_num_threads(int(os.getenv('CPU_THREADS', '6')))
        self.device = os.getenv('SAM_DEVICE', 'cpu')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('SAM_DEVICE must be cpu or cuda (HIP uses cuda API)')
        if self.device == 'cuda':
            from .gpu_check import check_gpu
            self.gpu_probe = check_gpu()
        ckpt = os.getenv('SAM_CHECKPOINT', '/models/sam2.1_hiera_tiny.pt')
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f'Missing SAM checkpoint: {ckpt}. Run the models compose service first.')
        from .performance import settings, StageTimer
        size, memories, pointers = settings()
        if os.getenv('SAM_ENCODER_URL'):
            raise ValueError('V3 latency mode disables the NPU encoder. Remove SAM_ENCODER_URL.')
        self.predictor = build_sam2_video_predictor(
            'configs/sam2.1/sam2.1_hiera_t.yaml', ckpt,
            device=self.device, apply_postprocessing=False,
            hydra_overrides_extra=[f'++model.image_size={size}'])
        # Load checkpoint at original memory parameter shape, then retain a shorter
        # suffix of the learned temporal positions for a shorter inference horizon.
        original_memories = self.predictor.num_maskmem
        self.predictor.maskmem_tpos_enc = torch.nn.Parameter(
            self.predictor.maskmem_tpos_enc.detach()[original_memories-memories:].clone(),
            requires_grad=False)
        self.predictor.num_maskmem = memories
        self.predictor.max_obj_ptrs_in_encoder = pointers
        self.keep = max(memories + 2, pointers + 2)
        self.info = dict(device=self.device, torch=torch.__version__, hip=torch.version.hip,
                         gpu=torch.cuda.get_device_name(0) if self.device == 'cuda' else None,
                         precision='fp16 autocast' if self.device == 'cuda' else 'float32',
                         model='SAM-2.1 Hiera Tiny', memory_frames=self.keep)
        self.info.update(image_size=size, attention_memory_frames=memories,
                         object_pointers=pointers, profile=os.getenv('SAM_PROFILE', '1') == '1',
                         experimental_attention=os.getenv('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '0'))
        self.info['gpu_probe'] = getattr(self, 'gpu_probe', None)
        self.info['encoder_backend'] = 'pytorch'
        self.timer = StageTimer(torch, self.device, self.predictor)
        self.last_timings = {}
        self.state = None
        self.index = -1

    def context(self):
        return self.torch.autocast('cuda', dtype=self.torch.float16) if self.device == 'cuda' else nullcontext()

    def tensor(self, jpeg):
        size = self.predictor.image_size
        img = Image.open(io.BytesIO(jpeg)).convert('RGB').resize((size, size))
        array = np.asarray(img, dtype=np.float32) / 255.0
        array = (array - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        return self.torch.from_numpy(array.transpose(2, 0, 1).copy())

    def start(self, jpeg, box):
        self.state = None
        self.index = 0
        self.timer.begin()
        with self.torch.inference_mode(), self.context():
            # Let official init_state create its schema; only replace frame storage.
            with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
                with open(directory + '/00000.jpg', 'wb') as stream:
                    stream.write(jpeg)
                self.state = self.predictor.init_state(directory, offload_video_to_cpu=True,
                                                       offload_state_to_cpu=self.device == 'cpu')
            # init_state has already encoded this exact JPEG. Keep both its input
            # tensor and feature cache; clearing the cache duplicated encoder work.
            self.state['images'] = {0: self.state['images'][0]}
            self.predictor.add_new_points_or_box(self.state, frame_idx=0, obj_id=1,
                                                box=np.asarray(box, dtype=np.float32))
            return self._propagate()

    def step(self, jpeg):
        self.timer.begin()
        self.index += 1
        self.state['images'][self.index] = self.tensor(jpeg)
        self.state['num_frames'] = self.index + 1
        with self.torch.inference_mode(), self.context():
            return self._propagate()

    def _propagate(self):
        result = None
        for _, ids, logits in self.predictor.propagate_in_video(
                self.state, start_frame_idx=self.index, max_frame_num_to_track=0):
            if ids != [1] or not self.torch.isfinite(logits).all().item():
                raise RuntimeError('SAM output IDs/logits invalid')
            result = (logits[0, 0] > 0).cpu().numpy()
        if result is None:
            raise RuntimeError('SAM produced no frame')
        prune_state(self.state, self.index, self.keep)
        self.last_timings = self.timer.finish()
        return result
