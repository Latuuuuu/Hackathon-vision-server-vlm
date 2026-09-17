"""Configuration and optional stage timing for live SAM inference."""
import os
import time


def settings():
    size = int(os.getenv('SAM_IMAGE_SIZE', '512'))
    memories = int(os.getenv('SAM_MEMORY_FRAMES', '3'))
    pointers = int(os.getenv('SAM_OBJECT_POINTERS', '4'))
    if size not in (384, 512, 768, 1024):
        raise ValueError('SAM_IMAGE_SIZE must be 384, 512, 768 or 1024')
    if not 2 <= memories <= 7 or not 1 <= pointers <= 16:
        raise ValueError('Memory frames must be 2..7; pointers 1..16')
    return size, memories, pointers


class StageTimer:
    def __init__(self, torch, device, model):
        self.torch, self.device = torch, device
        self.enabled = os.getenv('SAM_PROFILE', '1') == '1'
        self.pending = []
        if self.enabled:
            for attr, name in [('forward_image', 'image_encoder_ms'),
                               ('_prepare_memory_conditioned_features', 'memory_attention_ms'),
                               ('_forward_sam_heads', 'mask_decoder_ms'),
                               ('_encode_new_memory', 'memory_encoder_ms')]:
                original = getattr(model, attr)
                setattr(model, attr, self.wrap(original, name))

    def wrap(self, original, name):
        def timed(*args, **kwargs):
            if self.device == 'cuda':
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                start.record()
                result = original(*args, **kwargs)
                end.record()
                self.pending.append((name, start, end))
            else:
                start = time.perf_counter()
                result = original(*args, **kwargs)
                self.pending.append((name, start, time.perf_counter()))
            return result
        return timed

    def begin(self):
        self.pending.clear()

    def finish(self):
        result = {}
        if self.pending and self.device == 'cuda':
            self.pending[-1][2].synchronize()
        for name, start, end in self.pending:
            ms = start.elapsed_time(end) if self.device == 'cuda' else (end-start)*1000
            result[name] = result.get(name, 0) + ms
        self.pending.clear()
        return result
