"""Dependency-light helpers. All masks/boxes refer to their source frame."""
from dataclasses import dataclass
import numpy as np


def clear_tracking_metrics(stats):
    for key in ('output_captured','output_seq','output_epoch','completion_age_ms','sam_step_ms',
                'sam_init_ms','step_p50_ms','step_p95_ms','mask_pixels','depth_median_m','stage_ms',
                'initialization_source_seq'):
        stats[key] = None
    stats['tracking_fps'] = 0
    stats['skipped_camera_frames'] = 0


def latest_eligible(frame, epoch, last_seq):
    return frame is not None and frame.epoch == epoch and frame.seq > last_seq


@dataclass(frozen=True)
class Frame:
    seq: int
    epoch: int
    captured: float
    jpeg: bytes
    depth: object = None
    depth_scale: float = 0.001


def valid_box(box, width, height):
    b = np.asarray(box, dtype=np.float32)
    if b.shape != (4,) or not np.isfinite(b).all():
        raise ValueError('Invalid box')
    b[[0, 2]] = np.clip(b[[0, 2]], 0, width - 1)
    b[[1, 3]] = np.clip(b[[1, 3]], 0, height - 1)
    if b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError('Empty box')
    return b.tolist()


def tint(rgb, mask):
    result = rgb.copy()
    result[mask] = (rgb[mask].astype(np.float32) * .5 + np.array([40, 235, 125]) * .5).astype(np.uint8)
    return result


def prune_state(state, current, keep):
    cutoff = current - keep + 1
    for outputs in state['output_dict_per_obj'].values():
        for key in list(outputs['non_cond_frame_outputs']):
            if key < cutoff:
                del outputs['non_cond_frame_outputs'][key]
    for tracked in state['frames_tracked_per_obj'].values():
        for key in list(tracked):
            if key < cutoff:
                del tracked[key]
    # Preserve conditioning outputs; raw images are unnecessary after encoding.
    for key in list(state['images']):
        if key != current:
            del state['images'][key]


def history_sample(frames, source, limit):
    candidates = [f for f in frames if f.epoch == source.epoch and f.seq > source.seq]
    if not candidates or limit <= 0:
        return []
    indices = np.linspace(0, len(candidates) - 1, min(limit, len(candidates)), dtype=int)
    return [candidates[i] for i in indices]
