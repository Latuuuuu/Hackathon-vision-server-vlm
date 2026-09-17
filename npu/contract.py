"""Wire contract for projected SAM image features."""
import hashlib
import numpy as np
CONTRACT = 'sam2.1-tiny-forward-image-v1'

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def validate_arrays(arrays, specs):
    if set(arrays) != {s['name'] for s in specs}:
        raise ValueError('Encoder output names mismatch')
    for spec in specs:
        value = arrays[spec['name']]
        if list(value.shape) != spec['shape'] or value.dtype != np.float32:
            raise ValueError(f'Encoder shape/dtype mismatch: {spec["name"]}')
        if not np.isfinite(value).all():
            raise ValueError('Encoder returned nonfinite values')

def device_counts(report):
    counts = {}
    for item in report.get('deviceStat', []):
        name = str(item.get('name', '')).upper()
        if name in ('NPU', 'CPU'):
            counts[name] = counts.get(name, 0) + int(item.get('nodeNum', 0))
    return counts
