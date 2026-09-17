"""Experimental Vitis AI service. Refuses CPU-only assignment."""
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, request
from waitress import serve
from .contract import CONTRACT, device_counts, sha256, validate_arrays

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
session_lock = threading.Lock()

def main():
    root = Path(os.getenv('ENCODER_DIR','/encoder'))
    manifest = json.loads((root/'manifest.json').read_text())
    model = root/'encoder.onnx'
    if manifest['contract'] != CONTRACT or sha256(model) != manifest['onnx_sha256']:
        raise RuntimeError('Encoder manifest/model mismatch')
    if 'VitisAIExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('AMD Ryzen AI ONNX Runtime required; generic onnxruntime is insufficient')
    # Unique assignment report prevents acceptance of a stale prior model report.
    report_name = 'assignment-'+uuid.uuid4().hex+'.json'
    os.environ['XLNX_ONNX_EP_REPORT_FILE'] = report_name
    cache = Path('/cache')
    cache.mkdir(parents=True,exist_ok=True)
    options = {'config_file':'/app/npu/bf16.json','cache_dir':str(cache),
               'cache_key':manifest['onnx_sha256'], 'enable_cache_file_io_in_mem':'0'}
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 1
    sess_options.enable_profiling = True
    sess_options.profile_file_prefix = '/cache/ort-profile'
    print('Compiling/loading NPU encoder; first compilation may take time',flush=True)
    session = ort.InferenceSession(str(model),sess_options=sess_options,
                                   providers=['VitisAIExecutionProvider','CPUExecutionProvider'],
                                   provider_options=[options,{}])
    session.disable_fallback()
    names = [s['name'] for s in manifest['outputs']]
    sample = np.load(root/'sample.npy',allow_pickle=False)
    with np.load(root/'reference.npz',allow_pickle=False) as ref:
        references = {name:ref[name] for name in names}
    start = time.monotonic()
    values = session.run(names,{'image':sample})
    first_ms = (time.monotonic()-start)*1000
    arrays = dict(zip(names,values))
    validate_arrays(arrays,manifest['outputs'])
    reports = list(cache.rglob(report_name))
    if not reports:
        raise RuntimeError('No fresh Vitis assignment report. Inspect /cache and compiler logs; NPU use unverified.')
    counts = {}
    for path in reports:
        for name,count in device_counts(json.loads(path.read_text())).items():
            counts[name] = counts.get(name,0)+count
    if counts.get('NPU',0) <= 0:
        raise RuntimeError(f'No NPU nodes in assignment report: {counts}')
    error = {}
    for name,value in arrays.items():
        ref = references[name]
        relative_l2 = float(np.linalg.norm(value-ref)/max(float(np.linalg.norm(ref)),1e-12))
        error[name] = dict(relative_l2=relative_l2,max_abs=float(np.max(np.abs(value-ref))))
        if relative_l2 > .05:
            raise RuntimeError(f'Encoder BF16 error gate exceeded: {name} {relative_l2}')
    timings = []
    for _ in range(3):
        start = time.monotonic()
        session.run(names,{'image':sample})
        timings.append((time.monotonic()-start)*1000)
    profile = session.end_profiling()
    info = dict(ready=True,manifest=manifest,npu_nodes=counts['NPU'],cpu_nodes=counts.get('CPU',0),
                first_run_ms=first_ms,warm_p50_ms=float(np.median(timings)),errors=error,
                assignment_reports=[str(p) for p in reports],profile=profile,
                note='NPU partition verified; CPU fallback nodes may remain. Mask quality still requires testing.')
    (cache/'encoder-validation.json').write_text(json.dumps(info,indent=2))
    print('NPU_ENCODER_READY',json.dumps(info),flush=True)

    @app.get('/health')
    def health():
        return jsonify(info)

    @app.post('/encode')
    def encode():
        try:
            image = np.load(io.BytesIO(request.get_data()),allow_pickle=False)
            if not isinstance(image,np.ndarray) or image.dtype != np.float32 or list(image.shape) != manifest['input_shape']:
                return jsonify(error='Expected normalized float32 NCHW input'),400
            if not np.isfinite(image).all():
                return jsonify(error='Nonfinite input'),400
            with session_lock:
                values = session.run(names,{'image':image})
            arrays = dict(zip(names,values))
            validate_arrays(arrays,manifest['outputs'])
            buffer = io.BytesIO()
            np.savez(buffer,**arrays)
            return Response(buffer.getvalue(),mimetype='application/octet-stream')
        except Exception as exc:
            app.logger.exception('Encoder request failed')
            return jsonify(error=str(exc)),500
    serve(app,host='0.0.0.0',port=8090,threads=2)

if __name__ == '__main__':
    main()
