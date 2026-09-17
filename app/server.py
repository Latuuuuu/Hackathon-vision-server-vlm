import csv
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from waitress import serve

from .core import Frame, tint, clear_tracking_metrics, latest_eligible
from .models import Locator, SegmentTracker

app = Flask(__name__, static_folder='static')
cv2.setNumThreads(1)
lock = threading.RLock()
latest = None
generation = 0
command = None
candidates = None
tracked_jpeg = None
tracked_meta = None
stats = dict(phase='starting', camera='connecting', camera_fps=0, tracking_fps=0,
             model_ready=False, error=None, output_captured=None)
samples = deque(maxlen=120)
completed = deque(maxlen=60)
output = Path('/output')


def update(**values):
    with lock:
        stats.update(values)


def decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def encode(image):
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError('JPEG encode failed')
    return data.tobytes()


def capture():
    global latest, generation, candidates, tracked_jpeg, command, tracked_meta
    import pyrealsense2 as rs
    epoch = 0
    seq = 0
    while True:
        pipeline = rs.pipeline()
        try:
            config = rs.config()
            w, h, fps = (int(os.getenv(k, d)) for k, d in
                         [('CAMERA_WIDTH', '640'), ('CAMERA_HEIGHT', '480'), ('CAMERA_FPS', '30')])
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            profile = pipeline.start(config)
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)
            epoch += 1
            ticks = deque(maxlen=60)
            with lock:
                generation += 1
                command = candidates = tracked_jpeg = None
                tracked_meta = None
                clear_tracking_metrics(stats)
            update(camera='connected', camera_error=None, phase='idle' if stats['model_ready'] else 'loading_models')
            while True:
                frames = align.process(pipeline.wait_for_frames(5000))
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                captured = time.monotonic()
                seq += 1
                packet = Frame(seq, epoch, captured, encode(np.asanyarray(color.get_data())),
                               np.asanyarray(depth.get_data()).copy(), scale)
                ticks.append(captured)
                with lock:
                    latest = packet
                    stats['camera_fps'] = (len(ticks) - 1) / max(.001, ticks[-1] - ticks[0])
                    stats['camera_seq'] = seq
        except Exception as exc:
            with lock:
                latest = candidates = tracked_jpeg = command = None
                generation += 1
                tracked_meta = None
                clear_tracking_metrics(stats)
            update(camera='disconnected', camera_error=str(exc), phase='camera_disconnected', camera_fps=0,
                   output_captured=None, tracking_fps=0)
            time.sleep(2)
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass


def worker():
    global command, candidates, tracked_jpeg
    try:
        update(phase='loading_models')
        tracker = SegmentTracker()
        locator = Locator()
        update(model_ready=True, backend=tracker.info, phase='idle',
               locate_requested_device=os.getenv('LA_DEVICE', 'auto'),
               locate_backend_verification='請查看容器啟動日誌中的實際 backend／tensor 配置')
    except Exception as exc:
        update(phase='model_error', error=repr(exc))
        app.logger.exception('Model initialization failed')
        return
    active = None
    last_seq = -1
    active_epoch = None
    while True:
        with lock:
            job = command
            command = None
            gen = generation
            frame = latest
        if active != gen:
            active = None
            tracker.state = None
        try:
            if job and job[0] == 'detect':
                _, job_gen, description, source = job
                update(phase='locating', error=None)
                start = time.monotonic()
                boxes, raw = locator.locate(source.jpeg, description)
                elapsed = (time.monotonic() - start) * 1000
                with lock:
                    if job_gen != generation:
                        continue
                    stats.update(locate_ms=elapsed, boxes=boxes, locate_raw=raw, phase='choose' if boxes else 'not_found')
                    candidates = (job_gen, source, boxes)
                    if len(boxes) == 1:
                        command = ('select', job_gen, source, boxes[0])
                continue
            if job and job[0] == 'select':
                _, job_gen, source, box = job
                with lock:
                    if job_gen != generation or latest is None or source.epoch != latest.epoch:
                        continue
                    if time.monotonic() - source.captured > float(os.getenv('HISTORY_SECONDS', '30')):
                        stats.update(phase='expired', error='選取影格已過期，請重新定位')
                        continue
                    stats.update(phase='initializing', error=None)
                    samples.clear()
                    completed.clear()
                start = time.monotonic()
                mask = tracker.start(source.jpeg, box)
                active = job_gen
                active_epoch = source.epoch
                last_seq = source.seq
                # This source is necessarily old after Locate. Record initialization,
                # but publish only after processing a newly captured camera image.
                with lock:
                    if active == generation:
                        stats.update(sam_init_ms=(time.monotonic()-start)*1000,
                                     phase='waiting_live_frame', initialization_source_seq=source.seq)
                continue
            if active is None or frame is None:
                time.sleep(.02)
                continue
            phase = 'tracking'
            # Re-read at the last possible moment; never drain historical packets.
            with lock:
                frame = latest
            if not latest_eligible(frame, active_epoch, last_seq):
                time.sleep(.005)
                continue
            skipped = max(0, frame.seq - last_seq - 1)
            start = time.monotonic()
            mask = tracker.step(frame.jpeg)
            last_seq = frame.seq
            publish(frame, mask, start, active, phase, skipped, tracker.last_timings)
        except Exception as exc:
            active = None
            tracker.state = None
            with lock:
                # A stopped/replaced request must not overwrite the new UI state.
                if gen == generation:
                    stats.update(phase='tracking_error', error=repr(exc))
            app.logger.exception('Inference failed')


def publish(frame, mask, started, gen, phase, skipped, stage_ms=None):
    global tracked_jpeg, tracked_meta
    rgb = cv2.cvtColor(decode(frame.jpeg), cv2.COLOR_BGR2RGB)
    if mask.shape != rgb.shape[:2]:
        raise RuntimeError(f'Mask/image mismatch: {mask.shape}, {rgb.shape}')
    jpeg = encode(cv2.cvtColor(tint(rgb, mask), cv2.COLOR_RGB2BGR))
    now = time.monotonic()
    ms = (now - started) * 1000
    pixels = int(mask.sum())
    distance = None
    if frame.depth is not None and pixels:
        values = frame.depth[mask]
        values = values[values > 0]
        if len(values):
            distance = float(np.median(values) * frame.depth_scale)
    with lock:
        if gen != generation:
            return
        tracked_jpeg = jpeg
        tracked_meta = dict(seq=frame.seq, epoch=frame.epoch, captured=frame.captured, generation=gen)
        samples.append(ms)
        if phase == 'tracking':
            completed.append(now)
        rate = (len(completed) - 1) / (completed[-1] - completed[0]) if len(completed) > 1 else 0
        stats.update(phase=phase if pixels else 'empty_mask', output_captured=frame.captured,
                     output_seq=frame.seq, output_epoch=frame.epoch, tracking_fps=rate,
                     sam_step_ms=ms, step_p50_ms=float(np.median(samples)) if samples else None,
                     step_p95_ms=float(np.percentile(samples, 95)) if samples else None,
                     completion_age_ms=(now - frame.captured) * 1000, mask_pixels=pixels,
                     depth_median_m=distance,
                     skipped_camera_frames=stats.get('skipped_camera_frames', 0) + skipped)
        stats['stage_ms'] = stage_ms or {}
        row = dict(epoch=frame.epoch, frame=frame.seq, phase=phase, step_ms=round(ms, 2),
                   age_ms=round((now-frame.captured)*1000, 2), skipped=skipped, mask_pixels=pixels)
        for name in ('image_encoder_ms','memory_attention_ms','mask_decoder_ms','memory_encoder_ms'):
            row[name] = round((stage_ms or {}).get(name,0),2)
    # A bounded log; output is serialized by the inference worker.
    path = output / 'tracking-v3.csv'
    if path.exists() and path.stat().st_size > 10_000_000:
        path.replace(output / 'tracking-v3.previous.csv')
    new = not path.exists()
    with path.open('a') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


@app.get('/')
def index():
    return app.send_static_file('index.html')


@app.get('/api/status')
def status():
    with lock:
        result = dict(stats)
        result['generation'] = generation
        result['output_age_ms'] = (time.monotonic() - stats['output_captured']) * 1000 if stats['output_captured'] else None
    return jsonify(result)


@app.post('/api/detect')
def detect():
    global generation, command, candidates, tracked_jpeg, tracked_meta
    description = str((request.get_json(silent=True) or {}).get('target', '')).strip()
    if not description or len(description) > 300:
        return jsonify(error='請輸入 1–300 字的目標描述'), 400
    with lock:
        if latest is None or not stats['model_ready']:
            return jsonify(error='相機或模型尚未就緒'), 409
        generation += 1
        candidates = tracked_jpeg = None
        tracked_meta = None
        clear_tracking_metrics(stats)
        stats.update(phase='queued', boxes=[], output_captured=None, tracking_fps=0)
        command = ('detect', generation, description, latest)
    return jsonify(ok=True)


@app.post('/api/select')
def select():
    global command
    data = request.get_json(silent=True) or {}
    with lock:
        if candidates is None or data.get('generation') != candidates[0]:
            return jsonify(error='候選已更新，請重新選擇'), 409
        index = data.get('index')
        if not isinstance(index, int) or not 0 <= index < len(candidates[2]):
            return jsonify(error='無效候選'), 400
        command = ('select', candidates[0], candidates[1], candidates[2][index])
        stats['phase'] = 'queued_selection'
    return jsonify(ok=True)


@app.post('/api/stop')
def stop():
    global generation, command, candidates, tracked_jpeg, tracked_meta
    with lock:
        generation += 1
        command = candidates = tracked_jpeg = None
        tracked_meta = None
        clear_tracking_metrics(stats)
        stats.update(phase='idle', boxes=[], output_captured=None, tracking_fps=0)
    return jsonify(ok=True)


@app.get('/candidate.jpg')
def candidate_image():
    with lock:
        selection = candidates
    if selection is None:
        return '', 404
    image = decode(selection[1].jpeg)
    for i, box in enumerate(selection[2]):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(image, str(i+1), (x1, max(24, y1)), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 255), 2)
    return Response(encode(image), mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})


@app.get('/video/<kind>')
def video(kind):
    if kind not in ('raw', 'tracked'):
        return '', 404
    def frames():
        previous = None
        while True:
            with lock:
                jpeg = latest.jpeg if kind == 'raw' and latest else tracked_jpeg if kind == 'tracked' else None
            if jpeg is not None and jpeg is not previous:
                previous = jpeg
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            time.sleep(.03)
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})


@app.get('/frame/<kind>')
def snapshot(kind):
    if kind not in ('raw','tracked'):
        return '',404
    with lock:
        if kind == 'raw' and latest:
            jpeg = latest.jpeg
            meta = dict(seq=latest.seq,epoch=latest.epoch,captured=latest.captured,generation=generation)
        elif kind == 'tracked' and tracked_jpeg and tracked_meta:
            jpeg, meta = tracked_jpeg, dict(tracked_meta)
        else:
            return Response(status=204,headers={'Cache-Control':'no-store'})
    key = f'{meta["generation"]}:{meta["epoch"]}:{meta["seq"]}'
    if request.args.get('after') == key:
        return Response(status=204,headers={'Cache-Control':'no-store'})
    return Response(jpeg,mimetype='image/jpeg',headers={
        'Cache-Control':'no-store, no-cache, must-revalidate',
        'X-Frame-Key':key,'X-Generation':str(meta['generation']),
        'X-Frame-Age-Ms':str((time.monotonic()-meta['captured'])*1000)})


@app.get('/api/log.csv')
def log():
    path = output / 'tracking-v3.csv'
    if not path.exists():
        return jsonify(error='尚無推論紀錄'), 404
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    output.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=capture, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    serve(app, host='0.0.0.0', port=8080, threads=12)
