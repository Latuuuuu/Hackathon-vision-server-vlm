"""Longer steady-state benchmark; tiny smoke tests miss memory-attention growth."""
import argparse
import io
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from app.models import SegmentTracker

parser=argparse.ArgumentParser()
parser.add_argument('--frames',type=int,default=24)
parser.add_argument('--warmup',type=int,default=8)
parser.add_argument('--output',default='/output/benchmark-sam.json')
args=parser.parse_args()
if not 0 <= args.warmup < args.frames:
    parser.error('Require 0 <= warmup < frames')
model=SegmentTracker()
print(json.dumps(model.info,indent=2),flush=True)
rows=[]
for i in range(args.frames+1):
    image=Image.new('RGB',(640,480),(160,160,160))
    ImageDraw.Draw(image).ellipse((180+i%20,130,320+i%20,280),fill=(220,35,40))
    buf=io.BytesIO();image.save(buf,format='JPEG')
    start=time.perf_counter()
    mask=model.start(buf.getvalue(),[175,125,325,285]) if i==0 else model.step(buf.getvalue())
    ms=(time.perf_counter()-start)*1000
    assert mask.shape==(480,640) and mask.dtype==np.bool_
    row=dict(frame=i,step_ms=ms,mask_pixels=int(mask.sum()),stage_ms=model.last_timings)
    rows.append(row)
    print(json.dumps(row),flush=True)
values=[r['step_ms'] for r in rows[1+args.warmup:]]
result=dict(backend=model.info,frames=rows,warmup_excluded=args.warmup,
            steady_p50_ms=float(np.median(values)),steady_p95_ms=float(np.percentile(values,95)),
            steady_fps=1000/float(np.mean(values)),
            note='Synthetic model-only benchmark; not camera FPS or mask-quality validation')
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
Path(args.output).write_text(json.dumps(result,indent=2))
print('BENCHMARK_PASS',args.output,flush=True)
