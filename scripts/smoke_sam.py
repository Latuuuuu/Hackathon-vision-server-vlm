"""Real model smoke test, no camera or Locate model required. Not a quality benchmark."""
import argparse
import io
import json
import time
from PIL import Image, ImageDraw
from app.models import SegmentTracker

parser = argparse.ArgumentParser()
parser.add_argument('--frames', type=int, default=3)
args = parser.parse_args()
if args.frames < 2:
    parser.error('--frames must be >= 2')
model = SegmentTracker()
print(json.dumps(model.info, indent=2), flush=True)
for i in range(args.frames):
    image = Image.new('RGB', (640, 480), (160, 160, 160))
    draw = ImageDraw.Draw(image)
    offset = i % 30
    draw.ellipse((180+offset, 130, 320+offset, 280), fill=(220, 35, 40))
    stream = io.BytesIO()
    image.save(stream, format='JPEG')
    start = time.monotonic()
    mask = model.start(stream.getvalue(), [175, 125, 325, 285]) if i == 0 else model.step(stream.getvalue())
    assert mask.shape == (480, 640)
    assert mask.dtype.kind == 'b'
    assert len(model.state['images']) == 1
    assert len(model.state['output_dict_per_obj'][0]['non_cond_frame_outputs']) <= model.keep
    print(json.dumps(dict(frame=i, seconds=time.monotonic()-start, pixels=int(mask.sum()))), flush=True)
print('PASS: real SAM initialization, propagation, output shape and storage bounds. Synthetic mask accuracy is not asserted.')
