"""Verify the actual resident Locate C API without opening RealSense."""
import argparse
import json
import time
from pathlib import Path
from app.models import Locator

parser = argparse.ArgumentParser()
parser.add_argument('image',help='Image path inside container, e.g. /output/test.jpg')
parser.add_argument('--target',default='the cup')
args = parser.parse_args()
model = Locator()
start = time.monotonic()
boxes, raw = model.locate(Path(args.image).read_bytes(),args.target)
print(json.dumps(dict(seconds=time.monotonic()-start,boxes=boxes,raw=raw)),flush=True)
print('LOCATE_CALL_PASS; inspect boxes and Vulkan backend log for detection quality/device.',flush=True)
