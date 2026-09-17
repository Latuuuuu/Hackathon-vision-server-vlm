import io
import json
import threading
import time
import unittest

import numpy as np
import zmq
from PIL import Image

from vlm_server import protocol as P
from vlm_server.http_api import create_app
from vlm_server.pipeline import Detection, TargetFinder, encode_mask_crop, to_pixel_box
from vlm_server.query import QueryStore
from vlm_server.zmq_server import Stats, Worker, ZmqServer


def jpeg(width=64, height=48):
    stream = io.BytesIO()
    Image.new('RGB', (width, height)).save(stream, format='JPEG')
    return stream.getvalue()


def header(request_id, kind='detect'):
    return json.dumps(dict(protocol_version=1, type=kind, request_id=request_id)).encode()


class Protocol(unittest.TestCase):
    def test_ping_and_detect(self):
        self.assertIsNone(P.parse_request([header(1, 'ping')])[1])
        self.assertEqual(P.parse_request([header(2), b'x'])[1], b'x')

    def test_errors_keep_request_id_when_possible(self):
        with self.assertRaises(P.ProtocolError) as ctx:
            P.parse_request([header(7)])
        self.assertEqual(ctx.exception.request_id, 7)
        for frames in ([b'not json'], [b'[]'], [json.dumps(dict(request_id=-1)).encode()], []):
            with self.assertRaises(P.ProtocolError) as ctx:
                P.parse_request(frames)
            self.assertIsNone(ctx.exception.request_id)


class Query(unittest.TestCase):
    def test_version_changes_only_on_new_text(self):
        store = QueryStore()
        self.assertEqual(store.get(), (None, 0))
        self.assertEqual(store.set(' cup '), 1)
        self.assertEqual(store.set('cup'), 1)
        self.assertEqual(store.set('red cup'), 2)
        self.assertEqual(store.clear(), 3)
        self.assertEqual(store.clear(), 3)
        with self.assertRaises(ValueError):
            store.set('  ')

    def test_http_api(self):
        store = QueryStore()
        client = create_app(store, Stats()).test_client()
        self.assertEqual(client.post('/api/query', json=dict(text='cup')).get_json()['query_version'], 1)
        self.assertEqual(client.post('/api/query', json={}).status_code, 400)
        self.assertEqual(client.delete('/api/query').get_json(), dict(text=None, query_version=2))


class Geometry(unittest.TestCase):
    def test_pixel_box_is_exclusive_and_clipped(self):
        self.assertEqual(to_pixel_box([10.4, 5.6, 20.2, 30.9], 64, 48), [10, 5, 21, 31])
        self.assertEqual(to_pixel_box([-5, -5, 100, 100], 64, 48), [0, 0, 64, 48])

    def test_oversized_image_rejected_before_locate(self):
        finder = TargetFinder.__new__(TargetFinder)
        with self.assertRaises(ValueError):
            finder.find(jpeg(2000, 100), 'cup')

    def test_mask_crop_matches_bbox(self):
        mask = np.zeros((48, 64), bool)
        mask[10:20, 30:40] = True
        png = encode_mask_crop(mask, [30, 10, 40, 20])
        crop = np.asarray(Image.open(io.BytesIO(png)))
        self.assertEqual(crop.shape, (10, 10))
        self.assertTrue((crop == 255).all())


class SlowFinder:
    def __init__(self, delay):
        self.delay, self.calls = delay, []

    def find(self, data, text):
        self.calls.append(text)
        time.sleep(self.delay)
        if text == 'nothing':
            return None
        if text == 'boom':
            raise RuntimeError('boom')
        return Detection([1, 2, 11, 22], 0.5, 3, encode_mask_crop(np.ones((48, 64), bool), [1, 2, 11, 22]))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.queries, self.stats = QueryStore(), Stats()
        self.server = ZmqServer('tcp://127.0.0.1:*', self.queries, self.stats, self.context)
        self.finder = SlowFinder(0.3)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=Worker(self.server, self.finder).run_forever, daemon=True).start()
        self.assertTrue(self.server.bound.wait(2))
        self.client = self.context.socket(zmq.DEALER)
        self.client.setsockopt(zmq.LINGER, 0)
        self.client.connect(self.server.endpoint)

    def tearDown(self):
        self.server.stop.set()
        self.client.close(0)
        time.sleep(0.05)
        self.context.term()

    def recv(self, timeout=2.0):
        self.assertTrue(self.client.poll(int(timeout * 1000)), 'no reply')
        frames = self.client.recv_multipart()
        return json.loads(frames[0]), frames[1:]

    def detect(self, request_id):
        self.client.send_multipart([header(request_id), jpeg()])

    def test_immediate_statuses(self):
        self.client.send_multipart([header(1, 'ping')])
        self.assertEqual(self.recv()[0]['status'], P.PONG)
        self.detect(2)
        self.assertEqual(self.recv()[0]['status'], P.NO_QUERY)
        self.queries.set('cup')
        self.detect(3)
        reply = self.recv()[0]
        self.assertEqual((reply['request_id'], reply['status'], reply['error']), (3, P.ERROR, 'model loading'))
        self.client.send_multipart([header(4)])
        self.assertEqual(self.recv()[0]['status'], P.ERROR)

    def test_found_not_found_error(self):
        self.stats.model_ready = True
        for text, status in (('cup', P.FOUND), ('nothing', P.NOT_FOUND), ('boom', P.ERROR)):
            self.queries.set(text)
            self.detect(10)
            reply, extra = self.recv()
            self.assertEqual(reply['status'], status)
            if status == P.FOUND:
                self.assertEqual((reply['bbox'], reply['num_candidates'], reply['has_mask']), ([1, 2, 11, 22], 3, True))
                self.assertEqual(np.asarray(Image.open(io.BytesIO(extra[0]))).shape, (20, 10))
                self.assertGreaterEqual(reply['server_ms'], 250)
            else:
                self.assertEqual(extra, [])

    def test_only_newest_pending_request_is_processed(self):
        self.stats.model_ready = True
        self.queries.set('cup')
        self.detect(1)
        time.sleep(0.1)  # worker is now busy with request 1
        for request_id in (2, 3, 4):
            self.detect(request_id)
        # Pings are answered while inference runs.
        self.client.send_multipart([header(99, 'ping')])
        self.assertEqual(self.recv(0.2)[0]['request_id'], 99)
        ids = [self.recv()[0]['request_id'], self.recv()[0]['request_id']]
        self.assertEqual(ids, [1, 4])
        self.assertFalse(self.client.poll(500))
        self.assertEqual(self.server.slot.dropped, 2)


if __name__ == '__main__':
    unittest.main()
