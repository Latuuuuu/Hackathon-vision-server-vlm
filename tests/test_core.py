import unittest
import numpy as np
from app.core import Frame, history_sample, prune_state, tint, valid_box


class Invariants(unittest.TestCase):
    def test_only_mask_pixels_change(self):
        image = np.full((3, 4, 3), 100, dtype=np.uint8)
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 2] = True
        result = tint(image, mask)
        np.testing.assert_array_equal(result[~mask], image[~mask])
        self.assertFalse(np.array_equal(result[mask], image[mask]))
        self.assertTrue((image == 100).all())

    def test_replay_never_crosses_camera_session(self):
        frames = [Frame(i, 1 if i < 10 else 2, float(i), b'') for i in range(20)]
        source = frames[2]
        replay = history_sample(frames, source, 3)
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay[-1].seq, 9)
        self.assertTrue(all(f.seq > source.seq and f.epoch == source.epoch for f in replay))

    def test_memory_is_bounded_and_anchor_survives(self):
        anchor = object()
        outputs = {'cond_frame_outputs': {0: anchor}, 'non_cond_frame_outputs': dict.fromkeys(range(1, 101))}
        state = {'output_dict_per_obj': {0: outputs}, 'frames_tracked_per_obj': {0: dict.fromkeys(range(101))},
                 'images': dict.fromkeys(range(101))}
        prune_state(state, 100, 32)
        self.assertIs(outputs['cond_frame_outputs'][0], anchor)
        self.assertEqual(len(outputs['non_cond_frame_outputs']), 32)
        self.assertEqual(list(state['images']), [100])

    def test_invalid_boxes_rejected(self):
        for box in ([1, 2, 1, 5], [0, 0, float('nan'), 4], [1, 2]):
            with self.assertRaises(ValueError):
                valid_box(box, 640, 480)
        self.assertEqual(valid_box([-2, -5, 700, 600], 640, 480), [0., 0., 639., 479.])


if __name__ == '__main__':
    unittest.main()
