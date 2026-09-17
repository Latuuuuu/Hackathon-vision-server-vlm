import os
import unittest
from unittest.mock import patch
from app.core import Frame, clear_tracking_metrics, latest_eligible
from app.performance import settings


class LatencyChecks(unittest.TestCase):
    def test_stop_clears_stale_metrics(self):
        stats=dict(output_seq=3000,output_captured=42,completion_age_ms=1900,
                   sam_step_ms=1800,tracking_fps=.5,mask_pixels=10,camera_fps=30,
                   stage_ms={'memory_attention_ms':1000})
        clear_tracking_metrics(stats)
        self.assertIsNone(stats['output_seq'])
        self.assertIsNone(stats['completion_age_ms'])
        self.assertIsNone(stats['stage_ms'])
        self.assertEqual(stats['tracking_fps'],0)
        self.assertEqual(stats['camera_fps'],30)

    def test_live_frame_rejects_old_or_reconnected_source(self):
        self.assertFalse(latest_eligible(None,1,5))
        self.assertFalse(latest_eligible(Frame(5,1,0,b''),1,5))
        self.assertFalse(latest_eligible(Frame(99,2,0,b''),1,5))
        self.assertTrue(latest_eligible(Frame(99,1,0,b''),1,5))

    def test_performance_bounds(self):
        with patch.dict(os.environ,{'SAM_IMAGE_SIZE':'512','SAM_MEMORY_FRAMES':'3','SAM_OBJECT_POINTERS':'4'}):
            self.assertEqual(settings(),(512,3,4))
            for key,value in [('SAM_IMAGE_SIZE','500'),('SAM_MEMORY_FRAMES','0'),('SAM_OBJECT_POINTERS','17')]:
                with patch.dict(os.environ,{key:value}):
                    with self.assertRaises(ValueError):
                        settings()

if __name__ == '__main__':
    unittest.main()
