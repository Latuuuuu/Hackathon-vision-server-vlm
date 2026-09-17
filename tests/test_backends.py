import json
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
from app.gpu_check import check_gpu
from npu.contract import device_counts, validate_arrays

class BackendChecks(unittest.TestCase):
    @patch('app.gpu_check.subprocess.run')
    def test_native_fault_is_not_success(self, run):
        run.return_value = subprocess.CompletedProcess([], -11, 'FILL\n','Segmentation fault')
        with self.assertRaisesRegex(RuntimeError, 'returncode=-11'):
            check_gpu()
    @patch('app.gpu_check.subprocess.run')
    def test_empty_success_is_rejected(self, run):
        run.return_value = subprocess.CompletedProcess([],0,'','')
        with self.assertRaisesRegex(RuntimeError,'without GPU_PASS'):
            check_gpu()
    @patch('app.gpu_check.subprocess.run')
    def test_verified_child_result(self,run):
        run.return_value = subprocess.CompletedProcess([],0,'GPU_PASS '+json.dumps({'arch':'gfx1152'}),'')
        self.assertEqual(check_gpu()['arch'],'gfx1152')
    def test_cpu_report_does_not_imply_npu(self):
        self.assertEqual(device_counts({'deviceStat':[{'name':'CPU','nodeNum':25}]}).get('NPU',0),0)
        self.assertEqual(device_counts({'deviceStat':[{'name':'NPU','nodeNum':12}]}),{'NPU':12})
    def test_bad_encoder_result_rejected(self):
        specs=[{'name':'fpn_0','shape':[1,2]}]
        validate_arrays({'fpn_0':np.ones((1,2),np.float32)},specs)
        for value in [np.ones((2,1),np.float32),np.ones((1,2),np.float16),np.full((1,2),np.nan,np.float32)]:
            with self.assertRaises(ValueError):
                validate_arrays({'fpn_0':value},specs)

if __name__ == '__main__':
    unittest.main()
