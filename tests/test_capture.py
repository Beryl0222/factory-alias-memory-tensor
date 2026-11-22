"""核对合成采集的交换记录与原始尺寸。"""
import json
from pathlib import Path
import unittest

import numpy as np

from caseio import load_case


class CaptureTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1] / "data"
        self.records, self.tensors = load_case(data / "capture.json")
        self.capacity = json.loads((data / "capacity.json").read_text(encoding="utf-8"))

    def test_capture_contract(self):
        r, a = self.records, self.tensors
        self.assertEqual(a["p3"].size, r["page_width"])
        self.assertLess(r["views"]["twist"]["strides"][0], 0)
        self.assertEqual(r["step"]["operations"][0][3], "read")
        self.assertEqual(r["upstream"][0], r["upstream"][-1])
