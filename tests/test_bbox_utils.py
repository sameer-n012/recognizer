import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import embed_faces  # noqa: E402
import face_detection  # noqa: E402


class TestFaceDetectionBboxUtils(unittest.TestCase):
    def test_clamp_bbox_valid(self):
        bbox = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(
            face_detection.clamp_bbox(bbox, width=100, height=100),
            [10.0, 20.0, 30.0, 40.0],
        )

    def test_clamp_bbox_invalid(self):
        bbox = [10.0, 20.0, 10.0, 25.0]
        self.assertIsNone(face_detection.clamp_bbox(bbox, width=100, height=100))

    def test_offset_bbox(self):
        bbox = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(
            face_detection.offset_bbox(bbox, (10.0, 20.0)),
            [11.0, 22.0, 13.0, 24.0],
        )

    def test_absolute_bbox_conversion(self):
        person_bbox = [100.0, 200.0, 300.0, 500.0]
        face_bbox = [10.0, 20.0, 60.0, 80.0]
        abs_bbox = face_detection.offset_bbox(
            face_bbox, (person_bbox[0], person_bbox[1])
        )
        abs_bbox = face_detection.clamp_bbox(abs_bbox, width=1000, height=1000)
        self.assertEqual(abs_bbox, [110.0, 220.0, 160.0, 280.0])


class TestEmbedFacesBboxUtils(unittest.TestCase):
    def test_bbox_iou(self):
        box_a = [0.0, 0.0, 10.0, 10.0]
        box_b = [5.0, 5.0, 15.0, 15.0]
        iou = embed_faces.bbox_iou(box_a, box_b)
        self.assertAlmostEqual(iou, 25.0 / 175.0, places=6)

    def test_match_face_to_detection(self):
        face_a = SimpleNamespace(bbox=[0.0, 0.0, 10.0, 10.0], det_score=0.9)
        face_b = SimpleNamespace(bbox=[20.0, 20.0, 30.0, 30.0], det_score=0.9)
        match = embed_faces.match_face_to_detection(
            [1.0, 1.0, 9.0, 9.0], [face_a, face_b], min_iou=0.2
        )
        self.assertIs(match, face_a)

    def test_clamp_bbox_with_padding(self):
        bbox = [10.0, 10.0, 30.0, 30.0]
        padded = embed_faces.clamp_bbox(bbox, width=100, height=100, pad_ratio=0.1)
        self.assertEqual(padded, [8, 8, 32, 32])

    def test_clamp_bbox_out_of_bounds(self):
        bbox = [-10.0, -5.0, 5.0, 10.0]
        clamped = embed_faces.clamp_bbox(bbox, width=100, height=100, pad_ratio=0.0)
        self.assertEqual(clamped, [0, 0, 5, 10])

    def test_clamp_bbox_invalid(self):
        bbox = [10.0, 10.0, 10.0, 9.0]
        self.assertIsNone(
            embed_faces.clamp_bbox(bbox, width=100, height=100, pad_ratio=0.0)
        )


if __name__ == "__main__":
    unittest.main()
