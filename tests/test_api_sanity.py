import unittest
import numpy as np
import cv2

from fastapi.testclient import TestClient
from api.main import app


def _png_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode failed"
    return buf.tobytes()


class TestAPISanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertTrue(j["ok"], msg=f"Service init failed: {j.get('error')}")

    def _post_match(self, method: str, *, return_overlay: bool = True):
        files = {
            "img_a": ("a.png", _png_bytes(1), "image/png"),
            "img_b": ("b.png", _png_bytes(2), "image/png"),
        }
        data = {
            "method": method,
            "return_overlay": "true" if return_overlay else "false",
            "capture_a": "plain",
            "capture_b": "plain",
        }
        return self.client.post("/match", data=data, files=files)

    def test_match_classic(self):
        r = self._post_match("classic", return_overlay=True)
        self.assertEqual(r.status_code, 200, msg=r.text)
        j = r.json()
        self.assertIn("score", j)
        self.assertIn("latency_ms", j)
        self.assertIn("meta", j)
        self.assertIn("overlay", j)
        self.assertIsNotNone(j["overlay"])

    def test_match_dl(self):
        r = self._post_match("dl", return_overlay=False)
        self.assertEqual(r.status_code, 200, msg=r.text)
        j = r.json()
        self.assertIn("score", j)
        self.assertIn("latency_ms", j)
        self.assertIn("meta", j)
        self.assertIsNone(j["overlay"])

    def test_match_dedicated(self):
        r = self._post_match("dedicated", return_overlay=True)
        self.assertEqual(r.status_code, 200, msg=r.text)
        j = r.json()
        self.assertIn("score", j)
        self.assertIn("latency_ms", j)
        self.assertIn("meta", j)
        self.assertIn("overlay", j)
        self.assertIsNotNone(j["overlay"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
