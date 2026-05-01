import unittest
import requests
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestMultimodalAPI(unittest.TestCase):
    def setUp(self):
        # הגדרת הנתיבים לקבצים שלך
        self.face_img = str(ROOT / "data" / "visual_samples" / "faces" / "misha_1.jpg")
        self.finger_img = str(ROOT / "data" / "visual_samples" / "after_preprocess" / "001_sid1040_f03_plain_train.png")
        self.base_url = "http://localhost:8000"

        if not os.path.exists(self.face_img) or not os.path.exists(self.finger_img):
            self.fail("Test images not found. Please verify paths.")

    def test_1_enroll_multimodal(self):
        print("\n--- Enrolling multimodal user ---")
        with open(self.face_img, "rb") as f_face, open(self.finger_img, "rb") as f_finger:
            res = requests.post(
                f"{self.base_url}/enroll",
                data={"username": "misha_multimodal_test"},
                files={"face_image": f_face, "finger_image": f_finger}
            )

        self.assertEqual(res.status_code, 200, res.text)
        j = res.json()
        self.assertEqual(j["status"], "success")
        self.assertIn("user_id", j)
        print("Enroll Response:", j)

    def test_2_authenticate_multimodal(self):
        print("\n--- Authenticating multimodal user ---")
        with open(self.face_img, "rb") as f_face, open(self.finger_img, "rb") as f_finger:
            res = requests.post(
                f"{self.base_url}/authenticate",
                data={"face_weight": 0.23, "finger_weight": 0.77},
                files={"face_image": f_face, "finger_image": f_finger}
            )

        self.assertEqual(res.status_code, 200, res.text)
        j = res.json()
        self.assertIn("score", j)
        self.assertIn("decision", j)
        self.assertIn("latency_ms", j)
        print("Auth Response:", j)


if __name__ == "__main__":
    unittest.main(verbosity=2)