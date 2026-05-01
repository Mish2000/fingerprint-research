import sys
import unittest
from pathlib import Path
from scipy.spatial.distance import cosine

from archive.frozen_multimodal.src.fingerprint_extractor import extract_fingerprint_vector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestFingerExtractorLogic(unittest.TestCase):
    def setUp(self):
        # תיקון הנתיב לתיקייה האמיתית לפי עץ הפרויקט
        base_dir = ROOT / "data" / "visual_samples" / "after_preprocess"

        # שימוש בזוג תמונות (Plain ו-Roll) השייכות לאותו נבדק ואותה אצבע (sid1040, finger 03)
        self.img_plain = str(base_dir / "001_sid1040_f03_plain_train.png")
        self.img_roll_same = str(base_dir / "028_sid1040_f03_roll_train.png")
        # תמונת Plain של אדם אחר לגמרי למטרת ההשוואה השלילית (sid1443, finger 09)
        self.img_plain_other = str(base_dir / "000_sid1443_f09_plain_train.png")

    def test_1_extraction_and_dimensionality(self):
        vec1 = extract_fingerprint_vector(self.img_plain, capture_type="plain")
        self.assertIsNotNone(vec1, "Failed to extract fingerprint vector")
        # עדכון אורך הוקטור ל-512 כפי שקבענו
        self.assertEqual(len(vec1), 512, f"Expected vector length 512, got {len(vec1)}")

    def test_2_discrimination_cosine_distance(self):
        vec_plain = extract_fingerprint_vector(self.img_plain, capture_type="plain")
        vec_roll_same = extract_fingerprint_vector(self.img_roll_same, capture_type="roll")
        vec_plain_other = extract_fingerprint_vector(self.img_plain_other, capture_type="plain")

        self.assertIsNotNone(vec_plain, "Missing vector for plain")
        self.assertIsNotNone(vec_roll_same, "Missing vector for roll (same identity)")
        self.assertIsNotNone(vec_plain_other, "Missing vector for other identity")

        dist_same_finger = cosine(vec_plain, vec_roll_same)
        dist_different_finger = cosine(vec_plain, vec_plain_other)

        print(f"\nDistance Plain <-> Roll (Same Finger): {dist_same_finger:.4f}")
        print(f"Distance Plain <-> Plain (Other Finger): {dist_different_finger:.4f}")

        self.assertLess(
            dist_same_finger,
            dist_different_finger,
            "Discrimination Failed! Model confused identities."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)