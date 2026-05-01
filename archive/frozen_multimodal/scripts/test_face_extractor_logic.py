import sys
import unittest
from pathlib import Path
from scipy.spatial.distance import cosine

# ייבוא נכון של מודולים מתוך הפרויקט
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive.frozen_multimodal.src.face_extractor import extract_face_vector

class TestFaceExtractorLogic(unittest.TestCase):
    def setUp(self):
        # הגדרת נתיבים לתמונות
        self.img_misha_1 = str(ROOT / "data" / "visual_samples" / "faces" / "misha_1.jpg")
        self.img_misha_2 = str(ROOT / "data" / "visual_samples" / "faces" / "misha_2.jpg")
        self.img_other = str(ROOT / "data" / "visual_samples" / "faces" / "other_person.jpg")
        self.img_not_a_face = str(ROOT / "data" / "visual_samples" / "summary.json")

    def test_1_extraction_and_dimensionality(self):
        """מוודא חילוץ תקין ווקטור באורך 512"""
        vec1 = extract_face_vector(self.img_misha_1)
        self.assertIsNotNone(vec1, "Failed to extract face from misha_1.jpg")
        self.assertEqual(len(vec1), 512, f"Expected vector length 512, got {len(vec1)}")

    def test_2_negative_no_face(self):
        """מוודא שקובץ ללא פנים מחזיר None ולא קורס"""
        vec_none = extract_face_vector(self.img_not_a_face)
        self.assertIsNone(vec_none, "Should return None for an image without a face")

    def test_3_discrimination_cosine_distance(self):
        """מוודא שהמרחק בין תמונות של אותו אדם קטן מהמרחק לאדם אחר"""
        vec1 = extract_face_vector(self.img_misha_1)
        vec2 = extract_face_vector(self.img_misha_2)
        vec3 = extract_face_vector(self.img_other)

        self.assertIsNotNone(vec1, "Missing vector for misha_1")
        self.assertIsNotNone(vec2, "Missing vector for misha_2")
        self.assertIsNotNone(vec3, "Missing vector for other_person")

        dist_same_person = cosine(vec1, vec2)
        dist_different_person = cosine(vec1, vec3)

        print(f"\nDistance Misha 1 <-> Misha 2: {dist_same_person:.4f}")
        print(f"Distance Misha 1 <-> Other: {dist_different_person:.4f}")

        self.assertLess(
            dist_same_person,
            dist_different_person,
            "Discrimination Failed! Model confused identities."
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)