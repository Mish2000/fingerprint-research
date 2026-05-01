import logging
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fpbench.matchers.baseline_dl import BaselineDL
from src.fpbench.preprocess.preprocess import PreprocessConfig

logger = logging.getLogger(__name__)


class FingerprintVectorExtractor:
    def __init__(self):
        # מעבר למודל ה-Deep Learning הגלובלי שלך משבוע 4-5 (מותאם בצורה טבעית ל-Vector DB)
        prep_cfg = PreprocessConfig(target_size=512)
        self.dl_matcher = BaselineDL(prep_cfg=prep_cfg)

    def extract_fingerprint_vector(self, img_path: str, capture_type: str = "plain") -> list[float] | None:
        try:
            # emb הוא מערך Numpy המייצג את התמונה כולה
            emb, ms = self.dl_matcher.embed_path(img_path)

            if emb is None or len(emb) == 0:
                logger.warning(f"No fingerprint features extracted from {img_path}")
                return None

            # הרשת לרוב מחזירה 2048 ממדים, pgvector תומך בעד 2000 לאינדקס HNSW.
            # ניקח 512 כדי להיות מתואמים עם פונקציית הפנים, או פשוט כדי לחסוך מקום
            vector = emb.flatten()
            if len(vector) > 512:
                vector_512 = vector[:512]
            else:
                vector_512 = vector

            # נרמול L2 חובה עבור חישובי Cosine Distance
            norm = np.linalg.norm(vector_512)
            if norm > 0:
                normalized_vector = vector_512 / norm
            else:
                normalized_vector = vector_512

            return normalized_vector.tolist()

        except Exception as e:
            logger.error(f"Fingerprint DL extraction failed for {img_path}: {e}")
            return None


finger_extractor = FingerprintVectorExtractor()


def extract_fingerprint_vector(img_path: str, capture_type: str = "plain") -> list[float] | None:
    return finger_extractor.extract_fingerprint_vector(img_path, capture_type)