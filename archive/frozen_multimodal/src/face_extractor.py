import logging
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

logger = logging.getLogger(__name__)

# זיהוי אוטומטי של החומרה - יבחר ב-GPU (cuda) אם קיים, אחרת ירוץ על המעבד
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Face Extractor initialized using device: {device}")

# טעינת המודלים לזיכרון פעם אחת כדי לחסוך זמן בריצות הבאות
# MTCNN אחראי על חיתוך הפנים מתוך התמונה המקורית
mtcnn = MTCNN(keep_all=False, device=device)
# Resnet אחראי על הפיכת הפנים לווקטור באורך 512
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def extract_face_vector(img_path: str) -> list[float] | None:
    try:
        # טעינת התמונה
        img = Image.open(img_path).convert('RGB')

        # שלב 1: זיהוי וחיתוך הפנים
        face_tensor = mtcnn(img)

        if face_tensor is None:
            logger.warning(f"No face detected in {img_path}")
            return None

        # הוספת מימד Batch כי המודל מצפה לטנזור בצורה (batch, channels, height, width)
        face_tensor = face_tensor.unsqueeze(0).to(device)

        # שלב 2: יצירת ההטמעה (Embedding)
        with torch.no_grad():
            embedding = resnet(face_tensor)

        # המרה חזרה מטנזור על ה-GPU לרשימת פייתון רגילה שה-DB שלנו יודע לקבל
        feature_vector = embedding[0].cpu().numpy().tolist()

        if len(feature_vector) != 512:
            logger.error(f"Expected vector length 512, got {len(feature_vector)}")
            return None

        return feature_vector

    except Exception as e:
        logger.error(f"Face extraction failed for {img_path}: {e}")
        return None