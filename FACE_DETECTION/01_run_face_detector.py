from detect_face import extract_face
from FACE_DETECTION.detect_face import extract_face
from FACE_DETECTION.detector_config import DetectorConfig


CONFIG = DetectorConfig()

configuration = CONFIG.configuration()

print(configuration)