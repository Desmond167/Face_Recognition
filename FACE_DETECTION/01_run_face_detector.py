from detect_face import extract_face
from detector_config import DetectorConfig


CONFIG = DetectorConfig()

DETECTOR_SETTINGS = CONFIG.configuration()

print(type(DETECTOR_SETTINGS))

extract_face(image_path=DETECTOR_SETTINGS.INPUT_IMAGE ,
                output_image_size=DETECTOR_SETTINGS.OUTPUT_IMAGE_SIZE ,
                    device=DETECTOR_SETTINGS.DEVICE)