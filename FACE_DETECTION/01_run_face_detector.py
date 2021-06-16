from PIL import Image
from detector import DetectFace
from detector_config import DetectorConfig

#_______ Initialize the CONFIGURE Class _______#
CONFIG = DetectorConfig()

#_______ Import the Configuration Object _______#
DETECTOR_SETTINGS = CONFIG.configuration()

#_______ Initialize the Detector Class _______#
DETECTOR = DetectFace(input_image=DETECTOR_SETTINGS.INPUT_IMAGE,
                        device=DETECTOR_SETTINGS.DEVICE,
                            output_image=DETECTOR_SETTINGS.OUTPUT_IMAGE,
                                output_image_size=DETECTOR_SETTINGS.OUTPUT_IMAGE_SIZE)

#_______ Run the main Face Detection function _______#
# GET_FACE_DETAILS = DETECTOR.run_MTCNN()
GET_FACE_DETAILS = DETECTOR.run_MTCNN(bool_get_bounding_box=DETECTOR_SETTINGS.BOOL_GET_BOUNDING_BOX , 
                                        bool_get_probability=DETECTOR_SETTINGS.BOOL_GET_PROBABILITY ,
                                            bool_get_landmarks=DETECTOR_SETTINGS.BOOL_GET_LANDMARKS)

print(GET_FACE_DETAILS.BOUNDING_BOX)
