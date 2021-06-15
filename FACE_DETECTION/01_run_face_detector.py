from PIL import Image
from detect_face import extract_face
from detector_config import DetectorConfig

#------- Initialize the CONFIGURE Class -------#
CONFIG = DetectorConfig()

DETECTOR_SETTINGS = CONFIG.configuration()


#------- Run the main Face Detection function -------#
GET_FACE_DETAILS = extract_face(image_path=DETECTOR_SETTINGS.INPUT_IMAGE ,
                output_image_size=DETECTOR_SETTINGS.OUTPUT_IMAGE_SIZE ,
                    device=DETECTOR_SETTINGS.DEVICE)

print(GET_FACE_DETAILS.BOUNDING_BOX)
CROPPED_FACE_IMAGE = Image.fromarray(GET_FACE_DETAILS.BOUNDING_BOX)

if CROPPED_FACE_IMAGE != 'RGB':
    CROPPED_FACE_IMAGE = CROPPED_FACE_IMAGE.convert('RGB')
    print("1")

# ------- Resize pixels to required size ------- #
CROPPED_FACE_IMAGE = CROPPED_FACE_IMAGE.resize(DETECTOR_SETTINGS.OUTPUT_IMAGE_SIZE)
CROPPED_FACE_IMAGE.save(DETECTOR_SETTINGS.OUTPUT_IMAGE)
