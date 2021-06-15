from detect_face import extract_face
from detector_config import DetectorConfig

#------- Initialize the CONFIGURE Class -------#
CONFIG = DetectorConfig()

DETECTOR_SETTINGS = CONFIG.configuration()


#------- Run the main Face Detection function -------#
GET_FACE_DETAILS = extract_face(image_path=DETECTOR_SETTINGS.INPUT_IMAGE ,
                output_image_size=DETECTOR_SETTINGS.OUTPUT_IMAGE_SIZE ,
                    device=DETECTOR_SETTINGS.DEVICE)

    print(GET_FACE_DETAILS)


    # resize pixels to the model size
    # image = im.fromarray(face)
    # image = image.resize((160,160))
    # image.save('new.jpg')
    # print(image)