from PIL import Image
from facenet_pytorch import MTCNN
from detector_config import DetectorConfig
from collections import namedtuple
from matplotlib import pyplot


CONFIG = DetectorConfig()

DETECTOR_SETTINGS = CONFIG.configuration()

################ Extract a face from a given photograph ################
def extract_face(image_path, output_image_size, device):
    BOUNDING_BOX = []
    PROBABILITY = []
    LANDMARKS = []

    #------- Initialize the MTCNN Model using default weights -------
    MTCNN_MODEL = MTCNN(keep_all=True, device=device)

    # ------- Load the Image ------- #
    # INPUT_IMAGE = Image.open(image_path)
    INPUT_IMAGE = pyplot.imread(image_path)
    print(INPUT_IMAGE)
    
    #---------------- Get the bounding box co-ordinates, Probability and Landmarks on the face ----------------#
    boxes, prob, landmarks = MTCNN_MODEL.detect(INPUT_IMAGE, landmarks=DETECTOR_SETTINGS.GET_LANDMARKS)

    #------- Check if GET_BOUNDING_BOX is True in Config file -------#
    if DETECTOR_SETTINGS.GET_BOUNDING_BOX == True:
        BOUNDING_BOX = boxes

    #------- Check if GET_PROBABILITY is True in Config file -------#
    if DETECTOR_SETTINGS.GET_PROBABILITY == True:
        PROBABILITY = prob

    #------- Check if GET_LANDMARKS is True in Config file -------#
    if DETECTOR_SETTINGS.GET_LANDMARKS == True:
        LANDMARKS = landmarks

    for box in BOUNDING_BOX:
        x1 = int(box[0])
        x2 = int(x1 + box[2])
        y1 = int(box[1])
        y2 = int(y1 + box[3])
        CROPPED_FACE_IMAGE = INPUT_IMAGE[y1:y2, x1:x2]
        CROPPED_FACE_IMAGE = Image.fromarray(CROPPED_FACE_IMAGE)

    if CROPPED_FACE_IMAGE != 'RGB':
        CROPPED_FACE_IMAGE = CROPPED_FACE_IMAGE.convert('RGB')

    # ------- Resize pixels to required size ------- #
    CROPPED_FACE_IMAGE = CROPPED_FACE_IMAGE.resize(DETECTOR_SETTINGS.OUTPUT_IMAGE_SIZE)
    CROPPED_FACE_IMAGE.save(DETECTOR_SETTINGS.OUTPUT_IMAGE)


    # ------- Face Dict containing Bounding Box, Probability and Landmarks -------#
    face_dict ={
            "BOUNDING_BOX" : BOUNDING_BOX,
            "PROBABILITY" : PROBABILITY,
            "LANDMARKS" : LANDMARKS
        }

    #------- Convert Face Info dict to python object -------#
    FACE_DETAILS = namedtuple("FACE_DETAILS", face_dict.keys())(*face_dict.values())

    return FACE_DETAILS