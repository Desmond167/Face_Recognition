from PIL import Image
from facenet_pytorch import MTCNN
from detector_config import DetectorConfig
from collections import namedtuple


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
    INPUT_IMAGE = Image.open(image_path)
    
    #---------------- Get the bounding box co-ordinates, Probability and Landmarks on the face ----------------#
    boxes, prob, landmarks = MTCNN_MODEL.detect(INPUT_IMAGE, landmarks=DETECTOR_SETTINGS.GET_LANDMARKS)

    #------- Check if GET_BOUNDING_BOX is True in Config file -------#
    if DETECTOR_SETTINGS.GET_BOUNDING_BOX == True:
        BOUNDING_BOX = boxes

    #------- Check if GET_PROBABILITY is True in Config file -------#
    if DETECTOR_SETTINGS.GET_PROBABILTY == True:
        PROBABILITY = prob

    #------- Check if GET_LANDMARKS is True in Config file -------#
    if DETECTOR_SETTINGS.GET_LANDMARKS == True:
        LANDMARKS = landmarks

    # ------- Face Dict containing Bounding Box, Probability and Landmarks -------#
    face_dict ={
            "BOUNDING_BOX" : BOUNDING_BOX,
            "PROBABILITY" : PROBABILITY,
            "LANDMARKS" : LANDMARKS
        }

    #------- Convert Face Info dict to python object -------#
    FACE_DETAILS = namedtuple("FACE_DETAILS", face_dict.keys())(*face_dict.values())