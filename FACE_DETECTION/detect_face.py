from PIL import Image
from facenet_pytorch import MTCNN

################ Extract a face from a given photograph ################
def extract_face(image_path, output_image_size, device):

    #------- Initialize the MTCNN Model using default weights -------
    MTCNN_MODEL = MTCNN(keep_all=True, device=device)

    # ------- Load the Image ------- #
    INPUT_IMAGE = Image.open(image_path)
    
    # Detect face
    boxes, probs, landmarks = MTCNN_MODEL.detect(INPUT_IMAGE, landmarks=True)

    print(boxes)
    print(probs)
    print(landmarks)

    # resize pixels to the model size
    # image = im.fromarray(face)
    # image = image.resize((160,160))
    # image.save('new.jpg')
    # print(image)