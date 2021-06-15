from PIL import Image as im

################ Extract a face from a given photograph ################
def extract_face():
    # load image from file
    pixels = im.open("sharon.jpg")
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = im.fromarray(face)
    image = image.resize((160,160))
    image.save('new.jpg')
    print(image)

extract_face()