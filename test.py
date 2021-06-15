from PIL import Image
im = Image.open("sharon.jpg")
im.resize((160,160), 0)
print(im)