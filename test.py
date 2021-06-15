from PIL import Image
im = Image.open("sharon.jpg")
new_image = im.resize((160,160))
print(im)
print(new_image)