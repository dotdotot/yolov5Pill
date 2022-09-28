import os
import glob
from posixpath import split
from PIL import Image

file_list = os.listdir('C:\pillImage\img') 
img_files = [file for file in file_list if file.endswith('.jpg')]
img_files = glob.glob('C:\pillImage\img\\*.jpg')

image_number = input("image number : ")
image_number = image_number.split()

image_numbers = []
index = 0
for i in image_number:
    image_numbers.append(int(image_number[index]) - 3)
    index += 1

image_number = 0
for i in image_numbers:
    img = Image.open('C:\pillImage\pill' + str(i) + '.jpg')
    img.save('C:\\newImage\\shape\\' + str(3) + '\\triangle\pill' + str(image_number) + '.jpg' , 'JPEG')
    image_number += 1
        