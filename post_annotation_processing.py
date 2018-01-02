import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process(i,img):
    temp = cv2.imread(img,1)
    image = np.zeros(temp.shape, dtype="uint8")
    # convert red portions in the image to green and rest with black
    image[np.where((temp==[0,0,255]).all(axis=2))] = [0,255,0] 
    img_name = os.path.basename(img)
    img_name_final = os.path.splitext(img_name)[0] + '.jpg'
    path = 'labels1/' + img_name_final
    cv2.imwrite(path,image)


for (i,image_file) in enumerate(glob.iglob('images1/*.tif')):
        process(image_file)
