
# coding: utf-8

# In[ ]:


import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import newaxis
from scipy.misc import imresize
import pickle


# In[ ]:


a = []
b = []

for (i,image_file) in enumerate(glob.iglob('input_images_2/*.jpg')):
        img = cv2.imread(image_file)
        a.append(imresize(img, (80, 160, 3)))
        if(i%100==0):
            print(i)


# In[ ]:


f = open('input_final_1.p','wb')
pickle.dump(a, f, protocol=2)
f.flush()
a = []


# In[ ]:


for (i,image_file) in enumerate(glob.iglob('../output_data/labels1/*.jpg')):
        img = cv2.imread(image_file)
        temp = imresize(img, (80, 160, 3))
        temp = temp[:,:,1]
        temp = temp[:,:,newaxis]
        b.append(temp)
        if(i%100==0):
            print(i)


# In[ ]:


g = open('output_final_2.p','wb')
pickle.dump(b, g, protocol=2)
g.flush()
b=[]

