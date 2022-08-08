# -*- coding: utf-8 -*-
#%%

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from skimage import draw
from matplotlib import pyplot as plt
#from matplotlib import cm





#%%

sample1 = np.loadtxt("../data/1_001.txt")

'''
# Column 1-3:   right hand average orientation (phi1, PHI, phi2 in radians)
# Column 4-6:   left hand average orientation (phi1, PHI, phi2 in radians)
# Column 7:     Misorientation Angle
# Column 8-10:  Misorientation Axis in Right Hand grain
# Column 11-13: Misorientation Axis in Left Hand grain
# Column 14:    length (in microns)
# Column 15:    trace angle (in degrees)
# Column 16-19: x,y coordinates of endpoints (in microns)
# Column 20-21: IDs of right hand and left hand grains

'''


df = pd.DataFrame(  data = sample1, 
                    columns = ["right_phi1","right_PHI","right_phi2",                 #1-3
                               "left_phi1","left_PHI","left_phi2",                    #4-6 
                               "ori_angle",                                           #7
                               "right_ori_x","right_ori_y","right_ori_z",              #8-10
                               "left_ori_x","leff_ori_y","left_ori_z",                 #11-13  
                               "length",                                              #14
                               "trace_angle",                                         #15
                               "x_start", "y_start", "x_end", "y_end",                #16-19
                               "grain_right","grain_left"                             #20-21
                               ]                    
                 )

#df.head()
#%%
# Creating image

width = int(df.x_end.max())
height = int(df.y_end.max())

coordinates = df[["x_start","y_start","x_end","y_end"]]
coordinates_array = coordinates.to_numpy()

coordinates_array = (coordinates_array.astype(int))

rows = coordinates_array.shape[0]




#%%
'''

    Drawing the image using pillow
    
'''
img = Image.new("RGB", size = (width, height))
#img.show()

img1 = ImageDraw.Draw(img)

for x in range(rows):
    
    #print(coordinates_array[x])
    img1.line(tuple(coordinates_array[x]),"white",1)
    
#img1.line((0,0,100,100),"white",1)
img.save(fp = "./001", format = "png")
img.show()


#%% Same, but whitout pillow
'''

    Drawing the image using numpy and scikit
    
'''
np_img = np.zeros((width+1,height+1))
for x in range(rows):
    rr,cc = draw.line(coordinates_array[x][0],coordinates_array[x][1],coordinates_array[x][2],coordinates_array[x][3])
    np_img[rr,cc] = 1

#rr,cc = draw.line(0,0,100,100)
#np_img = np.rot90(np_img)
np_img = np.flip(np_img)
plt.imshow(np_img,cmap= "Greys")
 
#img2 = Image.fromarray(np_img*255)
#img2.show()


#%%
'''
    Inverting Image

'''


plt.imshow(1-np_img, cmap="Greys")


img_invert = ImageOps.invert(img)
img_invert.show()


               
#%%
#TODO: draw the boundaries in the jpg file


'''

    Drawing the image using pillow
    
'''
img_jpg = Image.open("../data/1_001.jpg")
img_jpg = img_jpg.resize((width,height))


#img.show()

img_jpg1 = ImageDraw.Draw(img_jpg)

for x in range(rows):
    
    #print(coordinates_array[x])
    img_jpg1.line(tuple(coordinates_array[x]),"white",1)
    
#img1.line((0,0,100,100),"white",1)
#img.save(fp = "./001", format = "png")
img_jpg.show()





