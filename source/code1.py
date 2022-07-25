# -*- coding: utf-8 -*-
#%%

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


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

img = np.zeros((width,height)
               
               
#%%
#TODO: draw the boundaries in the jpg file






