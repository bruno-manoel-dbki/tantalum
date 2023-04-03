#!/usr/bin/env python
# coding: utf-8

# ## GB energy with WIELD

# In[1]:


import pandas as pd
import numpy as np
from skimage import draw,io
from skimage.segmentation import flood, flood_fill
import cv2
import math
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
import wield_input as wield


# In[6]:


if __name__ == "__main__":
    file = sys.argv[1]
    
folder = "../data/"
#file = "1_001"
path = folder + file

#%%

sample = np.loadtxt(path+ ".txt")

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


df = pd.DataFrame(  data = sample, 
                    columns = ["right_phi1","right_PHI","right_phi2",                 #1-3
                               "left_phi1","left_PHI","left_phi2",                    #4-6 
                               "ori_angle",                                           #7
                               "right_ori_x","right_ori_y","right_ori_z",              #8-10
                               "left_ori_x","left_ori_y","left_ori_z",                 #11-13  
                               "length",                                              #14
                               "trace_angle",                                         #15
                               "x_start", "y_start", "x_end", "y_end",                #16-19
                               "grain_right","grain_left"                             #20-21
                               ]                    
                 )


width = int(max([max(df.x_end),max(df.x_start)]))+1
height = int(max([max(df.y_end),max(df.y_start)]))+1

print("Dataframe " + file+ " sucessfully imported")


#df_wield = df[['right_phi1','right_PHI','right_phi2','left_phi1','left_PHI','left_phi2','trace_angle']]

np_img = np.zeros([height+1, width+1, 3])

for index, row in df.iterrows():
    #if(index <200):
    energy = wield.energy1d(row[['right_phi1','right_PHI','right_phi2','left_phi1','left_PHI','left_phi2','trace_angle']].to_dict())
    df.loc[index,'gb_energy'] = energy
    print(energy)
df.gb_energy = (df.gb_energy - df.gb_energy.min()) / (df.gb_energy.max() - df.gb_energy.min())

for index, row in df.iterrows():   
    rr,cc,a = draw.line_aa(row.x_start.astype("uint16"),row.y_start.astype("uint16"),row.x_end.astype("uint16"),row.y_end.astype("uint16"))
    np_img[cc,rr,0] = row.gb_energy
print("ETL in Dataframe sucessfully done")


#plt.figure()
io.imsave("../ml_sets/"+file+"_energy.png",np_img)
#plt.show()

# In[44]:


