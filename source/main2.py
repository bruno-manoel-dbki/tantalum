#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:18:51 2023

@author: bmdbki
"""

import pandas as pd
import numpy as np
from skimage import draw,io
from skimage.segmentation import flood, flood_fill
import cv2
import math
import os
from matplotlib import pyplot as plt
import sys
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

#import wield_input as wield


#TODO: remove this global variables from here. Actually we can't do that because origin changes for each grain and I don't know how to pass parameters when use sorted() function

origin = [0, 0]
refvec = [0, 1]



''' 
Method:     find_voids_2 
Parameters: original: np_array
Return:     
            centers: list of the center of all voids found, sorted by coordinates.
            radii:   list of radius of all voids found, sorted by coordinates.
            vheight: 
            image:   The original image
            drawing: The image with circle over all voids detected.

Description: This method works with classical image processing to detect circles. It applies to edge detection with bilateralFiler and then filter by threshold. Then all countors are detect to 
'''
def find_voids_2(original):
	### Read image

	
	### Save temp file for comparison
	#cv2.imwrite(picname+"_original.png", original)
	
	### Run several filter over image to achieve a more distinct void structure.
	### Voids will be more of a round shape this way, which reduces error.
	### Depending on grain structure, some filters may have no effect on particular samples.
    original = cv2.bilateralFilter(original,9,75,75)
    
	#original = cv2.medianBlur(original,5)
	#original = cv2.GaussianBlur(original,(5,5),0)
	
	### TODO: Determine threshold for binary image. First value is gray-value used for cut-off, second is white.
    retval, image = cv2.threshold(original, 60, 255, cv2.THRESH_BINARY)

	
	### Find eliptical shapes with a minimum size and dilate picutre
    
	
	### Read contours from dilated binary picture
    contours, hierarchy = cv2.findContours(image,
	    cv2.RETR_LIST,
	    cv2.CHAIN_APPROX_SIMPLE
	) ### Simplify contour 

	
	### Load bakcground image for contour plot
    drawing = original

	### Get height of the picture to set up an approximate area (height*height)
    vheight = len(drawing)
	
    centers = []
    radii = []
	
    for contour in contours:
        area = cv2.contourArea(contour)
		### there is one contour that contains all others (perimeter of image), filter it out considering 80% of full image size
 		### do not run the rest of the loop, jump straight to next contour
        
        ## The contour that contains all other is all one that is bigger than 80% of image size. This one is droped 
        if area > (original.size*0.8):
                continue
		
		### bound contour with a rectangle. 	
        br = cv2.boundingRect(contour)
		
		#el2 = cv2.fitEllipse(contour)

		### TODO: Determine radius of circle by taking half of a side (which one??) of the bounding Rectangle, then multiply by a factor if desired to make  up for mismatch due to color treshold in line 14.
		
        radius = (br[2]/2)*1.4
        radii.append(radius)
		
		### Find moments of countour
        m = cv2.moments(contour)
		
		### Avoid error due to division by zero
        if m['m00'] == 0.0:
            m['m00'] = 1.0
					
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)
        cv2.circle(drawing, center, int(radius), (0, 0, 255), 3)
	
    print("The program has detected {} voids".format(len(centers)))
	
    i = 0
    for center in centers:
        cv2.circle(drawing, center, 3, (255, 0, 0), -1)
        cv2.circle(drawing, center, int(radii[i]), (0, 255, 0), 1)
        i = i + 1

	### Save image with recognized voids
   # cv2.imwrite("_drawing.png", drawing)
   # cv2.imshow('finished', drawing)
   # cv2.waitKey()
   # cv2.destroyAllWindows()
    	
    return centers, radii, vheight, image, drawing

''' 
Method:     GB_reconstruction 
Parameters: 
            df:     A pandas dataframe contaning all information about grain and grain boundaries
            prefix: The name of original file to be used in the output file name


Output:     prefix.pkl: A pickle file named after the input file saved in the output folder

Description: This method aims to reconstruct boundaries in a set o selected voids detected by find_voids_2().
'''


def GB_reconstruction(df, prefix: str):

    width = int(df.x_end.max())+1 
    height = int(df.y_end.max())+1 

    '''

        Drawing the image using numpy and scikit
        
        The 3rd dimension is an index of the boundarie, so after I can refer to the df.
        
    '''

    gb_img = cv2.imread("../data/"+prefix+ '.jpg', 0)
    gb_img = cv2.resize(gb_img,(width,height),interpolation = cv2.INTER_AREA)

    centers, radii, vheight, image, drawing = find_voids_2(gb_img)

    bd_info = df.copy() 

    np_img = np.zeros([height+1, width+1, 3])

    for idx, row in bd_info.iterrows():
        rr,cc,a = draw.line_aa(row.x_start.astype("uint16"),row.y_start.astype("uint16"),row.x_end.astype("uint16"),row.y_end.astype("uint16"))
        np_img[cc,rr,1] = 255


    void = np_img.copy()
    voids_detected = np_img.copy()
    to_drop = []
    bd_new = bd_info.copy()


    useful_void = []

    # for num in [num]:
    for idx,num in enumerate(range(len(centers))):

        radi_0 = radii[num]*1.05
        center_0 = centers[num]
        radi_0 = round(radi_0)

        #TODO:The next steps aims to deal in each void area, but we're working in a square area, what means we can have unexpeted lines in the image. Change a square to a mask with a circular area is a must to simplify it.


        x_start, y_start = int(center_0[0] - radi_0) , int(center_0[1] - radi_0 )
        x_end, y_end = center_0[0] + radi_0 , center_0[1] + radi_0 
        
    
        # Select all lines in dataframe that are inside the square void area
        bd_start_in_void = bd_info[["x_start","y_start"]][
                                (bd_info["x_start"]>x_start) & (bd_info["x_start"]<x_end) 
                                &
                                (bd_info["y_start"]>y_start) & (bd_info["y_start"]<y_end)]
        
        
        
        bd_end_in_void = bd_info[["x_end","y_end"]][
                                (bd_info["x_end"]>x_start) & (bd_info["x_end"]<x_end) 
                                &
                                (bd_info["y_end"]>y_start) & (bd_info["y_end"]<y_end)]
        
        
        #Security step to verify if the area is correct
        void_view = void[y_start-50 : y_end+50 , x_start-50 : x_end+50]

        # Full Outer Join operation to combine all lines from previous operation. 
        #TODO: Check if the operation didn't drop any good boundary
        bd_inside = pd.concat([bd_start_in_void, bd_end_in_void])
        
        # Drop all boundaries that start and end inside the area. This boundaries usually exists because the edge detection used to create the original data
        bd_to_drop = bd_inside[bd_inside.index.duplicated()]
        
        to_drop += bd_to_drop.index.tolist()
        
        # We are interested in all boundaries that start or end over the selected area must, so everthing that wasn't dropped in the step above 
        bd_to_keep = bd_inside[~bd_inside.index.duplicated(keep=False)]
        
        # All voids that connect more than 4 grain boundaries around it are considered big enough to be dropped. 
        if (len(bd_to_keep) <5) & (len(bd_to_keep) >0):
            useful_void += [[idx,True]]
            cv2.rectangle(voids_detected,(x_start-25,y_start-25),(x_end+25,y_end+25), (255,255,255), 2)
            cv2.circle(voids_detected, center_0, int(radi_0), (255, 255, 255), 1)
            
            start_points = bd_to_keep[["x_start","y_start"]].dropna().values.astype("int32").tolist()
            end_points = bd_to_keep[["x_end","y_end"]].dropna().values.astype("int32").tolist()
        
            
        # Write a line from all start_points to all end_points. This step close grains destroyed by voids and insert these new lines to a new dataframe.
            for s in (start_points):
                for e in (end_points):
                    cv2.line(void, s, e, (255, 255, 255), 2)
                
                    new_line = [{'x_start':s[0],
                                    'y_start':s[1], 
                                    'x_end':e[0], 
                                    'y_end':e[1]}]
                    df_new_line = pd.DataFrame.from_records(new_line)   
                    bd_new = pd.concat([bd_new,df_new_line] , ignore_index=True)
                    
        
            
            if( not os.path.exists("../output")):
                print("Creating folder")
                os.mkdir("../output")
                
           # plt.imsave("../output/"+ prefix+ "_void_"+ str(idx) + "_base.jpg", void_view.astype("uint8"))
    #else:
        #    useful_void += [[idx,False]]
        #    cv2.rectangle(voids_detected,(x_start,y_start),(x_end,y_end), (255,255), 1)



    #Removing void edges from final detaframe

    bd_clean = bd_info.drop(index = to_drop)

    bd_new = bd_new.drop(index = to_drop)

    #Security step to verify if the new sample is ok
    void_clean = np.zeros([height+1,width+1, 3])

    for idx, row in bd_new.iterrows():
        rr,cc,a = draw.line_aa(row.x_start.astype("uint16"),row.y_start.astype("uint16"),row.x_end.astype("uint16"),row.y_end.astype("uint16"))

        void_clean[cc,rr] = (0,255,0)


            
    #plt.figure(10)
    #plt.imshow(void_clean)
    #plt.imsave("Boundaries.png",void_clean.astype("uint8"))
    #plt.show()
    bd_new.to_pickle("../output/"+ prefix + "_remake.pkl")  
        
    return 0


''' 
Method:     clockwiseangle_and_distance 
Parameters: point: This parameter is the point we need to compare with the origin to evaluate the lenght and angle of the point.
Return:     angle: The angle between the input point and the origin informed
            lenvector: the distance between the input point and the origin informed
Description: This method is part of the colorizing method. Here we obtain the position of the point to obtain a clockwise oriantation based on angle and lenght of vector. 
'''

def clockwiseangle_and_distance(point):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
   
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


    
###############################
    
# file = "1_001"        
if __name__ == "__main__":
    file = sys.argv[1]
    
folder = "../data/"

#file = sys.argv[1]
path = folder + file


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

flooded_grains = np.zeros([height, width, 3])
overflood = np.sum(flooded_grains==0) * 0.8
over = []
out = []
print("Dataframe sucessfully imported")


prefix = file
suffix = "remake"
 
GB_reconstruction(df, prefix = file)


sample = pd.read_pickle("../output/" + file + "_remake.pkl")
  
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
 





df_left = df[['left_phi1','left_PHI','left_phi2','grain_left']]
df_left = df_left.rename(columns={"grain_left": "grain"})

df_left = df_left[~df_left.grain.duplicated()].sort_values('grain')
df_left = df_left.set_index('grain')



df_right = df[['right_phi1','right_PHI','right_phi2','grain_right']]
df_right = df_right.rename(columns={"grain_right": "grain"})

df_right = df_right[~df_right.grain.duplicated()].sort_values('grain')
df_right = df_right.set_index('grain')

#TODO: Check the completeness of this join

df_grains = df_left.join(df_right)

df_grains_norm = (df_grains - df_grains.min()) / (df_grains.max() - df_grains.min())

print("ETL in Dataframe sucessfully done")

print("Running flood method")

for grain in df_grains.index.dropna():
        One_grain = df[(df["grain_right"] == grain) | (df["grain_left"] == grain)]
        grain_info = df_grains_norm.loc[grain,:]
        np_img = np.zeros([height, width, 3])

        x_center = math.ceil(One_grain[['x_start','x_end']].mean().mean())
        y_center = math.ceil(One_grain[['y_start','y_end']].mean().mean())

        phi1,Phi,phi2 = grain_info[["right_phi1","right_PHI","right_phi2"]]
#        if(x_center >0 and y_center <200):

                #over is a list to check all grains where overflow happened.
        One_grain = One_grain[One_grain["length"]>5]
         
        start = pd.DataFrame(columns=["x","y"])
        end = pd.DataFrame(columns=["x","y"])
        start[["x","y"]] = One_grain[['x_start','y_start']]
        end[["x","y"]] = One_grain[['x_end','y_end']]
        points = pd.concat([start,end])
         
        # points = points.drop_duplicates()
        p1 = points.to_numpy()
         
         #TODO: This origin is a global variable, need to be substituted by a parameter in this method, but the issue is that the sorted method doesn't allow us to change send more than one parameter (maybe just I didn't realize how to do that).
         
        origin = [x_center,y_center]
         
         # With all points of a grain packed in p1, we get a clockwise sorte based on angle and distance of the centroid.
        sort = sorted(p1, key=clockwiseangle_and_distance)
        a = []
        for b in sort:
            a.append(tuple((int(b[0]),int(b[1]))))
         
         # Use the polylines method with the sorted list of point to ensure that the polygon will be closed.
        cv2.polylines(np_img, np.array([a]), True, (phi1,Phi,phi2), 2)
         
         # Flood again with the garantee of a closed grain.
        mask = flood(np_img, (y_center, x_center,0))
        # print("Over: "+ str(overflood))
        if(np.sum(mask==1)<overflood):
        
             flooded_grains[mask[:,:,1] !=0] = [phi1,Phi,phi2]
             flooded_grains[np_img[:,:,1] !=0] = [phi1,Phi,phi2]
            
             
        else:
             #print(np.sum(mask==1))
             out.append(grain)
             
             #cv2.putText(flooded_grains, text=str(int(grain)), org=(x_center,y_center),fontFace=2, fontScale=0.2, color=(255,255,255), thickness=1)
print(out)
print("Flood method done")


io.imsave("../ml_sets/" + prefix + ".png",flooded_grains) 

#%%  
# PART 2 - SLICE METHOD
   

print("Running Slice method")
width = int(max([max(df.x_end),max(df.x_start)]))+1
height = int(max([max(df.y_end),max(df.y_start)]))+1

N = width//8
M = height//8



grey_img = cv2.imread("../data/"+ prefix + '.jpg', 0)
grey_img = cv2.resize(grey_img,(width,height),interpolation = cv2.INTER_AREA)

tiles = [flooded_grains[x:x+M,y:y+N] for x in range(0,flooded_grains.shape[0],M) for y in range(0,flooded_grains.shape[1],N)]

tiles_grey = [grey_img[x:x+M,y:y+N] for x in range(0,grey_img.shape[0],M) for y in range(0,grey_img.shape[1],N)]

n_voids = []

for idx in range(len(tiles)):
    centers, radii, vheight, image, drawing = find_voids_2(tiles_grey[idx])
    n_voids.append([idx,len(centers)])
    io.imsave("../ml_sets/"+ prefix + '_'+ str(idx) + '_' + str(len(centers)) + suffix + '.png',tiles[idx])
    # io.imsave("../ml_sets/"+ prefix + '_'+ str(idx) + '_' + str(len(centers)) + 'proof.png',tiles_grey[idx])

#save n_voids as csv

print("Divide et Vince Done "+ prefix)





    

