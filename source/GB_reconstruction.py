# -*- coding: utf-8 -*-
#%%

import numpy as np
import pandas as pd
from skimage import draw
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
import os

#from matplotlib import cm


#%%


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
	
	### TO CHANGE:
	### Determine treshold for binary image. First value is gray-value used for cut-off, second is white.
    retval, image = cv2.threshold(original, 60, 255, cv2.THRESH_BINARY)

	
	### Find eliptical shapes with a minimum size and dilate picutre
    #el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    #image = cv2.dilate(image, el, iterations=1)

	### Save picture temporarily for comparison
	#cv2.imwrite(pa_pic+"_dilated.png", image)
	#cv2.imshow('dilated', image)
	
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
        if area > (original.size*0.8):
                continue
		
		### bound contour with a rectangle. 	
        br = cv2.boundingRect(contour)
		
		#el2 = cv2.fitEllipse(contour)

		### TO CHANGE:
		### Determine radius of circle by taking half of a side (which one??) 
		### of the bounding Rectangle, then multiply by a factor if desired to make 
		### up for mismatch due to color treshold in line 14.
        radius = (br[2]/2)*1.4
        radii.append(radius)
		
		### Find moments of countour
        m = cv2.moments(contour)
		
		### Avoid error due to division by zero
        if m['m00'] == 0.0:
            m['m00'] = 1.0
					
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)
		#cv2.circle(drawing, center, 3, (255, 0, 0), -1)
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



#%%

folder = "../data/"
file = "1_001"
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


# To increase the image resolution just multiply width, height and 
# coordinates_array and bd_info (before casting) by the necessary increase. 
# 


width = int(df.x_end.max())+1 #multply here
height = int(df.y_end.max())+1 #multply here


#%%

'''

    Drawing the image using numpy and scikit
    
    The 3rd dimension is an index of the boundarie, so after I can refer to the df.
    
'''


#%%


gb_img = cv2.imread(path+ '.jpg', 0)
gb_img = cv2.resize(gb_img,(width,height),interpolation = cv2.INTER_AREA)

centers, radii, vheight, image, drawing = find_voids_2(gb_img)

plt.imshow(drawing)

#%%
# bd_info = df[["x_start","y_start","x_end","y_end"]].copy() # if want to increase resolution, multiply here
bd_info = df.copy() 
#bd_info = bd_info.astype('int32')




np_img = np.zeros([height+1, width+1, 3])

for idx, row in bd_info.iterrows():
    rr,cc,a = draw.line_aa(row.x_start.astype("uint16"),row.y_start.astype("uint16"),row.x_end.astype("uint16"),row.y_end.astype("uint16"))
    np_img[cc,rr,1] = 255



#plt.imshow(np_img)


#
#
#   HOW TO CONSIDER THE VOID AS A CIRCLE AND NOT A RECTANGLE?
#       We need a mask to compare if the start/end point is inside the void
# 
#
#
#  void_0 = np_img[y_start : y_end , x_start : x_end]
#  mask = np.zeros([width,height], dtype="uint8")
#  cv2.circle(mask, center_0, radi_0, 255, -1)
# 
#  TODO: NEXT STEP IS FIND A WAY TO COMPARE THE CIRCLE COORDINATES WITH THE DATASET.
#    MAYBE BITWISE WORKS, BUT ONLY IF DO NOT NEED TO CREATE AN IMAGE WITH POINTS
#

# num = 8
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

    #HOW TO LOOK INSIDE A CIRCLE AND NOT INSIDE A SQUARE?


    x_start, y_start = int(center_0[0] - radi_0) , int(center_0[1] - radi_0 )
    x_end, y_end = center_0[0] + radi_0 , center_0[1] + radi_0 
    
  
    
    bd_start_in_void = bd_info[["x_start","y_start"]][
                            (bd_info["x_start"]>x_start) & (bd_info["x_start"]<x_end) 
                            &
                            (bd_info["y_start"]>y_start) & (bd_info["y_start"]<y_end)]
    
    
    
    bd_end_in_void = bd_info[["x_end","y_end"]][
                            (bd_info["x_end"]>x_start) & (bd_info["x_end"]<x_end) 
                            &
                            (bd_info["y_end"]>y_start) & (bd_info["y_end"]<y_end)]
    
    
    
    #bd_to_drop = bd_start_in_void.merge(bd_end_in_void, how ="inner")
    #bd_to_keep = bd_start_in_void.merge(bd_end_in_void, how ="outer")

# Another method to keep index
    
    void_view = void[y_start-50 : y_end+50 , x_start-50 : x_end+50]

    
    bd_inside = pd.concat([bd_start_in_void, bd_end_in_void])
    
#TODO: use pd.concat([bd_start_in_void, bd_end_in_void],axis=1) insted of 
#      operation above could reduce operations and make code cleaner
      


#    print(bd_to_keep)   
#    bd_to_keep = bd_to_keep.drop_duplicates(keep = False)
    bd_to_drop = bd_inside[bd_inside.index.duplicated()]
    
    to_drop += bd_to_drop.index.tolist()
    
    bd_to_keep = bd_inside[~bd_inside.index.duplicated(keep=False)]
   # print(bd_to_keep)
    
    
    if (len(bd_to_keep) <5) & (len(bd_to_keep) >0):
        useful_void += [[idx,True]]
        cv2.rectangle(voids_detected,(x_start-25,y_start-25),(x_end+25,y_end+25), (255,255,255), 2)
        cv2.circle(voids_detected, center_0, int(radi_0), (255, 255, 255), 1)
    
   
    
        
        start_points = bd_to_keep[["x_start","y_start"]].dropna().values.astype("int32").tolist()
        end_points = bd_to_keep[["x_end","y_end"]].dropna().values.astype("int32").tolist()
    
        
    
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
            
        plt.imsave("../output/"+ file+ "_void_"+ str(idx) + "_base.jpg", void_view.astype("uint8"))
   
    
   
    
   #else:
    #    useful_void += [[idx,False]]
    #    cv2.rectangle(voids_detected,(x_start,y_start),(x_end,y_end), (255,255), 1)



'''
Creating a new DF without any boundary that only exists near a void and is not 
conected to other boundaries

'''


bd_clean = bd_info.drop(index = to_drop)

bd_new = bd_new.drop(index = to_drop)

void_clean = np.zeros([height+1,width+1, 3])

for idx, row in bd_new.iterrows():
    rr,cc,a = draw.line_aa(row.x_start.astype("uint16"),row.y_start.astype("uint16"),row.x_end.astype("uint16"),row.y_end.astype("uint16"))

    void_clean[cc,rr] = (0,255,0)


        
plt.figure(10)
plt.imshow(void_clean)
#plt.imsave("Boundaries.png",void_clean.astype("uint8"))
plt.show()
bd_new.to_pickle("../output/"+ file + "_remake.pkl")  
    
    
# #%%


# # fig, axs = plt.subplots(2,2)
# # fig.suptitle('GB Layers')
# # axs[0,0].set_title("Original")
# # axs[0,0].imshow(np_img)
# # axs[0,1].imshow(void[:,:,0])
# # axs[1,0].imshow(void[:,:,2])
# # axs[1,1].imshow(void[:,:,1])
# #%%

# # new bds = (0,255,255) # Light blue
# # removed bds = (255,0,0) # Red
# # bds_keeped = ()



# # plt.figure(5)
# # plt.imshow(void)
# # plt.title("Final")

# # plt.figure(4)
# # plt.imshow(void[:,:,2])
# # plt.title("New Lines")

# # plt.figure(3)
# # plt.imshow(void[:,:,0])
# # plt.title("Dropped")

# # plt.figure(2)
# # plt.imshow(voids_detected)
# # plt.title("Voids Detected")


# # plt.figure(1)
# # plt.imshow(np_img)
# # plt.title("Original")



# # plt.show()

# #%%
# #      FOR ALL VOID, CREATE A SLICE OF ORIGINAL DF CONSIDERING ONLY BOUNDARIES 
# #      AROUND THE VOID. SAVE IT AS A GOOD DF

# for idx,useful in useful_void:
    
#     if (useful is True):
#         radi_0 = radii[idx]*5
#         center_0 = centers[idx]
#         radi_0 = round(radi_0)
    
    
#         # x_start, y_start = center_0[0] - radi_0 , center_0[1] - radi_0 
#         # x_end, y_end = center_0[0] + radi_0 , center_0[1] + radi_0 
#         x_start, y_start = center_0[0] - 50, center_0[1] - 50 
#         x_end, y_end = center_0[0] + 50 , center_0[1] + 50
        
        
#         bd_start_in_area = bd_new[
#                                 (bd_new["x_start"]>x_start) & (bd_new["x_start"]<x_end) 
#                                 &
#                                 (bd_new["y_start"]>y_start) & (bd_new["y_start"]<y_end)]
        
        
        
#         bd_end_in_area = bd_new[
#                                 (bd_new["x_end"]>x_start) & (bd_new["x_end"]<x_end) 
#                                 &
#                                 (bd_new["y_end"]>y_start) & (bd_new["y_end"]<y_end)]
        
        
       
#         bd_inside = pd.concat([bd_start_in_area, bd_end_in_area])
#         bd_inside = bd_inside[~bd_inside.index.duplicated()]

         
#         bd_inside.loc[:,"x_start"] = bd_inside.x_start - x_start
#         bd_inside.loc[:,"x_end"] = bd_inside.x_end - x_start
        
#         bd_inside.loc[:,"y_start"] = bd_inside.y_start - y_start
#         bd_inside.loc[:,"y_end"] = bd_inside.y_end - y_start
        
#         x_min = bd_inside[["x_start","x_end"]].values.min()
#         y_min = bd_inside[["y_start","y_end"]].values.min()
        
#         bd_inside.loc[:,"x_start"] = bd_inside.x_start - x_min
#         bd_inside.loc[:,"x_end"] = bd_inside.x_end - x_min
        
#         bd_inside.loc[:,"y_start"] = bd_inside.y_start - y_min
#         bd_inside.loc[:,"y_end"] = bd_inside.y_end -y_min
        
        
#         x_size = bd_inside[["x_start","x_end"]].values.max()
#         y_size = bd_inside[["y_start","y_end"]].values.max()
        
        
        
        
        
        
        
#         void_new = np.zeros([y_size+1,x_size+1, 3])
#         # void_new = np.zeros([height+1,width+1, 3])

#         for aux, row in bd_inside.iterrows():
#             rr,cc = draw.line(row.x_start.astype("uint8"),row.y_start.astype("uint8"),row.x_end.astype("uint8"),row.y_end.astype("uint8"))
#             void_new[cc,rr] = (0,255,0)
            
#             # void_new[cc-y_start,rr-x_start] = (0,255,0)
        
        
        
        
#         # Pickle File    
        
        
#         if( not os.path.exists("../output")):
#             print("Creating folder")
#             os.mkdir("../output")
            
            
            
#         plt.imsave("../output/"+ file+ "_void_"+ str(idx) + "_new.jpg", void_new.astype("uint8"))
#         bd_inside.to_pickle("../output/"+ file+ "_void_"+ str(idx) + "_new.pkl")  
        
        
        
        
# #%%

# for idx,useful in useful_void:
    
#     if (useful is True):
#         radi_0 = radii[idx]*5
#         center_0 = centers[idx]
#         radi_0 = round(radi_0)
    
    
#         # x_start, y_start = center_0[0] - radi_0 , center_0[1] - radi_0 
#         # x_end, y_end = center_0[0] + radi_0 , center_0[1] + radi_0 
#         x_start, y_start = center_0[0] - 50, center_0[1] - 50 
#         x_end, y_end = center_0[0] + 50 , center_0[1] + 50
        
        
#         bd_start_in_area = bd_clean[
#                                 (bd_clean["x_start"]>x_start) & (bd_clean["x_start"]<x_end) 
#                                 &
#                                 (bd_clean["y_start"]>y_start) & (bd_clean["y_start"]<y_end)]
        
        
        
#         bd_end_in_area = bd_clean[
#                                 (bd_clean["x_end"]>x_start) & (bd_clean["x_end"]<x_end) 
#                                 &
#                                 (bd_clean["y_end"]>y_start) & (bd_clean["y_end"]<y_end)]
        
        
       
#         bd_inside = pd.concat([bd_start_in_area, bd_end_in_area])
#         bd_inside = bd_inside[~bd_inside.index.duplicated()]

         
#         bd_inside.loc[:,"x_start"] = bd_inside.x_start - x_start
#         bd_inside.loc[:,"x_end"] = bd_inside.x_end - x_start
        
#         bd_inside.loc[:,"y_start"] = bd_inside.y_start - y_start
#         bd_inside.loc[:,"y_end"] = bd_inside.y_end - y_start
        
#         x_min = bd_inside[["x_start","x_end"]].values.min()
#         y_min = bd_inside[["y_start","y_end"]].values.min()
        
#         bd_inside.loc[:,"x_start"] = bd_inside.x_start - x_min
#         bd_inside.loc[:,"x_end"] = bd_inside.x_end - x_min
        
#         bd_inside.loc[:,"y_start"] = bd_inside.y_start - y_min
#         bd_inside.loc[:,"y_end"] = bd_inside.y_end -y_min
        
        
#         x_size = bd_inside[["x_start","x_end"]].values.max()
#         y_size = bd_inside[["y_start","y_end"]].values.max()
        
        
        
        
        
        
#         # Pickle File    
        
        
#         if( not os.path.exists("../output")):
#             print("Creating folder")
#             os.mkdir("../output")
            
            
            
#         #plt.imsave("../output/"+ file+ "_void_"+ str(idx) + "_base.jpg", void_new.astype("uint8"))
#         bd_inside.to_pickle("../output/"+ file+ "_void_"+ str(idx) + "_base.pkl")  
        
       
        
        

# # TODO: MAYBE WE HAVE INTEREST IN CONSIDER THE BEHAVIOR OF BIG HOLES, TO DO
# #       WE'LL START WITH ALL FALSES ELEMENTS IN useful_voids