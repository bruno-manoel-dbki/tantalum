# -*- coding: utf-8 -*-
#%%

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from skimage import draw
from matplotlib import pyplot as plt
import cv2
import os
import math
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

file = "../data/1_002"

#%%

sample1 = np.loadtxt(file+ ".txt")

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


# To increase the image resolution just multiply width, height and 
# coordinates_array and bd_info (before casting) by the necessary increase. 
# 


width = int(df.x_end.max())+1 #multply here
height = int(df.y_end.max())+1 #multply here


#%%

'''

    Drawing the image using numpy and scikit
    
    The 3rd dimension is an index of the boundarie, so after I can refer to the df.
    TODO: create an ID in dataframe to avoid an order dependent reference
    
'''


# #np_img = np.zeros([width+1,height+1,3])
# np_img = np.zeros([height, width, 3])
# for idx, row in enumerate(coordinates_array):
#     rr,cc = draw.line(row[0],row[1],row[2],row[3])
# #    np_img[cc,rr,0:2] = [255, idx/3934*255]
    
# #    np_img[cc,rr,0:2] = [1, idx/3934]
#     np_img[cc,rr,0:2] = [1, idx]


# #np_img = np.transpose(np_img)
# plt.imshow(np_img)


# #img2 = Image.fromarray(np_img*255)
# #img2.show()

#%%

#---------------------------------------#
#                                       #
#                   TO DROP             #
#                                       #
#                                       #
#---------------------------------------#

# '''

#     Drawing the image 
    
#     The image can be described as a binary matrix, but we need to 
#   improve the description in it, so we add a new dimension defined
#   with a combination of right and left grain
  
#   Using the information from dataset, we create an index for each boundary.
#   It's not possible to index with dataset because each element describes
#   a line, but we can have a lot of lines in a boundary. To contour this 
#   situation is necessary to refer the index to grain around the boundaries
    
    
# '''

# bd_info = df[["x_start","y_start","x_end","y_end", "grain_left", "grain_right"]].copy() # if want to increase resolution, multiply here
# bd_info = bd_info.astype('int32')



# bd_unique =bd_info[["grain_left", "grain_right"]].drop_duplicates()

# bd_unique = bd_unique.reset_index()


# #


# bd_info["bd_index"] = None
# bd_info.loc[bd_unique["index"],"bd_index"]= bd_unique.index
# match = pd.DataFrame()
# for idx,info_row in bd_info[bd_info.bd_index.isnull()].iterrows():
#     match = bd_unique.loc[
#                         (info_row["grain_left"]== bd_unique["grain_left"])
#                         & 
#                         (info_row["grain_right"] == bd_unique[ "grain_right"])
#                         ]
#     bd_info.loc[idx,"bd_index"] = match.index.item()
    



# np_img = np.zeros([height+1, width+1, 3])
# for idx, row in bd_info.iterrows():
#     rr,cc = draw.line(row[0],row[1],row[2],row[3])

#     np_img[cc,rr,0] = row.bd_index


# #np_img = np.transpose(np_img)
# plt.imshow(np_img)





#---------------------------------------#
#                                       #
#                   TO DROP             #
#                                       #
#                                       #
#---------------------------------------#


#%%

# '''

#    Write bindary boundaries in jpg image
    
# '''
# path = '../data/'
# sample = "1_001.jpg"
# img_jpg = Image.open(path+sample)
# img_jpg = img_jpg.resize((width,height))



# coordinates = df[["x_start","y_start","x_end","y_end"]]
# coordinates_array = coordinates.to_numpy() #multply here
# coordinates_array = (coordinates_array.astype(int))



# img_jpg1 = ImageDraw.Draw(img_jpg)

# for row in coordinates_array:
    
#     #print(coordinates_array[x])
#     img_jpg1.line(tuple(row),"white",1)
    
# #img1.line((0,0,100,100),"white",1)
# #img.save(fp = "./001", format = "png")
# img_jpg.show()
#%%


gb_img = cv2.imread(file+ '.jpg', 0)
gb_img = cv2.resize(gb_img,(width,height),interpolation = cv2.INTER_AREA)

centers, radii, vheight, image, drawing = find_voids_2(gb_img)

plt.imshow(drawing)

#%%


# cv2.imwrite('temp.jpg',np_img)
# #img = cv2.cvtColor(np_img.astype('uint16'), cv2.COLOR_GRAY2BGR)
# img = cv2.imread('temp.jpg')
# os.remove("temp.jpg")

# for center, rad in zip(centers, radii):
#     cv2.circle(img,center,int(rad),(255,5,255), 5)



# cv2.imshow("image",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#%%
# #
# # 1 ->
# # 2 ->
# # 3 -> to much
# # 4 -> 7
# # 5 -> 2 boundaries
# # 6 -> 0 boundaries
# # 7 
# # 8
# # 9 -> 4
# # 10 ->

# #---------------------------------------#
# #                                       #
# #                   TO DROP             #
# #                                       #
# #                                       #
# #---------------------------------------#

# font = cv2.FONT_HERSHEY_SIMPLEX



# num = 0
# bd_inside_mask = []

# void = np_img.copy()

# for num in [num]:
# # for num in range(len(centers)):
        
#     radi_0 = radii[num]*1.1
#     center_0 = centers[num]
#     radi_0 = round(radi_0)
    
#     x_size = radi_0*2
#     y_size = radi_0*2
    
#     x_start, y_start = center_0[0] - radi_0 , center_0[1] - radi_0 
#     x_end, y_end = center_0[0] + radi_0 , center_0[1] + radi_0 
    

#     cv2.rectangle(void,(x_start,y_start),(x_end,y_end), (0,255,0), 1)
    
#     cv2.putText(void, str(num), (x_start,y_start), font, 1, (0,255, 0), 2, cv2.LINE_AA)
    
    
    
#     void_0 = np_img[y_start : y_end , x_start : x_end]
#     mask = np.zeros([x_size,y_size], dtype="uint8")
#     cv2.circle(mask, [radi_0,radi_0], radi_0, 255, -1)
#     masked = cv2.bitwise_and(void_0,void_0, mask=mask)
    
#     void_masked = masked
    

#     bd_inside_mask= np.unique(void_masked[:,:,0]).tolist()
    
#     if 0 in bd_inside_mask:
#         bd_inside_mask.remove(0)
    
#     tuples_start = [np.array(x) for x in bd_info[["x_start","y_start"]][(bd_info["bd_index"].isin(bd_inside_mask))].to_numpy()]
#     tuples_end = [np.array(x) for x in bd_info[["x_end","y_end"]][(bd_info["bd_index"].isin(bd_inside_mask))].to_numpy()]

#     cv2.line(void, tuples_end[0].astype(int), tuples_start[1].astype(int), (0, 255, 0), 1)

# print(bd_inside_mask)

# plt.figure()
# plt.imshow(void)
# #plt.imshow(np_img)
# plt.show()

# a = bd_info[(bd_info["x_start"]>x_start) &(bd_info["x_start"]<x_end) & (bd_info["y_start"]>y_start) &(bd_info["y_start"]<y_end)]

# #---------------------------------------#
# #                                       #
# #                   TO DROP             #
# #                                       #
# #                                       #
# #---------------------------------------#


#%%
bd_info = df[["x_start","y_start","x_end","y_end"]].copy() # if want to increase resolution, multiply here
bd_info = bd_info.astype('int32')




np_img = np.zeros([height+1, width+1, 3])

for idx, row in bd_info.iterrows():
    rr,cc = draw.line(row[0],row[1],row[2],row[3])

    np_img[cc,rr,0] = idx



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
#   NEXT STEP IS FIND A WAY TO COMPARE THE CIRCLE COORDINATES WITH THE DATASET.
#   MAYBE BITWISE WORKS, BUT ONLY IF DO NOT NEED TO CREATE AN IMAGE WITH POINTS
#

# num = 8
void = np_img.copy()
voids_detected = np_img.copy()
to_drop = []

# for num in [num]:
for num in range(len(centers)):

    radi_0 = radii[num]*1.05
    center_0 = centers[num]
    radi_0 = round(radi_0)

    #HOW TO LOOK INSIDE A CIRCLE AND NOT INSIDE A SQUARE?


    x_start, y_start = center_0[0] - radi_0 , center_0[1] - radi_0 
    x_end, y_end = center_0[0] + radi_0 , center_0[1] + radi_0 
    
    cv2.rectangle(voids_detected,(x_start,y_start),(x_end,y_end), (0,0,255), 1)
    
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
    
    void_0 = np_img[y_start : y_end , x_start : x_end]

    
    bd_inside = pd.concat([bd_start_in_void, bd_end_in_void])
    
#TODO: use pd.concat([bd_start_in_void, bd_end_in_void],axis=1) insted of 
#      operation above could reduce operations and make code clean
      


   # print(bd_to_keep)   
    #bd_to_keep = bd_to_keep.drop_duplicates(keep = False)
    bd_to_drop = bd_inside[bd_inside.index.duplicated()]
    
    to_drop += bd_to_drop.index.tolist()
    
    bd_to_keep = bd_inside[~bd_inside.index.duplicated(keep=False)]
   # print(bd_to_keep)
    
    
    if len(bd_to_keep) <5:
    
    # At this point we have a list of bds that belongs to the void area
    
        
        start_points = bd_to_keep[["x_start","y_start"]].dropna().values.astype("int32").tolist()
        
        end_points = bd_to_keep[["x_end","y_end"]].dropna().values.astype("int32").tolist()
    
        
        for s in (start_points):
            for e in (end_points):
                cv2.line(void, s, e, (0, 255, 255), 1)
                

    
'''
        NOVA MISSAO
            ADICIONAR ELEMENTOS DE bd_to_keep em bd_clean
       done REMOVER ELEMENTOS DE to_drop de bd_info e salvar em bd_clean
        
'''



bd_clean = bd_info.drop(index = to_drop)

np_clean = np.zeros([height+1, width+1, 3])

for idx, row in bd_clean.iterrows():
    rr,cc = draw.line(row[0],row[1],row[2],row[3])

    void[cc,rr] = (0,255,0)
        
# plt.figure()
# plt.imshow(void)
# #plt.imshow(np_img)
# plt.show()

    
    
#%%


# fig, axs = plt.subplots(2,2)
# fig.suptitle('GB Layers')
# axs[0,0].set_title("Original")
# axs[0,0].imshow(np_img)
# axs[0,1].imshow(void[:,:,0])
# axs[1,0].imshow(void[:,:,2])
# axs[1,1].imshow(void[:,:,1])
#%%




plt.figure(5)
plt.imshow(void[:,:,1])
plt.title("Final")

plt.figure(4)
plt.imshow(void[:,:,2])
plt.title("New Lines")

plt.figure(3)
plt.imshow(void[:,:,0])
plt.title("Dropped")

plt.figure(2)
plt.imshow(void_detected[:,:,0])
plt.title("Voids Detected")


plt.figure(1)
plt.imshow(np_img)
plt.title("Original")



plt.show()





# gray = cv2.cvtColor(void_masked.astype('uint8'),cv2.COLOR_BGR2GRAY)


# # edges = cv2.Canny(gray,50,150,apertureSize = 3)

# void_line = void_masked.copy()

# lines = cv2.HoughLines(gray,1,math.pi/360,2)

# # Draw lines on the image
# for r_theta in lines:
#     arr = np.array(r_theta[0], dtype=np.float64)
#     r, theta = arr
#     # Stores the value of cos(theta) in a
#     a = np.cos(theta)
 
#     # Stores the value of sin(theta) in b
#     b = np.sin(theta)
 
#     # x0 stores the value rcos(theta)
#     x0 = a*r
 
#     # y0 stores the value rsin(theta)
#     y0 = b*r
 
#     # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
#     x1 = int(x0 + 1000*(-b))
 
#     # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
#     y1 = int(y0 + 1000*(a))
 
#     # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
#     x2 = int(x0 - 1000*(-b))
 
#     # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
#     y2 = int(y0 - 1000*(a))
 
#     # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
#     # (0,0,255) denotes the colour of the line to be
#     # drawn. In this case, it is red.
#     cv2.line(void_0, (x1, y1), (x2, y2), (255, 0, 0), 1)
# # Show result
# plt.figure()
# plt.imshow(void)
# plt.show()