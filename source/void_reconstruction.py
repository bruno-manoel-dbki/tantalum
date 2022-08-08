import cv2

def find_voids(pa_pic, maxarea):
	### Read image
	original = cv2.imread(pa_pic, 0)
	
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
	el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
	image = cv2.dilate(image, el, iterations=1)

	### Save picture temporarily for comparison
	#cv2.imwrite(pa_pic+"_dilated.png", image)
	#cv2.imshow('dilated', image)
	
	### Read contours from dilated binary picture
	contours, hierarchy = cv2.findContours(image,
	    cv2.RETR_LIST,
	    cv2.CHAIN_APPROX_SIMPLE
	) ### Simplify contour 

	
	### Load bakcground image for contour plot
	drawing = cv2.imread(pa_pic)

	### Get height of the picture to set up an approximate area (height*height)
	vheight = len(drawing)
	
	centers = []
	radii = []
	
	for contour in contours:
		area = cv2.contourArea(contour)
		### there is one contour that contains all others (perimeter of image), filter it out
		### do not run the rest of the loop, jump straight to next contour
		if area > maxarea:
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
	
	#print("The program has detected {} voids".format(len(centers)))
	
	i = 0
	for center in centers:
		cv2.circle(drawing, center, 3, (255, 0, 0), -1)
		cv2.circle(drawing, center, int(radii[i]), (0, 255, 0), 1)
		i = i + 1

	### Save image with recognized voids
	#cv2.imwrite(name+"_drawing.png", drawing)
	#cv2.imshow('finished', drawing)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	return centers, radii, vheight, image, drawing


def open_gb_files(path: str):
    
    samples_list = os.listdir(path)    
    
    return samples_list


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import draw

def gb_reconstruction(sample: str):
    
    sample = np.loadtxt(sample + ".txt")
    
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

    
        
    width = int(df.x_end.max())
    height = int(df.y_end.max())
    
    coordinates = df[["x_start","y_start","x_end","y_end"]]
    coordinates_array = coordinates.to_numpy()
    
    coordinates_array = (coordinates_array.astype(int))
    
    rows = coordinates_array.shape[0]
    
    
    '''

        Drawing the image using numpy and scikit
        
    '''
    
    draw_image()

    os.mkdir("output")

    
    
    return 0


def draw_image(width: int, height: int ):
    
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


import os

def main():
    path = os.getcwd()
    samples_list = open_gb_files(path + "/data/")
    for sample in samples_list:        
        gb_reconstruction(sample)
        find_voids(sample)
    
        
    return 0

if __name__ == "__main__":
    main()


