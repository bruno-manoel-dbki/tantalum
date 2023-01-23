

import pandas as pd
import numpy as np
from skimage import draw,io
from skimage.segmentation import flood, flood_fill
import cv2
import math
import os
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats
import sys
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400

import wield_input as wield


#TODO: remove this global variables from here. Actually we can't do that because origin changes for each grain and I don't know how to pass parameters when use sorted() function

origin = [0, 0]
refvec = [0, 1]



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


def GB_reconstruction(df, prefix: str):

    width = int(df.x_end.max())+1 #multply here
    height = int(df.y_end.max())+1 #multply here

    '''

        Drawing the image using numpy and scikit
        
        The 3rd dimension is an index of the boundarie, so after I can refer to the df.
        
    '''

    gb_img = cv2.imread(path+ '.jpg', 0)
    gb_img = cv2.resize(gb_img,(width,height),interpolation = cv2.INTER_AREA)

    centers, radii, vheight, image, drawing = find_voids_2(gb_img)

    #plt.imshow(drawing)

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

        #TODO:HOW TO LOOK INSIDE A CIRCLE AND NOT INSIDE A SQUARE?


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
        
        
        
        void_view = void[y_start-50 : y_end+50 , x_start-50 : x_end+50]

        
        bd_inside = pd.concat([bd_start_in_void, bd_end_in_void])
        
        bd_to_drop = bd_inside[bd_inside.index.duplicated()]
        
        to_drop += bd_to_drop.index.tolist()
        
        bd_to_keep = bd_inside[~bd_inside.index.duplicated(keep=False)]
        
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
                
            plt.imsave("../output/"+ prefix+ "_void_"+ str(idx) + "_base.jpg", void_view.astype("uint8"))
    #else:
        #    useful_void += [[idx,False]]
        #    cv2.rectangle(voids_detected,(x_start,y_start),(x_end,y_end), (255,255), 1)




    bd_clean = bd_info.drop(index = to_drop)

    bd_new = bd_new.drop(index = to_drop)

    void_clean = np.zeros([height+1,width+1, 3])

    for idx, row in bd_new.iterrows():
        rr,cc,a = draw.line_aa(row.x_start.astype("uint16"),row.y_start.astype("uint16"),row.x_end.astype("uint16"),row.y_end.astype("uint16"))

        void_clean[cc,rr] = (0,255,0)


            
    #plt.figure(10)
    #plt.imshow(void_clean)
    #plt.imsave("Boundaries.png",void_clean.astype("uint8"))
    #plt.show()
    bd_new.to_pickle("../output/"+ file + "_remake.pkl")  
        
    return 1


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



def Divide_et_Vince(df, prefix: str, suffix: str):

    width = int(max([max(df.x_end),max(df.x_start)]))+1
    height = int(max([max(df.y_end),max(df.y_start)]))+1

    flooded_grains = np.zeros([height, width, 3])
    overflood = np.sum(flooded_grains==0) * 0.8
    over = []
    out = []
    print("Dataframe sucessfully imported")

    # In[5]:



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

            for idx, row in One_grain.iterrows():

                rr,cc,a = draw.line_aa(row.x_start.astype("uint16"),row.y_start.astype("uint16"),row.x_end.astype("uint16"),row.y_end.astype("uint16"))
                np_img[cc,rr] = (1,1,1)

            mask = flood(np_img, (y_center, x_center,0))
            #        print(str(grain) + " len "+ str(np.count_nonzero(mask)))
            #        print(str(grain) + " len 0 "+ str(np.sum(mask==1)))
            np_img[np_img[:,:,1] !=0] =  [phi1,Phi,phi2]

            if (np.sum(mask==1)<overflood):
                flooded_grains[mask[:,:,1] !=0] = [phi1,Phi,phi2]
                flooded_grains[np_img[:,:,1] !=0] =  [phi1,Phi,phi2]

            else:
                over.append(grain)
                One_grain = One_grain[One_grain["length"]>2]

                start = pd.DataFrame(columns=["x","y"])
                end = pd.DataFrame(columns=["x","y"])
                start[["x","y"]] = One_grain[['x_start','y_start']]
                end[["x","y"]] = One_grain[['x_end','y_end']]
                points = pd.concat([start,end])

                p = points.drop_duplicates()
                p1 = p.to_numpy()

                origin = [x_center,y_center]
                
                sort = sorted(p1, key=clockwiseangle_and_distance)
                a = []
                for b in sort:
                    a.append(tuple((int(b[0]),int(b[1]))))

                cv2.polylines(np_img, np.array([a]), True, (phi1,Phi,phi2), 2)


                mask = flood(np_img, (y_center, x_center,0))
                if(np.sum(mask==1)<overflood):
    #                 cv2.imshow('f',flood_grains)
    #                 cv2.waitKey(0)
    #                 cv2.destroyAllWindows()
                    
                    flooded_grains[mask[:,:,1] !=0] = [phi1,Phi,phi2]
                    flooded_grains[np_img[:,:,1] !=0] =  [phi1,Phi,phi2]
                else:
                    out.append(grain)
                #cv2.putText(flooded_grains, text=str(int(grain)), org=(x_center,y_center),fontFace=2, fontScale=0.2, color=(255,255,255), thickness=1)
  
    print("Flood method done")
    ## PART 2 - SLICE METHOD


    print("Running Slice method")
    width = int(max([max(df.x_end),max(df.x_start)]))+1
    height = int(max([max(df.y_end),max(df.y_start)]))+1

    N = width//4
    M = height//4



    grey_img = cv2.imread(path+ '.jpg', 0)
    grey_img = cv2.resize(grey_img,(width,height),interpolation = cv2.INTER_AREA)

    tiles = [flooded_grains[x:x+M,y:y+N] for x in range(0,flooded_grains.shape[0],M) for y in range(0,flooded_grains.shape[1],N)]

    tiles_grey = [grey_img[x:x+M,y:y+N] for x in range(0,grey_img.shape[0],M) for y in range(0,grey_img.shape[1],N)]

    n_voids = []

    for idx in range(len(tiles_grey)):
        centers, radii, vheight, image, drawing = find_voids_2(tiles_grey[idx])
        n_voids.append([idx,len(centers)])
        io.imsave("../ml_sets/"+ prefix + '_'+ str(idx) + '_' + str(len(centers)) + suffix + '.png',tiles[idx])
        io.imsave("../ml_sets/"+ prefix + '_'+ str(idx) + '_' + str(len(centers)) + 'proof.png',tiles_grey[idx])

    #save n_voids as csv

    print("Divide et Vince Done")


def main():

    
    folder = "../data/"
    file = sys.argv[1]
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
    Divide_et_Vince(df, prefix = file, suffix="void")


    GB_reconstruction(df, prefix = file)

    sample = pd.read_pickle("../output/" + file + "_remake.pkl")

    df_remake = pd.DataFrame(  data = sample, 
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
    Divide_et_Vince(df_remake, prefix = file , suffix="remake")

    
if __name__ == "__main__":
    main()
