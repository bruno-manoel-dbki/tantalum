# -*- coding: utf-8 -*-
#%%

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from skimage import draw
from matplotlib import pyplot as plt
from matplotlib import cm


'''
OBJECTIVE: 
    Find all boundary interesections in image:
        1 - Select all boundaries interesections
        2 - Extract each intersection considering an area(i) around it.
        3 - Use this areas as reconstruction reference
        4 - Extract an area around void
        5 - Apply Method XXX to reconstruct void based on actual intersections
        
        *This hypotesis considered just the boundary geometry information, rather than grain properties.
'''