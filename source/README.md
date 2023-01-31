# Source
This folder contains all learning process and the lastest implementation.


**GBEnergy.py** - Open txt file from data folder and extract Bungle Angle from Dataframe to apply in WIELD and obtain the energy. As output it save a file *gb.png* in source folder. This files is an image where the R channel desbribes the energy of each Grain boundary

**delorean_project.py** - This is an old file with a refactor unfinished. The original file is GB_reconstruction.

**Divide_et_Vince.py** - This script open the datafame from *output* folder (pickle file) or from *data* (txt file) process it filling all grains with color based on ph1, Phi, and phi2 value as RGB channels and split the filled images in 16 tiles each. The output will be in *ml_sets* folder the file name is a combination of *original name* + *tile position* + *number of void in the image*

**find_voids.py** - It is a funcition to process image and detect voids.

**Flood_grains.py** - Flood grains contains the learning process to find a good method to fill the grains based on its form.

**GB_reconstruction.py** - In this script we built the method to recontruct the voids. It uses find_voids adapted method to indicate void position, put away all voids that have more than 4 grain boundaries in it's verge. For each good void we looked for the all boundaries around and create a insert a new boundary to recreate the destroied grain. Also we removed all boundaries attached to the voids. The output is a pickle file of the dataframe with new boundaries and without the voids boundaries.

**main.py** - This file is a refactor of all methods and learning process grouped. If you need to run all development we did, you can run this file. It get the original file, recreate the boundaries and also fill grains and export all tiles of each image. To run this file you need to do:

    python main.py prefix_of_file

Example:

    python main.py 0_001

After all done you'll find the results in ml_sets folder.

**wield_input** - Here you'll find a method to manipulate the entry from dataframe to work with WIELD. This method needs the wield installed.





