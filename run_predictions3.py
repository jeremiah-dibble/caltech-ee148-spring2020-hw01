# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:23:30 2021

@author: Jerem
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:00:58 2021

@author: Jerem
"""

import os
import numpy as np
import json
from PIL import Image
from matplotlib import patches
from matplotlib import pyplot
import matplotlib.image as mpimg

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    scores = []
    top_boxes = []
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    file_num = -1
    for file in anotated_files:
        file_num += 1
        red_light =  np.asarray(Image.open(os.path.join(anotated_path, file)))
        box_height, box_width, box_channels = np.shape(red_light)
        top_boxes = []
       # num_boxes = np.random.randint(1,5) 
        (n_rows,n_cols,n_channels) = np.shape(I)
        num_boxes = int((n_rows*n_cols)/(box_height*box_width))
        
        
    
        for i in range(num_boxes*10):
            tl_row = np.random.randint(n_rows - box_height)
            tl_col = np.random.randint(n_cols - box_width)
            br_row = tl_row + box_height
            br_col = tl_col + box_width
            
            top_boxes.append([tl_row,tl_col,br_row,br_col]) 
        
        light_vector = red_light.reshape((-1,))
        #print(np.shape(light_vector))
        for box in top_boxes:
            sub_image = I[box[0]:box[2],box[1]:box[3]]
    
            score = (np.dot(sub_image.reshape((-1,)), light_vector))
            scores.append(score)
        bounding_boxes += top_boxes
        '''
        END YOUR CODE
        '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes, scores
##################################

# This code is used to generate the examples for Q5
# Continued at the end
###################################################
def show_box(key, boxs):
    index = file_names.index(key)
    I = Image.open(os.path.join(data_path,file_names[index]))
    I = np.asarray(I)
    img = mpimg.imread(data_path+'/'+key)

    for box in boxs:
        box_height = box[2] - box[0]
        box_width = box[3] - box[1]
        figure, ax = pyplot.subplots(1)
        rect = patches.Rectangle((box[0],box[1]),box_width,box_height, edgecolor='r', facecolor="none")
        ax.imshow(img)
        ax.add_patch(rect)
        #img = Image.fromarray(I[box[0]:box[2],box[1]:box[3]])
        
        #img.show()
 ####################################### #      
##########################################
# set the path to the downloaded data: 
data_path = "C:/Users/Jerem/OneDrive - California Institute of Technology/Caltech-/Classes/Spring 2021/EE 148/HW1/RedLights2011_Medium/RedLights2011_Medium"
anotated_path = "C:/Users/Jerem/OneDrive - California Institute of Technology/Caltech-/Classes/Spring 2021/EE 148/HW1/RedLights2011_Medium/Anotated"
# set a path for saving predictions: 
preds_path = 'C:/Users/Jerem/OneDrive - California Institute of Technology/Caltech-/Classes/Spring 2021/EE 148/HW1/predictions/'
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 
anotated_files = sorted(os.listdir(anotated_path))
anotated_files = [f for f in anotated_files if '.jpg' in f] 
# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
scores = {}
all_scores = []
mean = 0
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    mean += np.mean(I)
print(i)    
mean/i
    
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    
    preds[file_names[i]], scores[file_names[i]] = detect_red_light(I)
    all_scores.append(scores[file_names[i]])
    
#red_light =  np.asarray(Image.open(os.path.join(anotated_path, anotated_files[2])))    
#box_height, box_width, box_channels = np.shape(red_light)
cutoff = np.percentile(np.reshape(all_scores,(-1,)), 99.2)
final_preds= {}
high_scores = []
for key in preds:
    #for i in range(len(preds[key])):
    best_guess = scores[key].index(np.max(scores[key]))
    #best_guess = np.sort(scores[keys)[-1]]
    if scores[key][best_guess] > cutoff:
        high_scores.append(preds[key][best_guess])
    final_preds[key] = high_scores
    high_scores = []
            
        
 #   preds[file_names[i]] = detect_red_light(I)
# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)


red_light =  np.asarray(Image.open(os.path.join(anotated_path, anotated_files[0])))
img = Image.fromarray(red_light)
img.show()

        

######################################
# Here is the rest of the code to generate examples for Q5
######################################
i = 0
for p in final_preds:
    i+=1
    show_box(p, final_preds[p])
    if i > 100:
        break