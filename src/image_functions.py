from PIL import Image, ImageDraw
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random



def draw_bounding_rectangle(coords, img, color='#ff0000', width=5, alpha=1.0):
    """Draws a box around the image_object.
    
    coords is a tuple of bbox corners. 
    top left coords are the first 2 elements. 
    bottom right coords are the last 2 elements
    
    width is the width of the line of the drawn box.
    
    color is a hex color code.
    
    alpha is the opacity - range is 0 to 1."""

    # Check transparency
    if alpha > 1 or alpha < 0: 
        alpha = 1
    
    color_opaque = color + hex(int(alpha * 255))[-2:]
    
    # Draw rectangle
    draw = ImageDraw.Draw(img, 'RGBA')
    
    p1 = coords[0:2]
    p2 = coords[2:4]

    draw.rectangle([p1, p2], outline=color_opaque, width=width)
    
    
def new_bbox(img_object):
    '''
    Returns a dataframe that contains the bounding box coordinates and window size
    for all potential bounding boxes as defined by the size of the image and 
    a hard-coded set of window sizes

    Dataframe is also saved as a CSV file is saved in the same directory as the large maps
    '''
    img_width,  img_height = img_object.size
    overlap = 0.4

    df_list = []
    

    for winsize in [(277 * 0.5), (277 * 0.75), (277), (277 * 1.5), (277 * 2), (277 * 2.5), (277 * 3)]:
        winsize = round(winsize)        
        
        if (winsize > img_width) or (winsize > img_height):
            break            
        
        stride = round(winsize * (1 - overlap))

        # When winsize is odd, we increase by 1 pixel
        if (winsize % 2) != 0:
            winsize+=1
        x_values = np.arange((winsize / 2), (img_width - winsize / 2), stride)
        y_values = np.arange((winsize / 2), (img_height - winsize / 2), stride)
        centroids = [(x, y) for x in x_values for y in y_values]

        coords_list = [coords_bbox(x_ct, y_ct, winsize) for x_ct,y_ct in centroids]

        # Appends new records for current winsize and bbox
        df_list.append(pd.DataFrame([coords_list, [winsize] * len(coords_list)]).T)

    bboxes = pd.concat(df_list, ignore_index=True, axis=0)
    bboxes.columns = ['bbox_bounds', 'winsize']
    bboxes.to_csv(f'map/bbox_{img_width}_{img_height}.csv',index = False)
    return bboxes

def coords_bbox(x_cnt, y_cnt, winsize):
    '''
    x_cnt : x coordinate of center of window
    y_cnt : y coordinate of center of window
    winsize: Size of window (bounding box)

    Returns a tuple of (x-min, y-min, x-max, y-max) of the bounding box
    '''
    return int(x_cnt - winsize / 2), int(y_cnt - winsize / 2), int(x_cnt + winsize / 2), int(y_cnt + winsize / 2)


def new_bbox(img_object, create_new = False):
    '''
    Checks to see if a file containing bounding boxes already exists for the size of the image passed
    as an argument. 
    If already exists, the csv is loaded, if not a fresh file is created.

    create_new =    ignores check of file and creates (and overwrites) csv file of potential 
                        bounding boxes
    '''
    img_width,  img_height = img_object.size
    
    if create_new:
        print('creating new')
        bboxes = new_bbox(img_object)
    else:
        try:
            print('reading')
            bboxes = pd.read_csv(f'map/bbox_{img_width}_{img_height}.csv',
            converters={"bbox_bounds": eval})
        except FileNotFoundError:
            print('no document - creating')
            df_bboxPotentials = new_bbox(img_object)
                
    return bboxes

def class_pred(pred, threshold):
    '''
    pred : predicted probability of all classes
    threshold : predicted probability threshold that needs to be met to be considered a valid prediction

    Returns the predicted class 
    '''
    if max(pred) > threshold :
        return pred.argmax()
    return -1

def img_slice_save(img_list, file):
    '''
    Saves list of images to a specific folder
    Recreates folder on each run.
    listOfImages : list of images to save
    fName : used in the naming convention of the slices
    '''
    slice_path = f"./map/_slices_img/"
    
    if (os.path.exists(slice_path) and os.path.isdir(slice_path)):
        shutil.rmtree(slice_path)
    os.makedirs(slice_path)

    for count, img in enumerate(img_list):
        img.save(''.join([slice_path, str(count),'_',(file.split('/')[-1])]))