import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import random
from PIL import Image, ImageDraw
import pandas as pd
import os
import shutil

import numpy as np
np.random.seed(42)

def draw_bounding_rectangle(coords, img_object, color='#ff0000', width=5, alpha=1.0):
    """Draws a box around the image_object.
    
    coords is a tuple of bbox corners. 
    top left coords are the first 2 elements. 
    bottom right coords are the last 2 elements
    
    width is the width of the line of the drawn box.
    
    color is a hex color code.
    
    alpha is the opacity - range is 0 to 1."""

    # Check if transparent
    if alpha > 1 or alpha < 0: 
        alpha = 1
    
    color_opaque = color + hex(int(alpha * 255))[-2:]
    
    # Draw a rectangle
    draw = ImageDraw.Draw(img_object,'RGBA')
    
    p1 = coords[0:2]
    p2 = coords[2:4]

    draw.rectangle([p1, p2], outline=color_opaque, width=width)



    
def new_bboxes(img_object):
    img_width,  img_height = img_object.size
    overlap = 0.4

    df_list = []
    
    for winsize in [(277 * 0.75), (277), (277 * 1.5)]:
        winsize = round(winsize)        
        
        if (winsize > img_width) or (winsize > img_height):
            break            
        
        stride = round(winsize * (1 - overlap))

        # When window size is odd, we increase by 1 pixel
        if (winsize % 2) != 0:
            winsize += 1
        x_values = np.arange((winsize / 2), (img_width - winsize / 2), stride)
        y_values = np.arange((winsize / 2), (img_height - winsize / 2), stride)
        centroids = [(x, y) for x in x_values for y in y_values]

        coords_list = [coords_bbox(x_cnt, y_cnt, winsize) for x_cnt, y_cnt in centroids]

        # Append winsize and bbox records
        df_list.append(pd.DataFrame([coords_list, 
                                    [winsize] * len(coords_list)]).T)

    df_bbox = pd.concat(df_list, 
                        ignore_index=True, 
                        axis=0)
    df_bbox.columns = ['bbox_bounds', 
                       'winsize']
    df_bbox.to_csv(f'./map/bbox_{img_width}_{img_height}.csv', 
                   index=False)
    
    return df_bbox




def coords_bbox(x_cnt, y_cnt, winsize):
    return int(x_cnt - winsize / 2), int(y_cnt - winsize / 2), int(x_cnt + winsize / 2), int(y_cnt + winsize / 2)




def new_bbox(img_object, create_new=False):
    
    img_width,  img_height = img_object.size
    
    if create_new:
        print('creating new')
        df_bbox = new_bboxes(img_object)
    else:
        try:
            print('reading')
            df_bbox = pd.read_csv(f'./map/bbox_{img_width}_{img_height}.csv')
        except FileNotFoundError:
            print('no document - creating')
            df_bbox = new_bboxes(img_object)
            
    return df_bbox




def img_slice(img_list):

    file_name = img_file.name
    slice_path = f"./map/_slices_img/"
    st.write(slice_path)

    if (os.path.exists(slice_path) and os.path.isdir(slice_path)):
        shutil.rmtree(slice_path)
    os.makedirs(slice_path)

    for count, img in enumerate(img_list):
        img.save(''.join([slice_path, str(count), '_', (file_name.split('/')[-1])]))




def class_pred(pred, threshold):
    if max(pred) > threshold :
        return pred.argmax()
    return -1


# ------------------------------------- #
# ----------------App------------------ #
# ------------------------------------- #


color_map = {}
pred_visual = False
img_path = None
disable_button = True
img_annot = None
model = tf.keras.models.load_model('tf_TransferLearning_8class_VGG16.hfpy')
class_names = {'bright dune': 0,
 'crater': 1,
 'dark dune': 2,
 'impact ejecta': 3,
 'other': 4,
 'slope streak': 5,
 'spider': 6,
 'swiss cheese': 7}


with st.sidebar:
    st.header('Landmark Classification on Mars')
    st.write('[Alvaro Mendizabal](https://github.com/alvaromendizabal/Landmark_Classification_on_Mars)')
    st.title('select feature color:')    
    color_map[class_names['bright dune']] = st.color_picker('Bright Dune', '#0096FF')
    color_map[class_names['crater']] = st.color_picker('Crater', '#EE4B2B')
    color_map[class_names['dark dune']] = st.color_picker('Dark Dune', '#FF69B4')
    color_map[class_names['impact ejecta']] = st.color_picker('Impact Ejecta', '#FFEA00')
    color_map[class_names['other']] = st.color_picker('Other', '#AAFF00')
    color_map[class_names['slope streak']] = st.color_picker('Slope Streak', '#E0B0FF')
    color_map[class_names['spider']] = st.color_picker('Spider', '#FFAC1C')
    color_map[class_names['swiss cheese']] = st.color_picker('Swiss Cheese', '#FFF5EE')



st.title("Landmark Classification on Mars")
col1, col2 = st.columns(2)

# Image loader
# imgFile = st.file_uploader("Choose an image file", type="jpg")
img_file = st.file_uploader("select an image", type="jpg")


with col1:
    if img_file:
        img = Image.open(img_file)
        st.image(img, caption='Red Planet')
        disable_button = False
        img_width,  img_height = img.size
        dfbbox = new_bbox(img, create_new=True)
        img_classify = [img.crop(dfbbox.iloc[row]['bbox_bounds']) 
                            for row in range(0,dfbbox.shape[0])]


# Controls. Disabled until image is loaded
# Multiselect toolbar
options = st.multiselect(
     'select features',
     ['Crater', 'Bright Dunes', 'Dark Dune', 'Impact Ejecta',
     'Slope Streak','Spider','Swiss Cheese','Other'],
     ['Crater', 'Bright Dunes', 'Dark Dune', 'Impact Ejecta',
     'Slope Streak','Spider','Swiss Cheese'],
     disabled=disable_button
     )
options = [str.lower(x) for x in options]
class_visual = [v for k, v in class_names.items() if k in options]


# Confidence threshold slider
thresh = st.slider('Confidence Threshold',
value=0.80,
min_value = 0.01, 
max_value= 1.0,
disabled=disable_button)




if st.button('Predict', disabled=disable_button):
    img_slice(img_classify)

    predict_datagen = ImageDataGenerator(rescale=1./255)

    predict_generator = predict_datagen.flow_from_directory(
        './map/',
        target_size = (227,227),
        batch_size=100,
        color_mode='rgb',
        class_mode=None
    )
    predict_generator.reset()

    preds = model.predict(predict_generator)

    dfbbox['pred_class'] = [class_pred(pred, thresh) for pred in preds]

    mask_img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    df_valid = dfbbox[dfbbox['pred_class'].isin(class_visual)][['bbox_bounds', 'pred_class']]

    for idx in range(0, df_valid.shape[0]):        
        draw_bounding_rectangle(df_valid.iloc[idx]['bbox_bounds'],
                            mask_img,
                            color=color_map[df_valid.iloc[idx]['pred_class']]
                            )

    img_annot = img.convert('RGB')

    img_annot.paste(mask_img, (0,0), mask_img)

with col2:
    if img_annot is not None:
        st.image(img_annot)