import numpy as np
import re
import os
import pandas as pd
import datetime
from pandas.core.frame import DataFrame
import tensorflow as tf
import glob
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import compute_class_weight

import dl_models

img_dir = "data/images"
out_dir = "data/augmented"

def viagra_no_dataset(dataframe, target, image_dims):

  augmented_dataframe = DataFrame(columns=['id', 'attribute_ids'])
  counts = dataframe['attribute_ids'].value_counts().to_dict()

  data_up_generator = ImageDataGenerator(
    # horizontal_flip=True,
    # brightness_range=[15,30],
    zoom_range=[0.8,1.0], #FIXME check this
    # width_shift_range=[-25,25], 
    # height_shift_range=[-25,25], 
    # rotation_range=20,
  )

  data_down_generator = ImageDataGenerator(
  )

  for key, value in counts.items():

    if key != '0':
      continue

    images_from_class = dataframe[dataframe['attribute_ids'] == key]
    
    generator = data_down_generator.flow_from_dataframe(
        dataframe=images_from_class, 
        directory=img_dir, 
        target_size=image_dims,
        x_col='id', 
        y_col='attribute_ids', 
        class_mode='categorical',
        shuffle=True, 
        batch_size=1,
        save_to_dir=out_dir,
        save_format='png',
        save_prefix=key
      )

    for _ in range(min(target, value)):
      generator.next()
    
    # Care for indecsses
    for path_name in glob.glob(os.path.join(out_dir, f'{key}_*.png')):
      match = re.findall(f'/({key}.+).png',path_name)[0]
      augmented_dataframe = augmented_dataframe.append({'id': match, 'attribute_ids': key}, ignore_index = True)

    if value < target:

      generator = data_up_generator.flow_from_dataframe(
        dataframe=images_from_class, 
        directory=img_dir, 
        target_size=image_dims,
        x_col='id', 
        y_col='attribute_ids', 
        class_mode='categorical',
        shuffle=True, 
        batch_size=1,
        save_to_dir=out_dir,
        save_format='png',
        save_prefix=key
      )
      for _ in range(target-value):
        generator.next()
      for path_name in glob.glob(os.path.join(out_dir, f'{key}_*.png')):
        match = re.findall(f'/({key}.+).png',path_name)[0]
        augmented_dataframe = augmented_dataframe.append({'id': match, 'attribute_ids': key}, ignore_index = True)