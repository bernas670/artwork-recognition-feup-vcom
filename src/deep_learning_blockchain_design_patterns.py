import numpy as np
import os
import pandas as pd
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import compute_class_weight

import dl_models

img_dir = "data/images"
out_dir = "output"

IMAGE_DIMS = (224,224,3)
BATCH_SIZE = 128
CLASS_COUNT = 50

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, update_freq='batch')

if __name__ == '__main__':
  df = pd.read_csv('data/multiclass.csv')
  df['id'] = df['id'] + ".png"
  df['attribute_ids'] = df['attribute_ids'].astype(str)
  data_generator = ImageDataGenerator(
    validation_split=0.2,
    horizontal_flip=True,
    brightness_range=[0.5,1.0],
    zoom_range=[0.8,1.0], #FIXME check this
    width_shift_range=[-25,25], 
    height_shift_range=[-25,25], 
    rotation_range=20,
  )
  
  train_gen = data_generator.flow_from_dataframe(
    dataframe=df, 
    directory=img_dir, 
    x_col='id', 
    y_col='attribute_ids', 
    class_mode='categorical',
    shuffle=True, 
    target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]), 
    batch_size=BATCH_SIZE,
    subset='training'
  )
  
  validation_gen = data_generator.flow_from_dataframe(
    dataframe=df, 
    directory=img_dir, 
    x_col='id', 
    y_col='attribute_ids', 
    class_mode='categorical', 
    target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]), 
    batch_size=BATCH_SIZE,
    subset='validation'
  )

  model = dl_models.get_paper_net(IMAGE_DIMS, CLASS_COUNT)

  model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9),metrics=['accuracy'])
  history = model.fit(
    train_gen, 
    validation_data=validation_gen, 
    epochs=100,
    verbose=1,
    callbacks=[tensorboard_callback],
    class_weight= compute_class_weight('balanced', df['attribute_ids'].tolist())
    )