import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from augment_data import viagra_no_dataset
import dl_models

img_dir = "data/images"
out_dir = "output"

IMAGE_DIMS = (224,224,3)
BATCH_SIZE = 32
CLASS_COUNT = 70

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, update_freq='batch')

if __name__ == '__main__':
  df = pd.read_csv('data/multiclass.csv')       # get original dataset
  df['id'] = df['id'] + ".png"
  df['attribute_ids'] = df['attribute_ids'].astype(str)
  
  aug_df = viagra_no_dataset(df, 1, (224,224))    # augment dataset

  aug_df['id'] = aug_df['id'] + ".png"
  aug_df['attribute_ids'] = aug_df['attribute_ids'].astype(str)

  data_generator = ImageDataGenerator(validation_split=0.2)

  train_gen = data_generator.flow_from_dataframe(
    dataframe=aug_df,
    directory="data/augmented",
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

  model = dl_models.get_paper_net(IMAGE_DIMS, 50)

  model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
  history = model.fit(
    train_gen,
    validation_data=validation_gen,
    epochs=100,
    verbose=1,
    callbacks=[tensorboard_callback]
    )