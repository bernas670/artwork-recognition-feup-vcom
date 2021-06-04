import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet50

from tensorflow.python.keras.models import Sequential

def get_self_model(image_shape): 
  nbFilters = 32
  poolSize = (2, 2)
  kernelSize = (3, 3)

  model = Sequential()
  model.add(Conv2D(nbFilters, kernelSize, input_shape=image_shape, activation='relu'))
  model.add(Conv2D(nbFilters, kernelSize, activation='relu'))
  model.add(MaxPooling2D(pool_size=poolSize))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation='softmax'))
  
  return model

def get_resnet_model(input, image_shape):
    
  input = tf.keras.applications.resnet.preprocess_input()
  model = ResNet50()

  
def get_vgg19_model(image_shape, class_count):
  input = tf.keras.applications.vgg19.preprocess_input()

  # model = VGG19(
  #   input_shape=image_shape,
 
  # return model

def get_mobile_model(image_shape, class_count):
  model = MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    input_tensor=None,
    pooling="max",
    weights=None,
    classes=class_count,
    classifier_activation="softmax",
  )

  return model

def get_inception_v3(image_shape, class_count):
  model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling="max",
    classes=class_count,
    classifier_activation="softmax",
  )

  return model

def get_paper_net(image_shape, class_count):
  model = Sequential()
  model.add(Conv2D(96, (11,11), input_shape=image_shape, strides=(4,4), activation='relu'))   #1
  model.add(BatchNormalization())   
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
  model.add(Conv2D(256, (5,5), activation='relu'))  #2
  model.add(BatchNormalization())  # FIXME check paper
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
  model.add(Conv2D(384, (3,3), activation='relu')) #3
  model.add(Conv2D(384, (3,3), activation='relu')) #4 
  model.add(Conv2D(256, (3,3), activation='relu')) #5
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dense(4096, activation='relu'))
  model.add(Dense(class_count, activation='softmax'))

  return model