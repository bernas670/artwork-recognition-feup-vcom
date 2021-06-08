import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential

def get_mobile_model(class_count, activation='softmax'): 
  model = MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    input_tensor=None,
    pooling="max",
    weights=None,
    classes=class_count,
    classifier_activation=activation,
  )

  return model, (224, 224), tf.keras.applications.mobilenet.preprocess_input 

def get_inception_v3(class_count, activation='softmax'): 
  model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling="max",
    classes=class_count,
    classifier_activation=activation,
  )

  return model, (299, 299), tf.keras.applications.inception_v3.preprocess_input

def get_paper_net(class_count, activation='softmax'): 
  model = Sequential()
  model.add(Conv2D(96, (11,11),input_shape=(224, 224,3), strides=(4,4), activation='relu'))   #1
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
  model.add(Dense(class_count, activation=activation))

  return model, (224, 224), None

def get_paper_net_cam(class_count, activation='softmax'): 
  model = Sequential()
  model.add(Conv2D(96, (11,11),input_shape=(224, 224,3),strides=(4,4), activation='relu'))   #1
  model.add(BatchNormalization())   
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
  model.add(Conv2D(256, (5,5), activation='relu'))  #2
  model.add(BatchNormalization())  # FIXME check paper
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
  model.add(Conv2D(384, (3,3), activation='relu')) #3
  model.add(Conv2D(384, (3,3), activation='relu')) #4 
  model.add(Conv2D(256, (3,3), activation='relu')) #5
  model.add(Conv2D(1024, (3,3), padding='same')) #5
  model.add(GlobalAveragePooling2D())
  model.add(Dense(class_count, activation=activation))

  return model, (224, 224), None


def get_vgg19_cam(class_count, activation='softmax'): 
  model = tf.keras.applications.VGG19(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling="max",
    classes=class_count,
  )
  new_model = tf.keras.models.Sequential()  

  for layer in model.layers[:-3]:
    new_model.add(layer)

  new_model.add(Conv2D(1024, (3,3), padding='same')) #5
  new_model.add(GlobalAveragePooling2D())
  new_model.add(Dense(class_count, activation=activation))
  new_model.summary()

  return new_model, (224, 224), tf.keras.applications.vgg19.preprocess_input