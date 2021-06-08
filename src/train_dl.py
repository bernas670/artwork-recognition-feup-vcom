import os
import pickle
import pandas as pd
import tensorflow as tf
import datetime
from keras_preprocessing.image import ImageDataGenerator
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import dl_models

BATCH_SIZE = 32
CLASS_COUNT = 2
EPOCHS = 4
MODEL_DIR = "models"
MODEL_NAME = "alex"
TRAINING_SET_CSV = "data_task3/dataset/training_set.csv"
VALIDATION_SET_CSV = "data_task3/dataset/validation_set.csv"
TRAINING_IMAGE_FOLDER = "data_task3/dataset/training_set"
VALIDATION_IMAGE_FOLDER = "data_task3/dataset/validation_set"
SAVE_MODEL = True
SAVE_HISTORY = False
SAVE_LOGS = False

# Metrics

metrics = [
  # tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
  # tf.keras.metrics.Recall(),
  tf.keras.metrics.TruePositives(),
  tf.keras.metrics.TrueNegatives(),
  tf.keras.metrics.FalseNegatives(),
  tf.keras.metrics.FalsePositives(),
  # tf.keras.metrics.AUC(),
  tf.keras.metrics.Accuracy(name="accuracy", dtype=None),
  tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None),
  # tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
  # tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_categorical_accuracy", dtype=None)
]

# Tensorboard Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if SAVE_LOGS:
  LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1, update_freq='batch')
else:
  tensorboard_callback = None

if __name__ == '__main__':

  # Get training dataset csv
  train_df = pd.read_csv(TRAINING_SET_CSV)       
  train_df['id'] = train_df['id'] + ".jpg"
  train_df['attribute_ids'] = train_df['attribute_ids'].astype(str)

  # Get validation dataset csv
  val_df = pd.read_csv(VALIDATION_SET_CSV)      
  val_df['id'] = val_df['id'] + ".jpg"
  val_df['attribute_ids'] = val_df['attribute_ids'].astype(str)

  # Get model
  model, image_dims, preprocessing_func = dl_models.get_paper_net_cam(CLASS_COUNT, activation="softmax")

  # Prepare data
  data_generator = ImageDataGenerator(preprocessing_function=preprocessing_func)

  train_gen = data_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAINING_IMAGE_FOLDER,
    x_col='id',
    y_col='attribute_ids',
    class_mode='categorical',
    shuffle=True,
    target_size=image_dims,
    batch_size=BATCH_SIZE,
  )

  validation_gen = data_generator.flow_from_dataframe(
    dataframe=val_df,
    directory=VALIDATION_IMAGE_FOLDER,
    x_col='id',
    y_col='attribute_ids',
    class_mode='categorical',
    shuffle=True,
    target_size=image_dims,
    batch_size=BATCH_SIZE,
  )

  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=metrics)

  # Train model
  history = model.fit(
    train_gen,
    validation_data=validation_gen,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[tensorboard_callback] if SAVE_LOGS else None
    )

  # Save model and training history

  if SAVE_MODEL:
    model.save(os.path.join(MODEL_DIR, f'{MODEL_NAME}.h5'))

  if SAVE_HISTORY:
    with open(os.path.join(MODEL_DIR, f'{MODEL_NAME}.history'), "wb") as output_file:
      pickle.dump(history.history, output_file)