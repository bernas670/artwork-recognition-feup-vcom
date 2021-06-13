import os
import pickle
import pandas as pd
import tensorflow as tf
import datetime
import math
from keras_preprocessing.image import ImageDataGenerator
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import dl_models
import helper.augment_data as augment_data
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

BATCH_SIZE = 32
CLASS_COUNT = 50
EPOCHS = 100
MODEL_DIR = "models"
MODEL_NAME = "incep_3" + "_" + datetime.datetime.now().strftime("%d-%H%M%S")
TRAIN_DATASET_CSV = "data/multiclass_train.csv"
TEST_DATASET_CSV = "data/multiclass_test.csv"
AUG_DATASET_CSV = "data/aug.csv"
IMAGE_FOLDER = "data/images"
SAVE_MODEL = True
SAVE_HISTORY = True
SAVE_LOGS = False
AUGMENT_DATA = False
AUGMENT_COUNT = 100
AUGMENT_GEN = False

# Metrics
metrics = [
  "accuracy",
  tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
  tf.keras.metrics.Recall(),
  tf.keras.metrics.TruePositives(),
  tf.keras.metrics.TrueNegatives(),
  tf.keras.metrics.FalseNegatives(),
  tf.keras.metrics.FalsePositives(),
  tf.keras.metrics.AUC(),
  # tf.keras.metrics.Accuracy(name="accuracy", dtype=None),
  tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None),
  # tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
  tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_categorical_accuracy", dtype=None)
]

# Tensorboard Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Callbacks
callbacks = [
  tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, f'{MODEL_NAME}.csv'), separator=",", append=False),
  tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0,patience=5,verbose=1,mode="auto",baseline=None,restore_best_weights=True),
  tf.keras.callbacks.ModelCheckpoint(
    save_freq= 'epoch' ,
    monitor = 'val_loss',
    save_best_only=True,
    mode='min',
    verbose=1,
    filepath = os.path.join(MODEL_DIR, f'{MODEL_NAME}.h5',
  ))
]

if SAVE_LOGS:
  LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1, update_freq='batch')
  callbacks.append(tensorboard_callback)  

if __name__ == '__main__':

    # Get dataset csv
  df = pd.read_csv(TRAIN_DATASET_CSV)
  df['id'] = df['id'] + ".png"
  df['attribute_ids'] = df['attribute_ids'].astype(str)

  # Get model
  model, image_dims, preprocessing_func = dl_models.get_inception_v3_transfer(CLASS_COUNT, activation="softmax")
    
  if AUGMENT_DATA: 
    if AUGMENT_GEN:
      aug_df = augment_data.augment_dataset(df, AUGMENT_COUNT, image_dims, src_dir=IMAGE_FOLDER, out_dir=IMAGE_FOLDER, csv_dir=AUG_DATASET_CSV)    # augment dataset
    else:
      aug_df = pd.read_csv(AUG_DATASET_CSV)       

    aug_df['id'] = aug_df['id'] + ".png"
    aug_df['attribute_ids'] = aug_df['attribute_ids'].astype(str)

  test_df = pd.read_csv(TEST_DATASET_CSV)
  test_df['id'] = test_df['id'] + ".png"
  test_df['attribute_ids'] = test_df['attribute_ids'].astype(str)


  # Get val/train set
  size = math.floor(len(df) * 0.2) if AUGMENT_DATA else 0.2
  x_train, x_validation, y_train, y_validation = train_test_split(df['id'], df['attribute_ids'], test_size=size, stratify=df['attribute_ids'])
  train_df = pd.DataFrame({"id": x_train, "attribute_ids": y_train})
  validation_df = pd.DataFrame({"id": x_validation, "attribute_ids": y_validation})
  
  if AUGMENT_DATA:
    train_df = train_df.append(aug_df)

  train_df["attribute_ids"].astype(int).plot.hist(bins=70)
  plt.show()
  # Prepare data
  data_generator = ImageDataGenerator(preprocessing_function=preprocessing_func)

  train_gen = data_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=IMAGE_FOLDER,
    x_col='id',
    y_col='attribute_ids',
    class_mode='categorical',
    shuffle=True,
    target_size=image_dims,
    batch_size=BATCH_SIZE,
  )

  validation_gen = data_generator.flow_from_dataframe(
    dataframe=validation_df,
    directory=IMAGE_FOLDER,
    x_col='id',
    y_col='attribute_ids',
    class_mode='categorical',
    shuffle=True,
    target_size=image_dims,
    batch_size=BATCH_SIZE,
  )

  test_gen = data_generator.flow_from_dataframe(
    dataframe=test_df,
    directory=IMAGE_FOLDER,
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
    callbacks=callbacks 
    )

  # Test model
  test_results = model.evaluate(
      x=test_gen,
      verbose=1,
  )

  # Save model and training history
  if SAVE_MODEL:
    model.save(os.path.join(MODEL_DIR, f'{MODEL_NAME}.h5'))

  if SAVE_HISTORY:
    with open(os.path.join(MODEL_DIR, f'{MODEL_NAME}.history'), "wb") as output_file:
      pickle.dump({"history": history.history, "test": test_results}, output_file)