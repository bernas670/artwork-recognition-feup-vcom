import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import dl_models
from keras_preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

MODEL_PATH = 'models/incep_4_10-153503.h5'
TEST_DATASET_CSV = "data/multiclass_test.csv"
CLASS_COUNT = 50
IMAGE_FOLDER = "data/images"

# Get model
_, image_dims, preprocessing_func = dl_models.get_inception_v3_transfer(CLASS_COUNT, activation="softmax")

model = tf.keras.models.load_model(MODEL_PATH)

test_df = pd.read_csv(TEST_DATASET_CSV)
test_df['id'] = test_df['id'] + ".png"
test_df['attribute_ids'] = test_df['attribute_ids'].astype(str)
data_generator = ImageDataGenerator(preprocessing_function=preprocessing_func)
test_gen = data_generator.flow_from_dataframe(
  dataframe=test_df,
  directory=IMAGE_FOLDER,
  x_col='id',
  y_col='attribute_ids',
  class_mode='categorical',
  shuffle=False,
  target_size=image_dims,
  batch_size=32,
)
predictions = model.predict(test_gen, verbose=1)

print('Confusion Matrix')

inv_map = {v: k for k, v in test_gen.class_indices.items()}
mapped_classes = list(map(lambda x: int(inv_map[int(x)]), test_gen.classes))
mapped_predictions = list(map(lambda x: int(inv_map[int(x)]), predictions.argmax(axis=1)))
plt.matshow(confusion_matrix(mapped_classes, mapped_predictions, [i for i in range(0,70)]))
plt.xticks([i for i in range(0,80,10)], rotation='vertical')
plt.yticks([i for i in range(0,80,10)])
plt.show()
