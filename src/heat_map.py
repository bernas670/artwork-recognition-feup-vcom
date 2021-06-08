import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
import dl_models
import heat_map_helper
import heat_map_keras
import heat_map_paper

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IN_FOLDER = 'data_task3/tests'
OUT_FOLDER = 'data_task3/results'
MODEL_PATH = 'models/alex.h5'
EXPERIMENT_ID = 'ALEX'
CONV_LAYER_NAME = "conv2d_5"
CLASS_COUNT = 2
LABELS = ['Painting', 'Statue']

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

_, img_size, preprocessing = dl_models.get_paper_net_cam(CLASS_COUNT, 'sigmoid')
preprocessing = preprocessing if preprocessing else lambda x:x

# Remove last layer's softmax
model.layers[-1].activation = None

for image_path in glob.glob(f'{IN_FOLDER}/*'):
    img_array = preprocessing(heat_map_helper.get_img_array(image_path, size=img_size))

    # Get result
    preds = model.predict(img_array)
    preds_sig = tf.sigmoid(preds).numpy()[0]
    label = LABELS[tf.argmax(preds_sig).numpy()]
    print(f'{image_path} - {label}  - {preds_sig}')

    # Create heatmap
    heatmap = heat_map_keras.make_gradcam_heatmap(img_array, model, CONV_LAYER_NAME)

    # Save heatmap and image with heatmap
    plt.matshow(heatmap)
    plt.savefig(f'{OUT_FOLDER}/{EXPERIMENT_ID}_{CONV_LAYER_NAME}_heat_{os.path.basename(image_path)}')
    heat_map_helper.save_and_display_cam(image_path,heatmap, f'{OUT_FOLDER}/{EXPERIMENT_ID}_{CONV_LAYER_NAME}_cam_{os.path.basename(image_path)}')
