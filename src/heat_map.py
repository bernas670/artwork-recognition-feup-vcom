import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
import dl_models
import task3.heat_map_paper as heat_map_paper
import task3.heat_map_helper as heat_map_helper

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IN_FOLDER = 'data_task3/tests'
OUT_FOLDER = 'data_task3/results'
MODEL_PATH = 'models/vgg_cam_11-161942.h5'
EXPERIMENT_ID = 'VGG'
CONV_LAYER_NAME = "conv2d"
CLASS_COUNT = 2
LABELS = ['Painting', 'Sculpture']

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

_, img_size, preprocessing = dl_models.get_vgg19_cam(CLASS_COUNT, 'sigmoid')
preprocessing = preprocessing if preprocessing else lambda x:x

for image_path in glob.glob(f'{IN_FOLDER}/*'):
    img_array = preprocessing(heat_map_helper.get_img_array(image_path, size=img_size))

    # Get result
    preds = model.predict(img_array)
    label = LABELS[tf.argmax(preds[0]).numpy()]
    print(f'{image_path} - {label}  - {preds}')

    # Create heatmap
    heatmap = heat_map_paper.make_gradcam_heatmap(img_array, model, CONV_LAYER_NAME)
    # Save heatmap and image with heatmap
    x = plt.matshow(heatmap)
    plt.colorbar()
    plt.savefig(f'{OUT_FOLDER}/{EXPERIMENT_ID}_{CONV_LAYER_NAME}_heat_{os.path.basename(image_path)}')
    heat_map_helper.save_and_display_cam(image_path,heatmap, f'{OUT_FOLDER}/{EXPERIMENT_ID}_{CONV_LAYER_NAME}_cam_{os.path.basename(image_path)}')
    heat_map_helper.save_side_by_side(image_path,heatmap,img_size, max(preds[0]), label, f'{OUT_FOLDER}/{EXPERIMENT_ID}_{CONV_LAYER_NAME}_cam_side_{os.path.basename(image_path)}')