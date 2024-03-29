from tensorflow import keras
import matplotlib.cm as cm
import numpy as np
import cv2 as cv

def get_img_array(img_path, size):
    # `img` is a PIL image of size WxH
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (W, H, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch" of size (1, W, H, 3)
    # 
    array = np.expand_dims(array, axis=0)
    return array

def save_and_display_cam(img_path, heatmap, cam_path="cam.jpg", alpha=0.8):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

def save_side_by_side(img_path, heatmap, size, classification, label, cam_path):

    img = keras.preprocessing.image.load_img(img_path)
    img = img.resize(size)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(size)
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.8 + img
    # Write classification
    img = cv.putText(img,f'{label} - {str(round(classification, 2))}',(0,50), cv.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),3)
    image = np.hstack((img, superimposed_img))
    image = keras.preprocessing.image.array_to_img(image)
    image.save(cam_path)