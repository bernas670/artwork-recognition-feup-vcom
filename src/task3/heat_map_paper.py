import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def calculateCAM(feature_conv, weight_softmax):
    h, w, nc = feature_conv.shape

    # Multiply each feature map with it's importance for the classification
    cam = np.matmul(feature_conv.numpy().reshape((h*w, nc)), weight_softmax)
    cam = cam.reshape(h, w)

    # Normalize the heatmap
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cam_img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    
    # Create model that fetches predictions and the activations of the last layer
    cam_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    [last_conv_layer_output], preds = cam_model(img_array)

    idx = tf.argmax(preds.numpy()[0]).numpy()

    weights = model.layers[-1].get_weights()[0]
    weights = np.squeeze(weights)
    weights = weights[:,idx]
    CAM = calculateCAM(last_conv_layer_output, weights)
    return CAM