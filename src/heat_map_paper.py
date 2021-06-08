import numpy as np
import tensorflow as tf
import cv2
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    h, w, nc = feature_conv.shape
    cam =  np.dot(weight_softmax,feature_conv.numpy().reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cv2.resize(cam_img, size_upsample)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    [last_conv_layer_output], preds = grad_model(img_array)

    idx = tf.nn.softmax(preds).numpy()
    idx = tf.argmax(idx[0]).numpy()

    weights = model.layers[-1].get_weights()[0]
    weights = np.squeeze(weights)
    weights = weights[:,idx]
    CAM = returnCAM(last_conv_layer_output, [weights], [idx])
    height, width, _ = img_array[0].shape
    heatmap = cv2.resize(CAM,(width, height))

    return heatmap