import torch
from PIL import Image
import io
import tensorflow as tf
import numpy as np
global graph, model
graph = tf.compat.v1.get_default_graph()
import pickle as pk

from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import img_to_array, array_to_img, load_img
import keras.utils.data_utils

model2 = load_model("backend/models/severity.h5")

with open('backend/models/vgg16_cat_list.pk', 'rb') as f:
    cat_list = pk.load(f)

# detecting severity
def read_input(image_encoded):
    pil_Image = Image.open(io.BytesIO(image_encoded))
    return pil_Image

def prepare_img_256(img_path):
    # urllib.request.urlretrieve(img_path, 'save.jpg')
    img = load_img(img_path, target_size=(256, 256))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) / 255
    return x

def pipe3_sev(img_256, model):
    pred = model.predict(img_256)
    pred_labels = np.argmax(pred, axis=1)
    d = {0: 'minor', 1: 'moderate', 2: 'severe'}
    for key in d.keys():
        if pred_labels[0] == key:
            # print("Result:{} damage".format(d[key]))
            return d[key]

def pipe(img_path):
    img_256 = prepare_img_256(img_path)
    y = pipe3_sev(img_256, model2)
    result = {'severity':y}
    return result    

# detecting bounding boxes
def get_yolov5():
    model = torch.hub.load('backend/yolov5', 'custom', path='backend/models/best.pt', source='local')
    model.conf = 0.5
    return model

def get_image_from_bytes(binary_image, max_size=1024):
    input_image =Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize((
        int(input_image.width * resize_factor),
        int(input_image.height * resize_factor)
    ))
    return resized_image


