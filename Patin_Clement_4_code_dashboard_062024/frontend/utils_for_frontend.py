import tensorflow as tf
import json
import requests
import numpy as np
import os
os.environ["SM_FRAMEWORK"] = 'tf.keras'
from segmentation_models.metrics import IOUScore


def call_seg_api(image_path, API_URL):
    headers = {"accept" : "application/json"}
    # files = [
    #     ('img', (open(image_path, 'rb'), 'image/png'))
    # ]
    files = {'img' : open(image_path, 'rb')}
    response = requests.post(url = API_URL+"/predict", headers=headers, files=files)

    segformer_pred = json.loads(response.json()["segformer_pred"])
    unet_resnet18_pred = json.loads(response.json()["unet_resnet18_pred"])

    return segformer_pred, unet_resnet18_pred


def prep_for_display(input_image, input_mask, input_pred, alpha=0.7) :
    '''
    from a trained segmentation model and from images and their masks :
        - resize
        - prepare
        - predict
        - apply color to each class
    
    parameters :
    ------------
    input_image - 3D array-like, channel last

    returns :
    ---------
    images, masks, preds - tuple of 3 4D arrays

    '''

    # resize
    size = (256, 512)
    image = tf.image.resize(input_image, size=size, method="bilinear").numpy().astype("uint8")
    mask = tf.image.resize(input_mask, size=size, method="nearest").numpy()
    pred = tf.image.resize(input_pred, size=size, method="nearest").numpy()

    # merge masks and preds with original images
    mask = (alpha * mask + (1 - alpha) * image).astype('uint8')
    pred = (alpha * pred + (1 - alpha) * image).astype('uint8')

    return image, mask, pred



# def IOU_per_class(ground_truth, pred) :

