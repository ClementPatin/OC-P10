import os
os.environ["SM_FRAMEWORK"] = 'tf.keras'

import segmentation_models as sm
import albumentations as A

import tensorflow as tf
import keras

from transformers import TFAutoModelForSemanticSegmentation

import numpy as np

import gc





# dictionnary macro / micro classes
cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}



# array with RGB colors, one for each macro class
cats_colors = np.array([
        [0, 0, 0],
        [128, 64, 128],
        [70, 70, 70],
        [153, 153, 153],
        [107, 142, 35],
        [70, 130, 180],
        [220, 20, 60],
        [0, 0, 142]
])



def load_unet_model() :
    # input shape
    input_shape = (256, 512, 3)
    # the number of classes is given by the "cats" dictionnary
    classes = 8

    # unet model with resnet backbone
    unet_resnet18 = sm.Unet(
        backbone_name="resnet18",
        input_shape=input_shape,
        classes=classes,
        activation="softmax"
    )

    # load weights
    unet_resnet18.load_weights("models/unet_resnet18.h5")

    return unet_resnet18


def load_segformer_model() :
    # checkpoint
    checkpoint = "nvidia/mit-b1"

    # label2id and idtolabel
    label2id = {label : i for i,label in enumerate(cats.keys())}
    id2label = {i : label for i,label in enumerate(cats.keys())}

    # unet model with resnet backbone
    segformer = TFAutoModelForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        image_size = 384,
        num_labels = len(label2id),
        id2label = id2label,
        label2id = label2id,
    )

    # load weights
    segformer.load_weights("models/segformer.h5")

    return segformer



def predict_with_unet_model(unet_model, input_images) :
    '''
    from a trained segmentation model and from images and their masks :
        - resize
        - prepare
        - predict
        - apply color to each class
    
    parameters :
    ------------
    unet_model - segmentation model
    input_images - 4D array-like, channel last
    input_masks - 4D array-like channel last
    alpha - float : when merging mask and image, handle intensity of mask. By default : 0.7

    returns :
    ---------
    images, masks, preds - tuple of 3 4D arrays

    '''

    # preprocessing
    preprocessor = sm.get_preprocessing('resnet18')
    preprocessor = A.Lambda(image=preprocessor)
    preprocessor = A.Compose([preprocessor])
    sample = preprocessor(image=input_images)#, mask=input_masks)
    # images, masks = sample["image"], sample["mask"]
    images = sample["image"]

    # resize
    images = tf.image.resize(images, size=(256, 512), method="bilinear").numpy().astype("uint8")
    # masks = tf.image.resize(masks, size=(256, 512), method="nearest").numpy()

    # predict
    if images.sum() == 0 :
        preds = images
    else :
        preds = unet_model.predict(images)
        # get class label for each pixel then put them in a channel
        preds = cats_colors[np.argmax(preds, axis=-1)]

    # # merge masks and preds with original images
    # masks = (alpha * masks + (1 - alpha) * images).astype('uint8')
    # preds = (alpha * preds + (1 - alpha) * images).astype('uint8')

    # return images, masks, preds

    del images
    gc.collect()

    return preds





def predict_with_segformer_model(segformer_model, input_images) :
    '''
    from a trained SegFormer model and from images and their masks :
        - resize
        - normalize
        - predict
        - apply color to each class
    
    parameters :
    ------------
    segformer_model - segformer huggingface TFmodel
    input_images - 4D array-like, channel last
    input_masks - 4D array-like channel last
    cats_colors - array-like of shape (num classes, 3) : RGB values for each class
    alpha - float : when merging mask and image, handle intensity of mask. By default : 0.7

    returns :
    ---------
    images, masks, preds - tuple of 3 4D arrays

    '''

    # normalize images, for compatibility with trained segformer
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    images = (input_images/255 - mean) / std

    # resize images to 384 x 384, for compatibility with trained segformer
    images = tf.image.resize(images, size=(384, 384), method="bilinear").numpy()

    # predict
    if np.array(input_images).sum() == 0 :
        preds = tf.image.resize(input_images, size=(256, 512), method="bilinear").numpy()
    else :
        # first transpose (segformer needs "channel first")
        preds = segformer_model.predict(tf.transpose(images, perm=(0, 3, 1, 2))).logits
        # resize to 256 x 512, but first transpose back
        preds = tf.transpose(preds, (0,2,3,1))
        preds = tf.image.resize(preds, size=(256, 512), method="bilinear").numpy()
        # get class label for each pixel then put them in a channel
        preds = cats_colors[np.argmax(preds, axis=-1)]

    # # resize also images to 256 x 512
    # output_images = tf.image.resize(input_images, size=(256, 512), method="bilinear").numpy().astype('uint8')
    # output_masks = tf.image.resize(input_masks, size=(256, 512), method="nearest").numpy()

    # # merge masks and preds with original images
    # output_masks = (alpha * output_masks + (1 - alpha) * output_images).astype('uint8')
    # preds = (alpha * preds + (1 - alpha) * output_images).astype('uint8')

    # return output_images, output_masks, preds
    return preds


