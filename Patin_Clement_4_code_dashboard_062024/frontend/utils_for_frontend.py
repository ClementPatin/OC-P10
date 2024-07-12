import tensorflow as tf
import json
import requests
import numpy as np
import pandas as pd
import os
os.environ["SM_FRAMEWORK"] = 'tf.keras'
from segmentation_models.metrics import IOUScore
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go



# segmentation macro categories
cats = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']
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




def compute_classes(mask):
    """
    From a ground truth segmentation mask compute the proportion of each class

    Parameters :
    ------------
    mask - np array : ground truth mask (H, W, 3)

    Returns :
    ---------
    classes_tab - DataFrame : with columns "class" and "percent"
    """
    # map each pixel to initial class
    mask_map = np.zeros((mask.shape[0], mask.shape[1], len(cats)), dtype="uint8")
    # Iterate through each color and create a mask
    for category, color in enumerate(cats_colors):
        mask_gt = np.all(mask == color, axis=-1)
        mask_map[:, :, category][mask_gt] = 1

    # count pixels
    pixel_class_count = np.sum(mask_map, axis=(0,1), dtype="float")

    # return pixel_class_count
    
    # compute proportions
    pixel_class_count /= mask.shape[0] * mask.shape[1]

    # put results in a dataframe
    classes_tab = pd.DataFrame()
    classes_tab["class"] = cats
    classes_tab["percent %"] = pixel_class_count * 100

    return classes_tab



def plot_classes(classes_tab) :

    colors = {
        f"{cat}" : f"rgb({c[0]}, {c[1]}, {c[2]})" for cat, c in zip(cats, cats_colors)
    }

    # Create a bar plot
    fig = go.Figure()
    # Add bars
    fig.add_trace(go.Bar(
        x=classes_tab['class'],
        y=classes_tab['percent %'],
        marker_color=[colors[cat] for cat in classes_tab['class']]
    ))

    # Update layout to color x-tick labels
    fig.update_layout(
        yaxis=dict(
            range=[0, 50]  # Set y-axis limits
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=classes_tab['class'],
            ticktext=[f'<span style="color:{colors[cat]}; font-weight:bold;">{cat}</span>' for cat in classes_tab['class']]
        )
    )

    return fig


def IoU_per_class(gt, pred, IoU_col_name):
    """
    From a ground truth segmentation mask and a prediction, already mapped to rgb colors, compute IoU_per_class

    Parameters :
    ------------
    gt - np array : ground truth mask (H, W, 3)
    pred - np array : prediction (H, W, 3)
    IoU_col_name - str : name of the column with IoU per class

    Returns :
    ---------
    IoU_tab - DataFrame : with columns "class" and IoU_col_name
    """
    # map each pixel to initial class
    gt_map = np.zeros((gt.shape[0], gt.shape[1], 1), dtype="uint8")
    pred_map = np.zeros((pred.shape[0], pred.shape[1], 1), dtype="uint8")
    # Iterate through each color and create a mask
    for category, color in enumerate(cats_colors):
        # for ground truth
        mask_gt = np.all(gt == color, axis=-1)
        gt_map[mask_gt] = category
        # for pred
        mask_pred = np.all(pred == color, axis=-1)
        pred_map[mask_pred] = category
    
    # initiate meanIoU metric
    meanIoU = tf.keras.metrics.MeanIoU(num_classes=len(cats))
    # compute confusion matrix
    meanIoU.update_state(y_true=gt_map, y_pred=pred_map)
    cm = meanIoU.get_weights()[0]

    # compute IoU per class
    IoU_classes = cm.diagonal()/(cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal())

    # put results in a dataframe
    IoU_tab = pd.DataFrame(index=range(len(cats)+1))
    IoU_tab["class"] = ["MEAN"] + cats
    IoU_tab[IoU_col_name] = [np.nanmean(IoU_classes)] + list(IoU_classes)

    return IoU_tab



def plot_IoU(IoU_tabs) :

    if len(IoU_tabs) == 2 :

        df = pd.merge(left=IoU_tabs[0], right=IoU_tabs[1], on="class")

        df = df.melt(id_vars="class", value_vars=df.columns[1:], value_name="IoU", var_name="model")

        color = 'model'

    if len(IoU_tabs) == 1 :
        df = IoU_tabs[0]
        df.columns = ["class", "IoU"]

        color = None

    fig = px.bar(
        data_frame=df,
        x = 'class',
        y = 'IoU',
        color = color,
        barmode = "group"
    )

    colors = {
        f"{cat}" : f"rgb({c[0]}, {c[1]}, {c[2]})" for cat, c in zip(cats, cats_colors)
    } | {"MEAN" : "rgb(0, 0, 0)"}


    # Create a bar plot
    # fig = go.Figure()
    # # Add bars
    # fig.add_trace(go.Bar(
    #     x=df['class'],
    #     y=df['percent %'],
    #     marker_color=[colors[cat] for cat in df['class']]
    # ))

    # Update layout to color x-tick labels
    fig.update_layout(
        yaxis=dict(
            range=[0, 1]  # Set y-axis limits
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=df['class'],
            ticktext=[f'<span style="color:{colors[cat]}; font-weight:bold;">{cat}</span>' for cat in df['class']]
        )
    )



    return fig
    



def call_seg_api(image_path, API_URL):
    headers = {"accept" : "application/json"}
    # files = [
    #     ('img', (open(image_path, 'rb'), 'image/png'))
    # ]
    files = {'img' : open(image_path, 'rb')}
    response = requests.post(url = API_URL+"/predict", headers=headers, files=files)

    segformer_pred = json.loads(response.json()["segformer_pred"])
    unet_resnet18_pred = json.loads(response.json()["unet_resnet18_pred"])

    return np.array(segformer_pred), np.array(unet_resnet18_pred)


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






# Re-create Xplique function "plot_attribution" 

def _normalize(image) :
    """
    Normalize an image in [0, 1].

    Parameters
    ----------
    image
        Image to prepare.

    Returns
    -------
    image
        Image ready to be used with matplotlib (in range[0, 1]).
    """
    image = np.array(image, np.float32)

    image -= image.min()
    image /= image.max()

    return image

def _clip_percentile(tensor, percentile) :
    """
    Apply clip according to percentile value (percentile, 100-percentile) of a tensor
    only if percentile is not None.

    Parameters
    ----------
    tensor
        tensor to clip.

    Returns
    -------
    tensor_clipped
        Tensor clipped accordingly to the percentile value.
    """

    assert 0. <= percentile <= 100., "Percentile value should be in [0, 100]"

    if percentile is not None:
        clip_min = np.percentile(tensor, percentile)
        clip_max = np.percentile(tensor, 100. - percentile)
        tensor = np.clip(tensor, clip_min, clip_max)

    return tensor

def _clip_normalize(explanation, clip_percentile = 0.1, absolute_value = False) :
    if absolute_value:
        explanation = np.abs(explanation)

    if clip_percentile:
        explanation = _clip_percentile(explanation, clip_percentile)

    explanation = _normalize(explanation)

    return explanation

def plot_attribution(explanation,
                      image = None,
                      cmap = "jet",
                      alpha = 0.5,
                      clip_percentile = 0.1,
                      absolute_value = False,
                      **plot_kwargs):
    """
    Displays a single explanation and the associated image (if provided).
    Applies a series of pre-processing to facilitate the interpretation of heatmaps.

    Parameters
    ----------
    explanation
        Attribution / heatmap to plot.
    image
        Image associated to the explanations.
    cmap
        Matplotlib color map to apply.
    alpha
        Opacity value for the explanation.
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers  in case of too
        extreme values.
    absolute_value
        Whether an absolute value is applied to the explanations.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if image is not None:
        image = _normalize(image)
        plt.imshow(image)

    if len(explanation.shape) == 4: # images channel are reduced
        explanation = np.mean(explanation, -1)

    explanation = _clip_normalize(explanation, clip_percentile, absolute_value)

    plt.imshow(explanation, cmap=cmap, alpha=alpha, **plot_kwargs)
    plt.axis('off')







def plot_explanation(image, explanation, target, alpha_mask, alpha_explain_plot, output_size=(256, 512)) :
    '''
    plot an images explanation with xplique "plot_attributions" function
    
    parameters :
    ------------
    image - tf tensor
    zone_name - string : name of the area of the image for which we wish to display the explanation
    explanantion - tf tensor : xplique image explanation, from an explainer
    target - tf tensor : xplique target, from xplique segmentation functions
    alpha_mask - float in [0, 1] : for merging images and predicted selected masks, alpha value for the mask
    alpha_explain_plot - float in [0, 1] : for the xplique "plot_attributions" function, opacity value for te explanation
    output_size - tuple of int
    '''
    # add mask to image for visualization
    mask = tf.expand_dims(tf.cast(tf.reduce_any(target != 0, axis=-1), tf.float32), -1)
    image_with_mask = (1.0 - alpha_mask) * image + alpha_mask * mask

    # resize
    explanation_resized = tf.image.resize(explanation, size=output_size, method="bilinear")
    image_with_mask_resized = tf.image.resize(image_with_mask, size=output_size, method="bilinear")

  
    fig = plt.figure(figsize=(14,7))
    # visualize explanation
    plot_attribution(
        explanation_resized, 
        image_with_mask_resized, 
        cmap='jet', 
        alpha=alpha_explain_plot, 
        absolute_value=False, 
        clip_percentile=0.2,
        )
    
    # remove axis and margins
    plt.axis("off")
    plt.margins(0, 0)

    return fig














