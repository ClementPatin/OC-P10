import streamlit as st
from streamlit_image_select import image_select
import pandas as pd
import numpy as np
import plotly.express as px
import json
import requests
from PIL import Image
import io
import os
import sys
import tensorflow as tf
from joblib import load

import utils_for_frontend as mf

# set wide layout
st.set_page_config(layout="wide")
# get API url from environnement variable
API_URL = os.environ['API_URL']

# get examples of Cityscapes images and their masks
images_folder = 'test_datapoints/test_images'
masks_folder = 'test_datapoints/test_masks'
images_paths = [os.path.join(images_folder, filename) for filename in os.listdir(images_folder)]
images_paths.sort()
masks_paths = [os.path.join(masks_folder, filename) for filename in os.listdir(masks_folder)]
masks_paths.sort()
# load them as little images for "image_select" function
images_icons = [Image.open(image_path).resize((128, 128)) for image_path in images_paths]


# app title
st.title('Streamlit App with Tabs')

# Define tabs
tab1, tab2, tab3 = st.tabs(['Dataset', 'SegFormer sv U-net', 'Explainability'])

# first tab - EDA
with tab1:
    st.header('Display CityScapes images examples')

    # show images and allow to select one
    selected_image = image_select(
        label="Select an image :",
        images=images_icons,
        index=-1,
        use_container_width=False,
        key="image_select1",
        return_value="index"

    )
    # load selected image and mask 
    image = np.array(Image.open(images_paths[selected_image])).astype("uint8")
    mask = np.array(Image.open(masks_paths[selected_image])).astype("uint8")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("Image")
        st.image(image)
        st.write("Mask")
        st.image(mask)
    with col2 :
        # display a bar plot of classes percentages
        st.write("Distribution of classes :")
        which_plot = st.radio(label="Which sample :", options=["This image", "Whole dataset"], index=None)

        if which_plot is not None :
            if which_plot == "This image" :
                # compute the percentage of each class
                classes_tab = mf.compute_classes(mask=mask)
            if which_plot ==  "Whole dataset" :
                # Load pixel classes proportion dataframe
                classes_tab = load("classes_proportion/classes_tab.joblib")

            fig = mf.plot_classes(classes_tab=classes_tab)

            st.plotly_chart(fig)


# second tab - SegFormer
with tab2:
    st.header('Test SegFromer, and compare with U-net Resnet18')
   
    # images = [Image.open(image_path) for image_path in images]

    # show images and allow to select one
    selected_image2 = image_select(
        label="Select an image :",
        images=images_icons,
        index=-1,
        use_container_width=False,
        key="image_select2",
        return_value="index"
    )

    # load selected image and mask 
    image = np.array(Image.open(images_paths[selected_image2])).astype("uint8")
    mask = np.array(Image.open(masks_paths[selected_image2])).astype("uint8")

    # predict, using the api
    segformer_pred, unet_resnet18_pred = mf.call_seg_api(image_path=images_paths[selected_image2], API_URL=API_URL)

    # compute IoU per class for each prediction
    # first resized mask
    mask_resized = tf.image.resize(images=mask, size=(256, 512), method="nearest").numpy().astype("uint8")
    # use IoU_per_class custom function
    IoU_tab_segformer = mf.IoU_per_class(gt=mask_resized, pred=segformer_pred[0], IoU_col_name="SegFormer")
    IoU_tab_unet_resnet18 = mf.IoU_per_class(gt=mask_resized, pred=unet_resnet18_pred[0], IoU_col_name="Unet_Resnet18")

    # merge image and mask, and image and preds
    image, mask, preds = mf.prep_for_display(input_image=image, input_mask=mask, input_pred=[segformer_pred[0], unet_resnet18_pred[0]], alpha=0.7)

    # create columns and display image, mask, Unet-Resnet prediction and SegFormer prediction
    col1, col2, col3 = st.columns([3, 3, 5])

    with col1 :
        # display base image
        st.write("Image")
        st.image(image)
        # display ground truth mask
        st.write("Ground truth :")
        st.image(mask)

    with col2 :
        st.write("Unet-Resnet18 :")
        st.image(preds[1])
        st.write("SegFormer :")
        st.image(preds[0])

    with col3 :
        # and IoU per class
        st.write("Metric - Intersection over Union :")
        # display IoU per class for selected image or whole validation dataset
        which_sample = st.radio(label="Which sample :", options=["This image", "Validation dataset"], index=None)
        if which_sample is not None :
            if which_sample == "This image" :
                IoU_tabs = [IoU_tab_unet_resnet18, IoU_tab_segformer]
            if which_sample == "Validation dataset" :
                IoU_tabs = [load("IoU/IoU_Unet_Resnet18.joblib"), load("IoU/IoU_SegFormer.joblib")]
            fig = mf.plot_IoU(IoU_tabs)
            st.plotly_chart(fig)
    




# third tab - Explainability
with tab3 :
    st.header('Explain Segformer predictions')

    # load image examples
    explain_images = load("explainability/images_tf.joblib")
    # # load predictions
    # explain_predictions = load("explainability/tf_outputs.joblib")
    # load targets
    explain_targets_1 = load("explainability/targets_1.joblib")
    explain_targets_2 = load("explainability/targets_2.joblib")
    explain_targets = [explain_targets_1, explain_targets_2]
    #load explanations
    explanations_1 = load("explainability/explanations_1.joblib")
    explanations_2 = load("explainability/explanations_2.joblib")
    explanations = [explanations_1, explanations_2]
    # define explained zones and their index in "explanations_x"
    image_1_zones = {
        "left couple :couple: - Zone :black_large_square:" : 0, 
        "left couple :couple: - Border :black_square_button:" : 3, 
        "left car :blue_car: - Zone :black_large_square:" : 1, 
        "left car :blue_car: - Border :black_square_button:" : 4, 
        "central bike :bike: - Zone :black_large_square:" : 2, 
        "central bike :bike: - Border :black_square_button:" : 5
        }
    image_2_zones = {
        "left car :red_car: - Zone :black_large_square:" : 0, 
        "left car :red_car: - Border :black_square_button:" : 3, 
        "half right car :blue_car: - Zone :black_large_square:" : 1, 
        "half right car :blue_car: - Border :black_square_button:" : 4, 
        "right tree :deciduous_tree: - Zone :black_large_square:" : 2, 
        "right tree :deciduous_tree: - Border :black_square_button:" : 5
        }
    images_zones = [image_1_zones, image_2_zones]

    col1, col2 = st.columns([0.3, 0.7])
    with col1 :
        # show images and allow to select one
        # first, prepare images : normalize, x 255, uint8
        images_for_selector = [explain_images[0].numpy(), explain_images[1].numpy()]
        images_for_selector = [(im - im.min()) / (im.max() - im.min()) for im in images_for_selector]
        images_for_selector = [(im * 255).astype("uint8") for im in images_for_selector]
        # use image_select
        selected_i_image = image_select(
            label="Select an image :",
            images=images_for_selector,
            index=0,
            use_container_width=False,
            key="image_select_explain",
            return_value="index"
            )
        # select the zone to explain
        if selected_i_image == 0 :
            selected_zone = st.radio(
                label="Select a zone to explain :",
                options=list(images_zones[0].keys()),
                index=0
                )
        else :
            selected_zone = st.radio(
                label="Select a zone to explain :",
                options=list(images_zones[1].keys()),
                index=0
                )

        i_zone = images_zones[selected_i_image][selected_zone]
        # select opacity of the mask
        selected_alpha_mask = st.slider(
            label="Select the opacity of the segmentation mask :",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=0.5
        )
        # select opacity of explanation
        selected_alpha_explanation = st.slider(
            label="Select the opacity of the explanation :",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=0.25
        )

    with col2 :
        fig = mf.plot_explanation(
            image=explain_images[selected_i_image],
            explanation=explanations[selected_i_image][i_zone],
            target=explain_targets[selected_i_image][i_zone],
            alpha_mask=selected_alpha_mask, 
            alpha_explain_plot=selected_alpha_explanation, 
            output_size=(256, 512), 
        )

        st.pyplot(fig, use_container_width=True)