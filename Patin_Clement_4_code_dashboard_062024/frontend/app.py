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

import utils_for_frontend as mf

st.set_page_config(layout="wide")

# API_URL = "https://testapip10.azurewebsites.net"
# API_URL = "http://localhost:8000"

API_URL = os.environ['API_URL']

# images_folder = 'Patin_Clement_4_code_dashboard_062024/frontend/test_datapoints/test_images'
# masks_folder = 'Patin_Clement_4_code_dashboard_062024/frontend/test_datapoints/test_masks'

images_folder = 'test_datapoints/test_images'
masks_folder = 'test_datapoints/test_masks'

images = [os.path.join(images_folder, filename) for filename in os.listdir(images_folder)]
images.sort()
masks = [os.path.join(masks_folder, filename) for filename in os.listdir(masks_folder)]
masks.sort()

images_icons = [Image.open(image_path).resize((128, 128)) for image_path in images]


# Load a sample dataframe
data = {'Category': ['A', 'B', 'C', 'D'], 'Values': [10, 20, 30, 40]}
df = pd.DataFrame(data)

# Streamlit app
st.title('Streamlit App with Tabs')

# Define tabs
tab1, tab2 = st.tabs(['Tab 1: Display Image and Plot', 'Tab 2: Image Transformation'])

with tab1:
    st.header('Tab 1: Display Image and Plot')

    # Left hand side: Image thumbnails
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_image = image_select(
            label="Select an image :",
            images=images_icons,
            index=-1,
            use_container_width=False,
            key="image_select1",
            return_value="index"

        )
        st.image(images[selected_image])

    # Right hand side: Dataframe and plotly graph
    with col2:
        st.subheader('Dataframe and Plotly Graph')
        st.write(df)
        fig = px.bar(df, x='Category', y='Values')
        st.plotly_chart(fig)


with tab2:
    st.header('Tab 2: Image Transformation')

    st.subheader('Choose an Image to Transform')

    
    # images = [Image.open(image_path) for image_path in images]

    selected_image2 = image_select(
        label="Select an image :",
        images=images_icons,
        index=-1,
        use_container_width=False,
        key="image_select2",
        return_value="index"
    )

    image = np.array(Image.open(images[selected_image2])).astype("uint8")
    mask = np.array(Image.open(masks[selected_image2])).astype("uint8")

    segformer_pred, unet_resnet18_pred = mf.call_seg_api(image_path=images[selected_image2], API_URL=API_URL)

    image, mask, preds = mf.prep_for_display(input_image=image, input_mask=mask, input_pred=[segformer_pred[0], unet_resnet18_pred[0]], alpha=0.7)

    col1, col2, col3 = st.columns(3)

    with col1 :
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.write("Image")
        st.image(image)

    with col2 :
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.write("Ground truth :")
        st.image(mask)
    
    with col3 :
        st.write("Unet-Resnet18 :")
        st.image(preds[1])
        st.write("SegFormer :")
        st.image(preds[0])



