# imports
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
import tensorflow as tf
import keras

import utils_for_backend as mf

import json
import numpy as np

# initiate the app
app = FastAPI()




# create index
@app.get('/')
def index() :
    return {"message" : "welcome to the SegFormer Testing API"}

# load models
unet_resnet18 = mf.load_unet_model()
segformer = mf.load_segformer_model()


# create predict
@app.post('/predict')
async def predict_mask(img : UploadFile = File(...)) :
    # handle errors
    extension = img.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension :
        raise HTTPException(status_code=400, detail="file should be an image : 'jpg', 'jpeg' or 'png'")
    
    # read image file
    image_bytes = await img.read()
    image = tf.io.decode_image(image_bytes)
    image = keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image, size=(384, 2*384))

    # put image in a tensors (for inference compatibility)
    image = tf.expand_dims(image, axis=0).numpy()

    # use custom functions
    segformer_pred = mf.predict_with_segformer_model(segformer, image)
    unet_resnet18_pred = mf.predict_with_unet_model(unet_resnet18, image)

    return {
        "segformer_pred" : json.dumps(segformer_pred.tolist()),
        "unet_resnet18_pred" : json.dumps(unet_resnet18_pred.tolist())
        }







if __name__ == '__main__' :
    uvicorn.run(app, host="127.0.0.1", port = 8000)

# uvicorn main:app --reload