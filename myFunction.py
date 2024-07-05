# imports
import tensorflow as tf
import keras
import segmentation_models as sm

import albumentations as A

from transformers import AutoImageProcessor

import evaluate

import os

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from xplique.plots import plot_attributions



class generator(keras.utils.Sequence) :
    '''
    data generator for cityscapes dataset :
        - gets every images (and masks) paths
        - splits them using scikit learn "train_test_split" if necessary
        - reads images per batch
        - applies data augmentation if needed
        - applies image preprocessing
        - regroups classes into macro categories using a given dictionnary {macro_class_name : list of classes}
        - put categories in their own channels

        
    parameters :
    ------------
    batch_size - int
    images_path - string : path to the root folder containing images
    masks_path - string : path to the root folder containing masks
    image_size - tuple : (heigh, width)
    which_set - string : name of folder, "train", "val" or "test"
    n_images - int or None : number of images to keep. By default : None (no sampling)
    cats - dict or None : {macro_class_name : list of classes}. By default : None (no classes modifications)
    augmentation - Albumentations transform or None : For image augmentation. by default : None
    backbone - string or None : Name of the model architecture (backbone) that the data generator will used to train. Used for image preprocessing. By default : None
    shuffle - bool : To decide wether or not to shuffle the images between each epoch. By default : False
    split - bool : wether or not to split the images/masks into two sets. By default : False
    split_test_size - float : for scikit learn "train_test_split". By default : 0.1
    split_keep - string : "split_train" or "split_test". To decide which part the generator will use. By default : "split_trian"
    split_rs - int : for scikit learn "train_test_split" parammeter "random_state". Important for keeping the same spliting between two generators. By default : 16

    '''

    def __init__(self, batch_size, images_path, masks_path, image_size, which_set, n_images=None, cats=None, augmentation=None, backbone=None, shuffle=False, split=False, split_test_size = 0.1, split_keep="split_train", split_rs=16) :
        '''
        creates lists of paths (for images and for masks) and handles sampling and splitting
        '''
        self.batch_size = batch_size
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = image_size
        self.which_set = which_set
        self.n_images = n_images
        self.cats = cats
        self.augmentation = augmentation
        self.backbone = backbone
        self.shuffle = shuffle
        self.split = split
        self.split_keep = split_keep
        self.split_rs = split_rs

        # initiate lists to store paths, 1 for images, 1 for masks
        self.images_path_list = []
        self.masks_path_list = []

        # for each folder and each list
        for path, l in zip(
            [self.images_path, self.masks_path],
            [self.images_path_list, self.masks_path_list]
        ) :
            # get set folder
            set_path = os.path.join(path,which_set)
            # put each city folder path in a list
            cities_list = os.listdir(set_path)
            cities_path_list = [os.path.join(set_path,city) for city in cities_list]
            # add images paths contained in each city folder
            # (carefull with mask, take PNG with "labelIds" in its name)
            l.extend([
                    os.path.join(city_path,img_file_name) \
                    for city_path in cities_path_list \
                        for img_file_name in os.listdir(city_path) \
                            if ("leftImg8bit" in img_file_name) or ("labelIds" in img_file_name)
            ])
            # sort
            l.sort()

        # handle n_images
        if n_images :
            self.images_path_list = self.images_path_list[:n_images]
            self.masks_path_list = self.masks_path_list[:n_images]
        
        # handle split if neccessary
        if split :
            im_train, im_test = train_test_split(self.images_path_list, test_size=split_test_size, random_state=split_rs)
            mk_train, mk_test = train_test_split(self.masks_path_list, test_size=split_test_size, random_state=split_rs)
            if split_keep == "split_train" :
                self.images_path_list = im_train
                self.masks_path_list = mk_train
            else :
                self.images_path_list = im_test
                self.masks_path_list = mk_test

        # indexes in an attribute
        self.indexes = np.arange(len(self.images_path_list))
        # apply "on_epoch_end" for shuffling (see below)
        self.on_epoch_end()

    def __len__(self) :
        '''
        Denotes the number of batches per epoch
        '''
        # return int(np.floor(len(self.images_path_list)/self.batch_size))
        return len(self.images_path_list) // self.batch_size
    
    def __getitem__(self, batch_idx) :
        '''
        Creates a batch : loading, augmentation, image processing, labels changing, putting labels in seprated channels and returning batch images and masks
        '''
        # indexes for this batch
        start = batch_idx * self.batch_size
        stop = (batch_idx +1) * self.batch_size
        indexes_for_this_batch = self.indexes[start:stop]
        # initiate lists for this batch
        batch_images = []
        batch_masks = []
        # build batch
        for i in indexes_for_this_batch :
            # load image and mask as arrays 
            image = keras.preprocessing.image.load_img(self.images_path_list[i], target_size=self.image_size)
            image = keras.preprocessing.image.img_to_array(image, dtype="uint8")
            mask = keras.preprocessing.image.load_img(self.masks_path_list[i], color_mode = "grayscale", target_size=self.image_size)
            mask = keras.preprocessing.image.img_to_array(mask)

            # apply augmentation
            if self.augmentation :
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # apply preprocessing, for backbone compatibility
            if self.backbone :
                preprocessor = sm.get_preprocessing(self.backbone)
                preprocessor = A.Lambda(image=preprocessor)
                preprocessor = A.Compose([preprocessor])
                sample = preprocessor(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # simplify categories using self.cats values
            if self.cats :
                for k, list_of_labels in enumerate(self.cats.values()) :
                    mask = np.where(
                        np.isin(
                            mask, 
                            list_of_labels
                            ), 
                        k, 
                        mask
                        ).astype("uint8")

            # cast mask to categorical
            mask = keras.utils.to_categorical(mask, num_classes=len(self.cats), dtype="float32")
            # add to lists
            batch_images.append(image)
            batch_masks.append(mask)

        return np.array(batch_images), np.array(batch_masks)
    
    def on_epoch_end(self):
        '''Callback function to shuffle indexes at each epoch'''
        if self.shuffle :
            self.indexes = np.random.permutation(self.indexes)






def plot_learning_curves(history_dict, list_of_metrics, title="Learning curves", save_path=None):
    """
    Function to plot learning curves from a Keras history.history dictionary.

    Parameters :
    ------------
    history_dict - dict : Dictionary containing training and validation metrics.
    list_of_metrics - list : List of metrics to plot.
    title - str : suptitle
    save_path - string : to save figure. By default, None

    """

    # Extract training and validation metrics
    history_dict_filtered = { k : v for k,v in history_dict.items() if k.split('val_')[-1] in list_of_metrics}

    # Define number of subplots
    n_subplots = len(list_of_metrics)

    # define xticks
    xticks = np.arange(
        1,
        len(
            history_dict_filtered[list_of_metrics[0]]
            )
            +1
    )

    # Create figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=n_subplots, figsize=(15, 6))

    # Plot curves for each metric
    for i, metric in enumerate(list_of_metrics):
        ax = axes[i]
        if metric in history_dict_filtered.keys() :
            ax.plot(xticks, history_dict_filtered[metric], label=metric, color="grey")
        if "val_" + metric in history_dict_filtered.keys() :
            ax.plot(xticks, history_dict_filtered["val_" + metric], label='val_' + metric, color="red")
        ax.set_xticks(xticks)
        ax.set_title(metric)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    # Title
    fig.suptitle(title)

    # Adjust layout
    plt.tight_layout()

    # save
    if save_path is not None :
        plt.savefig(save_path, format = "png")

    # Show plot
    plt.show()









def testModel(model, test_gen, n_images, cats_colors, random_state=16, save_path=None, alpha=0.7) :
    '''
    test a segmentation model. Display the image, the mask and the predicted mask

    parameters :
    ------------
    model - segmentation model
    test_gen - generator custom class instance
    n_images - int
    cats_colors - array-like of shape (num classes, 3) : RGB values for each class
    random_state - int : random seed. By default : 16
    save_path - string : to save figure. By default, None
    alpha - float : when merging mask and image, handle intensity of mask. By default : 0.7
    '''

    # imports 
    import matplotlib.pyplot as plt
    import numpy as np
    # random seed
    np.random.seed(seed=random_state)

    # pick one batch
    some_batch_index = np.random.choice(np.arange(max(1,len(test_gen))), size = 1)[0]
    some_batch = test_gen[some_batch_index]
    # pick n_images idx from this batch
    some_batch_examples_idx = np.random.choice(np.arange(len(some_batch[0])), size = n_images)
    some_images = np.array([some_batch[0][i] for i in some_batch_examples_idx])
    some_masks = np.array([some_batch[1][i] for i in some_batch_examples_idx])

    # get class label for each pixel then put them in a channel
    some_masks = cats_colors[np.argmax(some_masks, axis=-1)]
    # merge with original images 
    some_masks = (alpha * some_masks + (1 - alpha) * some_images).astype(int)

    # predict
    preds = model.predict(some_images)
    # get class label for each pixel then put them in a channel
    preds = cats_colors[np.argmax(preds, axis=-1)]
    # merge with original images
    preds = (alpha * preds + (1 - alpha) * some_images).astype(int)

    # create figure
    fig, axs = plt.subplots(n_images,3,figsize=(14,7*n_images/3))

    # imshow 
    for i in range(n_images) :
        for j, images_array in enumerate([some_images, some_masks, preds]) :
            axs[i,j].imshow(images_array[i])
            axs[i,j].set_axis_off()

    # titles
    axs[0,0].set_title("Images")
    axs[0,1].set_title("Masks")
    axs[0,2].set_title("Predicted masks")
    fig.suptitle("Segmentation model, visualize predictions")

    if save_path is not None :
        plt.savefig(save_path, format = "png")







class generator_for_transformers(keras.utils.Sequence) :
    '''
    data generator for cityscapes dataset :
        - gets every images (and masks) paths
        - splits them using scikit learn "train_test_split" if necessary
        - create keras dataset :
            - load images
            - resize
            - apply data augmentation if needed
            - regroups classes into macro categories using a given dictionnary {macro_class_name : list of classes}
            - normalize (pytorch way, using dataset mean and std)

        
    parameters :
    ------------
    batch_size - int
    images_path - string : path to the root folder containing images
    masks_path - string : path to the root folder containing masks
    image_size - tuple : (heigh, width)
    n_images - int or None : number of images to keep. By default : None (no sampling)
    which_set - string : name of folder, "train", "val" or "test"
    cats - dict or None : {macro_class_name : list of classes}. By default : None (no classes modifications)
    augmentation - Albumentations transform or None : For image augmentation. by default : None
    image_mean - list-like : list of floats with length of the number of channels in the image. The mean of dataset pixels
    image-std - list-like : list of floats with length of the number of channels in the image. The std of dataset pixels
    shuffle - bool : To decide wether or not to shuffle the images between each epoch. By default : False
    split - bool : wether or not to split the images/masks into two sets. By default : False
    split_test_size - float : for scikit learn "train_test_split". By default : 0.1
    split_keep - string : "split_train" or "split_test". To decide which part the generator will use. By default : "split_trian"
    split_rs - int : for scikit learn "train_test_split" parammeter "random_state". Important for keeping the same spliting between two generators. By default : 16

    '''

    def __init__(self, batch_size, images_path, masks_path, image_size, which_set, n_images=None, cats=None, augmentation=None, image_mean=None, image_std=None, shuffle=False, split=False, split_test_size = 0.1, split_keep="split_train", split_rs=16) :
        '''
        creates lists of paths (for images and for masks) and handles sampling and splitting
        '''
        self.batch_size = batch_size
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = image_size
        self.which_set = which_set
        self.n_images = n_images
        self.cats = cats
        self.augmentation = augmentation
        self.image_mean = image_mean
        self.image_std = image_std
        self.shuffle = shuffle
        self.split = split
        self.split_keep = split_keep
        self.split_rs = split_rs

        # initiate lists to store paths, 1 for images, 1 for masks
        self.images_path_list = []
        self.masks_path_list = []

        # for each folder and each list
        for path, l in zip(
            [self.images_path, self.masks_path],
            [self.images_path_list, self.masks_path_list]
        ) :
            # get set folder
            set_path = os.path.join(path,which_set)
            # put each city folder path in a list
            cities_list = os.listdir(set_path)
            cities_path_list = [os.path.join(set_path,city) for city in cities_list]
            # add images paths contained in each city folder
            # (carefull with mask, take PNG with "labelIds" in its name)
            l.extend([
                    os.path.join(city_path,img_file_name) \
                    for city_path in cities_path_list \
                        for img_file_name in os.listdir(city_path) \
                            if ("leftImg8bit" in img_file_name) or ("labelIds" in img_file_name)
            ])
            # sort
            l.sort()

        # # handle n_images
        # if self.n_images :
        #     self.images_path_list = self.images_path_list[:self.n_images]
        #     self.masks_path_list = self.masks_path_list[:self.n_images]
        
        
        # handle split if neccessary
        if split :
            im_train, im_test = train_test_split(self.images_path_list, test_size=split_test_size, random_state=split_rs)
            mk_train, mk_test = train_test_split(self.masks_path_list, test_size=split_test_size, random_state=split_rs)
            if split_keep == "split_train" :
                self.images_path_list = im_train
                self.masks_path_list = mk_train
            else :
                self.images_path_list = im_test
                self.masks_path_list = mk_test


    def load_data(self, image_path, mask_path) :
        '''
        load of prepare for Segformer a data point (one image and one label image)
        '''
        # load image and mask as arrays 
        image = tf.io.read_file(image_path,)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(images=image, size=self.image_size, method="bilinear")
        image = tf.cast(image, tf.uint8)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(images=mask, size=self.image_size, method="nearest")
        mask = tf.cast(mask, tf.uint8)

        # apply augmentation
        if self.augmentation :
            def aug_fn(im, mk) :
                # create tranform
                sample = self.augmentation(image=im, mask=mk)
                # apply and convert to tensor
                im_out, mk_out = tf.cast(sample["image"], tf.float32), tf.cast(sample["mask"], tf.uint8)
                # resize
                im_out, mk_out = tf.image.resize(images=im_out, size=self.image_size, method="bilinear"), tf.image.resize(images=mk_out, size=self.image_size, method="nearest")

                return im_out, mk_out
            
            image, mask = tf.numpy_function(func=aug_fn, inp=[image, mask], Tout = [tf.float32, tf.uint8])
        
        else :
            image = tf.cast(image, tf.float32)

        # simplify categories using self.cats values
        if self.cats :
            def simplify_cats(mk) :
                out_mk = mk.copy()
                for k, list_of_labels in enumerate(self.cats.values()) :
                    out_mk = np.where(
                        np.isin(
                            mk, 
                            list_of_labels
                            ), 
                        k, 
                        out_mk
                        ).astype("float32")
                    
                return out_mk
                
            mask = tf.numpy_function(func=simplify_cats, inp=[mask], Tout=[tf.float32])
            
        # normalize
        image = (image/255 - self.image_mean) / self.image_std

        # transpose 
        image = tf.transpose(image, perm=(2, 0, 1))
                
        return {
            "pixel_values" : image, 
            "labels" : tf.squeeze(mask)
        }

    def create_dataset(self) :
        '''
        Create a dataset from paths
        '''
        # create dataset
        dataset = tf.data.Dataset.from_tensor_slices((self.images_path_list, self.masks_path_list))
        # shuffle if needed
        if self.shuffle == True :
            dataset = dataset.shuffle(10 * self.batch_size, reshuffle_each_iteration=True)
        # handle n_images
        if self.n_images :
            dataset = dataset.take(self.n_images)
        # load images
        dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        # devide in batches
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=False)
        # prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
        







# load "mean_iou" from "evaluate"
metric = evaluate.load("mean_iou")
# create "id2label" 
id2label = {
    0: 'void',
    1: 'flat',
    2: 'construction',
    3: 'object',
    4: 'nature',
    5: 'sky',
    6: 'human',
    7: 'vehicle'}

def compute_metrics(eval_pred):
    '''
    from prediction logits, and labels, compute and return :
        - mean and per-class iou
        - global, mean and per-class accuracy

    parameters :
    ------------
    - eval_pred - tuple of logits and labels tensors
    return :
    --------
    dictionnary of metrics names and values
    '''
    # extract logits and labels
    logits, labels = eval_pred
    # transpose logits to have "channel last" configuration
    logits = tf.transpose(logits, perm=[0, 2, 3, 1])
    # resize logits (Segformer returs L/4 x L/4 resolution)
    # method "bilinear" because these are logits (should be "nearest" for integer labels)
    logits_resized = tf.image.resize(
        logits,
        size=tf.shape(labels)[1:],
        method="bilinear",
    )
    # get pixel class predictions
    pred_labels = tf.argmax(logits_resized, axis=-1)
    # compute metric
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=8,
        ignore_index=-1,
    )
    # remove and extract "pre_category" metrics values
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()
    # put them back, but this time with more specific keys, thansk to "id2label"
    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    # add "val_" prefix and return metrics dict
    return {"val_" + k: v for k, v in metrics.items()}





def testSegformerModel(model, test_dataset, n_images, output_size=(256, 512), cats_colors=None, alpha=0.7) :
    '''
    test a segmentation model on some images and return images, masks, predicted mask in a "channel last" way

    parameters :
    ------------
    model - segformer model
    test_dataset - image, mask keras dataset
    n_images - int
    output_size - tuple of int. By default : (256, 512)
    cats_colors - array-like of shape (num classes, 3) : RGB values for each class. By Default : None
    alpha - float : when merging mask and image, handle intensity of mask. By default : 0.7
    '''

    # pick one batch
    datapoints = list(test_dataset.unbatch().take(n_images).as_numpy_iterator())
    
    ## images
    # extract images
    images = np.array([datapoint['pixel_values'] for datapoint in datapoints])
    # resize
    # first, transpose
    images_out = tf.transpose(images, (0,2,3,1))
    # then resize
    images_out = tf.image.resize(images_out, size=(output_size), method="bilinear").numpy()
    # scale to 255
    images_out = (np.array([(im-im.min())/(im.max()-im.min()) for im in images_out])*255).astype(int)
 
    ## masks
    # extract masks
    masks = np.array([datapoint['labels'] for datapoint in datapoints])
    # resize
    # first, put class values in their own channel
    masks_out = keras.utils.to_categorical(masks)
    # then resize
    masks_out = tf.image.resize(masks_out, size=output_size, method="nearest").numpy()
    # get class label for each pixel then map to cats_colors
    if cats_colors is not None :
        # get class label for each pixel then map to cats_colors
        masks_out = cats_colors[np.argmax(masks_out, axis=-1).astype(int)]
        # merge with original images 
        masks_out = (alpha * masks_out + (1 - alpha) * images_out).astype(int)

    ## preds
    # get predictions and transpose (for "channel first")
    preds = model.predict(images).logits
    # resize
    # first, transpose
    preds = tf.transpose(preds, (0,2,3,1))
    # then resize
    preds_resized = tf.image.resize(preds, size=output_size, method="bilinear").numpy()
    if cats_colors is not None :
        # get class label for each pixel then map to cats_colors
        preds_resized = cats_colors[np.argmax(preds_resized, axis=-1)]
        # merge with original images 
        preds_resized = (alpha * preds_resized + (1 - alpha) * images_out).astype(int)

    return images_out, masks_out, preds_resized







def plotSegformerModel(model, test_dataset, n_images, cats_colors, output_size=(256, 512), save_path=None, alpha=0.7) :
    '''
    test a segmentation model. Display the image, the mask and the predicted mask

    parameters :
    ------------
    model - segmentation model
    test_dataset - images, labels keras dataset
    n_images - int
    cats_colors - array-like of shape (num classes, 3) : RGB values for each class
    output_size - tuple of int
    save_path - string : to save figure. By default, None
    '''

    # imports 
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    images_resized, masks_resized, preds_resized = testSegformerModel(model, test_dataset, n_images, output_size, cats_colors, alpha)

    

    # create figure
    fig, axs = plt.subplots(n_images,3,figsize=(14,7*n_images/3))

    # imshow
    if n_images > 1 :
        for i in range(n_images) :
            for j, images_array in enumerate([images_resized, masks_resized, preds_resized]) :
                axs[i,j].imshow(images_array[i])
                axs[i,j].set_axis_off()
        # titles
        axs[0,0].set_title("Images")
        axs[0,1].set_title("Masks")
        axs[0,2].set_title("Predicted masks")

    else :
        for j, images_array in enumerate([images_resized, masks_resized, preds_resized]) :
            axs[j].imshow(images_array[0])
            axs[j].set_axis_off()
        # titles
        axs[0].set_title("Images")
        axs[1].set_title("Masks")
        axs[2].set_title("Predicted masks")



    fig.suptitle("Segmentation model, visualize predictions")

    if save_path is not None :
        plt.savefig(save_path, format = "png")






class WrapperSegFormer(keras.Model):
    """
    A wrapper class to ensure compatibility between the Xplique library and the Hugging Face 
    TFSegformerForSemanticSegmentation model.

    This class adapts a Hugging Face TFSegformerForSemanticSegmentation model to work seamlessly 
    with the Xplique library by adjusting input and output shapes accordingly. The Xplique library 
    requires models to be subclasses of `keras.Model`.

    Attributes :
    ------------
        model: The TensorFlow Hugging Face TFSegformerForSemanticSegmentation model to be wrapped.
    """

    def __init__(self, TFhuggingfaceModel):
        """
        Initializes the WrapperSegFormer with a given Hugging Face TFSegformerForSemanticSegmentation model.

        Parameter :
        -----------
            TFhuggingfaceModel: A TensorFlow model from the Hugging Face library.
        """
        super(WrapperSegFormer, self).__init__()
        self.model = TFhuggingfaceModel

    def __call__(self, inputs: np.ndarray):
        """
        Invokes the wrapped model on the given inputs and returns the processed outputs.

        This method modifies the input shape to match the requirements of the Hugging Face TFSegformerForSemanticSegmentation model, 
        processes the inputs through the model, and then modifies the output logits to match the desired 
        shape of (n, h, w, nb_classes).

        Parameter :
        -----------
            inputs (np.ndarray): The input data with shape (n, h, w, nb_channels).

        Returns :
        ---------
            tf.Tensor: The processed prediction logits with shape (n, h, w, nb_classes).
        """
        # Modify inputs to match TFSegformerForSemanticSegmentation model
        inputs_tr = tf.transpose(inputs, perm=(0, 3, 1, 2))

        # Use TFSegformerForSemanticSegmentation model's call method and extract logits
        outputs = self.model(inputs_tr).logits

        # Modify outputs logits to match (n, h, w, nb_classes)
        outputs = tf.transpose(outputs, perm=(0, 2, 3, 1))

        # Resize outputs to match input height and width
        # Segformer output resolution is H/4, W/4; resize to H, W
        outputs = tf.image.resize(
            outputs,
            size=(tf.shape(inputs)[1], tf.shape(inputs)[2])
        )

        return outputs






def plot_explanations(explanations, targets, images, alpha_mask, alpha_explain_plot, output_size=(256, 512), save_path=None) :
    '''
    plot an images explanation with xplique "plot_attributions" function
    
    parameters :
    ------------
    explanantions - tf tensor : xplique explanations, from an explainer
    targets - tf tensor : xplique targets, from xplique segmentation functions
    alpha_mask - float in [0, 1] : for merging images and predicted selected masks, alpha value for the mask
    alpha_explain_plot - float in [0, 1] : for the xplique "plot_attributions" function, opacity value for te explanation
    output_size - tuple of int
    save_path - string : to save figure. By default, None
    '''
    # add mask to image for visualization
    masks = tf.expand_dims(tf.cast(tf.reduce_any(targets != 0, axis=-1), tf.float32), -1)
    images_with_mask = (1 - alpha_mask) * images + alpha_mask * masks

    # resize
    explanation_resized = tf.image.resize(explanations, size=output_size, method="bilinear")
    images_with_mask_resized = tf.image.resize(images_with_mask, size=output_size, method="bilinear")

    # visualize explanation
    plot_attributions(
        explanation_resized, 
        images_with_mask_resized, 
        img_size=12., 
        cols=images.shape[0]//2, 
        cmap='jet', 
        alpha=alpha_explain_plot, 
        absolute_value=False, 
        clip_percentile=0.5
        )
    # save
    if save_path is not None :
        plt.savefig(save_path, format = "png")








def predict_with_unet_model(unet_model, input_images, input_masks, cats_colors, alpha=0.7) :
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
    cats_colors - array-like of shape (num classes, 3) : RGB values for each class
    alpha - float : when merging mask and image, handle intensity of mask. By default : 0.7

    returns :
    ---------
    images, masks, preds - tuple of 3 4D arrays

    '''

    # preprocessing
    preprocessor = sm.get_preprocessing('resnet18')
    preprocessor = A.Lambda(image=preprocessor)
    preprocessor = A.Compose([preprocessor])
    sample = preprocessor(image=input_images, mask=input_masks)
    images, masks = sample["image"], sample["mask"]

    # resize
    images = tf.image.resize(images, size=(256, 512), method="bilinear").numpy()
    masks = tf.image.resize(masks, size=(256, 512), method="nearest").numpy()

    # for mask, get class label for each pixel then put them in a channel
    masks = cats_colors[np.argmax(masks, axis=-1)]
    # merge with original images
    masks = (alpha * masks + (1 - alpha) * images).astype('uint8')

    # predict
    preds = unet_model.predict(images)
    # get class label for each pixel then put them in a channel
    preds = cats_colors[np.argmax(preds, axis=-1)]
    # merge with original images
    preds = (alpha * preds + (1 - alpha) * images).astype('uint8')

    return images, masks, preds





def predict_with_segformer_model(segformer_model, input_images, input_masks, cats_colors, alpha=0.7) :
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

    # predict, but first transpose (segformer needs "channel first")
    preds = segformer_model.predict(tf.transpose(images, perm=(0, 3, 1, 2))).logits
    # resize to 256 x 512, but first transpose back
    preds = tf.transpose(preds, (0,2,3,1))
    preds = tf.image.resize(preds, size=(256, 512), method="bilinear").numpy()
    # get class label for each pixel then put them in a channel
    preds = cats_colors[np.argmax(preds, axis=-1)]
    # merge with original images
    preds = (alpha * preds + (1 - alpha) * preds).astype('uint8')

    # resize also images to 256 x 512
    output_images = tf.image.resize(input_images, size=(256, 512), method="bilinear").numpy()
    output_masks = tf.image.resize(input_masks, size=(256, 512), method="nearest").numpy()
    # for mask, get class label for each pixel then put them in a channel
    output_masks = cats_colors[np.argmax(output_masks, axis=-1)]
    # merge with original images
    output_masks = (alpha * output_masks + (1 - alpha) * output_images).astype('uint8')

    return output_images, output_masks, preds














# def prep_y_true_and_y_pred(y_true, y_pred, height, width, verbose=False) :
#     """
#     prepare segformer logits and ground truth tensors for loss or metric

#     Parameters :
#     ------------
#         y_true - 4D tensor : Ground truth labels (B, C, H, W)
#         y_pred - 4D tensor : Predicted labels (logits) (B, C, H, W)
#         height - int : height of y_true
#         width - int : widht of y_true
#         verbose - bool : for printing tensors shapes (for debugging) (carefull : needs "run_eagerly=True" in compile()). By default : False
        
#     Returns :
#     ---------
#         total_loss_value - 1D tenser : Batch losses values.
#     """

#     # print shape
#     if verbose == True :
#         print("--- Loss Verbose ---")
#         print("y_true :", y_true.shape)
#         print("y_pred :", y_pred.shape)
    

#     # transpose
#     # (transformers use "channel first", segmentation models use "channel last")
#     y_pred_transposed = tf.transpose(y_pred, perm=(0,2,3,1))
#     y_true_transposed = tf.transpose(y_true, perm=(0,2,3,1))
#     # print shape and a sample
#     if verbose == True :
#         print("y_pred_transposed :", y_pred_transposed.shape)
#         print(y_pred_transposed[10])
#         print('---------')
#         print("y_true_transposed :", y_true_transposed.shape)
#         print(y_true_transposed[10])
#         print('---------')

#     # # logits --> prediction
#     # y_pred_transposed_argmax = tf.argmax(y_pred_transposed, axis=-1)
#     # y_pred_transposed_prediction = keras.utils.to_categorical(y_pred_transposed_argmax, num_classes=8, dtype="float32")
#     # # print shape and a sample
#     # if verbose == True :
#     #     print("y_pred_transposed_argmax :", y_pred_transposed_argmax.shape)
#     #     print(y_pred_transposed_argmax[10])
#     #     print('---------')
#     #     print("y_pred_transposed_prediction :", y_pred_transposed_prediction.shape)
#     #     print(y_pred_transposed_prediction[10])
#     #     print('---------')
    
#     # Resize y_true to match y_pred shape
#     # (segformer outputs H/4 and W/4 perdictions)
#     # ("nearest" method, we do not want to create unrelevant values)
#     y_true_transposed_resized = tf.image.resize(y_true_transposed, size=(int(height/4),int(width/4)), method="nearest")
#     # print shape and a sample
#     if verbose == True :
#         print("y_true_transposed_resized :", y_true_transposed_resized.shape)
#         print(y_true_transposed_resized[10])
#         print('---------')

#     # return y_true_transposed_resized, y_pred_transposed_prediction
#     return y_true_transposed_resized, y_pred_transposed





# def custom_total_loss_for_segformer(y_true, y_pred, class_weights, height, width, verbose=False):
#     """
#     Custom total loss function (focal loss + dice loss) for transformers' segformer / tensorflow compatibility 
    
#     Parameters :
#     ------------
#         y_true - 4D tensor : Ground truth labels (B, C, H, W)
#         y_pred - 4D tensor : Predicted labels (logits) (B, C, H, W)
#         class_weights - list : weights for dice loss
#         height - int : height of y_true
#         width - int : widht of y_true
#         verbose - bool : for printing tensors shapes (for debugging) (carefull : needs "run_eagerly=True" in compile()). By default : False
        
#     Returns :
#     ---------
#         total_loss_value - 1D tenser : Batch losses values.
#     """

#     # instantiate losses
#     dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
#     focal_loss = sm.losses.CategoricalFocalLoss()

#     # use "prep_y_true_and_y_pred" function
#     y_true_prep, y_pred_prep = prep_y_true_and_y_pred(y_true, y_pred, height, width, verbose)
    
#     # Compute individual losses
#     dice_loss_value = dice_loss(y_true_prep, y_pred_prep)
#     focal_loss_value = focal_loss(y_true_prep, y_pred_prep)
    
#     # Sum the losses
#     total_loss_value = dice_loss_value + focal_loss_value

#     if verbose == True :
#         print("LOSS", total_loss_value)

#     return total_loss_value


# # Define the custom loss function wrapper
# def custom_total_loss_for_segformer_wrapper(class_weights, height, width, verbose=False):
#     """simple wrapper for 'custom_total_loss_for_segformer' function """
#     def loss_fn(y_true, y_pred):
#         return custom_total_loss_for_segformer(y_true, y_pred, class_weights, height, width, verbose)
#     return loss_fn






# def custom_jaccard_metric_for_segformer(y_true, y_pred, height, width, verbose=False) :
#     """
#     Custom jaccard metric function for transformers' segformer / tensorflow compatibility 
    
#     Parameters :
#     ------------
#         y_true - 4D tensor : Ground truth labels (B, C, H, W)
#         y_pred - 4D tensor : Predicted labels  (logits) (B, C, H, W)
#         height - int : height of y_true
#         width - int : widht of y_true
#         verbose - bool : for printing tensors shapes (for debugging) (carefull : needs "run_eagerly=True" in compile()). By default : False
        
#     Returns :
#     ---------
#         metric_value - 1D tenser : Batch metric values.
#     """
#     # instantiate jaccard metric
#     jaccard = sm.metrics.IOUScore(threshold=0.5)

#     # use "prep_y_true_and_y_pred" function
#     y_true_prep, y_pred_prep = prep_y_true_and_y_pred(y_true, y_pred, height, width, verbose)
    
#     # Compute individual metrics
#     metric_value = jaccard(y_true_prep, y_pred_prep)
    
#     return metric_value


   
# # Define the custom metric function wrapper
# def custom_jaccard_metric_for_segformer_wrapper(height, width, verbose=False):
#     """simple wrapper for 'custom_jaccard_metric_for_segformer' function """
#     def metric_jaccard(y_true, y_pred):
#         return custom_jaccard_metric_for_segformer(y_true, y_pred, height, width, verbose)
#     return metric_jaccard



# def custom_dice_metric_for_segformer(y_true, y_pred, height, width, verbose=False) :
#     """
#     Custom dice metric function for transformers' segformer / tensorflow compatibility 
    
#     Parameters :
#     ------------
#         y_true - 4D tensor : Ground truth labels (B, C, H, W)
#         y_pred - 4D tensor : Predicted labels  (logits) (B, C, H, W)
#         height - int : height of y_true
#         width - int : widht of y_true
#         verbose - bool : for printing tensors shapes (for debugging) (carefull : needs "run_eagerly=True" in compile()). By default : False
        
#     Returns :
#     ---------
#         metric_value - 1D tenser : Batch metric values.
#     """
#     # instantiate jaccard metric
#     dice = sm.metrics.FScore()

#     # use "prep_y_true_and_y_pred" function
#     y_true_prep, y_pred_prep = prep_y_true_and_y_pred(y_true, y_pred, height, width, verbose)
    
#     # Compute individual metrics
#     metric_value = dice(y_true_prep, y_pred_prep)
   
#     return metric_value


   
# # Define the custom metric function wrapper
# def custom_dice_metric_for_segformer_wrapper(height, width, verbose=False):
#     """simple wrapper for 'custom_dice_metric_for_segformer' function """
#     def metric_dice(y_true, y_pred):
#         return custom_dice_metric_for_segformer(y_true, y_pred, height, width, verbose)
#     return metric_dice
